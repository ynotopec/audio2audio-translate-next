"""
Audio-to-Audio Translator — Full Pipeline
ASR (Nemotron 3.5 streaming) → Text Translation (MarianMT) → TTS (OmniVoice)

Imports are package-relative: the app must be launched as ``backend.main``
(see run.sh / server.py). No sys.path manipulation, no absolute paths.

TTS backend
-----------
``TTS_BACKEND`` selects the synthesis backend. Only ``omnivoice`` is supported
by this environment: the legacy ``qwen`` backend (backend/tts_engine_qwen.py)
requires transformers==4.57.3, which conflicts with omnivoice (transformers>=5).
"""

import asyncio
import io
import json
import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from math import gcd
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from scipy.signal import resample_poly

_TTS_BACKEND = os.environ.get("TTS_BACKEND", "omnivoice").strip().lower()
if _TTS_BACKEND in ("qwen", "qwen3", "qwen3-tts"):
    raise RuntimeError(
        "TTS_BACKEND=qwen needs transformers==4.57.3 and cannot coexist with "
        "omnivoice (transformers>=5). Use a dedicated venv, or leave "
        "TTS_BACKEND unset."
    )
if _TTS_BACKEND != "omnivoice":
    raise RuntimeError(
        f"Unknown TTS_BACKEND={_TTS_BACKEND!r} (expected 'omnivoice')"
    )

from . import tts_engine as _tts_engine  # OmniVoice (zero-shot, 600+ languages)
from .streaming_asr import UtteranceStreamer
from .translation import SUPPORTED_LANGS, init_translation, translate_text

# One decode at a time on the shared ASR model (single GPU): concurrent
# conformer_stream_step calls from several sockets otherwise thrash the device.
ASR_DECODE_LOCK = threading.Lock()

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "50")) * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Globals ─────────────────────────────────────────
nemo_model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

ASR_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
SAMPLE_RATE = 16000

# ── FastAPI app ─────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Heavy models are loaded once, before the first request is served.
    await _startup()
    yield


app = FastAPI(title="Audio2Audio Translator", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Wildcard origin is incompatible with credentials (and this API has no
    # auth), so credentials are explicitly disabled.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


def resample_audio(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample with a polyphase anti-aliasing FIR.

    ``np.interp`` (the previous approach) is linear interpolation with no
    low-pass: decimating 48 kHz -> 16 kHz that way folds high frequencies back
    into the speech band and degrades ASR.
    """
    if sr_in == sr_out or len(audio) == 0:
        return np.asarray(audio, dtype=np.float32)
    g = gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out) // g, int(sr_in) // g
    return resample_poly(np.asarray(audio, dtype=np.float32), up, down).astype(np.float32)



def trim_silence(audio: np.ndarray, sr: int, threshold: float = 1e-3, pad_ms: int = 60) -> np.ndarray:
    """Trim leading/trailing silence from TTS output."""
    if len(audio) == 0:
        return audio
    win = max(1, sr // 100)
    rms = np.sqrt(np.convolve(audio ** 2, np.ones(win) / win, mode="same"))
    voiced = np.where(rms > threshold)[0]
    if len(voiced) == 0:
        return audio
    pad = sr * pad_ms // 1000
    return audio[max(0, voiced[0] - pad): min(len(audio), voiced[-1] + pad)]


def audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


import re
_LANG_TAG_RE = re.compile(r"\s*<([a-z]{2}(?:-[A-Za-z]{2})?)>\s*$")

def parse_transcript(raw: str) -> tuple[str, str | None]:
    """Strip Nemotron language tag `<en-US>`; return (text, detected_locale|None)."""
    m = _LANG_TAG_RE.search(raw)
    text = _LANG_TAG_RE.sub("", raw).strip()
    return (text, m.group(1) if m else None)


def transcribe_nemotron(audio: np.ndarray) -> tuple[str, str | None]:
    """Nemotron ASR in-memory (numpy) — no temp file, no dataloader overhead."""
    if nemo_model is None:
        return "", None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if len(audio) < SAMPLE_RATE // 4:  # <250ms: nothing usable
        return "", None
    try:
        result = nemo_model.transcribe([audio], verbose=False)
        if not result:
            return "", None
        first = result[0]
        if isinstance(first, str):
            raw = first
        elif isinstance(first, dict):
            raw = first.get("text", "") or first.get("pred_text", "")
        else:
            raw = getattr(first, "text", "") or ""
        return parse_transcript(raw)
    except Exception as e:
        logger.error(f"ASR error: {e}")
        return "", None


# ── Batch API ────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "tts_backend": _TTS_BACKEND,
        "asr_model": ASR_MODEL_ID if nemo_model else ASR_MODEL_ID + " (not loaded)",
        "tts_model": _tts_engine.model.__class__.__name__ if _tts_engine.model else "not loaded",
        "device": device,
        "supported_languages": list(SUPPORTED_LANGS.keys()),
    }


@app.post("/api/translate")
async def translate_file(
    file: UploadFile = File(...),
    src_lang: str = Form("auto"),
    tgt_lang: str = Form("fr-FR"),
):
    """Batch: upload audio → translated WAV."""
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
        )

    with tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(file.filename)[1] or ".wav", delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        audio_np, sr = sf.read(tmp_path)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio_np = resample_audio(audio_np, sr, SAMPLE_RATE)

        # ASR with Nemotron (auto-detected language tag available)
        logger.info(f"ASR: {src_lang} → translate → {tgt_lang}")
        transcript, detected = transcribe_nemotron(audio_np)
        effective_src = src_lang if src_lang != "auto" else (detected or "en-US")
        logger.info(f"Transcript: {transcript} (detected: {detected})")

        # Translate
        translated = translate_text(transcript, effective_src, tgt_lang)
        logger.info(f"Translated: {translated}")

        # TTS
        if _tts_engine.model is None:
            raise HTTPException(status_code=503, detail="TTS model not loaded")
        wavs, tts_sr = await _tts_engine.synthesize(translated, tgt_lang)
        wav_bytes = audio_to_wav_bytes(trim_silence(np.asarray(wavs, dtype=np.float32), tts_sr), tts_sr)

        return Response(content=wav_bytes, media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Batch translate error: {e}")
        return Response(content='{"detail":"internal error"}', status_code=500,
                        media_type="application/json")
    finally:
        os.unlink(tmp_path)


# ── Streaming WebSocket ──────────────────────────────
@app.websocket("/ws/translate")
async def ws_translate(ws: WebSocket):
    """
    Streaming audio→text→audio.
      1. JSON config: {src_lang, tgt_lang, latency_ms}
      2. Binary PCM from mic
      ← JSON {"type":"transcript","text":"..."}
      ← JSON {"type":"translated","text":"..."}
      ← Binary WAV (TTS)
    """
    await ws.accept()
    logger.info("Client connected to translation WS")

    config = {"src_lang": "auto", "tgt_lang": "fr-FR", "sample_rate": 16000}
    audio_chunks: list[np.ndarray] = []
    buffer_lock = threading.Lock()
    stop_event = threading.Event()

    async def _flush_and_respond(text: str, detected: str | None):
        effective_src = config["src_lang"] if config["src_lang"] != "auto" else (detected or "en-US")
        translated = translate_text(text, effective_src, config["tgt_lang"])
        await ws.send_json({"type": "transcript", "text": text})
        await ws.send_json({"type": "translated", "text": translated})
        if _tts_engine.model is None:
            logger.error("TTS model not loaded; audio for this utterance is dropped")
            await ws.send_json({"type": "error", "detail": "TTS model not loaded"})
            return
        wavs, sr = await _tts_engine.synthesize(translated, config["tgt_lang"])
        wav_bytes = audio_to_wav_bytes(trim_silence(np.asarray(wavs, dtype=np.float32), sr), sr)
        await ws.send_bytes(wav_bytes)

    async def _send_partial(text: str):
        await ws.send_json({"type": "transcript", "text": text})

    def _asr_thread():
        """
        Cache-aware streaming ASR (nemotron-3.5-asr-streaming). One encoder cache
        per utterance; partial hypotheses are pushed live; an utterance is closed
        only on a real pause (see backend/streaming_asr.py), then translated + voiced.
        """
        if nemo_model is None:
            return
        streamer = UtteranceStreamer(nemo_model)

        def _emit(events):
            for kind, raw in events:
                clean, detected = parse_transcript(raw)
                if not clean:
                    continue
                if kind == "partial":
                    asyncio.run_coroutine_threadsafe(_send_partial(clean), loop)
                else:
                    # Queue finals instead of firing them concurrently: a single
                    # consumer preserves the audio ordering (TTS is slow, so
                    # concurrent coroutines used to answer out of order).
                    loop.call_soon_threadsafe(finals_q.put_nowait, (clean, detected))

        try:
            while not stop_event.is_set():
                with buffer_lock:
                    new = np.concatenate(audio_chunks) if audio_chunks else None
                    audio_chunks.clear()
                if new is None:
                    stop_event.wait(0.02)
                    continue
                with ASR_DECODE_LOCK:
                    events = streamer.push(new)
                _emit(events)
            _emit(streamer._finalize())  # close the open utterance on stop
        except Exception as e:
            logger.error(f"ASR thread error: {e}")

    finals_q = asyncio.Queue()

    async def _finals_worker():
        """Serialize final utterances: translate + synthesize strictly in order."""
        while True:
            item = await finals_q.get()
            if item is None:
                return
            text, detected = item
            try:
                await _flush_and_respond(text, detected)
            except Exception as e:
                logger.error(f"Final response error: {e}")

    loop = asyncio.get_running_loop()
    finals_task = asyncio.create_task(_finals_worker())
    asr_thread = threading.Thread(target=_asr_thread, daemon=True)
    asr_thread.start()

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                logger.info("Client disconnected")
                break
            if msg["type"] != "websocket.receive":
                continue

            if "text" in msg and msg["text"] is not None:
                try:
                    cfg = json.loads(msg["text"])
                except (json.JSONDecodeError, TypeError):
                    continue

                if cfg.get("command") == "stop":
                    stop_event.set()
                    break

                if "src_lang" in cfg:
                    config["src_lang"] = cfg["src_lang"]
                if "tgt_lang" in cfg:
                    config["tgt_lang"] = cfg["tgt_lang"]
                if "sample_rate" in cfg:
                    try:
                        config["sample_rate"] = float(cfg["sample_rate"])
                    except (TypeError, ValueError):
                        pass
                    logger.info(f"Client configured: sr={config['sample_rate']} src={config['src_lang']} tgt={config['tgt_lang']}")
                continue

            if "bytes" in msg and msg["bytes"]:
                with buffer_lock:
                    arr = np.frombuffer(msg["bytes"], dtype=np.int16).astype(np.float32) / 32767.0
                    src_sr = config.get("sample_rate", 16000)
                    if src_sr and abs(src_sr - SAMPLE_RATE) > 1:
                        # resample client rate → pipeline rate (16k), anti-aliased
                        arr = resample_audio(arr, int(round(src_sr)), SAMPLE_RATE)
                    audio_chunks.append(arr)
                continue

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stop_event.set()
        # Let the ASR thread flush its open utterance into finals_q...
        asr_thread.join(timeout=15)
        # ...then let the worker send the remaining audio before closing.
        try:
            # Schedule the sentinel through the loop too: the ASR thread queues
            # finals with call_soon_threadsafe, so a direct put_nowait(None)
            # would overtake (and drop) the last utterance's audio.
            loop.call_soon_threadsafe(finals_q.put_nowait, None)
            await asyncio.wait_for(finals_task, timeout=15)
        except Exception:
            finals_task.cancel()


# ── Startup ────────────────────────────────────────
def _load_asr(attempts: int = 6, delay_s: float = 10.0):
    """Load the ASR model on CUDA, retrying on transient OOM.

    This host shares one GPU with other inference services: a model load can
    fail with cudaErrorMemoryAllocation while a neighbour is spiking, then
    succeed a few seconds later. Failing the whole startup on that is wrong.
    """
    from nemo.collections.asr.models import EncDecRNNTBPEModel

    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            return EncDecRNNTBPEModel.from_pretrained(
                model_name=ASR_MODEL_ID,
                map_location=device,
            ).eval()
        except (torch.AcceleratorError, torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e).lower()
            if "out of memory" not in msg and "memoryallocation" not in msg:
                raise
            last_err = e
            logger.warning(
                f"ASR load attempt {attempt}/{attempts} hit GPU OOM "
                f"({torch.cuda.mem_get_info()[0] / 1e9:.1f} GB free); retrying in {delay_s:.0f}s"
            )
            torch.cuda.empty_cache()
            time.sleep(delay_s)
    raise RuntimeError(f"ASR model could not be loaded after {attempts} attempts: {last_err}")


async def _startup():
    global nemo_model

    logger.info(f"Loading {ASR_MODEL_ID} on {device}…")
    nemo_model = _load_asr()
    logger.info(f"Nemotron ASR loaded ✓ on {device}")

    # Prompt-mode fix: fall back to language-agnostic 'auto' prompt when the
    # audio cut carries no language supervision (batch files, mic chunks).
    # This patches a private NeMo hook, so it is best-effort: a NeMo upgrade
    # must degrade the prompt choice, not prevent the server from starting.
    try:
        import nemo.collections.asr.data.audio_to_text_lhotse_prompt_index as _pidx

        _cls = _pidx.LhotseSpeechToTextBpeDatasetWithPromptIndex

        def _get_prompt_fallback(self, cut):
            mode = self._get_prompt_mode(cut)
            lang = getattr(cut.supervisions[0], "language", None)
            if mode == "langID" and lang:
                try:
                    return self._get_prompt_index(lang)
                except ValueError:
                    pass
            return self.auto_index

        _cls._get_prompt_index_for_cut = _get_prompt_fallback
        logger.info("Nemotron prompt-mode fallback (auto) installed ✓")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Nemotron prompt-mode fallback skipped ({e})")

    init_translation()
    logger.info(f"TTS backend: {_TTS_BACKEND}")
    await _tts_engine.load_model()