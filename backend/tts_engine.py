"""
OmniVoice audio synthesis backend (k2-fsa/OmniVoice).

Zero-shot, 600+ languages, output at 24 kHz.

Voice consistency
-----------------
OmniVoice *voice design* / *auto* modes re-sample the speaker timbre on every
``generate()`` call, so the voice drifts from sentence to sentence (measured
with a speaker encoder: ~0.5 cosine between consecutive sentences = different
speakers). A fixed ``torch.manual_seed`` does NOT fix this — the timbre is
resampled per call regardless.

The reliable way to keep one voice across utterances is **voice cloning**: build
a ``VoiceClonePrompt`` once and pass it to every ``generate()`` call (measured
~0.8 cosine across sentences = same speaker; transfers across languages too).
We build one prompt per target language at first use and cache it.

Language vs instruct
--------------------
``language`` (600+ names/codes) controls pronunciation and belongs in the
dedicated argument. ``instruct`` accepts a strict English keyword vocabulary
only (accent/age/gender/pitch/whisper); anything else raises ValueError.
"""
import asyncio
import logging
import os
import threading
import time
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# OmniVoice inference is not reentrant: serialize prompt creation and
# generation so concurrent utterances cannot interleave on the same weights.
_MODEL_LOCK = threading.RLock()

model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000

# Reference voice used to build the clone prompt. Deterministic (fixed seed),
# so the pipeline voice is stable across restarts too.
VOICE_DESIGN = os.environ.get("TTS_VOICE", "female, moderate pitch")
VOICE_REF_SEED = int(os.environ.get("TTS_VOICE_SEED", "1234"))

# Locale id -> OmniVoice language name (passed to generate(language=...)).
LANG_MAP = {
    "en-US": "English", "en-GB": "English", "en-AU": "English", "en-CA": "English",
    "fr-FR": "French", "fr-CA": "French",
    "de-DE": "German",
    "es-ES": "Spanish", "es-US": "Spanish",
    "it-IT": "Italian",
    "pt-BR": "Portuguese", "pt-PT": "Portuguese",
    "zh-CN": "Chinese",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "ru-RU": "Russian",
    "ar-AR": "Arabic",
    "hi-IN": "Hindi",
    "nl-NL": "Dutch",
    "pl-PL": "Polish",
    "tr-TR": "Turkish",
    "sv-SE": "Swedish",
    "cs-CZ": "Czech",
    "da-DK": "Danish",
    "fi-FI": "Finnish",
    "el-GR": "Greek",
    "he-IL": "Hebrew",
    "uk-UA": "Ukrainian",
    "ro-RO": "Romanian",
    "hu-HU": "Hungarian",
    "bg-BG": "Bulgarian",
    "hr-HR": "Croatian",
    "sk-SK": "Slovak",
    "lt-LT": "Lithuanian",
    "lv-LV": "Latvian",
    "et-EE": "Estonian",
    "vi-VN": "Vietnamese",
    "th-TH": "Thai",
    "nb-NO": "Norwegian",
}

# Neutral reference sentence per language: spoken once to bake the timbre into
# the clone prompt. Kept short so prompt creation is fast (~0.3 s, once/lang).
REF_SENTENCES = {
    "English": "Hello, this is a reference voice.",
    "French": "Bonjour, ceci est une voix de référence.",
    "German": "Hallo, dies ist eine Referenzstimme.",
    "Spanish": "Hola, esta es una voz de referencia.",
    "Italian": "Ciao, questa è una voce di riferimento.",
    "Portuguese": "Olá, esta é uma voz de referência.",
    "Russian": "Привет, это эталонный голос.",
    "Chinese": "你好，这是一个参考声音。",
    "Japanese": "こんにちは、これは参照音声です。",
    "Korean": "안녕하세요, 참조 음성입니다.",
}

# Cache: language name -> VoiceClonePrompt
_voice_prompts: dict = {}


def _tts_lang(locale_id: str) -> str:
    return LANG_MAP.get(locale_id, "English")


def _ref_sentence(lang: str) -> str:
    return REF_SENTENCES.get(lang, REF_SENTENCES["English"])


def _get_voice_prompt(lang: str):
    """Build (once) and return the clone prompt for `lang`."""
    with _MODEL_LOCK:
        if lang in _voice_prompts:
            return _voice_prompts[lang]
        ref_text = _ref_sentence(lang)
        torch.manual_seed(VOICE_REF_SEED)
        ref_audio = model.generate(
            text=ref_text,
            language=lang,
            instruct=VOICE_DESIGN,
        )[0]
        prompt = model.create_voice_clone_prompt(
            (torch.as_tensor(np.asarray(ref_audio, dtype=np.float32)), SAMPLE_RATE),
            ref_text=ref_text,
        )
        _voice_prompts[lang] = prompt
        logger.info(f"Voice prompt locked for {lang} (design={VOICE_DESIGN!r}, seed={VOICE_REF_SEED})")
        return prompt


async def load_model(attempts: int = 6, delay_s: float = 10.0):
    """Load OmniVoice, retrying on transient GPU OOM (shared device)."""
    global model
    from omnivoice import OmniVoice

    last_err = None
    for attempt in range(1, attempts + 1):
        logger.info(f"Loading OmniVoice {MODEL_ID} on {device}… (attempt {attempt})")
        try:
            model = OmniVoice.from_pretrained(
                MODEL_ID,
                device_map="cuda:0" if device == "cuda" else None,
                dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            logger.info(
                f"OmniVoice model loaded ✓ (sampling_rate={getattr(model, 'sampling_rate', SAMPLE_RATE)})"
            )
            return model
        except (torch.AcceleratorError, torch.cuda.OutOfMemoryError, RuntimeError) as e:
            msg = str(e).lower()
            if "out of memory" not in msg and "memoryallocation" not in msg:
                raise
            last_err = e
            logger.warning(f"TTS load attempt {attempt}/{attempts} hit GPU OOM; retrying in {delay_s:.0f}s")
            torch.cuda.empty_cache()
            time.sleep(delay_s)
    raise RuntimeError(f"OmniVoice could not be loaded after {attempts} attempts: {last_err}")


async def synthesize(text: str, language: str, speaker: Optional[str] = None):
    """Synthesize `text` in `language` (locale id or language name).

    Returns (float32 mono waveform, sample_rate). The speaker timbre is locked
    to the cached clone prompt, so consecutive utterances share one voice.
    """
    if not text or not text.strip():
        return np.array([], dtype=np.float32), SAMPLE_RATE

    lang = _tts_lang(language)
    logger.info(f"TTS OmniVoice: lang={lang}, text_len={len(text)}")

    loop = asyncio.get_running_loop()

    def _run():
        with _MODEL_LOCK:
            prompt = _get_voice_prompt(lang)
            audios = model.generate(
                text=text.strip(),
                language=lang,
                voice_clone_prompt=prompt,
                num_step=32,   # quality; 16 for lower latency
                speed=1.0,
            )
        wav = audios[0]
        if isinstance(wav, torch.Tensor):
            arr = wav.detach().cpu().float().numpy()
        else:
            arr = np.asarray(wav, dtype=np.float32)
        return arr, SAMPLE_RATE

    return await loop.run_in_executor(None, _run)


def get_supported_languages() -> list:
    return list(LANG_MAP.values())


def get_supported_speakers() -> list:
    return [VOICE_DESIGN]
