# Audio-to-Audio Translator

Lowest-latency audio-to-audio translation pipeline.

- **Primary**: source-language speech → target-language speech (language1 → language2)
- **Secondary**: any supported language → primary language (others → primary)

## Stack

- **ASR**: `nvidia/nemotron-3.5-asr-streaming-0.6b` — multilingual streaming speech recognition (40 languages, cache-aware FastConformer-RNNT, 80ms–1120ms latency)
- **Translation**: MarianMT (`facebook/marian-mt-*`) — direct language pairs with English intermediary fallback
- **TTS**: `k2-fsa/OmniVoice` — zero-shot, 600+ languages, 24 kHz output, RTF ~0.05.
  - Language goes in `generate(language=...)` (600+ names/codes), NOT in `instruct`.
  - `instruct` accepts a strict English keyword vocabulary only (accent/age/gender/pitch/whisper — e.g. `"male, moderate pitch"`); anything else raises ValueError. `instruct=None` = auto voice.
  - **Voice consistency**: design/auto modes re-sample the speaker on every `generate()` call → the voice drifts sentence to sentence; a fixed `torch.manual_seed` does NOT fix it (verified with a speaker encoder). `backend/tts_engine.py` builds one `VoiceClonePrompt` per target language from a deterministic reference clip and passes it to every call → one stable voice (same-voice cosine 0.83 vs 0.61 for design). `TTS_VOICE` (instruct keyword, default `"female, moderate pitch"`) and `TTS_VOICE_SEED` (default `1234`) control the reference.
  - Qwen3-TTS backup kept as `backend/tts_engine_qwen.py` (transformers 4.57.3, incompatible with OmniVoice's transformers>=5).

## Function

```
translate(language1, language2)
```

Streaming WebSocket endpoint: real-time audio-in → audio-out.
Batch REST endpoint: upload audio file → receive translated audio.

## Streaming ASR (backend/streaming_asr.py)

- Cache-aware encoder: one encoder cache per utterance, fed contiguous feature chunks via `conformer_stream_step` (chunk/pre-encode/shift from `encoder.streaming_cfg`).
- Partial hypotheses pushed live; an utterance closes only after ≥0.6 s of silence (or 10 s max), then is translated and voiced.
- Do not transcribe isolated segments with `model.transcribe()` for the live path: it discards encoder context and degrades the output.
- Translation pairs must exist on HuggingFace. Verified: `opus-mt-ROMANCE-en` (pt→en), `opus-mt-tc-big-en-pt`, `opus-mt-en-fr`, `opus-mt-en-jap`, `opus-mt-tc-big-en-ko`, `opus-mt-tc-big-en-tr`. Absent pairs fall back to the English pivot.

## Languages

ASR supports 40 locales (transcription-ready, broad-coverage, adaptation-ready).
TTS (OmniVoice): 600+ languages, zero-shot; locale ids mapped to language names in `backend/tts_engine.py`.

## Environment

- Single venv: `~/venv/audio2audio-translate-next-py312` (Python 3.12, transformers 5.19.0, torch 2.7.1+cu128).

## Service

```bash
./run.sh                 # port 8741
curl localhost:8741/health
curl -X POST "localhost:8741/api/translate?src_lang=en-US&tgt_lang=fr-FR" -F "file=@in.wav" -o out.wav
```
