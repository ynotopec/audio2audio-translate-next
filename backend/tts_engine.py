"""
OmniVoice audio synthesis backend (k2-fsa/OmniVoice).

Zero-shot, 600+ languages, output at 24 kHz, RTF ~0.05 on a modern GPU.

Three modes (see OmniVoice.generate):
  1. voice clone  -> ref_audio + ref_text
  2. voice design -> `instruct` string (strict keyword vocabulary, English only)
  3. auto         -> neither, model picks a voice

Language is passed through the dedicated ``language`` argument, NOT through
``instruct``: the model accepts 600+ language names/codes there. `instruct`
only accepts a fixed English keyword list (accent / age / gender / pitch /
whisper); anything else raises ValueError. Keep the two separate.
"""
import asyncio
import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000

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

# Strict keyword vocabulary accepted by OmniVoice `instruct` (English only).
# Anything outside this set raises ValueError at generate() time.
_VALID_PITCH = {"very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"}

# Voice presets: speaker id -> instruct string (valid keywords only).
VOICE_PRESETS = {
    "default":  "male, moderate pitch",
    "male":     "male, moderate pitch",
    "female":   "female, moderate pitch",
    "whisper":  "whisper, moderate pitch",
}


def _tts_lang(locale_id: str) -> str:
    return LANG_MAP.get(locale_id, "English")


def _build_instruct(speaker: Optional[str]) -> Optional[str]:
    """Map a speaker/preset name to a valid OmniVoice instruct string.

    Returns None to trigger auto-voice mode when the preset is unknown.
    """
    if not speaker:
        return None
    return VOICE_PRESETS.get(str(speaker).lower())


async def load_model():
    global model
    logger.info(f"Loading OmniVoice {MODEL_ID} on {device}…")
    from omnivoice import OmniVoice
    model = OmniVoice.from_pretrained(
        MODEL_ID,
        device_map="cuda:0" if device == "cuda" else None,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    logger.info(f"OmniVoice model loaded ✓ (sampling_rate={getattr(model, 'sampling_rate', SAMPLE_RATE)})")


async def synthesize(text: str, language: str, speaker: Optional[str] = None):
    """Synthesize `text` in `language` (locale id or language name).

    Returns (float32 mono waveform, sample_rate).
    """
    if not text or not text.strip():
        return np.array([], dtype=np.float32), SAMPLE_RATE

    lang = _tts_lang(language)
    instruct = _build_instruct(speaker)

    logger.info(f"TTS OmniVoice: lang={lang}, instruct={instruct!r}, text_len={len(text)}")

    loop = asyncio.get_running_loop()

    def _run():
        gen_kwargs = dict(
            text=text.strip(),
            language=lang,
            num_step=32,       # quality; 16 for lower latency
            speed=1.0,
        )
        # instruct=None => auto voice mode (model picks the voice).
        if instruct:
            gen_kwargs["instruct"] = instruct
        audios = model.generate(**gen_kwargs)
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
    return list(VOICE_PRESETS.keys())
