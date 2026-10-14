"""
OmniVoice audio synthesis backend.
Zero-shot, 600+ languages, RTF ~0.025, output at 24 kHz.
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

LANG_MAP = {
    "en-US": "English", "en-GB": "English",
    "fr-FR": "French", "fr-CA": "French",
    "de-DE": "German",
    "es-ES": "Spanish", "es-US": "Spanish",
    "it-IT": "Italian",
    "pt-BR": "Portuguese", "pt-PT": "Portuguese",
    "zh-CN": "Chinese",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "ru-RU": "Russian",
}

DEFAULT_SPEAKER = {
    "Chinese": "default_chinese",
    "English": "default_english",
    "Japanese": "default_japanese",
    "Korean": "default_korean",
    "German": "default_german",
    "French": "default_french",
    "Russian": "default_russian",
    "Portuguese": "default_portuguese",
    "Spanish": "default_spanish",
    "Italian": "default_italian",
}


def _tts_lang(locale_id: str) -> str:
    return LANG_MAP.get(locale_id, "English")


async def load_model():
    global model
    logger.info(f"Loading OmniVoice {MODEL_ID} on {device}…")
    from omnivoice import OmniVoice
    model = OmniVoice.from_pretrained(
        MODEL_ID,
        device_map="cuda:0" if device == "cuda" else None,
        dtype=torch.float16,
    )
    logger.info("OmniVoice model loaded ✓")


async def synthesize(text: str, language: str, speaker: Optional[str] = None):
    if not text or not text.strip():
        return np.array([], dtype=np.float32), 24000

    tts_lang = _tts_lang(language)

    # Voice design instruct for language + gender
    instruct = _build_instruct(language, speaker)

    logger.info(f"TTS OmniVoice: lang={tts_lang}, instruct={instruct!r}, text_len={len(text)}")

    loop = asyncio.get_running_loop()

    def _run():
        # Auto Voice / Voice Design hybrid
        audio_list = model.generate(
            text=text.strip(),
            instruct=instruct,
            num_step=32,  # quality; 16 for speed
            speed=1.0,
        )
        # audio_list is list[np.ndarray] at 24kHz
        wavs = audio_list[0] if isinstance(audio_list, list) else audio_list
        if isinstance(wavs, torch.Tensor):
            arr = wavs.detach().cpu().float().numpy()
        else:
            arr = np.array(wavs, dtype=np.float32)
        return arr, 24000

    return await loop.run_in_executor(None, _run)


def _build_instruct(language: str, speaker: Optional[str] = None) -> str:
    """Build voice-design instruct from locale."""
    parts = []
    # Language hint (the model is multilingual, language helps pronunciation)
    tts_lang = _tts_lang(language)
    if tts_lang and tts_lang != "English":
        parts.append(tts_lang.lower())
    # Gender default
    parts.append("neutral pitch")
    return ", ".join(parts) if parts else ""


def get_supported_languages() -> list:
    return list(LANG_MAP.values())


def get_supported_speakers() -> list:
    return ["auto"]  # Auto Voice mode, no fixed speaker IDs
