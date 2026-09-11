"""Lightweight language detection for TTS chunk routing.

Uses langdetect when available; falls back to assuming English.
Only flags unsupported languages — does not block synthesis.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Kokoro supported language codes
KOKORO_LANG_CODES = {"a", "b", "e", "f", "h", "i", "j", "p", "z"}

# Map langdetect ISO-693-1 codes to Kokoro lang_code
_LANG_MAP = {
    "en": "a",
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "pt": "p",
    "zh": "z",
}


def detect_lang_code(text: str) -> str:
    """Detect language of text and return Kokoro lang_code.

    Returns 'a' (English) as default if detection fails or language is unsupported.
    Logs a warning for unsupported languages.
    """
    try:
        from langdetect import detect as _detect, LangDetectException
        iso = _detect(text)
    except (ImportError, LangDetectException):
        return "a"

    kokoro_code = _LANG_MAP.get(iso)
    if kokoro_code is None:
        logger.warning(
            "Unsupported language '%s' detected in chunk — defaulting to English voice. "
            "Output may be mispronounced.",
            iso,
        )
        return "a"

    return kokoro_code


def is_supported_language(text: str) -> tuple[bool, str]:
    """Check if text's language is supported by Kokoro.

    Returns (is_supported, iso_code).
    """
    try:
        from langdetect import detect as _detect, LangDetectException
        iso = _detect(text)
    except (ImportError, LangDetectException):
        return True, "en"

    return iso in _LANG_MAP, iso
