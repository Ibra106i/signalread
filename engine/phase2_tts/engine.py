"""TTS engine abstraction.

Provides a common interface for Kokoro (primary) and Piper (fallback).
Both engines synthesize text to raw audio arrays; concatenation and
file I/O are handled by the pipeline.
"""
from __future__ import annotations

import logging
import re
import sys
import tempfile
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SynthResult:
    audio: np.ndarray  # float32 mono samples
    sample_rate: int


class TTSEngine(ABC):
    """Base class for TTS engines."""

    @abstractmethod
    def synthesize(self, text: str) -> SynthResult | None:
        """Synthesize text to audio. Returns None on failure."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Kokoro engine
# ---------------------------------------------------------------------------
class KokoroEngine(TTSEngine):
    """Kokoro TTS via KPipeline."""

    def __init__(self, lang_code: str = "a", voice: str = "af_heart", speed: float = 1.0) -> None:
        self.lang_code = lang_code
        self.voice = voice
        self.speed = speed
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code=self.lang_code)

    @property
    def name(self) -> str:
        return "kokoro"

    def synthesize(self, text: str) -> SynthResult | None:
        self._ensure_pipeline()
        try:
            # Kokoro handles its own sentence splitting; we pass our chunk as-is
            all_audio = []
            for _gs, _ps, audio in self._pipeline(
                text, voice=self.voice, speed=self.speed, split_pattern=r'(?<=[.!?])\s+'
            ):
                if audio is not None:
                    all_audio.append(audio.numpy())

            if not all_audio:
                return None

            combined = np.concatenate(all_audio)
            return SynthResult(audio=combined, sample_rate=24000)
        except Exception as exc:
            logger.warning("Kokoro synthesis failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Piper engine
# ---------------------------------------------------------------------------
class PiperEngine(TTSEngine):
    """Piper TTS via piper-tts."""

    def __init__(self, model_path: str | Path | None = None, voice_name: str = "en_US-lessac-medium") -> None:
        self.voice_name = voice_name
        self._model_path = model_path
        self._voice = None

    def _ensure_voice(self):
        if self._voice is None:
            from piper import PiperVoice
            if self._model_path:
                self._voice = PiperVoice.load(str(self._model_path), use_cuda=False)
            else:
                self._voice = PiperVoice.load(self.voice_name, use_cuda=False)

    @property
    def name(self) -> str:
        return "piper"

    def synthesize(self, text: str) -> SynthResult | None:
        self._ensure_voice()
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            with wave.open(tmp_path, "wb") as wav_file:
                self._voice.synthesize(
                    text=text,
                    wav_file=wav_file,
                    sentence_silence=0.2,
                )

            with wave.open(tmp_path, "rb") as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                sr = wav_file.getframerate()
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            Path(tmp_path).unlink(missing_ok=True)
            return SynthResult(audio=audio, sample_rate=sr)
        except Exception as exc:
            logger.warning("Piper synthesis failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_engine(engine_name: str, **kwargs) -> TTSEngine:
    """Create a TTS engine by name."""
    if engine_name == "kokoro":
        return KokoroEngine(**kwargs)
    elif engine_name == "piper":
        return PiperEngine(**kwargs)
    else:
        raise ValueError(f"Unknown TTS engine: {engine_name!r}")
