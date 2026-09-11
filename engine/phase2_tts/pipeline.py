"""TTS synthesis pipeline.

Orchestrates chunking, chapter detection, synthesis, concatenation,
and manifest generation. Handles errors per-chunk without aborting chapters.
"""
from __future__ import annotations

import json
import logging
import struct
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_blocks
from .chapters import Chapter, detect_chapters
from .engine import TTSEngine, SynthResult

logger = logging.getLogger(__name__)

_SILENCE_DURATION_S = 0.5
_DEFAULT_SAMPLE_RATE = 24000


@dataclass
class ManifestEntry:
    chunk_index: int
    start_time: float
    end_time: float
    source_block_index: int
    chapter_id: str


@dataclass
class PipelineResult:
    chapters_completed: int
    total_chunks: int
    successful_chunks: int
    failed_chunks: int
    manifest: list[ManifestEntry]
    phase1_degraded_count: int


def _make_silence(duration_s: float, sample_rate: int) -> np.ndarray:
    """Generate a silent audio array at the given sample rate."""
    n_samples = int(duration_s * sample_rate)
    return np.zeros(n_samples, dtype=np.float32)


def _concat_wavs(wav_paths: list[Path], output_path: Path, sample_rate: int) -> None:
    """Concatenate multiple WAV files into one, writing incrementally."""
    with wave.open(str(output_path), "wb") as out_wav:
        out_wav.setnchannels(1)
        out_wav.setsampwidth(2)  # 16-bit
        out_wav.setframerate(sample_rate)

        for wp in wav_paths:
            with wave.open(str(wp), "rb") as in_wav:
                frames = in_wav.readframes(in_wav.getnframes())
                out_wav.writeframes(frames)


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 audio array to WAV-format bytes."""
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    raw = audio_int16.tobytes()

    data_size = len(raw)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,           # chunk size
        1,            # PCM format
        1,            # mono
        sample_rate,
        sample_rate * 2,  # byte rate
        2,            # block align
        16,           # bits per sample
        b"data",
        data_size,
    )
    return header + raw


def run_pipeline(
    phase1_data: dict,
    engine: TTSEngine,
    output_dir: Path,
    voice: str | None = None,
) -> PipelineResult:
    """Run the full TTS pipeline on a Phase 1 output dict.

    Returns a PipelineResult with stats and manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = phase1_data.get("blocks", [])

    # --- Log degraded Phase 1 blocks ---
    degraded_sources = {"error_rate_limit", "error", "default_skip"}
    degraded_body_lost = sum(
        1 for b in blocks
        if b.get("resolution") == "skip"
        and b.get("classification_source") in degraded_sources
    )
    if degraded_body_lost > 0:
        logger.warning(
            "Phase 1 degradation: %d block(s) were defaulted to skip due to "
            "LLM errors — actual content may be higher than what will be synthesized.",
            degraded_body_lost,
        )

    # --- Filter to body blocks, preserving original indices ---
    body_blocks = []
    for i, b in enumerate(blocks):
        if b.get("resolution") == "body":
            block_with_idx = dict(b)
            block_with_idx["_index"] = i
            body_blocks.append(block_with_idx)

    if not body_blocks:
        logger.warning("No body-resolution blocks found — nothing to synthesize.")
        return PipelineResult(
            chapters_completed=0, total_chunks=0, successful_chunks=0,
            failed_chunks=0, manifest=[], phase1_degraded_count=degraded_body_lost,
        )

    logger.info("Body blocks: %d / %d total", len(body_blocks), len(blocks))

    # --- Detect chapters ---
    chapters = detect_chapters(body_blocks)
    logger.info("Chapters detected: %d", len(chapters))
    for ch in chapters:
        logger.info("  %s: blocks %d–%d", ch.id, ch.block_start, ch.block_end - 1)

    # --- Synthesize per chapter, chunking within each chapter ---
    manifest: list[ManifestEntry] = []
    total_chunks = 0
    successful = 0
    failed = 0
    chapters_done = 0

    for ch in chapters:
        chapter_body = body_blocks[ch.block_start:ch.block_end]
        if not chapter_body:
            continue

        ch_chunks = chunk_blocks(chapter_body)
        if not ch_chunks:
            continue

        total_chunks += len(ch_chunks)
        chapters_done += 1
        logger.info("Synthesizing chapter %s (%d chunks)...", ch.id, len(ch_chunks))

        chapter_wav_dir = output_dir / "_tmp_wav"
        chapter_wav_dir.mkdir(exist_ok=True)
        chapter_wav_files: list[Path] = []
        chapter_manifest: list[ManifestEntry] = []
        cumulative_time = 0.0
        chapter_sample_rate: int | None = None

        for ci, chunk in enumerate(ch_chunks):
            t0 = time.monotonic()
            result = engine.synthesize(chunk.text)
            dt = time.monotonic() - t0

            if result is not None:
                sr = result.sample_rate
                if chapter_sample_rate is None:
                    chapter_sample_rate = sr

                chunk_wav = chapter_wav_dir / f"chunk_{ci:05d}.wav"
                audio_bytes = _audio_to_wav_bytes(result.audio, sr)
                chunk_wav.write_bytes(audio_bytes)
                chapter_wav_files.append(chunk_wav)

                chunk_duration = len(result.audio) / sr
                chapter_manifest.append(ManifestEntry(
                    chunk_index=ci,
                    start_time=round(cumulative_time, 3),
                    end_time=round(cumulative_time + chunk_duration, 3),
                    source_block_index=chunk.source_block_indices[0] if chunk.source_block_indices else 0,
                    chapter_id=ch.id,
                ))
                cumulative_time += chunk_duration
                successful += 1
                logger.debug("  chunk %d: %.2fs audio (%.1fs synth time)", ci, chunk_duration, dt)
            else:
                # Use the chapter's detected sample rate, or default
                sr = chapter_sample_rate or _DEFAULT_SAMPLE_RATE
                silence = _make_silence(_SILENCE_DURATION_S, sr)
                chunk_wav = chapter_wav_dir / f"chunk_{ci:05d}.wav"
                audio_bytes = _audio_to_wav_bytes(silence, sr)
                chunk_wav.write_bytes(audio_bytes)
                chapter_wav_files.append(chunk_wav)

                chapter_manifest.append(ManifestEntry(
                    chunk_index=ci,
                    start_time=round(cumulative_time, 3),
                    end_time=round(cumulative_time + _SILENCE_DURATION_S, 3),
                    source_block_index=chunk.source_block_indices[0] if chunk.source_block_indices else 0,
                    chapter_id=ch.id,
                ))
                cumulative_time += _SILENCE_DURATION_S
                failed += 1
                logger.warning(
                    "  chunk %d FAILED: inserted %.1fs silence. Text: %r",
                    ci, _SILENCE_DURATION_S, chunk.text[:80],
                )

        # --- Concatenate chapter audio at the correct sample rate ---
        if chapter_wav_files:
            sr = chapter_sample_rate or _DEFAULT_SAMPLE_RATE
            chapter_audio_path = output_dir / f"{ch.id}.wav"
            _concat_wavs(chapter_wav_files, chapter_audio_path, sr)
            logger.info("  -> %s (%.1fs, %dHz)", chapter_audio_path.name, cumulative_time, sr)
            manifest.extend(chapter_manifest)

        # --- Cleanup temp WAVs ---
        for wf in chapter_wav_files:
            wf.unlink(missing_ok=True)
        if chapter_wav_dir.exists():
            chapter_wav_dir.rmdir()

    # --- Write manifest ---
    manifest_path = output_dir / "manifest.json"
    manifest_data = [
        {
            "chunk_index": m.chunk_index,
            "start_time": m.start_time,
            "end_time": m.end_time,
            "source_block_index": m.source_block_index,
            "chapter_id": m.chapter_id,
        }
        for m in manifest
    ]
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest_data, fh, indent=2, ensure_ascii=False)
    logger.info("Manifest: %s (%d entries)", manifest_path.name, len(manifest_data))

    return PipelineResult(
        chapters_completed=chapters_done,
        total_chunks=total_chunks,
        successful_chunks=successful,
        failed_chunks=failed,
        manifest=manifest,
        phase1_degraded_count=degraded_body_lost,
    )
