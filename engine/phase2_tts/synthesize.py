#!/usr/bin/env python3
"""
Phase 2 — TTS Synthesis Pipeline.

Converts Phase 1 classified text blocks into audio files with a manifest
mapping chunks to timestamps for resumable playback.

Usage:
    python -m engine.phase2_tts.synthesize input.json --output-dir ./audio
    python -m engine.phase2_tts.synthesize input.json --engine piper
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .engine import create_engine
from .pipeline import run_pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize Phase 1 classified blocks into audio.",
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to Phase 1 output JSON file (classified)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for audio + manifest (default: <input_stem>_audio/)",
    )
    parser.add_argument(
        "--engine",
        choices=["kokoro", "piper"],
        default="kokoro",
        help="TTS engine to use (default: kokoro)",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Voice name (kokoro: af_heart, piper: en_US-lessac-medium)",
    )
    parser.add_argument(
        "--lang",
        default="a",
        help="Language code for Kokoro (default: a = American English)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # --- Load input ---
    if not args.input_json.exists():
        logger.error("Input file not found: %s", args.input_json)
        sys.exit(1)

    with open(args.input_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if data.get("status") != "ok":
        logger.error(
            "Input file has status '%s' — only 'ok' files can be synthesized",
            data.get("status"),
        )
        sys.exit(1)

    # --- Output directory ---
    output_dir = args.output_dir or args.input_json.with_name(
        f"{args.input_json.stem}_audio"
    )

    # --- Create TTS engine ---
    engine_kwargs = {}
    if args.engine == "kokoro":
        engine_kwargs = {"lang_code": args.lang}
        if args.voice:
            engine_kwargs["voice"] = args.voice
    elif args.engine == "piper":
        if args.voice:
            engine_kwargs["voice_name"] = args.voice

    try:
        engine = create_engine(args.engine, **engine_kwargs)
    except Exception as exc:
        logger.error("Failed to initialize %s engine: %s", args.engine, exc)
        sys.exit(1)

    # --- Run pipeline ---
    print(f"Engine:  {engine.name}")
    print(f"Input:   {args.input_json.name}")
    print(f"Output:  {output_dir}/")
    print()

    try:
        result = run_pipeline(data, engine, output_dir)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)

    # --- Summary ---
    print()
    print("TTS Pipeline Summary:")
    print(f"  Chapters:        {result.chapters_completed}")
    print(f"  Chunks:          {result.total_chunks}")
    print(f"  Successful:      {result.successful_chunks}")
    print(f"  Failed:          {result.failed_chunks}")
    if result.phase1_degraded_count > 0:
        print(f"  Phase 1 degraded: {result.phase1_degraded_count} blocks lost to LLM errors")
    print(f"  Manifest:        {output_dir / 'manifest.json'}")
    print()


if __name__ == "__main__":
    main()
