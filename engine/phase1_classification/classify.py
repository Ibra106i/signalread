#!/usr/bin/env python3
"""
Phase 1 — AI Classification of ambiguous blocks.

Takes Phase 0 JSON output and resolves every block tagged 'ambiguous'
into a final 'body' or 'skip' label using a configurable LLM backend.

Usage:
    python -m engine.phase1_classification.classify <input.json> --config config.json
    python -m engine.phase1_classification.classify <input.json> --provider local
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import re
import sys
from pathlib import Path

from .config import ClassifierConfig, load_config
from .llm_client import LLMClient, LLMClientError, LLMRateLimitError, LLMResponseError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heuristic fast-path: classify very short ambiguous blocks without LLM
# ---------------------------------------------------------------------------
_TRAILING_NUM_RE = re.compile(r"[^\d]\d+$")
_SINGLE_TOKEN_RE = re.compile(r"^\S+$")


def _heuristic_classify(text: str) -> str | None:
    """Try to classify a very short ambiguous block by heuristics.

    Returns 'body' or 'skip' if confident, None if ambiguous needs LLM.
    """
    stripped = text.strip()

    # Pure numbers — almost always page numbers, section numbers, etc.
    if stripped.isdigit():
        return "skip"

    # Trailing number on a short string (e.g. "Chapter 3", "Section 1.2")
    # Requires a non-digit character before the digits to avoid overlap with
    # the pure-digit check above.  Strings like "12" never reach this branch.
    if len(stripped) <= 30 and _TRAILING_NUM_RE.search(stripped):
        return "skip"

    # Single token, all uppercase — likely an abbreviation or label
    if _SINGLE_TOKEN_RE.match(stripped) and stripped.isupper() and len(stripped) <= 10:
        return "skip"

    # Very short (< 5 chars) — likely a label, number, or fragment
    if len(stripped) < 5:
        return "skip"

    return None  # needs LLM


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _build_classify_prompt(block_text: str, context_before: str, context_after: str) -> str:
    """Build the user-facing prompt for a single ambiguous block."""
    parts = []
    if context_before:
        parts.append(f"--- Text before this block ---\n{context_before}\n")
    parts.append(f"--- Block to classify ---\n{block_text}\n")
    if context_after:
        parts.append(f"--- Text after this block ---\n{context_after}\n")
    parts.append(
        "Is this block body content meant to be read in sequence (READ), "
        "or non-body material to skip (SKIP)? Respond with exactly one word."
    )
    return "\n".join(parts)


def _parse_binary_response(raw: str) -> str | None:
    """Parse LLM response into 'body' or 'skip'. Returns None if unparseable."""
    cleaned = raw.strip().upper()
    # Strip quotes and punctuation
    cleaned = cleaned.strip("\"'.,;:!?")
    if cleaned == "READ":
        return "body"
    if cleaned == "SKIP":
        return "skip"
    return None


# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------
def classify_blocks(
    data: dict,
    client: LLMClient,
    config: ClassifierConfig,
    audit_log_path: Path | None = None,
) -> dict:
    """Classify all ambiguous blocks in a Phase 0 result dict.

    Returns a new dict with an added 'resolution' field per block.
    If audit_log_path is provided, writes one line per resolved ambiguous block.
    """
    result = copy.deepcopy(data)
    blocks = result["blocks"]

    # Open audit log early so failures still get written
    audit_fh = None
    if audit_log_path is not None:
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        audit_fh = open(audit_log_path, "w", encoding="utf-8")

    def _audit(idx: int, block: dict) -> None:
        """Write one audit line for an ambiguous block that was resolved."""
        if audit_fh is None:
            return
        src = block.get("classification_source", "unknown")
        res = block.get("resolution", "unknown")
        text_preview = block["text"].replace("\n", " ")[:80]
        audit_fh.write(
            f"block={idx}  category=ambiguous  resolution={res}  "
            f"source={src}  text={text_preview}\n"
        )
        audit_fh.flush()

    if not blocks:
        return result

    # --- Step 1: Count ambiguous blocks and print estimate ---
    ambiguous_indices = [i for i, b in enumerate(blocks) if b["category"] == "ambiguous"]
    total_ambiguous = len(ambiguous_indices)
    llm_calls_needed = 0
    heuristic_resolved = 0

    for idx in ambiguous_indices:
        text = blocks[idx]["text"]
        if _heuristic_classify(text) is not None:
            heuristic_resolved += 1
        else:
            llm_calls_needed += 1

    print(f"\n  Ambiguous blocks: {total_ambiguous}")
    print(f"  Heuristic-resolved (no LLM call): {heuristic_resolved}")
    print(f"  LLM calls needed: {llm_calls_needed}")
    if config.provider in ("groq", "openrouter") and llm_calls_needed > 0:
        print(
            f"  (Cloud provider '{config.provider}' — each call uses tokens. "
            f"Model: {config.model})"
        )
    print()

    # --- Step 2: Initialize resolution for all blocks ---
    for block in blocks:
        if block["category"] == "ambiguous":
            block["resolution"] = "pending"
        elif block["category"] in ("NarrativeText", "Title"):
            block["resolution"] = "body"
        else:
            block["resolution"] = "skip"

    # --- Step 3: Process ambiguous blocks ---
    for idx in ambiguous_indices:
        block = blocks[idx]
        text = block["text"]

        # Fast-path heuristics
        heur_result = _heuristic_classify(text)
        if heur_result is not None:
            block["resolution"] = heur_result
            block["classification_source"] = "heuristic"
            logger.debug("  Heuristic -> %s: %r", heur_result, text[:60])
            _audit(idx, block)
            continue

        # Build context window (up to 2 blocks before/after)
        ctx_before_parts = []
        for j in range(max(0, idx - 2), idx):
            if blocks[j].get("resolution") != "skip":
                ctx_before_parts.append(blocks[j]["text"])
        ctx_before = "\n".join(ctx_before_parts)

        ctx_after_parts = []
        for j in range(idx + 1, min(len(blocks), idx + 3)):
            if blocks[j].get("resolution") != "skip":
                ctx_after_parts.append(blocks[j]["text"])
        ctx_after = "\n".join(ctx_after_parts)

        prompt = _build_classify_prompt(text, ctx_before, ctx_after)

        # Call LLM with retry on malformed output
        resolved = False
        for attempt in range(1 + config.max_retries):
            try:
                raw = client.classify_binary(prompt)
                parsed = _parse_binary_response(raw)
                if parsed is not None:
                    block["resolution"] = parsed
                    block["classification_source"] = "llm"
                    logger.debug("  LLM -> %s (attempt %d): %r", parsed, attempt + 1, text[:60])
                    resolved = True
                    _audit(idx, block)
                    break
                else:
                    # Malformed response — retry with stricter prompt
                    logger.warning(
                        "  Malformed LLM response (attempt %d): %r — retrying",
                        attempt + 1,
                        raw[:100],
                    )
                    prompt = (
                        f"Classify this text as READ (body content) or SKIP (non-body).\n"
                        f"Text: {text[:500]}\n"
                        f"Respond with ONLY the word READ or SKIP."
                    )
            except LLMRateLimitError as exc:
                logger.error("Rate-limited by %s: %s", config.provider, exc)
                block["resolution"] = "skip"
                block["classification_source"] = "error_rate_limit"
                resolved = True
                _audit(idx, block)
                break
            except LLMClientError as exc:
                logger.error("LLM error: %s", exc)
                block["resolution"] = "skip"
                block["classification_source"] = "error"
                resolved = True
                _audit(idx, block)
                break

        if not resolved:
            logger.warning("  Defaulting to SKIP after retries: %r", text[:60])
            block["resolution"] = "skip"
            block["classification_source"] = "default_skip"
            _audit(idx, block)

    # --- Step 4: Sanity check — re-classify 5% sample of confident body blocks ---
    body_indices = [
        i for i, b in enumerate(blocks)
        if b.get("resolution") == "body" and b["category"] not in ("ambiguous",)
    ]
    sample_size = max(1, int(len(body_indices) * config.sanity_sample_pct))
    sample_indices = random.sample(body_indices, min(sample_size, len(body_indices)))

    if sample_indices and llm_calls_needed > 0:
        agreements = 0
        disagreements = 0
        print(f"  Sanity check: re-classifying {len(sample_indices)} confident body blocks...")

        for idx in sample_indices:
            block = blocks[idx]
            text = block["text"]

            ctx_before_parts = []
            for j in range(max(0, idx - 2), idx):
                if blocks[j].get("resolution") != "skip":
                    ctx_before_parts.append(blocks[j]["text"])
            ctx_before = "\n".join(ctx_before_parts)

            ctx_after_parts = []
            for j in range(idx + 1, min(len(blocks), idx + 3)):
                if blocks[j].get("resolution") != "skip":
                    ctx_after_parts.append(blocks[j]["text"])
            ctx_after = "\n".join(ctx_after_parts)

            prompt = _build_classify_prompt(text, ctx_before, ctx_after)
            try:
                raw = client.classify_binary(prompt)
                parsed = _parse_binary_response(raw)
                if parsed == "body":
                    agreements += 1
                else:
                    disagreements += 1
                    logger.warning(
                        "  Sanity disagree: block %d resolved as body, "
                        "LLM says %s — text: %r",
                        idx, parsed, text[:80],
                    )
            except LLMClientError:
                # Don't penalize sanity check for transient errors
                agreements += 1

        total_checked = agreements + disagreements
        rate = (agreements / total_checked * 100) if total_checked > 0 else 0
        print(f"  Sanity check result: {agreements}/{total_checked} agree ({rate:.0f}%)")
        if rate < 80:
            print("  WARNING: Low agreement rate — review classification quality.")
        print()

    if audit_fh is not None:
        audit_fh.close()

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve ambiguous Phase 0 blocks via LLM classification.",
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to Phase 0 output JSON file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to classifier config JSON (default: looks for config.json in phase1 dir)",
    )
    parser.add_argument(
        "--provider",
        choices=["local", "groq", "openrouter"],
        default=None,
        help="Override provider from config",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model from config",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: input_stem_classified.json)",
    )
    parser.add_argument(
        "--hardware-report",
        action="store_true",
        help="Print hardware detection report (local provider only)",
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
        logger.error("Input file has status '%s' — only 'ok' files can be classified", data.get("status"))
        sys.exit(1)

    # --- Load config ---
    config_path = args.config
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"

    config = load_config(config_path)

    # CLI overrides
    if args.provider:
        config.provider = args.provider
        # Re-apply defaults for the new provider
        config.__post_init__()
    if args.model:
        config.model = args.model

    # --- Validate ---
    errors = config.validate()
    if errors:
        for e in errors:
            logger.error("Config error: %s", e)
        sys.exit(1)

    # --- Hardware report (local only) ---
    if args.hardware_report or config.provider == "local":
        from .hardware import print_hardware_report
        print_hardware_report()

    # --- Run classification ---
    print(f"Provider: {config.provider}")
    print(f"Model:    {config.model}")
    print(f"Input:    {args.input_json.name}")

    output_path = args.output or args.input_json.with_name(
        f"{args.input_json.stem}_classified.json"
    )
    audit_log_path = output_path.with_name(f"{output_path.stem}_audit.log")

    try:
        with LLMClient(config) as client:
            result = classify_blocks(data, client, config, audit_log_path=audit_log_path)
    except LLMClientError as exc:
        logger.error("Fatal LLM error: %s", exc)
        sys.exit(1)

    # --- Write output ---
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    # --- Summary ---
    resolutions = {}
    for b in result["blocks"]:
        r = b.get("resolution", "unknown")
        resolutions[r] = resolutions.get(r, 0) + 1

    print(f"\nOutput: {output_path}")
    print("Classification summary:")
    for label, count in sorted(resolutions.items()):
        print(f"  {label}: {count}")
    print()


if __name__ == "__main__":
    main()
