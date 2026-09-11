"""Background worker that orchestrates Phase 0 → 1 → 2 for a single job."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from ..phase0_extraction.layout_classifier import process_pdf, process_epub
from ..phase1_classification.classify import classify_blocks
from ..phase1_classification.config import ClassifierConfig, load_config
from ..phase1_classification.llm_client import LLMClient, LLMClientError
from ..phase2_tts.pipeline import run_pipeline
from ..phase2_tts.engine import create_engine
from .jobs import Job, JobStage, update_job

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "phase1_classification" / "config.json"


def run_job(job: Job) -> None:
    """Execute the full pipeline for a single job. Called as a background task."""
    try:
        _run_pipeline(job)
    except Exception as exc:
        logger.exception("Job %s failed unexpectedly: %s", job.id, exc)
        update_job(job.id, stage=JobStage.FAILED, error=str(exc))


def _run_pipeline(job: Job) -> None:
    work_dir = job.work_dir
    input_file = work_dir / job.filename
    ext = input_file.suffix.lower()

    # --- Phase 0: Extraction ---
    update_job(job.id, stage=JobStage.EXTRACTING, progress=0.0)
    logger.info("Job %s: Phase 0 extracting %s", job.id, job.filename)

    try:
        if ext == ".pdf":
            phase0_result = process_pdf(input_file)
        elif ext == ".epub":
            phase0_result = process_epub(input_file)
        else:
            update_job(job.id, stage=JobStage.FAILED, error=f"Unsupported file type: {ext}")
            return
    except Exception as exc:
        update_job(job.id, stage=JobStage.FAILED, error=f"Phase 0 failed: {exc}")
        return

    if phase0_result.get("status") == "needs_ocr":
        update_job(job.id, stage=JobStage.FAILED, error="PDF has no extractable text layer (needs OCR)")
        return

    if phase0_result.get("status") != "ok":
        update_job(job.id, stage=JobStage.FAILED, error=f"Phase 0 returned status: {phase0_result.get('status')}")
        return

    # Save Phase 0 output
    phase0_path = work_dir / "phase0_output.json"
    with open(phase0_path, "w", encoding="utf-8") as fh:
        json.dump(phase0_result, fh, indent=2, ensure_ascii=False)

    update_job(job.id, progress=0.33)
    logger.info("Job %s: Phase 0 complete (%d blocks)", job.id, len(phase0_result.get("blocks", [])))

    # --- Phase 1: Classification ---
    update_job(job.id, stage=JobStage.CLASSIFYING, progress=0.33)
    logger.info("Job %s: Phase 1 classifying", job.id)

    config = load_config(_CONFIG_PATH)
    errors = config.validate()
    if errors:
        update_job(job.id, stage=JobStage.FAILED, error=f"Phase 1 config error: {'; '.join(errors)}")
        return

    audit_log_path = work_dir / "audit.log"
    try:
        with LLMClient(config) as client:
            phase1_result = classify_blocks(
                phase0_result, client, config, audit_log_path=audit_log_path
            )
    except LLMClientError as exc:
        update_job(job.id, stage=JobStage.FAILED, error=f"Phase 1 LLM error: {exc}")
        return
    except Exception as exc:
        update_job(job.id, stage=JobStage.FAILED, error=f"Phase 1 failed: {exc}")
        return

    # Save Phase 1 output
    phase1_path = work_dir / "phase1_output.json"
    with open(phase1_path, "w", encoding="utf-8") as fh:
        json.dump(phase1_result, fh, indent=2, ensure_ascii=False)

    # Read audit log for API exposure
    audit_content = None
    if audit_log_path.exists():
        audit_content = audit_log_path.read_text(encoding="utf-8")

    # Count degraded blocks
    degraded_sources = {"error_rate_limit", "error", "default_skip"}
    degraded_count = sum(
        1 for b in phase1_result.get("blocks", [])
        if b.get("resolution") == "skip"
        and b.get("classification_source") in degraded_sources
    )

    body_count = sum(1 for b in phase1_result.get("blocks", []) if b.get("resolution") == "body")
    total_count = len(phase1_result.get("blocks", []))

    degraded_detail = None
    if degraded_count > 0:
        degraded_detail = (
            f"{degraded_count} of {total_count} blocks were defaulted to skip "
            f"due to LLM errors — {body_count} body blocks available for synthesis."
        )

    update_job(
        job.id,
        progress=0.66,
        phase1_degraded_count=degraded_count,
        phase1_degraded_detail=degraded_detail,
        audit_log=audit_content,
    )
    logger.info("Job %s: Phase 1 complete (%d body, %d degraded)", job.id, body_count, degraded_count)

    # --- Phase 2: TTS Synthesis ---
    update_job(job.id, stage=JobStage.SYNTHESIZING, progress=0.66)
    logger.info("Job %s: Phase 2 synthesizing", job.id)

    audio_dir = work_dir / "audio"
    try:
        engine = create_engine("kokoro")
        phase2_result = run_pipeline(phase1_result, engine, audio_dir)
    except Exception as exc:
        update_job(job.id, stage=JobStage.FAILED, error=f"Phase 2 failed: {exc}")
        return

    # Collect audio file paths
    audio_files = {}
    for wav in audio_dir.glob("*.wav"):
        chapter_id = wav.stem
        audio_files[chapter_id] = wav

    update_job(
        job.id,
        stage=JobStage.DONE,
        progress=1.0,
        manifest=phase2_result.manifest if hasattr(phase2_result, "manifest") else [],
        audio_files=audio_files,
    )
    logger.info(
        "Job %s: Complete (%d chapters, %d/%d chunks)",
        job.id,
        phase2_result.chapters_completed,
        phase2_result.successful_chunks,
        phase2_result.total_chunks,
    )
