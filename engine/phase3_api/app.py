"""FastAPI application for the SignalRead pipeline.

Exposes Phase 0→1→2 as HTTP endpoints with async background processing.
"""
from __future__ import annotations

import io
import logging
import os
import struct
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from .jobs import JobStage, create_job, get_job
from .worker import run_job

logger = logging.getLogger(__name__)

app = FastAPI(title="SignalRead API", version="0.1.0")

# Working directory for all jobs — defaults to ./signalread_jobs relative to CWD
WORK_BASE = Path(os.environ.get("SIGNALREAD_WORK_DIR", Path.cwd() / "signalread_jobs"))

ALLOWED_EXTENSIONS = {".pdf", ".epub"}


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept a PDF or EPUB, start background processing, return job_id."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext!r}. Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    job = create_job(file.filename or f"upload{ext}", WORK_BASE)
    dest = job.work_dir / job.filename

    # Save uploaded file to job's working directory
    content = await file.read()
    dest.write_bytes(content)

    # Kick off background processing (non-blocking)
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run_job, job)

    return {"job_id": job.id, "filename": job.filename, "status": job.stage.value}


# ---------------------------------------------------------------------------
# GET /status/{job_id}
# ---------------------------------------------------------------------------
@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Return current stage, progress, and Phase 1 degradation info."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    result = {
        "job_id": job.id,
        "filename": job.filename,
        "stage": job.stage.value,
        "progress": job.progress,
    }

    if job.error:
        result["error"] = job.error

    if job.phase1_degraded_count is not None:
        result["phase1_degraded_count"] = job.phase1_degraded_count
    if job.phase1_degraded_detail:
        result["phase1_degraded_detail"] = job.phase1_degraded_detail

    return result


# ---------------------------------------------------------------------------
# GET /manifest/{job_id}
# ---------------------------------------------------------------------------
@app.get("/manifest/{job_id}")
async def get_manifest(job_id: str):
    """Return the manifest once processing is done."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.stage not in (JobStage.DONE,):
        raise HTTPException(
            status_code=202,
            detail={"message": "Not ready", "current_stage": job.stage.value},
        )

    if job.manifest is None:
        raise HTTPException(status_code=202, detail={"message": "Manifest not yet available"})

    return {"job_id": job.id, "manifest": job.manifest}


# ---------------------------------------------------------------------------
# GET /audio/{job_id}/{chapter_id}
# ---------------------------------------------------------------------------
@app.get("/audio/{job_id}/{chapter_id}")
async def get_audio(job_id: str, chapter_id: str, request: Request):
    """Stream a chapter's audio file with HTTP range request support."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.stage != JobStage.DONE:
        raise HTTPException(
            status_code=202,
            detail={"message": "Not ready", "current_stage": job.stage.value},
        )

    audio_path = job.audio_files.get(chapter_id)
    if audio_path is None or not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Chapter '{chapter_id}' not found")

    file_size = audio_path.stat().st_size

    # Parse Range header
    range_header = request.headers.get("range")
    if range_header:
        # Parse "bytes=start-end"
        try:
            ranges = range_header.replace("bytes=", "").split("-")
            start = int(ranges[0]) if ranges[0] else 0
            end = int(ranges[1]) if ranges[1] else file_size - 1
        except (ValueError, IndexError):
            raise HTTPException(status_code=416, detail="Invalid Range header")

        if start >= file_size or end >= file_size or start > end:
            raise HTTPException(status_code=416, detail="Range not satisfiable")

        content_length = end - start + 1

        def iter_range():
            with open(audio_path, "rb") as fh:
                fh.seek(start)
                remaining = content_length
                chunk_size = 64 * 1024
                while remaining > 0:
                    read_size = min(chunk_size, remaining)
                    data = fh.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_range(),
            status_code=206,
            media_type="audio/wav",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            },
        )

    # No range request — return full file
    def iter_full():
        with open(audio_path, "rb") as fh:
            while True:
                data = fh.read(64 * 1024)
                if not data:
                    break
                yield data

    return StreamingResponse(
        iter_full(),
        media_type="audio/wav",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        },
    )


# ---------------------------------------------------------------------------
# GET /audit-log/{job_id}
# ---------------------------------------------------------------------------
@app.get("/audit-log/{job_id}")
async def get_audit_log(job_id: str):
    """Return the Phase 1 audit log content."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.stage.value not in ("classifying", "synthesizing", "done", "failed"):
        if job.audit_log is None:
            raise HTTPException(
                status_code=202,
                detail={"message": "Not ready", "current_stage": job.stage.value},
            )

    if job.audit_log is None:
        raise HTTPException(
            status_code=202,
            detail={"message": "Audit log not yet available", "current_stage": job.stage.value},
        )

    return {"job_id": job.id, "audit_log": job.audit_log}
