"""In-memory job store for Phase 3 API.

v1: dict-backed, not persistent. Each job gets its own working directory.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class JobStage(str, Enum):
    EXTRACTING = "extracting"
    CLASSIFYING = "classifying"
    SYNTHESIZING = "synthesizing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    filename: str
    stage: JobStage = JobStage.EXTRACTING
    progress: float = 0.0
    error: str | None = None
    phase1_degraded_count: int | None = None
    phase1_degraded_detail: str | None = None
    work_dir: Path = field(default_factory=Path)
    manifest: list[dict] | None = None
    audit_log: str | None = None
    # Chapter audio files: chapter_id -> file path
    audio_files: dict[str, Path] = field(default_factory=dict)


# Global in-memory store
_jobs: dict[str, Job] = {}


def create_job(filename: str, work_base: Path) -> Job:
    """Create a new job with a unique ID and isolated working directory."""
    job_id = uuid.uuid4().hex[:12]
    work_dir = work_base / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    job = Job(id=job_id, filename=filename, work_dir=work_dir)
    _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def update_job(job_id: str, **kwargs: Any) -> Job | None:
    job = _jobs.get(job_id)
    if job is None:
        return None
    for k, v in kwargs.items():
        setattr(job, k, v)
    return job
