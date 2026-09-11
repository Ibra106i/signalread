"""Hardware detection for local model recommendations.

This module is ONLY invoked when provider=local and only meaningful
from the desktop-app context. It is NOT imported by the web frontend.
Detection failures default to the CPU-only path rather than crashing.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    ram_gb: float
    cpu_cores: int
    has_gpu: bool
    gpu_name: str
    gpu_vram_gb: float


@dataclass
class ModelRecommendation:
    model_id: str
    reason: str
    warning: str = ""


def _get_ram_gb() -> float:
    """Detect total system RAM in GB. Falls back to a conservative estimate."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass

    # Windows fallback
    try:
        result = subprocess.run(
            ["wmic", "memorychip", "get", "Capacity"],
            capture_output=True, text=True, timeout=5,
        )
        total_kb = 0
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                total_kb += int(line)
        if total_kb > 0:
            return total_kb / (1024 ** 2)
    except Exception:
        pass

    # Linux fallback
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:
        pass

    logger.warning("Could not detect RAM; assuming 8 GB")
    return 8.0


def _get_cpu_cores() -> int:
    """Detect logical CPU cores."""
    try:
        import os as _os
        return _os.cpu_count() or 2
    except Exception:
        return 2


def _detect_gpu() -> tuple[bool, str, float]:
    """Detect GPU via nvidia-smi. Returns (has_gpu, name, vram_gb).

    Falls back gracefully if nvidia-smi is unavailable.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False, "", 0.0

    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return False, "", 0.0

        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        name = parts[0] if len(parts) > 0 else "Unknown GPU"
        vram_mb = float(parts[1]) if len(parts) > 1 else 0.0
        return True, name, vram_mb / 1024
    except Exception as exc:
        logger.warning("nvidia-smi detection failed: %s", exc)
        return False, "", 0.0


def detect_hardware() -> HardwareInfo:
    """Detect system hardware. Never raises — returns conservative defaults on failure."""
    try:
        ram = _get_ram_gb()
    except Exception:
        ram = 8.0
    try:
        cores = _get_cpu_cores()
    except Exception:
        cores = 2
    try:
        has_gpu, gpu_name, vram = _detect_gpu()
    except Exception:
        has_gpu, gpu_name, vram = False, "", 0.0

    return HardwareInfo(
        ram_gb=round(ram, 1),
        cpu_cores=cores,
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        gpu_vram_gb=round(vram, 1),
    )


def recommend_model(hw: HardwareInfo | None = None) -> ModelRecommendation:
    """Recommend a model tier based on detected hardware.

    Returns a ModelRecommendation — does NOT download or switch models.
    """
    if hw is None:
        hw = detect_hardware()

    # GPU path
    if hw.has_gpu:
        if hw.gpu_vram_gb >= 8.0:
            return ModelRecommendation(
                model_id="qwen2.5:7b",
                reason=(
                    f"GPU detected ({hw.gpu_name}, {hw.gpu_vram_gb} GB VRAM). "
                    f"Qwen2.5-7B-Instruct will run well on this hardware."
                ),
            )
        elif hw.gpu_vram_gb >= 4.0:
            return ModelRecommendation(
                model_id="qwen2.5:3b",
                reason=(
                    f"GPU detected ({hw.gpu_name}, {hw.gpu_vram_gb} GB VRAM). "
                    f"Qwen2.5-3B or Llama-3.2-3B will fit in VRAM."
                ),
            )
        else:
            return ModelRecommendation(
                model_id="qwen2.5:1.5b",
                reason=(
                    f"GPU detected but only {hw.gpu_vram_gb} GB VRAM. "
                    f"Falling back to Qwen2.5-1.5B-Instruct (Q4)."
                ),
            )

    # CPU-only path
    if hw.ram_gb >= 16:
        return ModelRecommendation(
            model_id="qwen2.5:3b",
            reason=(
                f"No GPU detected. {hw.ram_gb} GB RAM available. "
                f"Qwen2.5-3B-Instruct (Q4 GGUF) should run at usable speed."
            ),
        )
    elif hw.ram_gb >= 8:
        return ModelRecommendation(
            model_id="qwen2.5:1.5b",
            reason=(
                f"No GPU detected. {hw.ram_gb} GB RAM available. "
                f"Qwen2.5-1.5B-Instruct (Q4 GGUF) is the recommended tier."
            ),
        )
    else:
        return ModelRecommendation(
            model_id="",
            reason=(
                f"No GPU detected and only {hw.ram_gb} GB RAM. "
                f"Local inference will likely be too slow or unreliable."
            ),
            warning="RECOMMEND_SWITCH_TO_CLOUD",
        )


def print_hardware_report() -> HardwareInfo:
    """Detect hardware, print a report, and return the info."""
    hw = detect_hardware()
    rec = recommend_model(hw)

    print("\n--- Hardware Detection (local provider) ---")
    print(f"  RAM:       {hw.ram_gb} GB")
    print(f"  CPU cores: {hw.cpu_cores}")
    if hw.has_gpu:
        print(f"  GPU:       {hw.gpu_name} ({hw.gpu_vram_gb} GB VRAM)")
    else:
        print("  GPU:       None detected")

    print(f"\n  Recommended model: {rec.model_id or '(none — use cloud provider)'}")
    print(f"  Reason: {rec.reason}")
    if rec.warning == "RECOMMEND_SWITCH_TO_CLOUD":
        print("  *** Local inference is NOT recommended for this hardware. ***")
        print("  *** Consider using Groq or OpenRouter instead. ***")
    print("-------------------------------------------\n")
    return hw
