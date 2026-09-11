"""Config loading for Phase 1 classifier.

Reads provider settings from a JSON config file with environment variable
overrides. Never commits or logs API keys.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


PROVIDER_DEFAULTS = {
    "local": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "",
        "model": "qwen2.5:1.5b",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": "",
        "model": "llama-3.1-8b-instant",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "",
        "model": "meta-llama/llama-3.1-8b-instruct",
    },
}

Provider = Literal["local", "groq", "openrouter"]


@dataclass
class ClassifierConfig:
    provider: Provider = "local"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    sanity_sample_pct: float = 0.05
    max_retries: int = 1
    request_timeout: float = 30.0

    def __post_init__(self) -> None:
        defaults = PROVIDER_DEFAULTS.get(self.provider, {})
        if not self.base_url:
            self.base_url = defaults.get("base_url", "")
        if not self.model:
            self.model = defaults.get("model", "")
        if not self.api_key:
            self.api_key = defaults.get("api_key", "")

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        if self.provider not in PROVIDER_DEFAULTS:
            errors.append(f"Unknown provider: {self.provider!r}")
        if not self.model:
            errors.append("No model specified")
        if self.provider in ("groq", "openrouter") and not self.api_key:
            errors.append(
                f"API key required for provider '{self.provider}'. "
                f"Set it in the config file or via the "
                f"{'GROQ_API_KEY' if self.provider == 'groq' else 'OPENROUTER_API_KEY'} env var."
            )
        return errors


def load_config(config_path: Path) -> ClassifierConfig:
    """Load config from JSON file, with env-var overrides for api_key."""
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    else:
        raw = {}

    provider: Provider = raw.get("provider", "local")

    # Environment variable overrides for API keys (never read from config file alone)
    api_key = raw.get("api_key", "")
    if not api_key:
        if provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "")
        elif provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")

    cfg = ClassifierConfig(
        provider=provider,
        model=raw.get("model", ""),
        api_key=api_key,
        base_url=raw.get("base_url", ""),
        sanity_sample_pct=raw.get("sanity_sample_pct", 0.05),
        max_retries=raw.get("max_retries", 1),
        request_timeout=raw.get("request_timeout", 30.0),
    )
    return cfg
