"""Unified HTTP client for OpenAI-compatible chat completion APIs.

Works with Ollama (local), Groq, and OpenRouter via the same interface.
Only the config values (base_url, api_key, model) differ between providers.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from .config import ClassifierConfig

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Raised for non-recoverable LLM errors (auth, unreachable, etc.)."""


class LLMRateLimitError(LLMClientError):
    """Raised specifically for 429 responses."""


class LLMResponseError(LLMClientError):
    """Raised when the model returns unusable output."""


class LLMClient:
    """Thin wrapper around the OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, config: ClassifierConfig) -> None:
        self.config = config
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"
        self._client = httpx.Client(timeout=config.request_timeout)

    def classify_binary(self, prompt: str) -> str:
        """Send a chat completion request and return the assistant's text response.

        Raises LLMClientError / LLMRateLimitError / LLMResponseError on failure.
        """
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a text classifier. For each block of text presented, "
                        "determine whether it is body content meant to be read in sequence, "
                        "or non-body material (headers, footnotes, page numbers, navigation, "
                        "metadata, etc.) that should be skipped. "
                        "Respond with exactly one word: READ or SKIP. Nothing else."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 4,
        }

        try:
            resp = self._client.post(url, json=body, headers=self._headers)
        except httpx.ConnectError as exc:
            raise LLMClientError(
                f"Cannot connect to {self.config.provider} at {self.config.base_url}. "
                f"Is the service running? Details: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMClientError(
                f"HTTP error from {self.config.provider}: {exc.response.status_code}"
            ) from exc

        if resp.status_code == 429:
            detail = resp.text[:200]
            raise LLMRateLimitError(
                f"Rate-limited by {self.config.provider} (429). Detail: {detail}"
            )
        if resp.status_code == 401 or resp.status_code == 403:
            raise LLMClientError(
                f"Authentication failed with {self.config.provider} "
                f"({resp.status_code}). Check your API key."
            )
        if resp.status_code >= 400:
            raise LLMClientError(
                f"HTTP {resp.status_code} from {self.config.provider}: {resp.text[:300]}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                f"Non-JSON response from {self.config.provider}: {resp.text[:200]}"
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(
                f"Malformed response structure from {self.config.provider}: {data}"
            ) from exc

        return content

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
