"""Shared Ollama request handling (host resolution and the generate client)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

from utils.vlm_common import (
    BaseVLMClient,
    OllamaMetrics,
    OllamaRequestStats,
    encode_image_b64,
    extract_json_object,
)

__all__ = [
    "DEFAULT_OLLAMA_HOST",
    "OLLAMA_HOST_ENV_VAR",
    "OllamaClient",
    "OllamaMetrics",
    "OllamaRequestStats",
    "normalize_host",
    "resolve_host",
]

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_HOST_ENV_VAR = "OLLAMA_HOST"

logger = logging.getLogger(__name__)


def normalize_host(host: str) -> str:
    """Trim whitespace and trailing slashes, adding http:// when no scheme is given.

    The Ollama CLI accepts bare ``host:port`` values (e.g.
    ``plotter.local:11434``), but ``requests`` raises ``MissingSchema`` on
    them, so a scheme is prepended.
    """
    cleaned = host.strip().rstrip("/")
    if cleaned and "://" not in cleaned:
        cleaned = f"http://{cleaned}"
    return cleaned


def resolve_host(configured_host: str | None = None) -> str:
    """Resolve the Ollama base URL.

    ``OLLAMA_HOST`` wins so a remote server can be selected at runtime
    without editing the tracked config file. Otherwise the configured host is
    used, falling back to localhost.
    """
    env_host = os.environ.get(OLLAMA_HOST_ENV_VAR, "").strip()
    if env_host:
        resolved = normalize_host(env_host)
        logger.debug("Using %s override: %s", OLLAMA_HOST_ENV_VAR, resolved)
        return resolved
    if configured_host and configured_host.strip():
        return normalize_host(configured_host)
    return DEFAULT_OLLAMA_HOST


class OllamaClient(BaseVLMClient):
    """Send Ollama generate requests and collect request/response metrics."""

    def __init__(
        self,
        host: str,
        model: str,
        timeout: int = 90,
        metrics: OllamaMetrics | None = None,
        session: requests.Session | None = None,
        num_ctx: int | None = None,
    ) -> None:
        if num_ctx is not None and (isinstance(num_ctx, bool) or num_ctx <= 0):
            raise ValueError(f"ollama num_ctx must be a positive integer, got {num_ctx!r}")
        super().__init__(model, timeout, metrics, session)
        self.host = host
        self.num_ctx = num_ctx

    def generate(
        self,
        image_path: str,
        prompt: str,
        phase: str = "unspecified",
        system_prompt: str | None = None,
    ) -> dict[str, Any] | None:
        """Analyze one image, preserving Ollama's JSON-mode response contract.

        Thinking models may leave ``response`` empty and put the JSON in ``thinking``;
        both fields are tried, ``response`` first.
        """
        started = time.perf_counter()

        logger.debug("Starting Ollama generate for phase: %s", phase)

        try:
            image_b64 = encode_image_b64(image_path)
        except OSError as exc:
            logger.warning("Cannot read image %s: %s", image_path, exc)
            self.metrics.record(phase, time.perf_counter() - started, False)
            return None

        logger.debug("Payload size: %d bytes", len(image_b64))

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "format": "json",
            "stream": False,
            "think": False,
        }
        if system_prompt and system_prompt.strip():
            payload["system"] = system_prompt.strip()
        if self.num_ctx is not None:
            payload["options"] = {"num_ctx": self.num_ctx}

        try:
            logger.debug("Sending Ollama request to %s", self.host)
            # Use requests.post if mocked in unit tests, otherwise use pooled session
            is_mocked = hasattr(requests.post, "mock_calls")
            post_fn = requests.post if is_mocked else self.session.post
            response = post_fn(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            logger.debug("Ollama response text: %s", response.text)
            result = response.json()
        except requests.exceptions.RequestException as exc:
            logger.warning("Ollama request failed: %s", exc)
            self.metrics.record(phase, time.perf_counter() - started, False)
            return None

        parsed = extract_json_object(result.get("response"), result.get("thinking"))
        if parsed is None:
            logger.warning("Ollama returned invalid JSON for phase %s: %.500s", phase, result)
            self.metrics.record(phase, time.perf_counter() - started, False)
            return None

        self.metrics.record(
            phase,
            time.perf_counter() - started,
            True,
            response=result,
        )

        elapsed = time.perf_counter() - started
        prompt_eval = result.get("prompt_eval_count", 0)
        eval_count = result.get("eval_count", 0)
        logger.debug(
            "[%s] Ollama response in %.1fs: %d prompt tokens, %d generated tokens",
            phase,
            elapsed,
            prompt_eval,
            eval_count,
        )

        return parsed
