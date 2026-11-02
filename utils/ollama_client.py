"""Shared Ollama request handling and run-level metrics."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
OLLAMA_HOST_ENV_VAR = "OLLAMA_HOST"

_SERVER_METRIC_FIELDS = (
    "prompt_eval_count",
    "eval_count",
    "prompt_eval_duration",
    "eval_duration",
    "load_duration",
    "total_duration",
)

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


@dataclass
class OllamaRequestStats:
    calls: int = 0
    successes: int = 0
    failures: int = 0
    elapsed_seconds: float = 0.0
    server_metrics: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        elapsed_seconds: float,
        succeeded: bool,
        response: dict[str, Any] | None = None,
    ) -> None:
        self.calls += 1
        self.elapsed_seconds += elapsed_seconds
        if succeeded:
            self.successes += 1
        else:
            self.failures += 1

        if response is None:
            return
        for field_name in _SERVER_METRIC_FIELDS:
            value = response.get(field_name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                self.server_metrics[field_name] = self.server_metrics.get(field_name, 0) + value


@dataclass
class OllamaMetrics:
    by_phase: dict[str, OllamaRequestStats] = field(default_factory=dict)

    def record(
        self,
        phase: str,
        elapsed_seconds: float,
        succeeded: bool,
        response: dict[str, Any] | None = None,
    ) -> None:
        stats = self.by_phase.setdefault(phase, OllamaRequestStats())
        stats.record(elapsed_seconds, succeeded, response)

    def format_summary(self) -> str:
        if not self.by_phase:
            return "Ollama request metrics: no requests made."

        lines = ["Ollama request metrics:"]
        for phase, stats in self.by_phase.items():
            average = stats.elapsed_seconds / stats.calls
            call_label = "call" if stats.calls == 1 else "calls"
            line = (
                f"  {phase}: {stats.calls} {call_label} "
                f"({stats.successes} successful, {stats.failures} failed), "
                f"{stats.elapsed_seconds:.1f}s total ({average:.1f}s average)"
            )
            server_metrics = stats.server_metrics
            if "prompt_eval_count" in server_metrics:
                line += f", {server_metrics['prompt_eval_count']} prompt tokens"
            if "eval_count" in server_metrics:
                line += f", {server_metrics['eval_count']} generated tokens"
            duration_parts = [
                f"{name.removesuffix('_duration')} {value / 1_000_000_000:.1f}s"
                for name, value in server_metrics.items()
                if name.endswith("_duration")
            ]
            if duration_parts:
                line += f", Ollama: {', '.join(duration_parts)}"
            lines.append(line)
        return "\n".join(lines)


class OllamaClient:
    """Send Ollama generate requests and collect request/response metrics."""

    def __init__(
        self,
        host: str,
        model: str,
        timeout: int = 90,
        metrics: OllamaMetrics | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.host = host
        self.model = model
        self.timeout = timeout
        self.metrics = metrics if metrics is not None else OllamaMetrics()
        self.session = session if session is not None else requests.Session()

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def generate(
        self,
        image_path: str,
        prompt: str,
        phase: str = "unspecified",
    ) -> dict[str, Any] | None:
        """Analyze one image, preserving Ollama's JSON-mode response contract."""
        started = time.perf_counter()

        logger.debug("Starting Ollama generate for phase: %s", phase)

        try:
            with open(image_path, "rb") as image_file:
                image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as exc:
            logger.warning("Cannot read image %s: %s", image_path, exc)
            self.metrics.record(phase, time.perf_counter() - started, False)
            return None

        logger.debug("Payload size: %d bytes", len(image_b64))

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "format": "json",
            "stream": False,
        }

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
            result = response.json()
            raw = result.get("response", "{}")
            parsed = json.loads(raw)
        except requests.exceptions.RequestException as exc:
            logger.warning("Ollama request failed: %s", exc)
            self.metrics.record(phase, time.perf_counter() - started, False)
            return None
        except json.JSONDecodeError as exc:
            logger.warning("Ollama returned invalid JSON: %s", exc)
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
