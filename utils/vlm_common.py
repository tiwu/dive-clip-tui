"""Shared building blocks for VLM backends (Ollama, OpenRouter).

Holds the per-phase request metrics, JSON reply extraction, image encoding and the
``BaseVLMClient`` lifecycle (session pooling, context manager) both clients share.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Self

import requests

_SERVER_METRIC_FIELDS = (
    "prompt_eval_count",
    "eval_count",
    "prompt_eval_duration",
    "eval_duration",
    "load_duration",
    "total_duration",
)


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


def extract_json_object(*candidates: str | None) -> dict[str, Any] | None:
    """Return the first JSON object found in ``candidates``, tried in order.

    Each candidate is parsed as-is, then via its first ``{...}`` block (handles markdown
    fences and prose around the JSON). Empty or ``None`` candidates are skipped, so
    ``extract_json_object(response, thinking)`` falls back to a thinking model's
    ``thinking`` field when ``response`` is empty.
    """
    for text in candidates:
        if not isinstance(text, str) or not text.strip():
            continue
        parsed = _parse_object(text)
        if parsed is None:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            parsed = _parse_object(match.group(0)) if match else None
        if parsed is not None:
            return parsed
    return None


def _parse_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def encode_image_b64(image_path: str) -> str:
    """Read an image file and return its base64 text (raises ``OSError`` if unreadable)."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class BaseVLMClient:
    """Model, timeout, metrics and pooled-session lifecycle shared by VLM clients."""

    def __init__(
        self,
        model: str,
        timeout: int,
        metrics: OllamaMetrics | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.metrics = metrics if metrics is not None else OllamaMetrics()
        self.session = session if session is not None else requests.Session()

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
