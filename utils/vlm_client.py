"""VLM client protocol, OpenRouter backend, and provider factory for the scan pipeline."""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from collections import deque
from typing import Any, Protocol

import requests

from utils.ollama_client import OllamaClient, resolve_host
from utils.vlm_common import (
    BaseVLMClient,
    OllamaMetrics,
    encode_image_b64,
    extract_json_object,
)

__all__ = [
    "OpenRouterClient",
    "RateLimiter",
    "VLMClient",
    "create_vlm_client",
    "extract_json_object",
]

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3.5-super-vl:free"
SUPPORTED_PROVIDERS = ("ollama", "openrouter")
_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_MAX_BACKOFF_SECONDS = 60.0


class VLMClient(Protocol):
    """Interface the scan pipeline needs from any VLM backend."""

    metrics: OllamaMetrics

    def generate(
        self,
        image_path: str,
        prompt: str,
        phase: str = "unspecified",
        system_prompt: str | None = None,
    ) -> dict[str, Any] | None:
        """Analyze one image and return the parsed JSON reply, or None on failure."""
        ...

    def close(self) -> None: ...


class RateLimiter:
    """Sliding-window limiter allowing at most ``per_minute`` calls per 60 seconds."""

    def __init__(self, per_minute: int, clock=time.monotonic, sleep=time.sleep) -> None:
        self.per_minute = per_minute
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.per_minute <= 0:
            return
        with self._lock:
            now = self._clock()
            while self._calls and now - self._calls[0] >= 60.0:
                self._calls.popleft()
            if len(self._calls) >= self.per_minute:
                delay = 60.0 - (now - self._calls[0])
                if delay > 0:
                    self._sleep(delay)
                now = self._clock()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
            self._calls.append(now)


class OpenRouterClient(BaseVLMClient):
    """Send image prompts to OpenRouter's OpenAI-compatible chat completions API."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_OPENROUTER_MODEL,
        timeout: int = 120,
        requests_per_minute: int = 20,
        max_retries: int = 3,
        metrics: OllamaMetrics | None = None,
        session: requests.Session | None = None,
        sleep=time.sleep,
        clock=time.monotonic,
    ) -> None:
        super().__init__(model, timeout, metrics, session)
        self.api_key = api_key
        self.max_retries = max_retries
        self._sleep = sleep
        self._limiter = RateLimiter(requests_per_minute, clock=clock, sleep=sleep)

    def _build_payload(
        self, image_b64: str, prompt: str, system_prompt: str | None
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        })
        return {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

    def _post_with_retry(self, payload: dict[str, Any]) -> requests.Response:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        attempt = 0
        while True:
            self._limiter.wait()
            response = self.session.post(
                OPENROUTER_URL, json=payload, headers=headers, timeout=self.timeout
            )
            if response.status_code not in _RETRY_STATUS_CODES or attempt >= self.max_retries:
                response.raise_for_status()
                return response
            delay = min(2.0**attempt, _MAX_BACKOFF_SECONDS)
            retry_after = response.headers.get("Retry-After", "")
            with contextlib.suppress(ValueError):
                delay = min(max(float(retry_after), 0.0), _MAX_BACKOFF_SECONDS)
            logger.warning(
                "OpenRouter returned %s; retrying in %.1fs (%d/%d)",
                response.status_code,
                delay,
                attempt + 1,
                self.max_retries,
            )
            self._sleep(delay)
            attempt += 1

    def generate(
        self,
        image_path: str,
        prompt: str,
        phase: str = "unspecified",
        system_prompt: str | None = None,
    ) -> dict[str, Any] | None:
        started = time.perf_counter()
        try:
            image_b64 = encode_image_b64(image_path)
        except OSError as exc:
            logger.warning("Cannot read image %s: %s", image_path, exc)
            self.metrics.record(phase, time.perf_counter() - started, False)
            return None

        payload = self._build_payload(image_b64, prompt, system_prompt)
        try:
            response = self._post_with_retry(payload)
            logger.trace("OpenRouter response text: %s", response.text)
            result = response.json()
            content = result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as exc:
            logger.warning("OpenRouter request failed: %s", exc)
            self.metrics.record(phase, time.perf_counter() - started, False)
            return None
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning("OpenRouter returned an unexpected response: %s", exc)
            self.metrics.record(phase, time.perf_counter() - started, False)
            return None

        parsed = extract_json_object(content)
        usage = result.get("usage") if isinstance(result, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        # Map OpenAI-style usage onto the shared per-phase token counters.
        metrics_response = {
            "prompt_eval_count": usage.get("prompt_tokens"),
            "eval_count": usage.get("completion_tokens"),
        }
        if parsed is None:
            logger.warning("OpenRouter returned invalid JSON for phase %s", phase)
            self.metrics.record(phase, time.perf_counter() - started, False, metrics_response)
            return None
        self.metrics.record(phase, time.perf_counter() - started, True, metrics_response)
        return parsed


def create_vlm_client(config: dict[str, Any]) -> VLMClient:
    """Build the scan VLM client from ``config['vlm']`` (default: Ollama)."""
    vlm_cfg = config.get("vlm") or {}
    provider = str(vlm_cfg.get("provider", "ollama")).strip().lower()

    if provider == "ollama":
        ollama_cfg = config.get("ollama") or {}
        return OllamaClient(
            resolve_host(ollama_cfg.get("host")),
            ollama_cfg.get("model", "qwen2.5-vl:7b"),
            ollama_cfg.get("timeout_seconds", 90),
            num_ctx=ollama_cfg.get("num_ctx"),
            num_predict=ollama_cfg.get("num_predict"),
        )

    if provider == "openrouter":
        api_key = os.environ.get(OPENROUTER_API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise ValueError(
                f"vlm.provider is 'openrouter' but {OPENROUTER_API_KEY_ENV_VAR} is not set"
            )
        or_cfg = vlm_cfg.get("openrouter") or {}
        return OpenRouterClient(
            api_key,
            or_cfg.get("model", DEFAULT_OPENROUTER_MODEL),
            or_cfg.get("timeout_seconds", 120),
            or_cfg.get("requests_per_minute", 20),
            or_cfg.get("max_retries", 3),
        )

    raise ValueError(
        f"Unknown vlm.provider {provider!r}; expected one of: {', '.join(SUPPORTED_PROVIDERS)}"
    )
