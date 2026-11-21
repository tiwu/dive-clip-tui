"""Unified pluggable text-LLM client for storyline planning.

Supports local Ollama, OpenRouter, Anthropic Claude, and Google Gemini behind a
single ``TextLLMClient`` protocol, cleanly separated from frame-based ``VLMClient``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections.abc import Mapping
from typing import Any, Protocol

import requests

from utils.ollama_client import resolve_host
from utils.vlm_client import (
    OPENROUTER_API_KEY_ENV_VAR,
    OPENROUTER_URL,
    RateLimiter,
)
from utils.vlm_common import extract_json_object

__all__ = [
    "ANTHROPIC_API_KEY_ENV_VAR",
    "DEFAULT_TEXT_LLM_MODELS",
    "GEMINI_API_KEY_ENV_VAR",
    "SUPPORTED_TEXT_LLM_PROVIDERS",
    "ClaudeTextClient",
    "GeminiTextClient",
    "OllamaTextClient",
    "OpenRouterTextClient",
    "TextLLMClient",
    "TextLLMError",
    "create_text_llm_client",
    "resolve_storyline_llm_settings",
]

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
SUPPORTED_TEXT_LLM_PROVIDERS: tuple[str, ...] = ("ollama", "openrouter", "claude", "gemini")

DEFAULT_TEXT_LLM_MODELS: dict[str, str] = {
    "ollama": "qwen3-vl:8b-instruct",
    "openrouter": "google/gemini-2.5-flash",
    "claude": "claude-3-5-sonnet-latest",
    "gemini": "gemini-2.5-flash",
}

_RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_MAX_BACKOFF_SECONDS = 60.0


class TextLLMError(RuntimeError):
    """Raised when a text-LLM backend is misconfigured or fails to return valid output."""


class TextLLMClient(Protocol):
    """Interface for text-only storyline planning backends."""

    provider: str
    model: str

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Send a prompt and return the parsed JSON object response."""
        ...


class OllamaTextClient:
    """Local Ollama text-generation client using ``/api/generate`` JSON mode."""

    provider = "ollama"

    def __init__(
        self,
        host: str,
        model: str,
        timeout: int = 120,
        num_ctx: int | None = None,
        num_predict: int | None = 4096,
        session: requests.Session | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.session = session if session is not None else requests.Session()

    def close(self) -> None:
        self.session.close()

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "think": False,
        }
        if system_prompt and system_prompt.strip():
            payload["system"] = system_prompt.strip()
        options: dict[str, Any] = {"temperature": temperature}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        predict_cap = max(max_tokens, self.num_predict or 0) if self.num_predict else max_tokens
        if predict_cap > 0:
            options["num_predict"] = predict_cap
        payload["options"] = options

        try:
            is_mocked = hasattr(requests.post, "mock_calls")
            post_fn = requests.post if is_mocked else self.session.post
            resp = post_fn(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            raise TextLLMError(f"Ollama storyline request failed: {exc}") from exc

        parsed = extract_json_object(data.get("response"), data.get("thinking"))
        if parsed is None:
            raise TextLLMError("Ollama storyline response did not contain a valid JSON object.")
        return parsed


class OpenRouterTextClient:
    """OpenRouter chat-completions client for text-only storyline planning."""

    provider = "openrouter"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_TEXT_LLM_MODELS["openrouter"],
        timeout: int = 120,
        requests_per_minute: int = 20,
        max_retries: int = 3,
        session: requests.Session | None = None,
        sleep=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session if session is not None else requests.Session()
        self._sleep = sleep
        self._limiter = RateLimiter(requests_per_minute, clock=clock, sleep=sleep)

    def close(self) -> None:
        self.session.close()

    def _post_with_retry(self, payload: dict[str, Any]) -> requests.Response:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        attempt = 0
        while True:
            self._limiter.wait()
            resp = self.session.post(
                OPENROUTER_URL, json=payload, headers=headers, timeout=self.timeout
            )
            if resp.status_code not in _RETRY_STATUS_CODES or attempt >= self.max_retries:
                resp.raise_for_status()
                return resp
            delay = min(2.0**attempt, _MAX_BACKOFF_SECONDS)
            retry_after = resp.headers.get("Retry-After", "")
            with contextlib.suppress(ValueError):
                delay = min(max(float(retry_after), 0.0), _MAX_BACKOFF_SECONDS)
            logger.warning(
                "OpenRouter returned %s; retrying in %.1fs (%d/%d)",
                resp.status_code,
                delay,
                attempt + 1,
                self.max_retries,
            )
            self._sleep(delay)
            attempt += 1

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = self._post_with_retry(payload)
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
        except (
            requests.exceptions.RequestException,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise TextLLMError(f"OpenRouter storyline request failed: {exc}") from exc

        parsed = extract_json_object(content)
        if parsed is None:
            raise TextLLMError("OpenRouter storyline response did not contain a valid JSON object.")
        return parsed


class ClaudeTextClient:
    """Anthropic Claude client for text storyline planning."""

    provider = "claude"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_TEXT_LLM_MODELS["claude"],
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key, timeout=float(self.timeout))
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt and system_prompt.strip():
            kwargs["system"] = system_prompt.strip()
        response = client.messages.create(**kwargs)
        return response.content[0].text if response.content else ""

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        raw = self.generate_text(
            prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens
        )
        parsed = extract_json_object(raw)
        if parsed is None:
            raise TextLLMError("Claude storyline response did not contain a valid JSON object.")
        return parsed


class GeminiTextClient:
    """Google Gemini client for text storyline planning."""

    provider = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_TEXT_LLM_MODELS["gemini"],
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model_instance = genai.GenerativeModel(self.model)
        full_prompt = (
            f"{system_prompt.strip()}\n\n{prompt}"
            if system_prompt and system_prompt.strip()
            else prompt
        )
        response = model_instance.generate_content(
            full_prompt, request_options={"timeout": self.timeout}
        )
        return response.text or ""

    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        raw = self.generate_text(
            prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens
        )
        parsed = extract_json_object(raw)
        if parsed is None:
            raise TextLLMError("Gemini storyline response did not contain a valid JSON object.")
        return parsed


def resolve_storyline_llm_settings(
    config: Mapping[str, Any],
    provider_override: str | None = None,
    model_override: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Resolve ``(provider, model, merged_settings)`` for storyline planning.

    Precedence for provider: ``provider_override`` -> ``storyline.provider`` ->
    ``cloud_storyline.provider`` -> ``"ollama"``.
    """
    storyline_cfg = config.get("storyline") if isinstance(config.get("storyline"), Mapping) else {}
    cloud_cfg = (
        config.get("cloud_storyline") if isinstance(config.get("cloud_storyline"), Mapping) else {}
    )
    ollama_cfg = config.get("ollama") if isinstance(config.get("ollama"), Mapping) else {}

    raw_provider = (
        provider_override or storyline_cfg.get("provider") or cloud_cfg.get("provider") or "ollama"
    )
    provider = str(raw_provider).strip().lower()
    if provider not in SUPPORTED_TEXT_LLM_PROVIDERS:
        raise TextLLMError(
            f"Unknown storyline provider {provider!r}; expected one of: "
            f"{', '.join(SUPPORTED_TEXT_LLM_PROVIDERS)}"
        )

    configured_model = None
    if model_override and str(model_override).strip():
        configured_model = str(model_override).strip()
    elif storyline_cfg.get("model") and (
        not provider_override
        or provider_override == str(storyline_cfg.get("provider") or "").strip().lower()
    ):
        configured_model = str(storyline_cfg["model"]).strip()
    elif (
        cloud_cfg.get("model") and str(cloud_cfg.get("provider") or "").strip().lower() == provider
    ):
        configured_model = str(cloud_cfg["model"]).strip()
    elif provider == "ollama" and ollama_cfg.get("model"):
        configured_model = str(ollama_cfg["model"]).strip()
    elif provider == "openrouter":
        or_storyline = storyline_cfg.get("openrouter")
        if isinstance(or_storyline, Mapping) and or_storyline.get("model"):
            configured_model = str(or_storyline["model"]).strip()

    model = configured_model or DEFAULT_TEXT_LLM_MODELS[provider]
    timeout_seconds = int(
        storyline_cfg.get("timeout_seconds") or ollama_cfg.get("timeout_seconds") or 120
    )
    settings = {
        "temperature": float(storyline_cfg.get("temperature", 0.4)),
        "timeout_seconds": timeout_seconds,
        "max_tokens": int(storyline_cfg.get("max_tokens", 4096)),
    }
    return provider, model, settings


def create_text_llm_client(
    config: Mapping[str, Any],
    provider: str | None = None,
    model: str | None = None,
) -> TextLLMClient:
    """Construct the configured ``TextLLMClient`` backend."""
    resolved_provider, resolved_model, settings = resolve_storyline_llm_settings(
        config, provider_override=provider, model_override=model
    )
    timeout = settings["timeout_seconds"]

    if resolved_provider == "ollama":
        ollama_cfg = config.get("ollama") if isinstance(config.get("ollama"), Mapping) else {}
        return OllamaTextClient(
            host=resolve_host(ollama_cfg.get("host")),
            model=resolved_model,
            timeout=timeout,
            num_ctx=ollama_cfg.get("num_ctx"),
            num_predict=max(int(ollama_cfg.get("num_predict") or 4096), 4096),
        )

    if resolved_provider == "openrouter":
        api_key = os.environ.get(OPENROUTER_API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise TextLLMError(
                f"Storyline provider is 'openrouter' but {OPENROUTER_API_KEY_ENV_VAR} is not set."
            )
        vlm_cfg = config.get("vlm") if isinstance(config.get("vlm"), Mapping) else {}
        or_cfg = vlm_cfg.get("openrouter") if isinstance(vlm_cfg.get("openrouter"), Mapping) else {}
        return OpenRouterTextClient(
            api_key=api_key,
            model=resolved_model,
            timeout=timeout,
            requests_per_minute=int(or_cfg.get("requests_per_minute", 20)),
            max_retries=int(or_cfg.get("max_retries", 3)),
        )

    if resolved_provider == "claude":
        api_key = os.environ.get(ANTHROPIC_API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise TextLLMError(
                f"Storyline provider is 'claude' but {ANTHROPIC_API_KEY_ENV_VAR} is not set."
            )
        return ClaudeTextClient(api_key=api_key, model=resolved_model, timeout=timeout)

    if resolved_provider == "gemini":
        api_key = os.environ.get(GEMINI_API_KEY_ENV_VAR, "").strip()
        if not api_key:
            raise TextLLMError(
                f"Storyline provider is 'gemini' but {GEMINI_API_KEY_ENV_VAR} is not set."
            )
        return GeminiTextClient(api_key=api_key, model=resolved_model, timeout=timeout)

    raise TextLLMError(f"Unsupported storyline provider: {resolved_provider!r}")
