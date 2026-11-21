"""Tests for utils.text_llm pluggable storyline LLM clients."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from utils import text_llm
from utils.text_llm import (
    ClaudeTextClient,
    GeminiTextClient,
    OllamaTextClient,
    OpenRouterTextClient,
    TextLLMError,
    create_text_llm_client,
    resolve_storyline_llm_settings,
)


def test_resolve_storyline_llm_settings_precedence() -> None:
    """CLI overrides win over storyline config, which falls back to cloud_storyline and ollama."""
    cfg = {
        "ollama": {"host": "http://localhost:11434", "model": "qwen3-vl:8b-instruct"},
        "cloud_storyline": {"provider": "claude", "model": "claude-legacy"},
    }
    provider, model, _ = resolve_storyline_llm_settings(cfg)
    assert provider == "claude"
    assert model == "claude-legacy"

    cfg["storyline"] = {"provider": "ollama", "model": "qwen3-text"}
    provider, model, _ = resolve_storyline_llm_settings(cfg)
    assert provider == "ollama"
    assert model == "qwen3-text"

    provider, model, _ = resolve_storyline_llm_settings(
        cfg, provider_override="openrouter", model_override="google/gemini-2.5-pro"
    )
    assert provider == "openrouter"
    assert model == "google/gemini-2.5-pro"

    with pytest.raises(TextLLMError, match="Unknown storyline provider"):
        resolve_storyline_llm_settings(cfg, provider_override="bogus")


def test_ollama_text_client_extracts_json_and_thinking() -> None:
    """OllamaTextClient parses JSON from response or thinking field."""
    resp = MagicMock()
    resp.json.return_value = {
        "response": "",
        "thinking": '```json\n{"proposals": [{"id": "t"}]}\n```',
    }
    client = OllamaTextClient("http://localhost:11434", "qwen3-vl:8b-instruct", timeout=30)

    with patch.object(text_llm.requests, "post", return_value=resp) as post:
        out = client.generate_json("Plan prompt", system_prompt="Be concise")

    assert out == {"proposals": [{"id": "t"}]}
    assert post.call_args.kwargs["json"]["format"] == "json"
    assert post.call_args.kwargs["json"]["system"] == "Be concise"


def test_openrouter_text_client_posts_text_payload() -> None:
    """OpenRouterTextClient sends text chat messages with json_object response_format."""
    session = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": {"content": '{"proposals": []}'}}]}
    session.post.return_value = resp

    client = OpenRouterTextClient("or-key", "google/gemini-2.5-flash", session=session)
    out = client.generate_json("Catalog prompt", system_prompt="System")
    assert out == {"proposals": []}
    payload = session.post.call_args.kwargs["json"]
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0]["role"] == "system"


def test_claude_and_gemini_text_clients_and_factory(monkeypatch) -> None:
    """create_text_llm_client validates env keys and constructs Claude and Gemini clients."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(TextLLMError, match="OPENROUTER_API_KEY"):
        create_text_llm_client({}, provider="openrouter")
    with pytest.raises(TextLLMError, match="ANTHROPIC_API_KEY"):
        create_text_llm_client({}, provider="claude")
    with pytest.raises(TextLLMError, match="GEMINI_API_KEY"):
        create_text_llm_client({}, provider="gemini")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ok-test")

    assert isinstance(create_text_llm_client({}, provider="ollama"), OllamaTextClient)
    assert isinstance(create_text_llm_client({}, provider="openrouter"), OpenRouterTextClient)
    claude_client = create_text_llm_client({}, provider="claude")
    gemini_client = create_text_llm_client({}, provider="gemini")
    assert isinstance(claude_client, ClaudeTextClient)
    assert isinstance(gemini_client, GeminiTextClient)

    mock_anthropic = MagicMock()
    mock_msg = MagicMock(text='{"proposals": [{"id": "highlight"}]}')
    mock_anthropic.Anthropic.return_value.messages.create.return_value = MagicMock(
        content=[mock_msg]
    )
    monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)
    assert claude_client.generate_json("hello", system_prompt="sys") == {
        "proposals": [{"id": "highlight"}]
    }

    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value.generate_content.return_value = MagicMock(
        text='{"proposals": [{"id": "teaser"}]}'
    )
    monkeypatch.setitem(sys.modules, "google.generativeai", mock_genai)
    assert gemini_client.generate_json("hello", system_prompt="sys") == {
        "proposals": [{"id": "teaser"}]
    }
