"""Tests for shared Ollama request handling and metrics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from utils import ollama_client


def test_generate_tracks_server_metrics(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake image data")
    response = MagicMock()
    response.json.return_value = {
        "response": '{"detected": true}',
        "prompt_eval_count": 20,
        "eval_count": 7,
        "prompt_eval_duration": 1_000_000_000,
        "eval_duration": 2_500_000_000,
        "load_duration": 300_000_000,
        "total_duration": 4_000_000_000,
    }
    client = ollama_client.OllamaClient("http://localhost:11434", "model", timeout=45)

    with patch.object(ollama_client.requests, "post", return_value=response) as post:
        result = client.generate(str(image_path), "Analyze", "coarse")

    assert result == {"detected": True}
    post.assert_called_once()
    assert post.call_args.kwargs["timeout"] == 45
    assert post.call_args.kwargs["json"]["format"] == "json"
    stats = client.metrics.by_phase["coarse"]
    assert (stats.calls, stats.successes, stats.failures) == (1, 1, 0)
    assert stats.server_metrics["prompt_eval_count"] == 20
    assert stats.server_metrics["eval_count"] == 7
    assert stats.server_metrics["eval_duration"] == 2_500_000_000


def test_generate_tracks_failed_request(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake image data")
    client = ollama_client.OllamaClient("http://localhost:11434", "model")

    with patch.object(
        ollama_client.requests,
        "post",
        side_effect=requests.exceptions.RequestException("connection error"),
    ):
        assert client.generate(str(image_path), "Analyze", "fine") is None

    stats = client.metrics.by_phase["fine"]
    assert (stats.calls, stats.successes, stats.failures) == (1, 0, 1)


def test_generate_tracks_invalid_json_as_failure(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake image data")
    response = MagicMock()
    response.json.return_value = {"response": "not json"}
    client = ollama_client.OllamaClient("http://localhost:11434", "model")

    with patch.object(ollama_client.requests, "post", return_value=response):
        assert client.generate(str(image_path), "Analyze", "coarse") is None

    stats = client.metrics.by_phase["coarse"]
    assert (stats.calls, stats.successes, stats.failures) == (1, 0, 1)


def test_metrics_aggregate_calls_by_phase() -> None:
    metrics = ollama_client.OllamaMetrics()
    metrics.record(
        "coarse",
        1.0,
        True,
        {"prompt_eval_count": 12, "eval_count": 4, "eval_duration": 2_000_000_000},
    )
    metrics.record("coarse", 3.0, False)
    metrics.record("fine", 2.0, True)

    summary = metrics.format_summary()
    assert "coarse: 2 calls (1 successful, 1 failed), 4.0s total (2.0s average)" in summary
    assert "12 prompt tokens, 4 generated tokens" in summary
    assert "Ollama: eval 2.0s" in summary
    assert "fine: 1 call (1 successful, 0 failed)" in summary


def test_metrics_summary_for_no_calls() -> None:
    assert ollama_client.OllamaMetrics().format_summary() == (
        "Ollama request metrics: no requests made."
    )


def test_resolve_host_prefers_env_over_configured(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://plotter.local:11434")
    assert ollama_client.resolve_host("http://localhost:11434") == "http://plotter.local:11434"


def test_resolve_host_adds_missing_scheme(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "plotter.local:11434")
    assert ollama_client.resolve_host(None) == "http://plotter.local:11434"


def test_resolve_host_strips_trailing_slash(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "plotter.local:11434/")
    assert ollama_client.resolve_host(None) == "http://plotter.local:11434"


def test_resolve_host_ignores_blank_env(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "   ")
    assert ollama_client.resolve_host("http://remote:11434") == "http://remote:11434"


def test_resolve_host_falls_back_to_configured(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert ollama_client.resolve_host("http://remote:11434") == "http://remote:11434"


def test_resolve_host_normalizes_configured_host(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert ollama_client.resolve_host("remote:11434/") == "http://remote:11434"


def test_resolve_host_defaults_to_localhost(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert ollama_client.resolve_host(None) == ollama_client.DEFAULT_OLLAMA_HOST
    assert ollama_client.resolve_host("") == ollama_client.DEFAULT_OLLAMA_HOST


def test_client_session_pooling_and_context_manager(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"data")

    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": '{"detected": true}'}
    mock_session.post.return_value = mock_resp

    with ollama_client.OllamaClient(
        "http://localhost:11434", "model", session=mock_session
    ) as client:
        res = client.generate(str(image_path), "Prompt", "phase1")
        assert res == {"detected": True}
        assert mock_session.post.call_count == 1

    mock_session.close.assert_called_once()


def test_generate_includes_system_prompt_when_provided(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"data")
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"response": '{"detected": true}'}
    mock_session.post.return_value = mock_resp

    client = ollama_client.OllamaClient("http://localhost:11434", "model", session=mock_session)

    # When system_prompt is provided and non-empty
    client.generate(
        str(image_path),
        "Prompt",
        "phase1",
        system_prompt="You are an expert video analyst.",
    )
    payload = mock_session.post.call_args.kwargs["json"]
    assert payload["system"] == "You are an expert video analyst."

    # When system_prompt is None or whitespace, "system" key must be omitted
    client.generate(str(image_path), "Prompt", "phase1", system_prompt=None)
    payload_none = mock_session.post.call_args.kwargs["json"]
    assert "system" not in payload_none

    client.generate(str(image_path), "Prompt", "phase1", system_prompt="   ")
    payload_blank = mock_session.post.call_args.kwargs["json"]
    assert "system" not in payload_blank


def _generate_with(tmp_path: Path, body: dict, **client_kwargs):
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"data")
    session = MagicMock()
    resp = MagicMock()
    resp.json.return_value = body
    session.post.return_value = resp
    client = ollama_client.OllamaClient(
        "http://localhost:11434", "model", session=session, **client_kwargs
    )
    result = client.generate(str(image_path), "Prompt", "phase1")
    return result, session.post.call_args.kwargs["json"], client


def test_generate_falls_back_to_thinking_when_response_empty(tmp_path: Path) -> None:
    result, _, client = _generate_with(
        tmp_path, {"response": "", "thinking": '{\n "domain": "underwater"\n}'}
    )
    assert result == {"domain": "underwater"}
    assert client.metrics.by_phase["phase1"].successes == 1


def test_generate_falls_back_to_thinking_when_response_invalid(tmp_path: Path) -> None:
    result, _, _ = _generate_with(tmp_path, {"response": "  ", "thinking": '{"a": 1}'})
    assert result == {"a": 1}
    result, _, _ = _generate_with(tmp_path, {"response": "nope", "thinking": '{"a": 2}'})
    assert result == {"a": 2}


def test_generate_extracts_fenced_json_from_thinking(tmp_path: Path) -> None:
    body = {"response": "", "thinking": 'Let me think.\n```json\n{"a": 3}\n```'}
    result, _, _ = _generate_with(tmp_path, body)
    assert result == {"a": 3}


def test_generate_invalid_everywhere_returns_none(tmp_path: Path) -> None:
    result, _, client = _generate_with(tmp_path, {"response": "", "thinking": "hmm"})
    assert result is None
    assert client.metrics.by_phase["phase1"].failures == 1


def test_generate_sends_think_false(tmp_path: Path) -> None:
    _, payload, _ = _generate_with(tmp_path, {"response": "{}"})
    assert payload["think"] is False


def test_generate_sends_num_ctx_option_when_set(tmp_path: Path) -> None:
    _, payload, _ = _generate_with(tmp_path, {"response": "{}"}, num_ctx=32768)
    assert payload["options"] == {"num_ctx": 32768}


def test_generate_omits_options_without_num_ctx(tmp_path: Path) -> None:
    _, payload, _ = _generate_with(tmp_path, {"response": "{}"})
    assert "options" not in payload


def test_num_ctx_must_be_positive() -> None:
    import pytest

    with pytest.raises(ValueError, match="num_ctx"):
        ollama_client.OllamaClient("http://localhost:11434", "model", num_ctx=0)


def test_generate_sends_num_predict_option_when_set(tmp_path: Path) -> None:
    _, payload, _ = _generate_with(tmp_path, {"response": "{}"}, num_predict=256)
    assert payload["options"] == {"num_predict": 256}


def test_generate_sends_both_num_ctx_and_num_predict(tmp_path: Path) -> None:
    _, payload, _ = _generate_with(tmp_path, {"response": "{}"}, num_ctx=16384, num_predict=256)
    assert payload["options"] == {"num_ctx": 16384, "num_predict": 256}


def test_num_predict_must_be_positive() -> None:
    import pytest

    with pytest.raises(ValueError, match="num_predict"):
        ollama_client.OllamaClient("http://localhost:11434", "model", num_predict=0)
    with pytest.raises(ValueError, match="num_predict"):
        ollama_client.OllamaClient("http://localhost:11434", "model", num_predict=-10)
    with pytest.raises(ValueError, match="num_predict"):
        ollama_client.OllamaClient("http://localhost:11434", "model", num_predict=True)


def test_warns_on_runaway_thinking_tokens(tmp_path: Path, caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        _generate_with(tmp_path, {"response": '{"detected": true}', "eval_count": 600})
    assert any(
        "qwen3-vl-thinking" in record.message or "600 tokens" in record.message
        for record in caplog.records
    )
