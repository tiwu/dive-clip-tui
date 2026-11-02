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
