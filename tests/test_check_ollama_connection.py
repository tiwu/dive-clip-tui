"""Tests for scripts/check_ollama_connection.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from scripts.check_ollama_connection import check_ollama_connection, main
from utils.config import ConfigError


def test_check_ollama_connection_success() -> None:
    """check_ollama_connection should return True on successful 200 response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "response": "Hello underwater world!",
        "eval_count": 12,
        "total_duration": 1_500_000_000,
    }
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response):
        result = check_ollama_connection("http://localhost:11434", "qwen2.5vl:3b", timeout=5)

    assert result is True


def test_check_ollama_connection_connection_error() -> None:
    """check_ollama_connection should return False on ConnectionError."""
    with patch(
        "requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")
    ):
        result = check_ollama_connection("http://localhost:11434", "qwen2.5vl:3b", timeout=5)

    assert result is False


def test_check_ollama_connection_timeout() -> None:
    """check_ollama_connection should return False on Timeout."""
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Read timeout")):
        result = check_ollama_connection("http://localhost:11434", "qwen2.5vl:3b", timeout=5)

    assert result is False


def test_check_ollama_connection_http_error() -> None:
    """check_ollama_connection should return False on HTTPError."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal server error"
    with patch("requests.post", side_effect=requests.exceptions.HTTPError(response=mock_resp)):
        result = check_ollama_connection("http://localhost:11434", "qwen2.5vl:3b", timeout=5)

    assert result is False


def test_check_ollama_connection_generic_exception() -> None:
    """check_ollama_connection should return False on unexpected exception."""
    with patch("requests.post", side_effect=RuntimeError("Unexpected")):
        result = check_ollama_connection("http://localhost:11434", "qwen2.5vl:3b", timeout=5)

    assert result is False


def test_main_success() -> None:
    """main should return 0 when connection succeeds."""
    config = {
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5vl:3b",
            "timeout_seconds": 10,
        }
    }
    with (
        patch("scripts.check_ollama_connection.load_config", return_value=config),
        patch("scripts.check_ollama_connection.check_ollama_connection", return_value=True),
    ):
        assert main() == 0


def test_main_connection_failure() -> None:
    """main should return 1 when connection check fails."""
    config = {
        "ollama": {
            "host": "http://localhost:11434",
            "model": "qwen2.5vl:3b",
            "timeout_seconds": 10,
        }
    }
    with (
        patch("scripts.check_ollama_connection.load_config", return_value=config),
        patch("scripts.check_ollama_connection.check_ollama_connection", return_value=False),
    ):
        assert main() == 1


def test_main_config_error() -> None:
    """main should return 1 when configuration fails to load."""
    with patch(
        "scripts.check_ollama_connection.load_config", side_effect=ConfigError("Invalid YAML")
    ):
        assert main() == 1


def test_main_missing_model() -> None:
    """main should return 1 when model is empty."""
    config = {"ollama": {"host": "http://localhost:11434", "model": ""}}
    with patch("scripts.check_ollama_connection.load_config", return_value=config):
        assert main() == 1
