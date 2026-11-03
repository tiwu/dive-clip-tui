"""Tests for the VLM client factory and OpenRouter backend."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import requests

from utils.ollama_client import OllamaClient
from utils.vlm_client import (
    OPENROUTER_URL,
    OpenRouterClient,
    RateLimiter,
    create_vlm_client,
    extract_json_object,
)


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "frame.jpg"
    path.write_bytes(b"\xff\xd8fakejpeg")
    return str(path)


def _response(status=200, content='{"detected": true}', usage=None, headers=None):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": usage if usage is not None else {"prompt_tokens": 10, "completion_tokens": 3},
    }
    if status >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status}")
    else:
        resp.raise_for_status.return_value = None
    return resp


def _client(session, **kwargs):
    return OpenRouterClient("key", session=session, sleep=MagicMock(), **kwargs)


# --- factory -----------------------------------------------------------------


def test_factory_defaults_to_ollama():
    client = create_vlm_client({"ollama": {"model": "m", "timeout_seconds": 5}})
    assert isinstance(client, OllamaClient)
    assert client.model == "m"
    assert client.timeout == 5


def test_factory_explicit_ollama():
    assert isinstance(create_vlm_client({"vlm": {"provider": "Ollama"}}), OllamaClient)


def test_factory_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    cfg = {"vlm": {"provider": "openrouter", "openrouter": {"model": "x/y", "max_retries": 1}}}
    client = create_vlm_client(cfg)
    assert isinstance(client, OpenRouterClient)
    assert client.model == "x/y"
    assert client.max_retries == 1
    assert client.api_key == "secret"


def test_factory_openrouter_defaults(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    client = create_vlm_client({"vlm": {"provider": "openrouter"}})
    assert client.model == "nvidia/nemotron-3.5-super-vl:free"


def test_factory_openrouter_missing_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        create_vlm_client({"vlm": {"provider": "openrouter"}})


def test_factory_unknown_provider():
    with pytest.raises(ValueError, match=r"Unknown vlm\.provider"):
        create_vlm_client({"vlm": {"provider": "nope"}})


# --- JSON extraction -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('Sure!\n```json\n{"a": 1}\n```', {"a": 1}),
        ("no json here", None),
        ("{broken", None),
        ("{not: json}", None),
        ("[1, 2]", None),
        (None, None),
    ],
)
def test_extract_json_object(text, expected):
    assert extract_json_object(text) == expected


# --- OpenRouterClient ------------------------------------------------------------


def test_generate_success_builds_payload_and_records_metrics(image):
    session = MagicMock()
    session.post.return_value = _response()
    client = _client(session, model="a/b", timeout=7)

    result = client.generate(image, "look", phase="coarse", system_prompt=" sys ")

    assert result == {"detected": True}
    args, kwargs = session.post.call_args
    assert args[0] == OPENROUTER_URL
    assert kwargs["headers"] == {"Authorization": "Bearer key"}
    assert kwargs["timeout"] == 7
    payload = kwargs["json"]
    assert payload["model"] == "a/b"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    parts = payload["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    stats = client.metrics.by_phase["coarse"]
    assert (stats.calls, stats.successes, stats.failures) == (1, 1, 0)
    assert stats.server_metrics == {"prompt_eval_count": 10, "eval_count": 3}


def test_generate_without_system_prompt(image):
    session = MagicMock()
    session.post.return_value = _response()
    _client(session).generate(image, "look")
    messages = session.post.call_args.kwargs["json"]["messages"]
    assert [m["role"] for m in messages] == ["user"]


def test_generate_recovers_json_from_prose(image):
    session = MagicMock()
    session.post.return_value = _response(content='Here: {"detected": false} done')
    assert _client(session).generate(image, "p") == {"detected": False}


def test_generate_invalid_json_returns_none(image):
    session = MagicMock()
    session.post.return_value = _response(content="sorry, cannot")
    client = _client(session)
    assert client.generate(image, "p", phase="fine") is None
    assert client.metrics.by_phase["fine"].failures == 1


def test_generate_missing_usage_is_tolerated(image):
    session = MagicMock()
    resp = _response()
    resp.json.return_value = {"choices": [{"message": {"content": "{}"}}], "usage": "bad"}
    session.post.return_value = resp
    client = _client(session)
    assert client.generate(image, "p") == {}
    assert client.metrics.by_phase["unspecified"].server_metrics == {}


def test_generate_unreadable_image_returns_none(tmp_path):
    session = MagicMock()
    client = _client(session)
    assert client.generate(str(tmp_path / "missing.jpg"), "p") is None
    session.post.assert_not_called()
    assert client.metrics.by_phase["unspecified"].failures == 1


def test_generate_http_error_returns_none(image):
    session = MagicMock()
    session.post.return_value = _response(status=401)
    client = _client(session)
    assert client.generate(image, "p") is None
    assert session.post.call_count == 1
    assert client.metrics.by_phase["unspecified"].failures == 1


def test_generate_connection_error_returns_none(image):
    session = MagicMock()
    session.post.side_effect = requests.exceptions.ConnectionError("down")
    client = _client(session)
    assert client.generate(image, "p") is None


def test_generate_malformed_body_returns_none(image):
    session = MagicMock()
    resp = _response()
    resp.json.return_value = {"error": "nope"}
    session.post.return_value = resp
    assert _client(session).generate(image, "p") is None


def test_retry_after_429_then_success(image):
    session = MagicMock()
    session.post.side_effect = [_response(429, headers={"Retry-After": "2"}), _response()]
    client = _client(session)
    assert client.generate(image, "p") == {"detected": True}
    assert session.post.call_count == 2
    client._sleep.assert_called_with(2.0)


def test_retry_backoff_without_retry_after(image):
    session = MagicMock()
    session.post.side_effect = [
        _response(503),
        _response(500, headers={"Retry-After": "x"}),
        _response(),
    ]
    client = _client(session)
    assert client.generate(image, "p") is not None
    assert [c.args[0] for c in client._sleep.call_args_list] == [1.0, 2.0]


def test_retry_gives_up_after_max_retries(image):
    session = MagicMock()
    session.post.return_value = _response(429)
    client = _client(session, max_retries=2)
    assert client.generate(image, "p") is None
    assert session.post.call_count == 3
    assert client.metrics.by_phase["unspecified"].failures == 1


def test_context_manager_closes_session():
    session = MagicMock()
    with _client(session):
        pass
    session.close.assert_called_once()


# --- RateLimiter -----------------------------------------------------------------


def test_rate_limiter_delays_over_limit():
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(2, clock=lambda: now[0], sleep=sleep)
    limiter.wait()
    now[0] = 10.0
    limiter.wait()
    limiter.wait()
    assert sleeps == [50.0]


def test_rate_limiter_window_expires():
    now = [0.0]
    sleeps = []
    limiter = RateLimiter(1, clock=lambda: now[0], sleep=sleeps.append)
    limiter.wait()
    now[0] = 61.0
    limiter.wait()
    assert sleeps == []


def test_rate_limiter_disabled():
    sleeps = []
    limiter = RateLimiter(0, sleep=sleeps.append)
    for _ in range(5):
        limiter.wait()
    assert sleeps == []


def test_payload_is_json_serializable(image):
    session = MagicMock()
    session.post.return_value = _response()
    _client(session).generate(image, "p")
    json.dumps(session.post.call_args.kwargs["json"])
