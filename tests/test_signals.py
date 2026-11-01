"""Tests for utils/signals.py."""

from __future__ import annotations

import signal
from unittest.mock import patch

import pytest

from utils.signals import setup_signal_handlers


def test_setup_signal_handlers_installs_handlers() -> None:
    """setup_signal_handlers should register handlers for SIGINT and SIGTERM."""
    with patch("signal.signal") as mock_signal:
        setup_signal_handlers()
        assert mock_signal.call_count == 2
        calls = [c[0][0] for c in mock_signal.call_args_list]
        assert signal.SIGINT in calls
        assert signal.SIGTERM in calls


def test_setup_signal_handlers_catches_value_error() -> None:
    """setup_signal_handlers should silently ignore ValueError (e.g., non-main thread)."""
    with patch("signal.signal", side_effect=ValueError("signal only works in main thread")):
        # Should not raise
        setup_signal_handlers()


def test_signal_handler_raises_keyboard_interrupt() -> None:
    """Installed handler should raise KeyboardInterrupt when invoked."""
    captured_handler = None

    def fake_signal(sig, handler):
        nonlocal captured_handler
        captured_handler = handler

    with patch("signal.signal", side_effect=fake_signal):
        setup_signal_handlers()

    assert captured_handler is not None
    with pytest.raises(KeyboardInterrupt):
        captured_handler(signal.SIGINT, None)
