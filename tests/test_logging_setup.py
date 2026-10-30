"""Tests for utils/logging_setup.py."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from utils.logging_setup import _OWNED_ATTR, configure_logging, default_log_path


def _owned_file_handlers() -> list[logging.FileHandler]:
    """File handlers installed by configure_logging, ignoring foreign ones."""
    return [
        h
        for h in logging.getLogger().handlers
        if isinstance(h, logging.FileHandler) and getattr(h, _OWNED_ATTR, False)
    ]


@pytest.fixture(autouse=True)
def restore_root_logging():
    """Save and restore the root logger so tests cannot leak handlers."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


def test_default_log_path_is_timestamped(tmp_path: Path) -> None:
    """The default logfile lives in <output>/logs and carries a timestamp."""
    log_path = default_log_path(tmp_path)

    assert log_path.parent == tmp_path / "logs"
    assert re.fullmatch(r"run-\d{8}-\d{6}\.log", log_path.name)


def test_configure_logging_without_logfile_returns_none() -> None:
    """No logfile requested means no file handler and a None return."""
    assert configure_logging() is None
    assert _owned_file_handlers() == []


def test_configure_logging_creates_parent_dirs(tmp_path: Path) -> None:
    """A nested logfile path should be created on demand."""
    log_path = tmp_path / "logs" / "run.log"

    resolved = configure_logging(log_file=log_path)

    assert resolved == log_path
    assert log_path.exists()


def test_configure_logging_writes_records(tmp_path: Path) -> None:
    """Messages logged after configuration should reach the logfile."""
    log_path = default_log_path(tmp_path)
    configure_logging(log_file=log_path)

    logging.getLogger("test").warning("something happened")

    assert "something happened" in log_path.read_text(encoding="utf-8")


def test_configure_logging_appends(tmp_path: Path) -> None:
    """Reconfiguring for the same path must not truncate an existing transcript."""
    log_path = tmp_path / "logs" / "run.log"

    configure_logging(log_file=log_path)
    logging.getLogger("first").info("step one")
    configure_logging(log_file=log_path)
    logging.getLogger("second").info("step two")

    content = log_path.read_text(encoding="utf-8")
    assert "step one" in content
    assert "step two" in content


def test_configure_logging_replaces_previous_file_handler(tmp_path: Path) -> None:
    """Switching logfiles should close the old handler instead of stacking them."""
    configure_logging(log_file=tmp_path / "a.log")
    configure_logging(log_file=tmp_path / "b.log")

    file_handlers = _owned_file_handlers()
    assert len(file_handlers) == 1
    assert Path(file_handlers[0].baseFilename) == tmp_path / "b.log"


def test_configure_logging_keeps_foreign_handlers() -> None:
    """Handlers owned by an embedding app or test runner must survive."""
    root = logging.getLogger()
    foreign = logging.StreamHandler()
    root.addHandler(foreign)

    configure_logging()

    assert foreign in root.handlers


def test_configure_logging_debug_level(tmp_path: Path) -> None:
    """debug=True should lower the root level to DEBUG."""
    configure_logging(debug=True)

    assert logging.getLogger().level == logging.DEBUG
