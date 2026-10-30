"""Tests for utils/version.py."""

from __future__ import annotations

from unittest.mock import patch

from utils.version import get_version


def test_get_version_metadata() -> None:
    """get_version should return package metadata version when available."""
    with patch("utils.version.version", return_value="1.2.3"):
        assert get_version() == "1.2.3"


def test_get_version_fallback(tmp_path) -> None:
    """get_version should fall back to pyproject.toml if metadata is missing."""
    with patch("utils.version.version", side_effect=Exception("not installed")):
        v = get_version()
        # In this repo, pyproject.toml exists and has a valid SemVer
        assert v != "unknown"
        assert "." in v
