"""Tests for utils/config.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from utils.config import ConfigError, load_config, load_general_config


def test_load_config_success(tmp_path: Path) -> None:
    """Valid YAML should load into a dictionary."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("general:\n  project_name: test\n  output_dir: ./out\n")

    data = load_config(config_file)
    assert data == {"general": {"project_name": "test", "output_dir": "./out"}}


def test_load_config_default_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """load_config without arguments should default to 'config.yaml' in current directory."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("general:\n  project_name: default\n")

    assert load_config() == {"general": {"project_name": "default"}}


def test_load_config_not_found(tmp_path: Path) -> None:
    """Missing config file should raise ConfigError by default."""
    missing = tmp_path / "nonexistent.yaml"
    with pytest.raises(ConfigError) as exc_info:
        load_config(missing)
    assert "Config file not found" in str(exc_info.value)


def test_load_config_default_to_empty(tmp_path: Path) -> None:
    """Missing config file returns {} when default_to_empty is True."""
    missing = tmp_path / "nonexistent.yaml"
    assert load_config(missing, default_to_empty=True) == {}


def test_load_config_empty_file(tmp_path: Path) -> None:
    """Empty or comment-only YAML file should return empty dictionary."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("# Only comments\n")
    assert load_config(config_file) == {}


def test_load_config_syntax_error(tmp_path: Path) -> None:
    """Malformed YAML (e.g. concatenated lines) should raise ConfigError with details."""
    config_file = tmp_path / "invalid.yaml"
    # Recreate the exact kind of error from the user: concatenated mapping lines
    config_file.write_text("general:\n  input_dir: /path/in  output_dir: /path/out\n")

    with pytest.raises(ConfigError) as exc_info:
        load_config(config_file)
    error_msg = str(exc_info.value)
    assert "Failed to parse config file" in error_msg
    assert "invalid.yaml" in error_msg


def test_load_config_non_dict(tmp_path: Path) -> None:
    """YAML file containing a scalar or list should raise ConfigError."""
    config_file = tmp_path / "list.yaml"
    config_file.write_text("- item1\n- item2\n")

    with pytest.raises(ConfigError) as exc_info:
        load_config(config_file)
    assert "expected a mapping/dictionary at top level" in str(exc_info.value)


def test_load_config_os_error(tmp_path: Path) -> None:
    """OSError during file read should raise ConfigError."""
    config_file = tmp_path / "unreadable.yaml"
    config_file.write_text("key: value\n")

    with patch("builtins.open", side_effect=PermissionError("Permission denied")):
        with pytest.raises(ConfigError) as exc_info:
            load_config(config_file)
        assert "Error reading config file" in str(exc_info.value)


def test_load_general_config_success(tmp_path: Path) -> None:
    """load_general_config should return the general section."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("general:\n  input_dir: /in\n  output_dir: /out\n")

    assert load_general_config(config_file) == {"input_dir": "/in", "output_dir": "/out"}


def test_load_general_config_missing_file(tmp_path: Path) -> None:
    """load_general_config returns {} when file does not exist."""
    missing = tmp_path / "nonexistent.yaml"
    assert load_general_config(missing) == {}


def test_load_general_config_no_general_section(tmp_path: Path) -> None:
    """load_general_config returns {} when general section is absent."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("ollama:\n  host: http://localhost:11434\n")
    assert load_general_config(config_file) == {}


def test_load_general_config_non_dict_section(tmp_path: Path) -> None:
    """load_general_config raises ConfigError if general is not a mapping."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("general: not_a_dict\n")

    with pytest.raises(ConfigError) as exc_info:
        load_general_config(config_file)
    assert "Invalid 'general' section" in str(exc_info.value)


def test_load_general_config_syntax_error(tmp_path: Path) -> None:
    """load_general_config raises ConfigError if YAML syntax is invalid."""
    config_file = tmp_path / "broken.yaml"
    config_file.write_text("general:\n  bad: [unclosed list\n")

    with pytest.raises(ConfigError) as exc_info:
        load_general_config(config_file)
    assert "Failed to parse config file" in str(exc_info.value)
