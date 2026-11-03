"""Configuration loading and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from utils.env import load_dotenv


class ConfigError(Exception):
    """Raised when configuration file cannot be read, parsed, or validated."""


def load_config(
    config_path: str | Path = "config.yaml",
    *,
    default_to_empty: bool = False,
) -> dict[str, Any]:
    """Load and parse a YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.
    default_to_empty:
        If True and the file does not exist, return an empty dictionary instead
        of raising ConfigError.

    Returns
    -------
    dict[str, Any]:
        Parsed configuration dictionary.

    Raises
    ------
    ConfigError:
        If the file cannot be found (when default_to_empty is False), cannot be read,
        contains invalid YAML syntax, or top-level content is not a mapping.
    """
    path = Path(config_path)
    # Secrets such as OPENROUTER_API_KEY may live in a .env next to the config or in the cwd.
    load_dotenv(path.resolve().parent / ".env")
    load_dotenv(".env")
    if not path.exists():
        if default_to_empty:
            return {}
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse config file '{path}':\n{exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Error reading config file '{path}': {exc}") from exc

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigError(
            f"Invalid configuration in '{path}': expected a mapping/dictionary at top level, got {type(data).__name__}"
        )

    return data


def load_general_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Return the `general` section of a config file, or {} if unavailable.

    If the file does not exist, an empty dictionary is returned.

    Raises
    ------
    ConfigError:
        If the file contains invalid YAML syntax, cannot be read, or if the
        general section is not a dictionary.
    """
    config = load_config(config_path, default_to_empty=True)
    general = config.get("general")
    if general is None:
        return {}
    if not isinstance(general, dict):
        raise ConfigError(
            f"Invalid 'general' section in '{config_path}': expected a dictionary, got {type(general).__name__}"
        )
    return general
