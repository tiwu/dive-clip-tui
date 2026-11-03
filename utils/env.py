"""Minimal ``.env`` loader so secrets like OPENROUTER_API_KEY can live outside config.yaml."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    key, sep, value = line.partition("=")
    key = key.strip()
    if not sep or not key:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    elif " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return key, value


def load_dotenv(path: str | Path = ".env") -> bool:
    """Load ``KEY=value`` pairs from ``path`` into ``os.environ``.

    Variables already set in the environment win over the file. Returns True when the
    file existed and was read.
    """
    env_path = Path(path)
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return False
    for raw in text.splitlines():
        parsed = _parse_line(raw)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)
    logger.debug("Loaded environment from %s", env_path)
    return True
