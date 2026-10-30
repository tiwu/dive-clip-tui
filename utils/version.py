"""Version utilities for dive-clip-tui."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def get_version() -> str:
    """Return package version from installed metadata or pyproject.toml fallback."""
    try:
        return version("dive-clip-tui")
    except PackageNotFoundError, Exception:
        pass

    try:
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            match = re.search(
                r'(?m)^version = "([^"]+)"',
                pyproject_path.read_text(encoding="utf-8"),
            )
            if match:
                return match.group(1)
    except Exception:
        pass

    return "unknown"
