"""Shared logging configuration for the pipeline scripts.

Every pipeline step logs through the standard library, so a single file handler
configured here is enough to persist a complete run transcript. The scripts do
not use `print()`, and ffmpeg output is captured rather than streamed, so no
stdout redirection is needed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Marks the handlers this module installs, so reconfiguring replaces only those
# and leaves handlers owned by an embedding application (or pytest) untouched.
_OWNED_ATTR = "_dive_clip_owned"
_CONSOLE: Console | None = None


def get_console() -> Console:
    """Return the shared stderr Console instance across logging and progress displays."""
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = Console(stderr=True)
    return _CONSOLE


def default_log_path(output_dir: Path) -> Path:
    """Return a timestamped logfile path under `<output_dir>/logs/`."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path(output_dir) / "logs" / f"run-{stamp}.log"


def configure_logging(
    debug: bool = False,
    log_file: Path | None = None,
    console: Console | None = None,
) -> Path | None:
    """Send root logging to stderr and, when given, to `log_file`.

    Terminal logging uses `RichHandler` for clean visual output. The logfile
    uses standard timestamped formatting in append mode so a step that is
    retried within the same run keeps a single transcript. Returns the resolved
    logfile path, or None when no logfile was requested.
    """
    active_console = console or get_console()
    console_handler = RichHandler(
        console=active_console,
        show_time=True,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    handlers: list[logging.Handler] = [console_handler]
    resolved: Path | None = None

    if log_file is not None:
        resolved = Path(log_file)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handlers.append(file_handler)

    root = logging.getLogger()

    # Close the previous configuration's file handles before dropping them.
    for existing in list(root.handlers):
        if getattr(existing, _OWNED_ATTR, False):
            root.removeHandler(existing)
            existing.close()

    root.setLevel(logging.DEBUG if debug else logging.INFO)
    for handler in handlers:
        setattr(handler, _OWNED_ATTR, True)
        root.addHandler(handler)

    return resolved
