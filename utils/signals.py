"""Signal handling utilities for graceful process shutdown."""

from __future__ import annotations

import signal


def setup_signal_handlers() -> None:
    """Ensure SIGINT and SIGTERM raise KeyboardInterrupt for clean shutdown."""

    def _handler(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    except ValueError, AttributeError:
        pass
