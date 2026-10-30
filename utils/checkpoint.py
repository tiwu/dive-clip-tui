"""Pipeline checkpointing and resume support.

Saves pipeline state to a JSON file so that a long-running batch can be
resumed from the last completed step after a crash or interruption.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PipelineState:
    """Tracks pipeline progress for resume support."""

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load state from disk, or return empty dict if not found."""
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def save(self) -> None:
        """Persist state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def is_step_done(self, step: str) -> bool:
        return bool(self.data.get(f"{step}_done", False))

    def mark_step_done(self, step: str) -> None:
        self.data[f"{step}_done"] = True
        self.save()

    def add_progress(self, step: str, item_id: str) -> None:
        """Record that an individual item has been processed."""
        key = f"{step}_progress"
        progress = self.data.get(key, [])
        if item_id not in progress:
            progress.append(item_id)
            self.data[key] = progress
            self.save()

    def is_item_done(self, step: str, item_id: str) -> bool:
        progress = self.data.get(f"{step}_progress", [])
        return item_id in progress

    def reset(self) -> None:
        """Clear all state."""
        self.data = {}
        if self.state_path.exists():
            self.state_path.unlink()
