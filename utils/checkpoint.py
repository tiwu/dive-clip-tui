"""Pipeline checkpointing and resume support.

Saves pipeline state to a JSON file so that a long-running batch can be
resumed from the last completed step after a crash or interruption.

Every write stamps ``updated_at`` so a stalled run can be told apart from a
slow one, and each step records ``<step>_started_at``/``<step>_finished_at``
plus its duration.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


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
            except json.JSONDecodeError, OSError:
                pass
        return {}

    def save(self) -> None:
        """Persist state to disk, stamping the write time.

        The same state file is written by the parent pipeline and by the worker
        scripts it spawns, so the on-disk contents are merged in before writing:
        progress lists are unioned and every other key is last-writer-wins.
        """
        merged = self._load()
        for key, value in self.data.items():
            if key.endswith("_progress") and isinstance(value, list):
                disk_items = merged.get(key)
                if isinstance(disk_items, list):
                    value = disk_items + [item for item in value if item not in disk_items]
            merged[key] = value

        merged["updated_at"] = _utc_now()
        self.data = merged
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def set_activity(self, message: str) -> None:
        """Record what the pipeline is currently working on and persist it."""
        self.data["activity"] = message
        self.data["activity_at"] = _utc_now()
        self.save()

    def is_step_done(self, step: str) -> bool:
        return bool(self.data.get(f"{step}_done", False))

    def mark_step_started(self, step: str) -> None:
        self.data[f"{step}_done"] = False
        self.data[f"{step}_started_at"] = _utc_now()
        self.save()

    def mark_step_done(self, step: str) -> None:
        finished_at = datetime.now(UTC)
        self.data[f"{step}_done"] = True
        self.data[f"{step}_finished_at"] = finished_at.isoformat(timespec="seconds")

        started_raw = self.data.get(f"{step}_started_at")
        if isinstance(started_raw, str):
            try:
                started_at = datetime.fromisoformat(started_raw)
            except ValueError:
                started_at = None
            if started_at is not None:
                elapsed = (finished_at - started_at).total_seconds()
                self.data[f"{step}_duration_seconds"] = round(elapsed, 1)

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
