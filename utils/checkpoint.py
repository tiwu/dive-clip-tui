"""Pipeline checkpointing and resume support.

Saves pipeline state to a JSON file so that a long-running batch can be
resumed from the last completed step after a crash or interruption.

Every write stamps ``updated_at`` so a stalled run can be told apart from a
slow one, and each step records ``<step>_started_at``/``<step>_finished_at``
plus its duration.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from utils.models import RefinedDetection

logger = logging.getLogger(__name__)


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

    def init_run_metadata(
        self,
        run_id: str | None = None,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        config_path: str | Path | None = None,
        cli_args: list[str] | None = None,
    ) -> None:
        """Initialize or update high-level run metadata."""
        if "created_at" not in self.data:
            self.data["created_at"] = _utc_now()
        if run_id:
            self.data["run_id"] = str(run_id)
        if input_dir:
            self.data["input_dir"] = str(input_dir)
        if output_dir:
            self.data["output_dir"] = str(output_dir)
        if config_path:
            self.data["config_path"] = str(config_path)
        if cli_args:
            self.data["cli_args"] = [str(a) for a in cli_args]
        self.save()

    def get_run_metadata(self) -> dict[str, Any]:
        """Return run context metadata stored in state."""
        return {
            "run_id": self.data.get("run_id"),
            "input_dir": self.data.get("input_dir"),
            "output_dir": self.data.get("output_dir"),
            "config_path": self.data.get("config_path"),
            "cli_args": self.data.get("cli_args"),
            "created_at": self.data.get("created_at"),
            "updated_at": self.data.get("updated_at"),
        }

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


class CoarseFrameResult(BaseModel):
    """Result of coarse screening on an individual video frame."""

    timestamp: float
    detected: bool
    confidence: float
    quality_score: int
    subject: str = ""
    candidate_window: tuple[float, float] | None = None
    elapsed: float = 0.0


class ClipScanCheckpoint(BaseModel):
    """Fine-grained checkpoint of in-progress scan for a single video file."""

    clip_id: str
    file_name: str
    pass_stage: Literal["coarse", "coarse_done", "fine", "completed"] = "coarse"
    coarse_fps: float = 0.5
    coarse_frames_analyzed: int = 0
    coarse_results: dict[str, CoarseFrameResult] = Field(default_factory=dict)
    candidate_windows: list[tuple[float, float]] = Field(default_factory=list)
    coarse_confidences: list[float] = Field(default_factory=list)
    coarse_detected: int = 0
    coarse_failures: int = 0
    coarse_elapsed_total: float = 0.0

    fine_windows_completed: list[int] = Field(default_factory=list)
    fine_frames_analyzed: int = 0
    fine_passed: int = 0
    fine_failures: int = 0
    fine_elapsed_total: float = 0.0
    fine_confidences: list[float] = Field(default_factory=list)
    refined_detections: list[RefinedDetection] = Field(default_factory=list)

    updated_at: str = Field(default_factory=_utc_now)

    def save_atomic(self, path: Path) -> None:
        """Atomically persist checkpoint state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _utc_now()
        temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> ClipScanCheckpoint | None:
        """Load checkpoint from disk, returning None if not found or corrupted."""
        path = Path(path)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return cls.model_validate(data)
        except Exception as exc:
            logger.warning("Failed to load checkpoint from %s: %s", path, exc)
            return None


def remove_clip_checkpoint(path: Path) -> None:
    """Safely remove a clip checkpoint file if it exists."""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not remove checkpoint %s: %s", path, exc)
