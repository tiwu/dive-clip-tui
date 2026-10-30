"""Tests for utils/checkpoint.py PipelineState."""

from __future__ import annotations

from pathlib import Path

from utils.checkpoint import PipelineState


def test_pipeline_state_save_and_load(tmp_path: Path) -> None:
    """State should persist to disk and reload correctly."""
    state_path = tmp_path / "state.json"
    state = PipelineState(state_path)
    state.set("ingest_done", True)
    state.set("scan_progress", ["clip001", "clip002"])
    state.save()

    # Reload from disk
    state2 = PipelineState(state_path)
    assert state2.get("ingest_done") is True
    assert state2.get("scan_progress") == ["clip001", "clip002"]


def test_pipeline_state_is_step_done_and_mark_done(tmp_path: Path) -> None:
    """is_step_done should reflect mark_step_done."""
    state_path = tmp_path / "state.json"
    state = PipelineState(state_path)

    assert state.is_step_done("ingest") is False
    state.mark_step_done("ingest")
    assert state.is_step_done("ingest") is True

    # Should persist
    state2 = PipelineState(state_path)
    assert state2.is_step_done("ingest") is True


def test_pipeline_state_add_and_is_item_done(tmp_path: Path) -> None:
    """add_progress / is_item_done should track individual items."""
    state_path = tmp_path / "state.json"
    state = PipelineState(state_path)

    assert state.is_item_done("scan", "clip001") is False
    state.add_progress("scan", "clip001")
    assert state.is_item_done("scan", "clip001") is True
    assert state.is_item_done("scan", "clip002") is False

    state.add_progress("scan", "clip002")
    assert state.get("scan_progress") == ["clip001", "clip002"]


def test_pipeline_state_reset(tmp_path: Path) -> None:
    """reset should clear all state and remove the file."""
    state_path = tmp_path / "state.json"
    state = PipelineState(state_path)
    state.set("ingest_done", True)
    state.mark_step_done("ingest")
    assert state_path.exists()

    state.reset()
    assert state.get("ingest_done") is None
    assert state.is_step_done("ingest") is False
    assert not state_path.exists()


def test_pipeline_state_nonexistent_file(tmp_path: Path) -> None:
    """Loading a non-existent state file should return empty defaults."""
    state_path = tmp_path / "nonexistent.json"
    state = PipelineState(state_path)
    assert state.get("anything") is None
    assert state.is_step_done("ingest") is False
    assert state.is_item_done("scan", "clip001") is False


def test_pipeline_state_corrupt_file(tmp_path: Path) -> None:
    """A corrupt JSON state file should fall back to empty defaults."""
    state_path = tmp_path / "state.json"
    state_path.write_text("not valid json {{{")
    state = PipelineState(state_path)
    assert state.get("anything") is None
    assert state.is_step_done("ingest") is False


def test_pipeline_state_save_stamps_updated_at(tmp_path: Path) -> None:
    """Every save should refresh the updated_at timestamp."""
    from datetime import datetime

    state_path = tmp_path / "state.json"
    state = PipelineState(state_path)
    state.set("anything", 1)
    state.save()

    first = state.get("updated_at")
    assert isinstance(first, str)
    datetime.fromisoformat(first)  # must be a valid ISO-8601 timestamp

    state.save()
    assert state.get("updated_at") == first or state.get("updated_at") > first


def test_pipeline_state_set_activity_records_timestamp(tmp_path: Path) -> None:
    """set_activity should record the message, its time, and persist immediately."""
    from datetime import datetime

    state_path = tmp_path / "state.json"
    state = PipelineState(state_path)
    state.set_activity("scan: clip001 pass 1")

    assert state.get("activity") == "scan: clip001 pass 1"
    datetime.fromisoformat(state.get("activity_at"))

    reloaded = PipelineState(state_path)
    assert reloaded.get("activity") == "scan: clip001 pass 1"
    assert reloaded.get("activity_at") == state.get("activity_at")


def test_pipeline_state_step_duration_is_recorded(tmp_path: Path) -> None:
    """mark_step_done should derive a duration from the recorded start time."""
    state_path = tmp_path / "state.json"
    state = PipelineState(state_path)
    state.mark_step_started("scan")
    assert state.get("scan_done") is False
    assert isinstance(state.get("scan_started_at"), str)

    state.mark_step_done("scan")
    assert state.get("scan_done") is True
    assert isinstance(state.get("scan_finished_at"), str)
    assert state.get("scan_duration_seconds") >= 0.0


def test_pipeline_state_save_merges_progress_from_disk(tmp_path: Path) -> None:
    """A stale instance must not drop progress written by another process."""
    state_path = tmp_path / "state.json"
    parent = PipelineState(state_path)
    parent.set_activity("parent")

    child = PipelineState(state_path)
    child.add_progress("scan", "clip001")

    parent.add_progress("scan", "clip002")

    reloaded = PipelineState(state_path)
    assert reloaded.get("scan_progress") == ["clip001", "clip002"]
