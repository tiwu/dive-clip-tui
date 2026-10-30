"""Tests for utils/checkpoint.py PipelineState."""

from __future__ import annotations

import json
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
