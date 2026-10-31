"""Unit tests for pipeline statistics models, formatting, and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.panel import Panel

from utils.stats import (
    Phase1Stats,
    Phase2Stats,
    PipelineStats,
    RenderStats,
    format_duration,
    format_stats_log,
    format_stats_table,
    load_stats_json,
    parse_duration,
    save_stats_json,
)


def test_parse_duration() -> None:
    """parse_duration should handle numeric and string timecodes."""
    assert parse_duration(15.5) == 15.5
    assert parse_duration("15.5") == 15.5
    assert parse_duration("01:30") == 90.0
    assert parse_duration("01:02:03") == 3723.0
    assert parse_duration("00:00:10:00") == 10.0
    assert parse_duration("00:00:10:15", fps=30.0) == 10.5
    assert parse_duration("") == 0.0
    assert parse_duration(None) == 0.0
    assert parse_duration("invalid") == 0.0


def test_format_duration() -> None:
    """format_duration should format seconds cleanly."""
    assert format_duration(0.0) == "0.0s"
    assert format_duration(12.4) == "12.4s"
    assert format_duration(65.0) == "1m 05s"
    assert format_duration(3665.0) == "1h 01m 05s"


def test_stats_models_defaults() -> None:
    """Stats models should instantiate with valid zero-defaults."""
    p1 = Phase1Stats()
    assert p1.candidate_windows == 0
    assert p1.candidate_duration == 0.0
    assert p1.frames_analyzed == 0

    p2 = Phase2Stats()
    assert p2.windows_scanned == 0
    assert p2.detections_dedup == 0
    assert p2.detections_dedup_duration == 0.0

    render = RenderStats()
    assert render.clips_rendered == 0
    assert render.total_duration == 0.0

    pipe = PipelineStats()
    assert pipe.video_count == 0
    assert pipe.total_video_duration == 0.0
    assert pipe.speed_ratio == 0.0
    assert pipe.fps_throughput == 0.0


def test_pipeline_stats_computations() -> None:
    """PipelineStats should compute speed ratio and fps throughput correctly."""
    pipe = PipelineStats(
        video_count=2,
        total_video_duration=120.0,
        total_elapsed_seconds=60.0,
        phase1=Phase1Stats(frames_analyzed=60),
        phase2=Phase2Stats(frames_analyzed=120),
    )
    # 120s footage in 60s execution = 2.0x realtime
    assert pytest.approx(pipe.speed_ratio, 0.01) == 2.0
    # 180 frames analyzed in 60s = 3.0 fps throughput
    assert pytest.approx(pipe.fps_throughput, 0.01) == 3.0


def test_save_and_load_stats_json(tmp_path: Path) -> None:
    """save_stats_json and load_stats_json should round-trip correctly."""
    stats = PipelineStats(
        run_id="run-20261031-120000",
        video_count=3,
        total_video_duration=300.5,
        total_elapsed_seconds=150.2,
        phase1=Phase1Stats(
            candidate_windows=5,
            candidate_duration=45.0,
            frames_extracted=150,
            frames_analyzed=150,
            detections_detected=8,
            ollama_calls=150,
            ollama_failures=0,
            elapsed_seconds=50.0,
        ),
        phase2=Phase2Stats(
            windows_scanned=5,
            frames_extracted=90,
            frames_analyzed=90,
            detections_found=12,
            detections_duration=24.0,
            detections_dedup=6,
            detections_dedup_duration=18.0,
            ollama_calls=90,
            ollama_failures=0,
            elapsed_seconds=70.0,
        ),
        render=RenderStats(
            clips_total=6,
            clips_rendered=6,
            clips_skipped=0,
            clips_failed=0,
            total_duration=18.0,
            elapsed_seconds=25.0,
        ),
        step_durations={"ingest": 5.2, "scan": 120.0, "render": 25.0},
    )

    json_path = tmp_path / "run_stats.json"
    save_stats_json(stats, json_path)
    assert json_path.exists()

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["run_id"] == "run-20261031-120000"
    assert data["video_count"] == 3
    assert data["phase1"]["candidate_windows"] == 5

    loaded = load_stats_json(json_path)
    assert loaded.run_id == stats.run_id
    assert loaded.video_count == stats.video_count
    assert loaded.phase1.candidate_duration == stats.phase1.candidate_duration
    assert loaded.phase2.detections_dedup == stats.phase2.detections_dedup
    assert loaded.render.clips_rendered == stats.render.clips_rendered


def test_format_stats_table() -> None:
    """format_stats_table should produce a renderable Rich Panel containing table."""
    stats = PipelineStats(
        run_id="run-123",
        video_count=2,
        total_video_duration=100.0,
        total_elapsed_seconds=50.0,
        phase1=Phase1Stats(
            candidate_windows=4,
            candidate_duration=32.0,
            frames_analyzed=50,
            elapsed_seconds=20.0,
        ),
        phase2=Phase2Stats(
            windows_scanned=4,
            detections_dedup=3,
            detections_dedup_duration=15.0,
            frames_analyzed=30,
            elapsed_seconds=25.0,
        ),
        render=RenderStats(
            clips_rendered=3,
            total_duration=15.0,
            elapsed_seconds=5.0,
        ),
    )
    panel = format_stats_table(stats)
    assert isinstance(panel, Panel)


def test_format_stats_log() -> None:
    """format_stats_log should generate clear multi-line string for logging."""
    stats = PipelineStats(
        run_id="run-123",
        video_count=2,
        total_video_duration=100.0,
        total_elapsed_seconds=50.0,
        phase1=Phase1Stats(
            candidate_windows=4,
            candidate_duration=32.0,
            frames_analyzed=50,
            elapsed_seconds=20.0,
        ),
        phase2=Phase2Stats(
            windows_scanned=4,
            detections_dedup=3,
            detections_dedup_duration=15.0,
            frames_analyzed=30,
            elapsed_seconds=25.0,
        ),
        render=RenderStats(
            clips_rendered=3,
            total_duration=15.0,
            elapsed_seconds=5.0,
        ),
    )
    log_text = format_stats_log(stats)
    assert "Run Statistics Summary" in log_text
    assert "Source footage:" in log_text
    assert "Phase 1 (Coarse):" in log_text
    assert "Phase 2 (Fine):" in log_text
    assert "Highlights Rendered:" in log_text
    assert "Speed:" in log_text
