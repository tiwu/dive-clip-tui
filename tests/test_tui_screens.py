"""Tests for TUI screen interactions and event handling."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import Button, Input, Select, Static

from tui.app import DiveClipTUI
from tui.screens.browse import BrowseScreen
from tui.screens.pipeline import PipelineScreen
from tui.screens.review import ReviewScreen


@pytest.mark.asyncio
async def test_browse_screen_load_manifest_nonexistent_folder() -> None:
    """BrowseScreen should show error when folder does not exist."""
    app = DiveClipTUI()
    async with app.run_test() as pilot:
        screen = app.screen
        assert isinstance(screen, BrowseScreen)

        folder_input = screen.query_one("#folder-input", Input)
        folder_input.value = "/nonexistent/path/for/sure"

        load_btn = screen.query_one("#load-manifest", Button)
        await pilot.click(load_btn)
        await pilot.pause()

        status = screen.query_one("#manifest-status", Static)
        assert "ERROR: Folder does not exist" in str(status.render())


@pytest.mark.asyncio
async def test_browse_screen_load_manifest_success(tmp_path: Path) -> None:
    """BrowseScreen should load manifest clips into DataTable."""
    manifest = {
        "clips": [
            {
                "file_name": "dive1.mp4",
                "duration": "10.0s",
                "resolution": "3840x2160",
                "fps": "50.0",
                "tags": ["macro", "turtle"],
            }
        ]
    }
    (tmp_path / "manifest_raw.json").write_text(json.dumps(manifest))

    app = DiveClipTUI()
    async with app.run_test() as pilot:
        screen = app.screen
        assert isinstance(screen, BrowseScreen)

        folder_input = screen.query_one("#folder-input", Input)
        folder_input.value = str(tmp_path)

        load_btn = screen.query_one("#load-manifest", Button)
        await pilot.click(load_btn)
        await pilot.pause()

        status = screen.query_one("#manifest-status", Static)
        assert "Loaded 1 clips" in str(status.render())

        review_btn = screen.query_one("#go-review", Button)
        assert review_btn.disabled is False

        # Click go-review to push review screen
        await pilot.click(review_btn)
        await pilot.pause()
        assert isinstance(app.screen, ReviewScreen)


@pytest.mark.asyncio
async def test_review_screen_load_and_filter(tmp_path: Path) -> None:
    """ReviewScreen should load detections and filter by quality."""
    detections = {
        "detections": [
            {
                "file_name": "dive1.mp4",
                "start_time": 0.0,
                "end_time": 5.0,
                "subject": "turtle",
                "quality_score": 8,
                "confidence": 0.9,
                "description": "turtle",
            },
            {
                "file_name": "dive2.mp4",
                "start_time": 10.0,
                "end_time": 15.0,
                "subject": "coral",
                "quality_score": 6,
                "confidence": 0.8,
                "description": "coral",
            },
        ]
    }
    det_file = tmp_path / "detections_refined.json"
    det_file.write_text(json.dumps(detections))

    app = DiveClipTUI()
    async with app.run_test() as pilot:
        app.push_screen(ReviewScreen())
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, ReviewScreen)

        det_input = screen.query_one("#detections-input", Input)
        det_input.value = str(det_file)

        load_btn = screen.query_one("#load-detections", Button)
        await pilot.click(load_btn)
        await pilot.pause()

        status = screen.query_one("#detections-status", Static)
        assert "Loaded 2 detections" in str(status.render())

        # Test filtering
        select = screen.query_one("#quality-filter", Select)
        select.value = "8"
        await pilot.pause()

        assert "Showing 1 detections" in str(status.render())

        # Test render button click
        render_btn = screen.query_one("#render-clips", Button)
        # Mock having accepted clips
        screen._accepted.add(0)
        render_btn.disabled = False
        await pilot.click(render_btn)
        await pilot.pause()
        assert "Rendering 1 accepted clips" in str(status.render())


@pytest.mark.asyncio
async def test_pipeline_screen_run_step_success() -> None:
    """PipelineScreen._run_step should run command and update log and status."""
    app = DiveClipTUI()
    async with app.run_test() as pilot:
        app.push_screen(PipelineScreen())
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, PipelineScreen)

        mock_proc = MagicMock()
        mock_proc.stdout = ["Processing step line 1\n", "Done\n"]
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with patch("subprocess.Popen", return_value=mock_proc):
            step_btn = screen.query_one("#step-ingest", Button)
            await pilot.click(step_btn)
            await pilot.pause()

        status = screen.query_one("#pipeline-status", Static)
        assert "ingest completed successfully" in str(status.render())


@pytest.mark.asyncio
async def test_pipeline_screen_run_step_failure() -> None:
    """PipelineScreen._run_step should handle failed commands."""
    app = DiveClipTUI()
    async with app.run_test() as pilot:
        app.push_screen(PipelineScreen())
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, PipelineScreen)

        mock_proc = MagicMock()
        mock_proc.stdout = ["Error encountered\n"]
        mock_proc.returncode = 1
        mock_proc.wait.return_value = 1

        with patch("subprocess.Popen", return_value=mock_proc):
            step_btn = screen.query_one("#step-scan", Button)
            await pilot.click(step_btn)
            await pilot.pause()

        status = screen.query_one("#pipeline-status", Static)
        assert "scan failed with exit code 1" in str(status.render())


@pytest.mark.asyncio
async def test_pipeline_screen_run_step_file_not_found() -> None:
    """PipelineScreen._run_step should handle FileNotFoundError."""
    app = DiveClipTUI()
    async with app.run_test() as pilot:
        app.push_screen(PipelineScreen())
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, PipelineScreen)

        with patch("subprocess.Popen", side_effect=FileNotFoundError):
            step_btn = screen.query_one("#step-edl", Button)
            await pilot.click(step_btn)
            await pilot.pause()

        status = screen.query_one("#pipeline-status", Static)
        assert "Script not found for edl step" in str(status.render())
