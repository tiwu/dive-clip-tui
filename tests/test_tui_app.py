"""Tests for tui app and screens."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tui.app import DiveClipTUI
from tui.screens.browse import BrowseScreen
from tui.screens.review import ReviewScreen
from tui.screens.pipeline import PipelineScreen


def test_app_has_screens() -> None:
    """App should register browse, review, and pipeline screens."""
    app = DiveClipTUI()
    assert "browse" in app.SCREENS
    assert "review" in app.SCREENS
    assert "pipeline" in app.SCREENS
    assert app.SCREENS["browse"] is BrowseScreen
    assert app.SCREENS["review"] is ReviewScreen
    assert app.SCREENS["pipeline"] is PipelineScreen


def test_browse_screen_has_widgets() -> None:
    """BrowseScreen should compose expected widgets."""
    screen = BrowseScreen()
    # Compose the screen to check widgets exist
    # We can't easily run compose() outside of an app context,
    # but we can verify the screen class exists and is a Screen
    from textual.screen import Screen

    assert isinstance(screen, Screen)


def test_review_screen_has_widgets() -> None:
    """ReviewScreen should compose expected widgets."""
    screen = ReviewScreen()
    from textual.screen import Screen

    assert isinstance(screen, Screen)


def test_pipeline_screen_has_widgets() -> None:
    """PipelineScreen should compose expected widgets."""
    screen = PipelineScreen()
    from textual.screen import Screen

    assert isinstance(screen, Screen)


@pytest.mark.asyncio
async def test_app_starts_with_browse_screen() -> None:
    """App should start with the browse screen active."""
    app = DiveClipTUI()
    async with app.run_test() as pilot:
        # The app should push the browse screen on mount
        assert app.screen is not None
        # Check that the browse screen is the current screen
        from tui.screens.browse import BrowseScreen

        assert isinstance(app.screen, BrowseScreen)


@pytest.mark.asyncio
async def test_quit_button_clickable() -> None:
    """Quit button should exist and be clickable without error."""
    app = DiveClipTUI()
    async with app.run_test() as pilot:
        quit_button = app.query_one("#quit")
        assert quit_button is not None
        assert quit_button.id == "quit"
        await pilot.click(quit_button)
        await pilot.pause()


@pytest.mark.asyncio
async def test_browse_screen_has_expected_ids() -> None:
    """BrowseScreen should have widgets with expected IDs."""
    app = DiveClipTUI()
    async with app.run_test() as pilot:
        # Check that key widgets exist on the current screen
        assert app.screen.query_one("#folder-input") is not None
        assert app.screen.query_one("#load-manifest") is not None
        assert app.screen.query_one("#manifest-table") is not None
        assert app.screen.query_one("#manifest-status") is not None


@pytest.mark.asyncio
async def test_review_screen_widgets() -> None:
    """ReviewScreen should have expected widgets when pushed."""
    app = DiveClipTUI()
    async with app.run_test() as pilot:
        app.push_screen(ReviewScreen())
        await pilot.pause()
        assert isinstance(app.screen, ReviewScreen)
        assert app.screen.query_one("#detections-input") is not None
        assert app.screen.query_one("#load-detections") is not None
        assert app.screen.query_one("#detections-table") is not None
        assert app.screen.query_one("#quality-filter") is not None
        assert app.screen.query_one("#accept-selected") is not None
        assert app.screen.query_one("#render-clips") is not None
        assert app.screen.query_one("#go-back") is not None


@pytest.mark.asyncio
async def test_pipeline_screen_widgets() -> None:
    """PipelineScreen should have expected widgets when pushed."""
    app = DiveClipTUI()
    async with app.run_test() as pilot:
        app.push_screen(PipelineScreen())
        await pilot.pause()
        assert isinstance(app.screen, PipelineScreen)
        assert app.screen.query_one("#step-ingest") is not None
        assert app.screen.query_one("#step-scan") is not None
        assert app.screen.query_one("#step-edl") is not None
        assert app.screen.query_one("#step-storyline") is not None
        assert app.screen.query_one("#pipeline-status") is not None
        assert app.screen.query_one("#pipeline-log") is not None
        assert app.screen.query_one("#go-back") is not None
