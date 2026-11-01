"""Tests for scripts/entrypoints.py console script wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from scripts import entrypoints


def test_entrypoint_scan_footage() -> None:
    """scan_footage entrypoint should delegate to scripts.02_scan_footage.main."""
    with patch.object(entrypoints, "_run") as mock_run:
        entrypoints.scan_footage()
        mock_run.assert_called_once_with("02_scan_footage")


def test_entrypoint_generate_edl() -> None:
    """generate_edl entrypoint should delegate to scripts.03_generate_edl.main."""
    with patch.object(entrypoints, "_run") as mock_run:
        entrypoints.generate_edl()
        mock_run.assert_called_once_with("03_generate_edl")


def test_entrypoint_plan_storyline() -> None:
    """plan_storyline entrypoint should delegate to scripts.04_plan_storyline.main."""
    with patch.object(entrypoints, "_run") as mock_run:
        entrypoints.plan_storyline()
        mock_run.assert_called_once_with("04_plan_storyline")


def test_entrypoint_ingest_media() -> None:
    """ingest_media entrypoint should delegate to scripts.00_ingest_media.main."""
    with patch.object(entrypoints, "_run") as mock_run:
        entrypoints.ingest_media()
        mock_run.assert_called_once_with("00_ingest_media")


def test_entrypoint_render_clips() -> None:
    """render_clips entrypoint should delegate to scripts.05_render_clips.main."""
    with patch.object(entrypoints, "_run") as mock_run:
        entrypoints.render_clips()
        mock_run.assert_called_once_with("05_render_clips")


def test_entrypoint_assemble_clips() -> None:
    """assemble_clips entrypoint should delegate to scripts.06_assemble_video.main."""
    with patch.object(entrypoints, "_run") as mock_run:
        entrypoints.assemble_clips()
        mock_run.assert_called_once_with("06_assemble_video")


def test_entrypoint_create_aftermovie() -> None:
    """create_aftermovie entrypoint should delegate to scripts.07_create_aftermovie.main."""
    with patch.object(entrypoints, "_run") as mock_run:
        entrypoints.create_aftermovie()
        mock_run.assert_called_once_with("07_create_aftermovie")


def test_run_helper_invokes_module_main() -> None:
    """_run helper should import module and invoke main()."""
    mock_mod = MagicMock()
    with patch("scripts.entrypoints.import_module", return_value=mock_mod) as mock_import:
        entrypoints._run("02_scan_footage")
        mock_import.assert_called_once_with(".02_scan_footage", package="scripts")
        mock_mod.main.assert_called_once()
