"""Tests for scripts/03_generate_edl.py."""

from __future__ import annotations

import importlib.util
import json
import os
import sys as sys_module
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Load scripts/03_generate_edl.py directly (name starts with a digit)
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_edl_spec = importlib.util.spec_from_file_location(
    "scripts_03_generate_edl", _SCRIPTS_DIR / "03_generate_edl.py"
)
edl = importlib.util.module_from_spec(_edl_spec)
_edl_spec.loader.exec_module(edl)


def test_generate_edl(tmp_path: Path) -> None:
    """generate_edl should create a CMX 3600 EDL file with marker events."""
    from utils.models import RefinedDetection, DetectionsRefined

    det1 = RefinedDetection(
        clip_id="clip001",
        file_name="dive1.mp4",
        file_path="/tmp/dive1.mp4",
        fps="25.0",
        start_time=0.0,
        end_time=5.0,
        subject="turtle",
        confidence=0.9,
        quality_score=8,
        description="Sea turtle swimming",
        color="Cyan",
        template="pelagics",
    )
    detections = DetectionsRefined(
        total_clips=1, total_detections=1, detections=[det1]
    )

    output_path = tmp_path / "markers.edl"
    edl.generate_edl(detections, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "TITLE: MADAGASCAR_AI_MARKERS" in content
    assert "FCM: NON-DROP FRAME" in content
    assert "CYAN" in content.upper() or "Cyan" in content


def test_generate_csv(tmp_path: Path) -> None:
    """generate_csv should create a CSV file with detection columns."""
    from utils.models import RefinedDetection, DetectionsRefined

    det1 = RefinedDetection(
        clip_id="clip001",
        file_name="dive1.mp4",
        file_path="/tmp/dive1.mp4",
        fps="25.0",
        start_time=0.0,
        end_time=5.0,
        subject="turtle",
        confidence=0.9,
        quality_score=8,
        description="Sea turtle swimming",
        color="Cyan",
        template="pelagics",
    )
    detections = DetectionsRefined(
        total_clips=1, total_detections=1, detections=[det1]
    )

    output_path = tmp_path / "markers.csv"
    edl.generate_csv(detections, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "Clip Name" in content
    assert "Start Timecode" in content
    assert "dive1.mp4" in content


def test_generate_edl_empty_detections(tmp_path: Path) -> None:
    """generate_edl should handle empty detections list."""
    from utils.models import DetectionsRefined

    detections = DetectionsRefined(total_clips=0, total_detections=0, detections=[])

    output_path = tmp_path / "empty.edl"
    edl.generate_edl(detections, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "TITLE: MADAGASCAR_AI_MARKERS" in content
    assert "FCM: NON-DROP FRAME" in content


def test_main_no_detections_file(tmp_path: Path) -> None:
    """main should exit with code 1 when detections_refined.json is missing."""
    import sys as sys_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text("general:\n  output_dir: \"./outputs\"\n")

    original_argv = sys_module.argv
    original_exit = sys_module.exit
    original_cwd = Path.cwd()

    exit_code = None

    def fake_exit(code=0):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    try:
        sys_module.argv = [
            "03_generate_edl.py",
            "--config",
            str(config_path),
        ]
        sys_module.exit = fake_exit
        os.chdir(tmp_path)
        try:
            edl.main()
        except SystemExit:
            pass
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    assert exit_code == 1


def test_main_with_detections(tmp_path: Path) -> None:
    """main should generate EDL and CSV from detections."""
    import sys as sys_module
    import os

    # Create config.yaml
    config_path = tmp_path / "config.yaml"
    config_path.write_text("general:\n  output_dir: \"./outputs\"\n")

    # Create outputs/detections_refined.json
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True)
    detections_data = {
        "total_clips": 1,
        "total_detections": 1,
        "detections": [
            {
                "clip_id": "clip001",
                "file_name": "dive1.mp4",
                "file_path": "/tmp/dive1.mp4",
                "fps": "25.0",
                "start_time": 0.0,
                "end_time": 5.0,
                "subject": "turtle",
                "confidence": 0.9,
                "quality_score": 8,
                "description": "Sea turtle swimming",
                "color": "Cyan",
                "template": "pelagics",
            }
        ],
    }
    (outputs_dir / "detections_refined.json").write_text(json.dumps(detections_data))

    original_argv = sys_module.argv
    original_exit = sys_module.exit
    original_cwd = Path.cwd()

    exit_code = None

    def fake_exit(code=0):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    try:
        sys_module.argv = [
            "03_generate_edl.py",
            "--config",
            str(config_path),
        ]
        sys_module.exit = fake_exit
        os.chdir(tmp_path)
        try:
            edl.main()
        except SystemExit:
            pass
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    assert exit_code is None or exit_code == 0
    # Check output files were created
    edl_path = outputs_dir / "markers_import.edl"
    csv_path = outputs_dir / "markers_import.csv"
    assert edl_path.exists(), f"EDL file not found at {edl_path}"
    assert csv_path.exists(), f"CSV file not found at {csv_path}"

