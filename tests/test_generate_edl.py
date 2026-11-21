"""Tests for scripts/03_generate_edl.py."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys as sys_module
from pathlib import Path

# Load scripts/03_generate_edl.py directly (name starts with a digit)
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_edl_spec = importlib.util.spec_from_file_location(
    "scripts_03_generate_edl", _SCRIPTS_DIR / "03_generate_edl.py"
)
edl = importlib.util.module_from_spec(_edl_spec)
_edl_spec.loader.exec_module(edl)


def test_generate_edl(tmp_path: Path) -> None:
    """generate_edl should create a CMX 3600 EDL file with marker events."""
    from utils.models import DetectionsRefined, RefinedDetection

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
    detections = DetectionsRefined(total_clips=1, total_detections=1, detections=[det1])

    output_path = tmp_path / "markers.edl"
    edl.generate_edl(detections, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert f"TITLE: {edl.DEFAULT_EDL_TITLE}" in content
    assert "MADAGASCAR" not in content.upper()
    assert "FCM: NON-DROP FRAME" in content
    assert "CYAN" in content.upper() or "Cyan" in content


def test_generate_csv(tmp_path: Path) -> None:
    """generate_csv should create a CSV file with detection columns."""
    from utils.models import DetectionsRefined, RefinedDetection

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
    detections = DetectionsRefined(total_clips=1, total_detections=1, detections=[det1])

    output_path = tmp_path / "markers.csv"
    edl.generate_csv(detections, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "Clip Name" in content
    assert "Start Timecode" in content
    assert "dive1.mp4" in content


def test_generate_csv_sanitizes_formula_injection(tmp_path: Path) -> None:
    """generate_csv should prepend single quote to formula-trigger characters."""
    from utils.models import DetectionsRefined, RefinedDetection

    det = RefinedDetection(
        clip_id="clip001",
        file_name="=cmd|' /C calc'!A0",
        file_path="/tmp/evil.mp4",
        fps="25.0",
        start_time=0.0,
        end_time=5.0,
        subject="+12345",
        confidence=0.9,
        quality_score=8,
        description="@SUM(1,1)",
        color="Cyan",
        template="pelagics",
    )
    detections = DetectionsRefined(total_clips=1, total_detections=1, detections=[det])

    output_path = tmp_path / "markers_sanitized.csv"
    edl.generate_csv(detections, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert "'=cmd|' /C calc'!A0" in content
    assert "'+12345" in content
    assert "'@SUM(1,1)" in content


def test_generate_edl_empty_detections(tmp_path: Path) -> None:
    """generate_edl should handle empty detections list."""
    from utils.models import DetectionsRefined

    detections = DetectionsRefined(total_clips=0, total_detections=0, detections=[])

    output_path = tmp_path / "empty.edl"
    edl.generate_edl(detections, output_path)

    assert output_path.exists()
    content = output_path.read_text()
    assert f"TITLE: {edl.DEFAULT_EDL_TITLE}" in content
    assert "FCM: NON-DROP FRAME" in content


def test_main_no_detections_file(tmp_path: Path) -> None:
    """main should exit with code 1 when detections_refined.json is missing."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text('general:\n  output_dir: "./outputs"\n')

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
        with contextlib.suppress(SystemExit):
            edl.main()
    finally:
        sys_module.argv = original_argv
        sys_module.exit = original_exit
        os.chdir(original_cwd)

    assert exit_code == 1


def test_main_with_detections(tmp_path: Path) -> None:
    """main should generate EDL and CSV from detections."""
    import os

    # Create config.yaml
    config_path = tmp_path / "config.yaml"
    config_path.write_text('general:\n  output_dir: "./outputs"\n')

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
        with contextlib.suppress(SystemExit):
            edl.main()
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


def test_format_edl_title() -> None:
    """format_edl_title should normalize project names to CMX 3600 uppercase ASCII."""
    assert edl.DEFAULT_EDL_TITLE == "DIVE_CLIP_AI_MARKERS"
    assert edl.format_edl_title(None) == "DIVE_CLIP_AI_MARKERS"
    assert edl.format_edl_title("") == "DIVE_CLIP_AI_MARKERS"
    assert edl.format_edl_title("   ---   ") == "DIVE_CLIP_AI_MARKERS"
    assert edl.format_edl_title("Bonaire Reef 2026!") == "BONAIRE_REEF_2026_AI_MARKERS"
    assert edl.format_edl_title("komodo_dives_ai_markers") == "KOMODO_DIVES_AI_MARKERS"
    assert edl.format_edl_title("Côte d'Azur & Red Sea") == "C_TE_D_AZUR_RED_SEA_AI_MARKERS"


def test_generate_edl_custom_title_and_newline_sanitization(tmp_path: Path) -> None:
    """generate_edl should accept explicit titles and block header newline injection."""
    from utils.models import DetectionsRefined

    detections = DetectionsRefined(total_clips=0, total_detections=0, detections=[])

    custom_path = tmp_path / "custom.edl"
    edl.generate_edl(detections, custom_path, title="CUSTOM_REEL")
    assert custom_path.read_text().startswith("TITLE: CUSTOM_REEL\n")

    injected_path = tmp_path / "injected.edl"
    edl.generate_edl(detections, injected_path, title="Reef\nFCM: DROP FRAME\r\n001 AX")
    lines = injected_path.read_text().splitlines()
    assert lines[0].startswith("TITLE: Reef")
    assert lines[1] == "FCM: NON-DROP FRAME"
    assert sum(1 for line in lines if line.startswith("FCM:")) == 1


def _write_minimal_detections(outputs_dir: Path) -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
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


def test_main_derives_edl_title_from_manifest_then_config(tmp_path: Path, monkeypatch) -> None:
    """main should resolve EDL title from manifest.json, manifest_raw.json, then config."""
    monkeypatch.chdir(tmp_path)

    # 1. Sibling manifest.json overrides config.general.project_name
    case1 = tmp_path / "case1"
    out1 = case1 / "outputs"
    _write_minimal_detections(out1)
    (out1 / "manifest.json").write_text(json.dumps({"project_name": "Bonaire 2026"}))
    cfg1 = case1 / "config.yaml"
    cfg1.write_text('general:\n  project_name: "Fallback Trip"\n  output_dir: "./outputs"\n')
    monkeypatch.chdir(case1)
    monkeypatch.setattr(sys_module, "argv", ["03_generate_edl.py", "--config", str(cfg1)])
    edl.main()
    assert (out1 / "markers_import.edl").read_text().startswith("TITLE: BONAIRE_2026_AI_MARKERS\n")

    # 2. Sibling manifest_raw.json used when manifest.json absent
    case2 = tmp_path / "case2"
    out2 = case2 / "outputs"
    _write_minimal_detections(out2)
    (out2 / "manifest_raw.json").write_text(json.dumps({"project_name": "Resolve Project"}))
    cfg2 = case2 / "config.yaml"
    cfg2.write_text('general:\n  project_name: "Fallback Trip"\n  output_dir: "./outputs"\n')
    monkeypatch.chdir(case2)
    monkeypatch.setattr(sys_module, "argv", ["03_generate_edl.py", "--config", str(cfg2)])
    edl.main()
    assert (
        (out2 / "markers_import.edl").read_text().startswith("TITLE: RESOLVE_PROJECT_AI_MARKERS\n")
    )

    # 3. Config project_name used when no manifest present
    case3 = tmp_path / "case3"
    out3 = case3 / "outputs"
    _write_minimal_detections(out3)
    cfg3 = case3 / "config.yaml"
    cfg3.write_text('general:\n  project_name: "Red Sea Trip"\n  output_dir: "./outputs"\n')
    monkeypatch.chdir(case3)
    monkeypatch.setattr(sys_module, "argv", ["03_generate_edl.py", "--config", str(cfg3)])
    edl.main()
    assert (out3 / "markers_import.edl").read_text().startswith("TITLE: RED_SEA_TRIP_AI_MARKERS\n")

    # 4. Empty config project_name falls back to DEFAULT_EDL_TITLE
    case4 = tmp_path / "case4"
    out4 = case4 / "outputs"
    _write_minimal_detections(out4)
    cfg4 = case4 / "config.yaml"
    cfg4.write_text('general:\n  project_name: ""\n  output_dir: "./outputs"\n')
    monkeypatch.chdir(case4)
    monkeypatch.setattr(sys_module, "argv", ["03_generate_edl.py", "--config", str(cfg4)])
    edl.main()
    assert (out4 / "markers_import.edl").read_text().startswith(f"TITLE: {edl.DEFAULT_EDL_TITLE}\n")
