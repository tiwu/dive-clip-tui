"""Tests for dive-clip review command and interactive feedback recording."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.dive_clip import (
    build_parser,
    cmd_review,
    load_feedback,
    open_clip_player,
    save_feedback,
    summarize_feedback,
)


def test_build_parser_review() -> None:
    """build_parser should properly parse the review subcommand and flags."""
    parser = build_parser()
    args = parser.parse_args(["review", "--run", "run-123", "--all", "--output", "./custom_out"])
    assert args.command == "review"
    assert args.run == "run-123"
    assert args.all is True
    assert args.output == "./custom_out"
    assert args.config == "config.yaml"


def test_open_clip_player(tmp_path: Path) -> None:
    """open_clip_player should invoke system player via Popen and handle errors."""
    clip = tmp_path / "clip.mp4"

    # Nonexistent file returns False
    assert open_clip_player(clip) is False

    clip.touch()
    with patch("scripts.dive_clip.subprocess.Popen") as mock_popen:
        mock_popen.return_value = MagicMock()
        assert open_clip_player(clip) is True
        assert mock_popen.called

    with patch("scripts.dive_clip.subprocess.Popen", side_effect=OSError("Player not found")):
        assert open_clip_player(clip) is False


def test_load_and_save_feedback(tmp_path: Path) -> None:
    """Feedback should be saved atomically and reloaded accurately."""
    feedback_file = tmp_path / "feedback.json"

    # Nonexistent loads empty structure
    state = load_feedback(feedback_file)
    assert state["feedback"] == {}

    state["feedback"]["clip1.mp4"] = {"decision": "keep", "note": "Great"}
    save_feedback(feedback_file, state)

    reloaded = load_feedback(feedback_file)
    assert reloaded["feedback"]["clip1.mp4"]["decision"] == "keep"
    assert "updated_at" in reloaded


def test_cmd_review_no_runs_found(tmp_path: Path) -> None:
    """cmd_review should return 1 when no runs directory exists."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f'general:\n  output_dir: "{tmp_path}"\n')

    parser = build_parser()
    args = parser.parse_args(["review", "--config", str(config_file)])
    assert cmd_review(args) == 1


def test_cmd_review_highlights_folder_missing_or_empty(tmp_path: Path) -> None:
    """cmd_review should handle missing or empty highlights directory gracefully."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-20261101-100000"
    run_dir.mkdir(parents=True)

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f'general:\n  output_dir: "{tmp_path}"\n')

    parser = build_parser()
    args = parser.parse_args(["review", "--config", str(config_file), "--run", run_dir.name])

    # Missing highlights folder
    assert cmd_review(args) == 1

    # Empty highlights folder
    (run_dir / "highlights").mkdir()
    assert cmd_review(args) == 0


def test_cmd_review_interactive_feedback_flow(tmp_path: Path) -> None:
    """cmd_review should interactively collect and save keep, reject, and wrong_subject feedback."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-20261101-120000"
    highlights_dir = run_dir / "highlights"
    highlights_dir.mkdir(parents=True)

    clip1 = highlights_dir / "clip1_0.0s-5.0s_turtle_q8.mp4"
    clip2 = highlights_dir / "clip2_5.0s-10.0s_shark_q9.mp4"
    clip3 = highlights_dir / "clip3_10.0s-15.0s_coral_q7.mp4"
    clip1.touch()
    clip2.touch()
    clip3.touch()

    detections = {
        "detections": [
            {
                "clip_id": "clip1",
                "file_name": "dive1.mp4",
                "start_time": 0.0,
                "end_time": 5.0,
                "subject": "turtle",
                "quality_score": 8,
                "domain": "underwater",
            },
            {
                "clip_id": "clip2",
                "file_name": "dive1.mp4",
                "start_time": 5.0,
                "end_time": 10.0,
                "subject": "shark",
                "quality_score": 9,
                "domain": "underwater",
            },
            {
                "clip_id": "clip3",
                "file_name": "dive2.mp4",
                "start_time": 10.0,
                "end_time": 15.0,
                "subject": "coral",
                "quality_score": 7,
                "domain": "underwater",
            },
        ]
    }
    (run_dir / "detections_refined.json").write_text(json.dumps(detections))

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f'general:\n  output_dir: "{tmp_path}"\n')

    # Simulation inputs:
    # Clip 1: 'k' (keep), note: 'clear turtle'
    # Clip 2: 'w' (wrong subject), corrected: 'manta ray', note: 'not shark'
    # Clip 3: 'r' (reject), note: ''
    mock_inputs = iter([
        "k",
        "clear turtle",
        "w",
        "manta ray",
        "not shark",
        "r",
        "",
    ])

    parser = build_parser()
    args = parser.parse_args(["review", "--config", str(config_file), "--run", run_dir.name])

    with (
        patch("builtins.input", side_effect=mock_inputs),
        patch("scripts.dive_clip.open_clip_player", return_value=True) as mock_player,
    ):
        rc = cmd_review(args)

    assert rc == 0
    assert mock_player.call_count == 3

    feedback_file = run_dir / "feedback.json"
    assert feedback_file.exists()
    saved = json.loads(feedback_file.read_text())["feedback"]
    assert len(saved) == 3

    assert saved[clip1.name]["decision"] == "keep"
    assert saved[clip1.name]["note"] == "clear turtle"
    assert saved[clip1.name]["detected_subject"] == "turtle"

    assert saved[clip2.name]["decision"] == "wrong_subject"
    assert saved[clip2.name]["corrected_subject"] == "manta ray"
    assert saved[clip2.name]["note"] == "not shark"

    assert saved[clip3.name]["decision"] == "reject"
    assert saved[clip3.name]["note"] is None


def test_cmd_review_skips_answered_and_re_reviews_with_all(tmp_path: Path) -> None:
    """cmd_review should skip previously answered clips by default, and include them with --all."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-20261101-140000"
    highlights_dir = run_dir / "highlights"
    highlights_dir.mkdir(parents=True)

    clip1 = highlights_dir / "c1_0.0s-5.0s_fish_q8.mp4"
    clip2 = highlights_dir / "c2_5.0s-10.0s_fish_q8.mp4"
    clip1.touch()
    clip2.touch()

    feedback_file = run_dir / "feedback.json"
    initial_feedback = {
        "run_id": run_dir.name,
        "feedback": {
            clip1.name: {
                "clip_file": clip1.name,
                "decision": "keep",
                "corrected_subject": None,
                "note": "previously reviewed",
            }
        },
    }
    feedback_file.write_text(json.dumps(initial_feedback))

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f'general:\n  output_dir: "{tmp_path}"\n')

    # Without --all: should only prompt for clip2
    mock_inputs_default = iter(["r", "bad visibility"])
    parser = build_parser()
    args_default = parser.parse_args([
        "review",
        "--config",
        str(config_file),
        "--run",
        run_dir.name,
    ])

    with (
        patch("builtins.input", side_effect=mock_inputs_default),
        patch("scripts.dive_clip.open_clip_player", return_value=True) as mock_player,
    ):
        rc = cmd_review(args_default)

    assert rc == 0
    assert mock_player.call_count == 1  # Only clip2 was opened

    saved = json.loads(feedback_file.read_text())["feedback"]
    assert len(saved) == 2
    assert saved[clip1.name]["decision"] == "keep"
    assert saved[clip2.name]["decision"] == "reject"

    # With --all: both clips are prompted again for editing
    mock_inputs_all = iter(["r", "changed mind", "k", "re-kept"])
    args_all = parser.parse_args([
        "review",
        "--config",
        str(config_file),
        "--run",
        run_dir.name,
        "--all",
    ])

    with (
        patch("builtins.input", side_effect=mock_inputs_all),
        patch("scripts.dive_clip.open_clip_player", return_value=True) as mock_player_all,
    ):
        rc = cmd_review(args_all)

    assert rc == 0
    assert mock_player_all.call_count == 2
    updated = json.loads(feedback_file.read_text())["feedback"]
    assert updated[clip1.name]["decision"] == "reject"
    assert updated[clip1.name]["note"] == "changed mind"


def test_cmd_review_skip_back_and_quit_controls(tmp_path: Path) -> None:
    """Controls: skip moves forward, back returns to previous, quit stops."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-20261101-160000"
    highlights_dir = run_dir / "highlights"
    highlights_dir.mkdir(parents=True)

    clip1 = highlights_dir / "c1_0.0s-5.0s_turtle_q8.mp4"
    clip2 = highlights_dir / "c2_5.0s-10.0s_shark_q9.mp4"
    clip1.touch()
    clip2.touch()

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f'general:\n  output_dir: "{tmp_path}"\n')

    # Sequence:
    # At clip 1: 's' (skip to clip 2)
    # At clip 2: 'b' (go back to clip 1)
    # At clip 1: 'k' (keep), note: ''
    # At clip 2: 'q' (quit)
    mock_inputs = iter(["s", "b", "k", "", "q"])

    parser = build_parser()
    args = parser.parse_args(["review", "--config", str(config_file), "--run", run_dir.name])

    with (
        patch("builtins.input", side_effect=mock_inputs),
        patch("scripts.dive_clip.open_clip_player", return_value=True),
    ):
        rc = cmd_review(args)

    assert rc == 0
    saved = json.loads((run_dir / "feedback.json").read_text())["feedback"]
    assert len(saved) == 1
    assert clip1.name in saved
    assert clip2.name not in saved


def test_cmd_review_player_error_preserves_state(tmp_path: Path) -> None:
    """Player error should notify user but allow review to continue and persist answer."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-20261101-180000"
    highlights_dir = run_dir / "highlights"
    highlights_dir.mkdir(parents=True)

    clip1 = highlights_dir / "c1_0.0s-5.0s_fish_q8.mp4"
    clip1.touch()

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f'general:\n  output_dir: "{tmp_path}"\n')

    parser = build_parser()
    args = parser.parse_args(["review", "--config", str(config_file), "--run", run_dir.name])

    with (
        patch("builtins.input", side_effect=["k", "judged anyway"]),
        patch("scripts.dive_clip.open_clip_player", return_value=False),
    ):
        rc = cmd_review(args)

    assert rc == 0
    saved = json.loads((run_dir / "feedback.json").read_text())["feedback"]
    assert saved[clip1.name]["decision"] == "keep"
    assert saved[clip1.name]["note"] == "judged anyway"


def test_summarize_feedback_output() -> None:
    """summarize_feedback should format decision totals and notes."""
    feedback = {
        "clip1.mp4": {
            "decision": "keep",
            "detected_subject": "turtle",
            "note": "nice",
        },
        "clip2.mp4": {
            "decision": "wrong_subject",
            "detected_subject": "shark",
            "corrected_subject": "dolphin",
            "note": "fin visible",
        },
    }
    with patch("scripts.dive_clip.logger.info") as mock_info:
        summarize_feedback(feedback)
        assert mock_info.called


def test_cmd_review_slugified_subject_and_clip_id_matching(tmp_path: Path) -> None:
    """Detection with spaces/special characters in subject and clip_id should match rendered clip."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-20261101-200000"
    highlights_dir = run_dir / "highlights"
    highlights_dir.mkdir(parents=True)

    # Rendered filename has slugified subject: green_sea_turtle, clip ID: clip_1
    clip = highlights_dir / "clip_1_0.0s-5.0s_green_sea_turtle_q8.mp4"
    clip.touch()

    # Detections file contains unslugified subject and clip_id
    detections = {
        "detections": [
            {
                "clip_id": "clip 1",
                "file_name": "dive_gopro.mp4",
                "start_time": 0.0,
                "end_time": 5.0,
                "subject": "green sea turtle",
                "quality_score": 8,
                "domain": "underwater",
            }
        ]
    }
    (run_dir / "detections_refined.json").write_text(json.dumps(detections))

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f'general:\n  output_dir: "{tmp_path}"\n')

    parser = build_parser()
    args = parser.parse_args(["review", "--config", str(config_file), "--run", run_dir.name])

    with (
        patch("builtins.input", side_effect=["k", "looks sharp"]),
        patch("scripts.dive_clip.open_clip_player", return_value=True),
    ):
        rc = cmd_review(args)

    assert rc == 0
    saved = json.loads((run_dir / "feedback.json").read_text())["feedback"]
    assert clip.name in saved
    item = saved[clip.name]
    assert item["detected_subject"] == "green sea turtle"
    assert item["source_file"] == "dive_gopro.mp4"
    assert item["domain"] == "underwater"
    assert item["start_time"] == 0.0
    assert item["end_time"] == 5.0


def test_cmd_review_ignores_temporary_files(tmp_path: Path) -> None:
    """cmd_review should exclude temporary hidden files (.tmp_*.mp4) from the review queue."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-20261101-210000"
    highlights_dir = run_dir / "highlights"
    highlights_dir.mkdir(parents=True)

    normal_clip = highlights_dir / "c1_0.0s-5.0s_fish_q8.mp4"
    temp_clip = highlights_dir / ".tmp_c2_5.0s-10.0s_shark_q9.mp4"
    normal_clip.touch()
    temp_clip.touch()

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f'general:\n  output_dir: "{tmp_path}"\n')

    parser = build_parser()
    args = parser.parse_args(["review", "--config", str(config_file), "--run", run_dir.name])

    with (
        patch("builtins.input", side_effect=["k", "done"]),
        patch("scripts.dive_clip.open_clip_player", return_value=True) as mock_player,
    ):
        rc = cmd_review(args)

    assert rc == 0
    # Only normal_clip should be opened, temp_clip must be excluded
    assert mock_player.call_count == 1
    saved = json.loads((run_dir / "feedback.json").read_text())["feedback"]
    assert normal_clip.name in saved
    assert temp_clip.name not in saved


def test_cmd_review_missing_clip_file_handling(tmp_path: Path) -> None:
    """Missing clip file should display an explicit error and allow user to skip."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-20261101-220000"
    highlights_dir = run_dir / "highlights"
    highlights_dir.mkdir(parents=True)

    clip = highlights_dir / "c1_0.0s-5.0s_fish_q8.mp4"
    clip.touch()

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f'general:\n  output_dir: "{tmp_path}"\n')

    parser = build_parser()
    args = parser.parse_args(["review", "--config", str(config_file), "--run", run_dir.name])

    # Remove the clip file before review opens it
    clip.unlink()

    with (
        patch("builtins.input", side_effect=["s"]),
        patch("scripts.dive_clip.open_clip_player") as mock_player,
    ):
        rc = cmd_review(args)

    assert rc == 0
    # open_clip_player should not be called for a nonexistent file
    assert not mock_player.called


def test_cmd_review_interrupt_at_prompt_pauses_cleanly(tmp_path: Path) -> None:
    """KeyboardInterrupt at decision, corrected, or note prompt pauses review cleanly."""
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-20261101-230000"
    highlights_dir = run_dir / "highlights"
    highlights_dir.mkdir(parents=True)

    clip1 = highlights_dir / "c1_0.0s-5.0s_fish_q8.mp4"
    clip2 = highlights_dir / "c2_5.0s-10.0s_shark_q9.mp4"
    clip1.touch()
    clip2.touch()

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f'general:\n  output_dir: "{tmp_path}"\n')

    parser = build_parser()
    args = parser.parse_args(["review", "--config", str(config_file), "--run", run_dir.name])

    def input_raising(prompt=""):
        val = next(mock_inputs)
        if isinstance(val, BaseException):
            raise val
        return val

    mock_inputs = iter(["k", "first clip done", "w", KeyboardInterrupt()])
    with (
        patch("builtins.input", side_effect=input_raising),
        patch("scripts.dive_clip.open_clip_player", return_value=True),
    ):
        rc = cmd_review(args)

    assert rc == 0
    saved = json.loads((run_dir / "feedback.json").read_text())["feedback"]
    assert clip1.name in saved
    assert saved[clip1.name]["decision"] == "keep"
    # Incomplete clip2 judgment was not written
    assert clip2.name not in saved
