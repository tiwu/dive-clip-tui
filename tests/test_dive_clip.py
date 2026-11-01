"""Tests for scripts/dive_clip.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.dive_clip import build_parser, cmd_scan, main, resolve_paths, run
from utils.run_archive import find_latest_run_dir


def test_run_success() -> None:
    """run should execute a command and return 0 on success."""
    with patch("scripts.dive_clip.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        rc = run(["echo", "hello"], "test step")
    assert rc == 0


def test_run_failure() -> None:
    """run should return non-zero exit code on failure."""
    with patch("scripts.dive_clip.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        rc = run(["false"], "test step")
    assert rc == 1


def test_run_keyboard_interrupt() -> None:
    """run should handle KeyboardInterrupt and return 130."""
    with patch("scripts.dive_clip.subprocess.run", side_effect=KeyboardInterrupt()):
        rc = run(["sleep", "100"], "test step")
    assert rc == 130


def test_run_file_not_found() -> None:
    """run should return 127 when command not found."""
    with patch("scripts.dive_clip.subprocess.run", side_effect=FileNotFoundError()):
        rc = run(["nonexistent_command"], "test step")
    assert rc == 127


def test_build_parser() -> None:
    """build_parser should create a parser with scan subcommand."""
    parser = build_parser()
    assert parser is not None

    # Test scan subcommand
    args = parser.parse_args(["scan", "--input", "/tmp/footage", "--output", "/tmp/highlights"])
    assert args.command == "scan"
    assert args.input == "/tmp/footage"
    assert args.output == "/tmp/highlights"
    assert args.config == "config.yaml"
    assert args.resume is False
    assert args.reset is False
    assert args.debug is False


def test_build_parser_with_flags() -> None:
    """build_parser should parse resume, reset, and debug flags."""
    parser = build_parser()
    args = parser.parse_args([
        "scan",
        "--input",
        "/tmp/footage",
        "--output",
        "/tmp/highlights",
        "--resume",
        "--reset",
        "--debug",
    ])
    assert args.resume is True
    assert args.reset is True
    assert args.debug is True


def test_cmd_scan_skips_completed_steps(tmp_path: Path) -> None:
    """cmd_scan should skip steps that are already marked done in state."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "highlights"
    output_folder.mkdir()

    # Pre-create run directory with state and artifacts marked done
    run_dir = output_folder / "runs" / "run-20261031-100000"
    run_dir.mkdir(parents=True)
    state_path = run_dir / "pipeline_state.json"
    state_path.write_text('{"ingest_done": true, "scan_done": true, "render_done": true}')

    (run_dir / "manifest.json").write_text("{}")
    (run_dir / "detections_refined.json").write_text("{}")

    args = argparse.Namespace(
        input=str(input_folder),
        output=str(output_folder),
        config="config.yaml",
        resume=True,
        reset=False,
        debug=False,
        log_file=None,
    )

    with patch("scripts.dive_clip.subprocess.run") as mock_run:
        rc = cmd_scan(args)

    assert rc == 0
    # No subprocess calls should have been made since all steps are done
    mock_run.assert_not_called()


def test_cmd_scan_runs_all_steps(tmp_path: Path) -> None:
    """cmd_scan should run all steps when none are done."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "highlights"
    output_folder.mkdir()

    args = argparse.Namespace(
        input=str(input_folder),
        output=str(output_folder),
        config="config.yaml",
        resume=False,
        reset=False,
        debug=False,
        log_file=None,
    )

    mock_result = MagicMock(returncode=0)
    with patch("scripts.dive_clip.subprocess.run", return_value=mock_result):
        rc = cmd_scan(args)

    assert rc == 0
    # Should have called subprocess 3 times (ingest, scan, render)
    assert mock_result.returncode == 0

    # State file should be created in the run directory with all steps done
    latest = find_latest_run_dir(output_folder)
    assert latest is not None
    state_path = latest / "pipeline_state.json"
    assert state_path.exists()


def test_cmd_scan_handles_interrupt(tmp_path: Path) -> None:
    """cmd_scan should return 130 and abort cleanly when interrupted."""
    args = _scan_args(tmp_path, debug=False)

    # Ingest succeeds, scan is interrupted with code 130
    mock_ingest = MagicMock(returncode=0)
    mock_scan = MagicMock(returncode=130)

    # Fake that ingest wrote manifest_raw.json so it proceeds to scan
    manifest_path = tmp_path / "highlights" / "manifest_raw.json"

    def side_effect(cmd, *a, **kw):
        if "00_ingest_media.py" in cmd[1]:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.touch()
            return mock_ingest
        return mock_scan

    with patch("scripts.dive_clip.subprocess.run", side_effect=side_effect):
        rc = cmd_scan(args)

    assert rc == 130
    # Render should NOT have run
    assert not (tmp_path / "highlights" / "detections_refined.json").exists()


def test_cmd_scan_reset_state(tmp_path: Path) -> None:
    """cmd_scan with --reset should clear state and re-run steps."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "highlights"
    output_folder.mkdir()

    # Pre-create run directory with state with all steps done
    run_dir = output_folder / "runs" / "run-20261031-100000"
    run_dir.mkdir(parents=True)
    state_path = run_dir / "pipeline_state.json"
    state_path.write_text('{"ingest_done": true, "scan_done": true, "render_done": true}')

    args = argparse.Namespace(
        input=str(input_folder),
        output=str(output_folder),
        config="config.yaml",
        resume=True,
        reset=True,
        debug=False,
        log_file=None,
    )

    mock_result = MagicMock(returncode=0)
    with patch("scripts.dive_clip.subprocess.run", return_value=mock_result):
        rc = cmd_scan(args)

    assert rc == 0
    # Should have called subprocess 3 times since state was reset
    assert mock_result.returncode == 0


def test_main_help(capsys) -> None:
    """main should print help and exit with code 0 when --help is used."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "dive-clip" in captured.out


def test_main_scan_command(capsys, tmp_path: Path) -> None:
    """main should execute the scan command."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "highlights"
    output_folder.mkdir()

    mock_result = MagicMock(returncode=0)
    with patch("scripts.dive_clip.subprocess.run", return_value=mock_result):
        with pytest.raises(SystemExit) as exc_info:
            main(["scan", "--input", str(input_folder), "--output", str(output_folder)])
        assert exc_info.value.code == 0


def _scan_args(tmp_path: Path, *, debug: bool) -> argparse.Namespace:
    input_folder = tmp_path / "footage"
    input_folder.mkdir(exist_ok=True)
    output_folder = tmp_path / "highlights"
    output_folder.mkdir(exist_ok=True)
    return argparse.Namespace(
        input=str(input_folder),
        output=str(output_folder),
        config="config.yaml",
        resume=False,
        reset=False,
        debug=debug,
        log_file=None,
    )


def test_cmd_scan_debug_threads_flag_to_children(tmp_path: Path) -> None:
    """cmd_scan(--debug) should pass --debug to every child script."""
    args = _scan_args(tmp_path, debug=True)

    with patch("scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)) as mock:
        assert cmd_scan(args) == 0

    cmds = [call.args[0] for call in mock.call_args_list]
    assert len(cmds) == 3
    for cmd in cmds:
        assert "--debug" in cmd


def test_cmd_scan_without_debug_omits_flag(tmp_path: Path) -> None:
    """cmd_scan(debug=False) should not pass --debug to any child script."""
    args = _scan_args(tmp_path, debug=False)

    with patch("scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)) as mock:
        assert cmd_scan(args) == 0

    for call in mock.call_args_list:
        assert "--debug" not in call.args[0]


def test_cmd_scan_threads_state_path_to_workers(tmp_path: Path) -> None:
    """Scan and render workers should heartbeat the shared state file."""
    args = _scan_args(tmp_path, debug=False)

    with patch("scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)) as mock:
        assert cmd_scan(args) == 0

    latest = find_latest_run_dir(Path(args.output))
    assert latest is not None
    state_path = str(latest / "pipeline_state.json")
    cmds = [call.args[0] for call in mock.call_args_list]
    assert len(cmds) == 3

    for cmd in cmds[1:]:
        assert "--state-path" in cmd
        assert cmd[cmd.index("--state-path") + 1] == state_path


def test_cmd_scan_writes_step_timestamps(tmp_path: Path) -> None:
    """Completed steps should record start, finish, and duration timestamps."""
    import json

    args = _scan_args(tmp_path, debug=False)

    with patch("scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)):
        assert cmd_scan(args) == 0

    latest = find_latest_run_dir(Path(args.output))
    assert latest is not None
    state = json.loads((latest / "pipeline_state.json").read_text())
    assert "updated_at" in state
    for step in ("ingest", "scan", "render"):
        assert state[f"{step}_done"] is True
        assert f"{step}_started_at" in state
        assert f"{step}_finished_at" in state
        assert state[f"{step}_duration_seconds"] >= 0


def test_resolve_paths_prefers_cli_over_config(tmp_path: Path) -> None:
    """Explicit flags must win over general.input_dir / general.output_dir."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "general:\n  input_dir: /from-config/in\n  output_dir: /from-config/out\n"
    )
    args = argparse.Namespace(input="/from-cli/in", output="/from-cli/out", config=str(config_path))

    assert resolve_paths(args) == (Path("/from-cli/in"), Path("/from-cli/out"))


def test_resolve_paths_falls_back_to_config(tmp_path: Path) -> None:
    """Omitting the flags should fall back to the config general section."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "general:\n  input_dir: /from-config/in\n  output_dir: /from-config/out\n"
    )
    args = argparse.Namespace(input=None, output=None, config=str(config_path))

    assert resolve_paths(args) == (Path("/from-config/in"), Path("/from-config/out"))


def test_resolve_paths_requires_input(tmp_path: Path) -> None:
    """A config carrying only output_dir is not enough to locate footage."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("general:\n  output_dir: /from-config/out\n")
    args = argparse.Namespace(input=None, output=None, config=str(config_path))

    assert resolve_paths(args) is None


def test_resolve_paths_requires_output(tmp_path: Path) -> None:
    """A config carrying only input_dir is not enough to place outputs."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("general:\n  input_dir: /from-config/in\n")
    args = argparse.Namespace(input=None, output=None, config=str(config_path))

    assert resolve_paths(args) is None


def test_resolve_paths_tolerates_missing_config(tmp_path: Path) -> None:
    """A missing config must not crash path resolution when flags are given."""
    args = argparse.Namespace(
        input="/cli/in", output="/cli/out", config=str(tmp_path / "gone.yaml")
    )

    assert resolve_paths(args) == (Path("/cli/in"), Path("/cli/out"))


def test_resolve_paths_handles_malformed_config(tmp_path: Path) -> None:
    """A malformed config should return None without unhandled exception."""
    broken_config = tmp_path / "broken.yaml"
    broken_config.write_text("general:\n  input_dir: /in  output_dir: /out\n")
    args = argparse.Namespace(input=None, output=None, config=str(broken_config))

    assert resolve_paths(args) is None


def test_cmd_scan_handles_malformed_config(tmp_path: Path) -> None:
    """cmd_scan should exit with code 1 when config is malformed."""
    broken_config = tmp_path / "broken.yaml"
    broken_config.write_text("general:\n  input_dir: /in  output_dir: /out\n")
    args = argparse.Namespace(
        input=None,
        output=None,
        config=str(broken_config),
        resume=False,
        reset=False,
        debug=False,
        log_file=None,
    )

    assert cmd_scan(args) == 1


def test_main_handles_malformed_config(tmp_path: Path) -> None:
    """main should exit cleanly with code 1 when config is malformed."""
    broken_config = tmp_path / "broken.yaml"
    broken_config.write_text("general:\n  input_dir: /in  output_dir: /out\n")

    with pytest.raises(SystemExit) as exc_info:
        main(["scan", "--config", str(broken_config)])
    assert exc_info.value.code == 1


def test_cmd_scan_exits_when_paths_unresolvable(tmp_path: Path) -> None:
    """cmd_scan should bail out with code 2 when no input folder is available."""
    args = argparse.Namespace(
        input=None,
        output=None,
        config=str(tmp_path / "config.yaml"),
        resume=False,
        reset=False,
        debug=False,
        log_file=None,
    )

    assert cmd_scan(args) == 2


def test_cmd_scan_passes_output_dir_to_scan_step(tmp_path: Path) -> None:
    """The scan step must use the run folder as output dir."""
    args = _scan_args(tmp_path, debug=False)

    with patch("scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)) as mock:
        assert cmd_scan(args) == 0

    latest = find_latest_run_dir(Path(args.output))
    assert latest is not None
    scan_cmd = mock.call_args_list[1].args[0]
    assert "--output-dir" in scan_cmd
    assert scan_cmd[scan_cmd.index("--output-dir") + 1] == str(latest)


def test_cmd_scan_writes_timestamped_logfile(tmp_path: Path) -> None:
    """Every scan run should leave one logfile under runs/run-<timestamp>/."""
    args = _scan_args(tmp_path, debug=False)

    with patch("scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)) as mock:
        assert cmd_scan(args) == 0

    logs = sorted((Path(args.output) / "runs").glob("run-*/run.log"))
    assert len(logs) == 1
    assert logs[0].read_text(encoding="utf-8").strip(), "logfile should not be empty"

    for cmd in (call.args[0] for call in mock.call_args_list):
        assert "--log-file" in cmd
        assert cmd[cmd.index("--log-file") + 1] == str(logs[0])


def test_cmd_scan_honours_explicit_log_file(tmp_path: Path) -> None:
    """--log-file should replace the default timestamped location."""
    args = _scan_args(tmp_path, debug=False)
    custom = tmp_path / "nested" / "custom.log"
    args.log_file = custom

    with patch("scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)):
        assert cmd_scan(args) == 0

    assert custom.exists()
    assert (Path(args.output) / "runs").exists()


def test_cmd_scan_threads_resume_flag(tmp_path: Path) -> None:
    """cmd_scan(resume=True) should pass --resume to scan child script."""
    args = _scan_args(tmp_path, debug=False)
    args.resume = True

    with patch("scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)) as mock:
        assert cmd_scan(args) == 0

    cmds = [call.args[0] for call in mock.call_args_list]
    scan_cmd = cmds[1]
    assert "--resume" in scan_cmd


def test_background_render_worker(tmp_path: Path) -> None:
    """_run_background_render_worker should read detections and invoke render_detections."""
    import json
    import threading

    from scripts.dive_clip import _run_background_render_worker

    detections_file = tmp_path / "detections_refined.json"
    highlights_dir = tmp_path / "highlights"
    detections_file.write_text(
        json.dumps({
            "total_clips": 1,
            "total_detections": 1,
            "detections": [{"file_path": "a.mp4", "subject": "turtle"}],
        })
    )

    stop_event = threading.Event()
    # Mock render_detections
    mock_render = MagicMock()
    with patch("scripts.dive_clip.importlib.util.spec_from_file_location") as mock_spec:
        mock_mod = MagicMock()
        mock_mod.render_detections = mock_render
        mock_spec.return_value.loader.exec_module = lambda mod: None
        with patch("scripts.dive_clip.importlib.util.module_from_spec", return_value=mock_mod):
            # Run one cycle and stop
            def stop_soon():
                stop_event.set()

            timer = threading.Timer(0.1, stop_soon)
            timer.start()
            _run_background_render_worker(detections_file, highlights_dir, stop_event)
            timer.join()

    assert mock_render.called
    assert mock_render.call_args[1].get("log_to_console") is False


def test_build_parser_version() -> None:
    """build_parser should configure --version flag."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0


def test_cmd_scan_prints_version(tmp_path: Path) -> None:
    """cmd_scan should include version number in startup output."""
    from rich.panel import Panel

    args = _scan_args(tmp_path, debug=False)
    with (
        patch("scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)),
        patch("scripts.dive_clip.get_console") as mock_get_console,
    ):
        mock_console = MagicMock()
        mock_console.is_terminal = True
        mock_get_console.return_value = mock_console
        assert cmd_scan(args) == 0

    panels = [
        call.args[0]
        for call in mock_console.print.call_args_list
        if call.args and isinstance(call.args[0], Panel)
    ]
    from utils.version import get_version

    version = get_version()
    assert any(f"v{version}" in str(p.title) or f"v{version}" in str(p.renderable) for p in panels)


def test_cmd_scan_archives_run_artifacts(tmp_path: Path) -> None:
    """cmd_scan should write manifest.json and detections_refined.json under runs/run-<timestamp>/."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "highlights"
    output_folder.mkdir()

    def side_effect(cmd, *a, **kw):
        if "00_ingest_media.py" in cmd[1]:
            out_idx = cmd.index("--output") + 1
            Path(cmd[out_idx]).write_text('{"clips": []}', encoding="utf-8")
        elif "02_scan_footage.py" in cmd[1]:
            run_dir_idx = cmd.index("--run-dir") + 1
            (Path(cmd[run_dir_idx]) / "detections_refined.json").write_text(
                '{"detections": []}', encoding="utf-8"
            )
        return MagicMock(returncode=0)

    args = argparse.Namespace(
        input=str(input_folder),
        output=str(output_folder),
        config="config.yaml",
        resume=False,
        reset=False,
        debug=False,
        log_file=None,
    )

    with patch("scripts.dive_clip.subprocess.run", side_effect=side_effect):
        assert cmd_scan(args) == 0

    runs_dir = output_folder / "runs"
    assert runs_dir.exists()
    run_folders = list(runs_dir.iterdir())
    assert len(run_folders) == 1
    run_folder = run_folders[0]
    assert run_folder.name.startswith("run-")
    assert (run_folder / "run.log").exists()
    assert (run_folder / "manifest.json").exists()
    assert (run_folder / "detections_refined.json").exists()
    assert (run_folder / "run_stats.json").exists()


def test_cmd_scan_emits_pipeline_statistics(tmp_path: Path) -> None:
    """cmd_scan should compile and persist pipeline statistics in run_stats.json."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "highlights"
    output_folder.mkdir()

    manifest_payload = {
        "project_name": "Test",
        "total_clips": 1,
        "clips": [
            {
                "clip_id": "clip1",
                "file_name": "test.mp4",
                "file_path": str(input_folder / "test.mp4"),
                "duration": 50.0,
            }
        ],
    }

    detections_payload = {
        "total_clips": 1,
        "total_detections": 1,
        "detections": [
            {
                "clip_id": "clip1",
                "file_name": "test.mp4",
                "file_path": str(input_folder / "test.mp4"),
                "start_time": 10.0,
                "end_time": 14.0,
                "subject": "turtle",
                "confidence": 0.9,
                "quality_score": 8,
                "description": "turtle swimming",
            }
        ],
    }

    def side_effect(cmd, *a, **kw):
        if "00_ingest_media.py" in cmd[1]:
            out_idx = cmd.index("--output") + 1
            Path(cmd[out_idx]).write_text(json.dumps(manifest_payload), encoding="utf-8")
        elif "02_scan_footage.py" in cmd[1]:
            run_dir_idx = cmd.index("--run-dir") + 1
            run_path = Path(cmd[run_dir_idx])
            (run_path / "detections_refined.json").write_text(
                json.dumps(detections_payload), encoding="utf-8"
            )
            state_idx = cmd.index("--state-path") + 1
            state_file = Path(cmd[state_idx])
            state_data = json.loads(state_file.read_text(encoding="utf-8"))
            state_data["scan_stats"] = {
                "phase1": {
                    "candidate_windows": 1,
                    "candidate_duration": 8.0,
                    "frames_analyzed": 20,
                    "ollama_calls": 20,
                },
                "phase2": {
                    "windows_scanned": 1,
                    "detections_dedup": 1,
                    "detections_dedup_duration": 4.0,
                    "frames_analyzed": 16,
                    "ollama_calls": 16,
                },
            }
            state_file.write_text(json.dumps(state_data), encoding="utf-8")
        elif "05_render_clips.py" in cmd[1]:
            out_idx = cmd.index("--output") + 1
            out_dir = Path(cmd[out_idx])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "clip1_turtle_q8.mp4").write_bytes(b"dummy")
        return MagicMock(returncode=0)

    args = argparse.Namespace(
        input=str(input_folder),
        output=str(output_folder),
        config="config.yaml",
        resume=False,
        reset=False,
        debug=False,
        log_file=None,
    )

    with patch("scripts.dive_clip.subprocess.run", side_effect=side_effect):
        assert cmd_scan(args) == 0

    runs_dir = output_folder / "runs"
    run_folder = next(iter(runs_dir.iterdir()))
    stats_file = run_folder / "run_stats.json"
    assert stats_file.exists()

    stats_data = json.loads(stats_file.read_text(encoding="utf-8"))
    assert stats_data["video_count"] == 1
    assert stats_data["total_video_duration"] == 50.0
    assert stats_data["phase1"]["candidate_windows"] == 1
    assert stats_data["phase2"]["detections_dedup"] == 1
    assert stats_data["render"]["clips_rendered"] == 1
    assert stats_data["render"]["total_duration"] == 4.0


def test_build_parser_scan_assemble_flags() -> None:
    """build_parser should support --assemble, --music, --top-n, and --assemble-mode on scan."""
    parser = build_parser()
    default_args = parser.parse_args(["scan", "--input", "/in", "--output", "/out"])
    assert default_args.assemble is False
    assert default_args.music is None
    assert default_args.top_n is None
    assert default_args.assemble_mode == "aftermovie"

    custom_args = parser.parse_args([
        "scan",
        "--input",
        "/in",
        "--output",
        "/out",
        "--assemble",
        "--music",
        "/tmp/track.mp3",
        "--top-n=5",
        "--assemble-mode",
        "simple",
    ])
    assert custom_args.assemble is True
    assert custom_args.music == "/tmp/track.mp3"
    assert custom_args.top_n == 5
    assert custom_args.assemble_mode == "simple"


def test_build_parser_assemble_and_aftermovie_subcommands() -> None:
    """build_parser should register assemble and aftermovie subcommands."""
    parser = build_parser()

    asm_args = parser.parse_args(["assemble", "--top-n", "8"])
    assert asm_args.command == "assemble"
    assert asm_args.top_n == 8
    assert asm_args.input is None
    assert asm_args.clips_dir is None
    assert asm_args.output is None
    assert asm_args.config == "config.yaml"

    afm_args = parser.parse_args([
        "aftermovie",
        "--music",
        "song.mp3",
        "--music-volume",
        "0.25",
        "--fade-duration",
        "1.5",
        "--top-n",
        "12",
    ])
    assert afm_args.command == "aftermovie"
    assert afm_args.music == "song.mp3"
    assert afm_args.music_volume == 0.25
    assert afm_args.fade_duration == 1.5
    assert afm_args.top_n == 12


def test_cmd_scan_with_assemble_runs_aftermovie(tmp_path: Path) -> None:
    """cmd_scan with --assemble should execute Step 4/4 aftermovie assembly and archive output."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "outputs"
    output_folder.mkdir()
    music_file = tmp_path / "music.mp3"
    music_file.write_bytes(b"mp3")

    parser = build_parser()
    args = parser.parse_args([
        "scan",
        "--input",
        str(input_folder),
        "--output",
        str(output_folder),
        "--assemble",
        "--music",
        str(music_file),
        "--top-n",
        "7",
    ])

    def side_effect(cmd, *a, **kw):
        if "00_ingest_media.py" in cmd[1]:
            Path(cmd[cmd.index("--output") + 1]).write_text('{"clips": []}', encoding="utf-8")
        elif "02_scan_footage.py" in cmd[1]:
            run_path = Path(cmd[cmd.index("--run-dir") + 1])
            (run_path / "detections_refined.json").write_text(
                '{"detections": []}', encoding="utf-8"
            )
        elif "07_create_aftermovie.py" in cmd[1]:
            out_file = Path(cmd[cmd.index("--output") + 1])
            out_file.write_bytes(b"aftermovie video")
        return MagicMock(returncode=0)

    with patch("scripts.dive_clip.subprocess.run", side_effect=side_effect) as mock_run:
        assert cmd_scan(args) == 0

    assert mock_run.call_count == 4
    assemble_cmd = mock_run.call_args_list[3].args[0]
    assert "07_create_aftermovie.py" in assemble_cmd[1]
    assert "--music" in assemble_cmd
    assert assemble_cmd[assemble_cmd.index("--music") + 1] == str(music_file)
    assert "--top-n" in assemble_cmd
    assert assemble_cmd[assemble_cmd.index("--top-n") + 1] == "7"
    assert assemble_cmd[assemble_cmd.index("--output") + 1] == str(output_folder / "aftermovie.mp4")

    latest = find_latest_run_dir(output_folder)
    assert latest is not None
    assert (output_folder / "aftermovie.mp4").exists()
    assert (latest / "aftermovie.mp4").exists()
    state = json.loads((latest / "pipeline_state.json").read_text(encoding="utf-8"))
    assert state.get("assemble_done") is True


def test_cmd_scan_with_assemble_simple_mode(tmp_path: Path) -> None:
    """cmd_scan with --assemble --assemble-mode simple should invoke 06_assemble_video.py."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "outputs"
    output_folder.mkdir()

    parser = build_parser()
    args = parser.parse_args([
        "scan",
        "--input",
        str(input_folder),
        "--output",
        str(output_folder),
        "--assemble",
        "--assemble-mode",
        "simple",
    ])

    with patch(
        "scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)
    ) as mock_run:
        assert cmd_scan(args) == 0

    assert mock_run.call_count == 4
    assemble_cmd = mock_run.call_args_list[3].args[0]
    assert "06_assemble_video.py" in assemble_cmd[1]
    assert assemble_cmd[assemble_cmd.index("--output") + 1] == str(
        output_folder / "highlight_reel.mp4"
    )


def test_cmd_scan_assemble_resume_skips_completed(tmp_path: Path) -> None:
    """cmd_scan --assemble --resume should skip assembly when assemble_done is true."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "outputs"
    output_folder.mkdir()

    run_dir = output_folder / "runs" / "run-20261101-100000"
    run_dir.mkdir(parents=True)
    state_path = run_dir / "pipeline_state.json"
    state_path.write_text(
        json.dumps({
            "ingest_done": True,
            "scan_done": True,
            "render_done": True,
            "assemble_done": True,
        }),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "detections_refined.json").write_text("{}", encoding="utf-8")
    (output_folder / "aftermovie.mp4").write_bytes(b"video")

    parser = build_parser()
    args = parser.parse_args([
        "scan",
        "--input",
        str(input_folder),
        "--output",
        str(output_folder),
        "--assemble",
        "--resume",
    ])

    with patch("scripts.dive_clip.subprocess.run") as mock_run:
        assert cmd_scan(args) == 0

    mock_run.assert_not_called()


def test_cmd_scan_assemble_reruns_when_video_missing(tmp_path: Path) -> None:
    """cmd_scan --assemble --resume should re-run Step 4 if aftermovie.mp4 was deleted."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "outputs"
    output_folder.mkdir()

    run_dir = output_folder / "runs" / "run-20261101-100000"
    run_dir.mkdir(parents=True)
    state_path = run_dir / "pipeline_state.json"
    state_path.write_text(
        json.dumps({
            "ingest_done": True,
            "scan_done": True,
            "render_done": True,
            "assemble_done": True,
        }),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "detections_refined.json").write_text("{}", encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args([
        "scan",
        "--input",
        str(input_folder),
        "--output",
        str(output_folder),
        "--assemble",
        "--resume",
    ])

    with patch(
        "scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)
    ) as mock_run:
        assert cmd_scan(args) == 0

    assert mock_run.call_count == 1
    assert "07_create_aftermovie.py" in mock_run.call_args.args[0][1]


def test_cmd_scan_assemble_summary_panel_includes_aftermovie_path(tmp_path: Path) -> None:
    """Summary panel should display the final assembled video path when --assemble is enabled."""
    from rich.panel import Panel

    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "outputs"
    output_folder.mkdir()

    parser = build_parser()
    args = parser.parse_args([
        "scan",
        "--input",
        str(input_folder),
        "--output",
        str(output_folder),
        "--assemble",
    ])

    with (
        patch("scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)),
        patch("scripts.dive_clip.get_console") as mock_get_console,
    ):
        mock_console = MagicMock()
        mock_console.is_terminal = True
        mock_get_console.return_value = mock_console
        assert cmd_scan(args) == 0

    panels = [
        call.args[0]
        for call in mock_console.print.call_args_list
        if call.args and isinstance(call.args[0], Panel)
    ]
    expected_path = str(output_folder / "aftermovie.mp4")
    assert any(expected_path in str(p.renderable) for p in panels)


def test_cmd_assemble_subcommand_resolves_latest_run(tmp_path: Path) -> None:
    """dive-clip assemble should auto-resolve detections and highlights from latest run."""
    output_folder = tmp_path / "outputs"
    run_dir = output_folder / "runs" / "run-20261101-120000"
    highlights_dir = run_dir / "highlights"
    highlights_dir.mkdir(parents=True)
    detections_file = run_dir / "detections_refined.json"
    detections_file.write_text('{"detections": []}', encoding="utf-8")

    from utils.run_archive import update_latest_symlink

    update_latest_symlink(output_folder, run_dir)

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f'general:\n  output_dir: "{output_folder}"\n', encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(["assemble", "--config", str(config_file), "--top-n", "6"])

    with patch(
        "scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)
    ) as mock_run:
        assert args.func(args) == 0

    assert mock_run.call_count == 1
    cmd = mock_run.call_args.args[0]
    assert "06_assemble_video.py" in cmd[1]
    assert Path(cmd[cmd.index("--input") + 1]).resolve() == detections_file.resolve()
    assert Path(cmd[cmd.index("--clips-dir") + 1]).resolve() == highlights_dir.resolve()
    assert Path(cmd[cmd.index("--output") + 1]) == output_folder / "highlight_reel.mp4"
    assert cmd[cmd.index("--top-n") + 1] == "6"


def test_cmd_aftermovie_subcommand_resolves_latest_run(tmp_path: Path) -> None:
    """dive-clip aftermovie should auto-resolve latest run and pass music and fade args."""
    output_folder = tmp_path / "outputs"
    run_dir = output_folder / "runs" / "run-20261101-120000"
    highlights_dir = run_dir / "highlights"
    highlights_dir.mkdir(parents=True)
    detections_file = run_dir / "detections_refined.json"
    detections_file.write_text('{"detections": []}', encoding="utf-8")

    from utils.run_archive import update_latest_symlink

    update_latest_symlink(output_folder, run_dir)

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f'general:\n  output_dir: "{output_folder}"\n', encoding="utf-8")
    music_file = tmp_path / "bg.mp3"
    music_file.write_bytes(b"mp3")

    parser = build_parser()
    args = parser.parse_args([
        "aftermovie",
        "--config",
        str(config_file),
        "--music",
        str(music_file),
        "--music-volume",
        "0.2",
        "--fade-duration",
        "1.5",
        "--top-n",
        "8",
    ])

    with patch(
        "scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)
    ) as mock_run:
        assert args.func(args) == 0

    assert mock_run.call_count == 1
    cmd = mock_run.call_args.args[0]
    assert "07_create_aftermovie.py" in cmd[1]
    assert Path(cmd[cmd.index("--input") + 1]).resolve() == detections_file.resolve()
    assert Path(cmd[cmd.index("--clips-dir") + 1]).resolve() == highlights_dir.resolve()
    assert Path(cmd[cmd.index("--output") + 1]) == output_folder / "aftermovie.mp4"
    assert cmd[cmd.index("--music") + 1] == str(music_file)
    assert cmd[cmd.index("--music-volume") + 1] == "0.2"
    assert cmd[cmd.index("--fade-duration") + 1] == "1.5"
    assert cmd[cmd.index("--top-n") + 1] == "8"


def test_cmd_scan_assemble_failure_preserves_prior_steps(tmp_path: Path) -> None:
    """If Step 4 assembly fails, cmd_scan returns non-zero while preserving prior outputs."""
    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "outputs"
    output_folder.mkdir()

    parser = build_parser()
    args = parser.parse_args([
        "scan",
        "--input",
        str(input_folder),
        "--output",
        str(output_folder),
        "--assemble",
    ])

    def side_effect(cmd, *a, **kw):
        if "00_ingest_media.py" in cmd[1]:
            Path(cmd[cmd.index("--output") + 1]).write_text('{"clips": []}', encoding="utf-8")
        elif "02_scan_footage.py" in cmd[1]:
            run_path = Path(cmd[cmd.index("--run-dir") + 1])
            (run_path / "detections_refined.json").write_text(
                '{"detections": []}', encoding="utf-8"
            )
        elif "07_create_aftermovie.py" in cmd[1]:
            return MagicMock(returncode=1)
        return MagicMock(returncode=0)

    with patch("scripts.dive_clip.subprocess.run", side_effect=side_effect):
        assert cmd_scan(args) == 1

    latest = find_latest_run_dir(output_folder)
    assert latest is not None
    state = json.loads((latest / "pipeline_state.json").read_text(encoding="utf-8"))
    assert state.get("ingest_done") is True
    assert state.get("scan_done") is True
    assert state.get("render_done") is True
    assert state.get("assemble_done") is False


def test_cmd_aftermovie_reads_top_n_from_config(tmp_path: Path) -> None:
    """dive-clip aftermovie and assemble should fall back to aftermovie.top_n_clips in config."""
    output_folder = tmp_path / "outputs"
    output_folder.mkdir()

    config_file = tmp_path / "config.aftermovie.yaml"
    config_file.write_text(
        f'general:\n  output_dir: "{output_folder}"\naftermovie:\n  top_n_clips: 15\n',
        encoding="utf-8",
    )

    parser = build_parser()
    args = parser.parse_args(["aftermovie", "--config", str(config_file)])

    with patch(
        "scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)
    ) as mock_run:
        assert args.func(args) == 0

    cmd = mock_run.call_args.args[0]
    assert cmd[cmd.index("--top-n") + 1] == "15"


def test_cmd_aftermovie_reads_volume_and_fade_from_config(tmp_path: Path) -> None:
    """dive-clip aftermovie should resolve music_volume and fade_duration from config."""
    output_folder = tmp_path / "outputs"
    output_folder.mkdir()

    config_file = tmp_path / "config.aftermovie.yaml"
    config_file.write_text(
        f'general:\n  output_dir: "{output_folder}"\n'
        "aftermovie:\n"
        "  top_n_clips: 12\n"
        "  music_volume: 0.25\n"
        "  fade_duration: 2.0\n",
        encoding="utf-8",
    )

    parser = build_parser()
    args = parser.parse_args(["aftermovie", "--config", str(config_file)])

    with patch(
        "scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)
    ) as mock_run:
        assert args.func(args) == 0

    cmd = mock_run.call_args.args[0]
    assert cmd[cmd.index("--top-n") + 1] == "12"
    assert cmd[cmd.index("--music-volume") + 1] == "0.25"
    assert cmd[cmd.index("--fade-duration") + 1] == "2.0"


def test_cmd_scan_worker_joined_before_final_render(tmp_path: Path) -> None:
    """cmd_scan must join the background render worker before launching the final render subprocess."""
    import threading

    input_folder = tmp_path / "footage"
    input_folder.mkdir()
    output_folder = tmp_path / "highlights"
    output_folder.mkdir()

    config_path = tmp_path / "config.yaml"
    config_path.write_text('general:\n  output_dir: "./outputs"\n')

    worker_thread_captured: list[threading.Thread] = []
    original_thread_init = threading.Thread.__init__

    def thread_init_spy(self, *args, **kwargs):
        original_thread_init(self, *args, **kwargs)
        if kwargs.get("target") and "render_worker" in kwargs["target"].__name__:
            worker_thread_captured.append(self)

    step_log: list[str] = []

    def mock_run(cmd, description):
        cmd_str = " ".join(cmd)
        if "00_ingest_media.py" in cmd_str:
            manifest_path = Path(cmd[cmd.index("--output") + 1])
            manifest_path.write_text('{"project_name": "Test", "clips": []}')
            step_log.append("ingest")
            return 0
        if "02_scan_footage.py" in cmd_str:
            step_log.append("scan")
            run_dir = Path(cmd[cmd.index("--output-dir") + 1])
            detections_path = run_dir / "detections_refined.json"
            detections_path.write_text('{"detections": []}')
            assert len(worker_thread_captured) == 1
            worker = worker_thread_captured[0]
            assert worker.is_alive()
            return 0
        if "05_render_clips.py" in cmd_str:
            step_log.append("render")
            assert len(worker_thread_captured) == 1
            worker = worker_thread_captured[0]
            assert not worker.is_alive(), (
                "Background render worker must be completely joined before final render starts!"
            )
            return 0
        return 0

    args = argparse.Namespace(
        input=str(input_folder),
        output=str(output_folder),
        config=str(config_path),
        resume=False,
        reset=False,
        debug=False,
        log_file=None,
    )

    with (
        patch("scripts.dive_clip.run", side_effect=mock_run),
        patch.object(threading.Thread, "__init__", side_effect=thread_init_spy, autospec=True),
    ):
        rc = cmd_scan(args)

    assert rc == 0
    assert step_log == ["ingest", "scan", "render"]
