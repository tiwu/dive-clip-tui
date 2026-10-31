"""Tests for scripts/dive_clip.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.dive_clip import build_parser, cmd_scan, main, resolve_paths, run


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

    # Pre-create state with all steps done
    state_path = output_folder / ".pipeline_state.json"
    state_path.write_text('{"ingest_done": true, "scan_done": true, "render_done": true}')

    # Pre-create manifest and detections so the "already done" check passes
    (output_folder / "manifest_raw.json").write_text("{}")
    (output_folder / "detections_refined.json").write_text("{}")

    args = argparse.Namespace(
        input=str(input_folder),
        output=str(output_folder),
        config="config.yaml",
        resume=False,
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

    # State file should be created with all steps done
    state_path = output_folder / ".pipeline_state.json"
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

    # Pre-create state with all steps done
    state_path = output_folder / ".pipeline_state.json"
    state_path.write_text('{"ingest_done": true, "scan_done": true, "render_done": true}')

    args = argparse.Namespace(
        input=str(input_folder),
        output=str(output_folder),
        config="config.yaml",
        resume=False,
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

    state_path = str(Path(args.output) / ".pipeline_state.json")
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

    state = json.loads((Path(args.output) / ".pipeline_state.json").read_text())
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
    """The scan step must use the resolved output folder, not the config one."""
    args = _scan_args(tmp_path, debug=False)

    with patch("scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)) as mock:
        assert cmd_scan(args) == 0

    scan_cmd = mock.call_args_list[1].args[0]
    assert "--output-dir" in scan_cmd
    assert scan_cmd[scan_cmd.index("--output-dir") + 1] == args.output


def test_cmd_scan_writes_timestamped_logfile(tmp_path: Path) -> None:
    """Every scan run should leave one timestamped logfile under the output dir."""
    args = _scan_args(tmp_path, debug=False)

    with patch("scripts.dive_clip.subprocess.run", return_value=MagicMock(returncode=0)) as mock:
        assert cmd_scan(args) == 0

    logs = sorted((Path(args.output) / "logs").glob("run-*.log"))
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
    assert not (Path(args.output) / "logs").exists()


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
