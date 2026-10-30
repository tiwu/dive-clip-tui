# Architecture Plan: [Issue #13] Add Rich console logging and inline scan progress bars

- **Issue**: [#13](https://github.com/tiwu/dive-clip-tui/issues/13)
- **Status**: In Progress
- **Author**: Autonomous Agent / Contributor
- **Date**: 2026-10-30
- **Target Branch**: `feature/issue-13-rich-cli-progress`

---

## 1. Problem Statement & Motivation

Currently, the CLI output during `dive-clip scan` uses standard Python logging formats with timestamps and plain monospace text. During footage scanning (`02_scan_footage.py`), it logs every single frame sequentially (`[Pass 1] Analyzing frame 1 of 23`, `[Pass 1] Analyzing frame 2 of 23`, etc.), causing rapid vertical terminal scrolling and visual clutter. In Pass 2, fine frames are analyzed without an inline progress indicator.

Using `rich` (already a dependency via Textual):
1. Terminal log records become colored, well-structured, and readable with `RichHandler`.
2. File logging preserves clean timestamps and unescaped plain text in `run-*.log`.
3. Frame analysis displays an inline updating progress bar with real-time stats (frame count, detection counts, elapsed, ETA) without scrolling line spam.
4. Pipeline step headers become clear visual section banners.

---

## 2. Requirements & Acceptance Criteria

- [x] `utils/logging_setup.py`: Configure terminal output with `rich.logging.RichHandler` for colored logs, clean timestamps, and syntax highlighting.
- [x] `utils/logging_setup.py`: Preserve plain text formatting for file logging (`run-*.log`) without ANSI escape sequences.
- [x] `scripts/02_scan_footage.py`: Display an inline updating progress bar for Pass 1 (coarse scan) showing clip name, frame progress, detection count, and ETA.
- [x] `scripts/02_scan_footage.py`: Display an inline updating progress bar for Pass 2 (fine scan) showing current window, frame progress, and detections.
- [x] `scripts/02_scan_footage.py`: Demote per-frame console logging during progress bar runs to avoid breaking the inline display, while ensuring full detail remains in debug/file logs.
- [x] `scripts/02_scan_footage.py`: Gracefully degrade to non-interactive mode when stderr is not a TTY or in headless environments.
- [x] `scripts/dive_clip.py`: Render colored step headers / panels for pipeline stages (Ingest, Scan, Render).
- [x] Automated tests cover `logging_setup` RichHandler configuration, fallback, and scan execution.

---

## 3. Change Log and Release Impact

- **Change category**: Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**: Add Rich console logging and inline progress indicators for video scanning

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Add tests in `tests/test_logging_setup.py` verifying `RichHandler` is used for stream logging and `FileHandler` retains plain text formatting.
   - Add/update tests in `tests/test_scan_footage.py` verifying progress display runs and finishes without breaking `process_clip` return values or error handling.
2. **Green (Implementation)**:
   - Update `utils/logging_setup.py` with `RichHandler` and console configuration.
   - Update `scripts/02_scan_footage.py` to use `rich.progress.Progress` with custom columns (spinner/bar, task description, frame counter, detection counter, ETA).
   - Update `scripts/dive_clip.py` with styled step banners.
3. **Refactor & Quality**:
   - Run focused tests: `uv run python -m pytest tests/test_logging_setup.py tests/test_scan_footage.py`
   - Run full test suite: `uv run python -m pytest`
   - Verify code formatting and linting: `uv run ruff check .` and `uv run ruff format --check .`

---

## 5. Verification & Rollout Checklist

- [x] Tests pass: `uv run python -m pytest`
- [x] Changed Python files pass Ruff lint and formatting checks
- [x] `CHANGELOG.md` updated under `[Unreleased]`
- [x] PR labeled with `semver:minor`
- [x] PR submitted with `Closes #13` and plan linked

