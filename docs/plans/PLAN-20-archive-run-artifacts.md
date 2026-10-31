# Architecture Plan: [Issue #20] Archive run artifacts (manifest, detections, log) under runs directory

- **Issue**: [#20](https://github.com/tiwu/dive-clip-tui/issues/20)
- **Status**: Review Ready
- **Author**: Autonomous Agent / Contributor
- **Date**: 2026-10-31
- **Target Branch**: `feature/issue-20-archive-run-artifacts`

---

## 1. Problem Statement & Motivation

When running the pipeline over time with refined prompts, parameters, and code adjustments, each run overwrites `manifest_raw.json` and `detections_refined.json` in the root output folder. While logs were retained in `<output>/logs/`, there was no unified record linking the detections and manifest from that specific execution.

To track detection quality evolution across runs, each pipeline execution should store a self-contained snapshot under `<output>/runs/run-<timestamp>/` containing:
- `run.log` (execution log)
- `manifest_raw.json` (ingested media snapshot)
- `detections_refined.json` (refined AI detections snapshot)

The root `<output>/manifest_raw.json` and `<output>/detections_refined.json` remain in place to preserve compatibility with downstream rendering and EDL generation tools.

---

## 2. Requirements & Acceptance Criteria

- [x] Add run management / archiving utilities (`utils/run_archive.py` or `utils/logging_setup.py`):
  - Generate timestamped run directory paths: `<output>/runs/run-<YYYYMMDD-HHMMSS>/`.
  - Provide helper `archive_run_artifacts(run_dir: Path, ...)` to snapshot `manifest_raw.json`, `detections_refined.json`, and copy log files if needed.
- [x] Update `utils/logging_setup.py`:
  - `default_log_path(output_dir: Path, timestamp: str | None = None) -> Path`: default log destination placed inside the run directory `<output>/runs/run-<timestamp>/run.log` (or backward-compatible alias).
- [x] Update `scripts/dive_clip.py`:
  - Initialize the run directory at the start of `cmd_scan`.
  - Archive `manifest_raw.json` into the run directory upon ingest completion.
  - Archive `detections_refined.json` into the run directory upon scan completion.
  - Archive final run artifacts at pipeline completion.
- [x] Update `scripts/02_scan_footage.py`:
  - When run standalone, if an archive / run directory is configured or created, archive the generated `detections_refined.json`.
- [x] Maintain backward compatibility:
  - Root `<output>/manifest_raw.json` and `<output>/detections_refined.json` continue to be written as the "latest" state for downstream tools (`render`, `assemble`, `aftermovie`, `edl`).
- [x] Automated tests cover run directory creation, artifact archiving, and CLI pipeline integration.
- [x] All relevant tests pass (`uv run python -m pytest`) and code passes Ruff lint and format checks.

---

## 3. Change Log and Release Impact

- **Change category**: Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**: Add per-run artifact archiving under `<output>/runs/run-<timestamp>/`, storing snapshots of the run log, `manifest_raw.json`, and `detections_refined.json` while maintaining latest files in the output directory.

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Add unit tests in `tests/test_run_archive.py` (or `tests/test_logging_setup.py` and `tests/test_dive_clip.py`) verifying:
     - Run directory generation with timestamp format `run-YYYYMMDD-HHMMSS`.
     - Archiving `manifest_raw.json` and `detections_refined.json` into `run_dir`.
     - `dive_clip.py` pipeline creates `<output>/runs/run-<timestamp>/` containing `run.log`, `manifest_raw.json`, and `detections_refined.json`.
2. **Green (Implementation)**:
   - Implement `utils/run_archive.py` (or extend `utils/logging_setup.py`).
   - Integrate run directory creation and artifact copying into `scripts/dive_clip.py` and `scripts/02_scan_footage.py`.
3. **Refactor & Quality**:
   - Run focused pytest tests.
   - Run full pytest test suite.
   - Run Ruff check and format.

---

## 5. Verification & Rollout Checklist

- [x] Tests pass: `uv run python -m pytest`
- [x] Changed Python files pass Ruff lint and formatting checks
- [x] `CHANGELOG.md` updated under `[Unreleased]`
- [x] PR labeled `semver:minor`
- [x] PR submitted with `Closes #20` and plan linked
