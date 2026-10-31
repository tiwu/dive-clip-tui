# Architecture Plan: [Issue #24] Self-contained run directories with resume support and manifest renaming

- **Issue**: [#24](https://github.com/tiwu/dive-clip-tui/issues/24)
- **Status**: Completed
- **Author**: Autonomous Agent / Contributor
- **Date**: 2026-10-31
- **Target Branch**: `feature/issue-24-self-contained-runs-and-resume`

---

## 1. Problem Statement & Motivation

Currently, pipeline artifacts are partially written to the output root (`.pipeline_state.json`, `manifest_raw.json`, `detections_refined.json`, `highlights/`) and partially archived into `<output>/runs/run-<timestamp>/` (`run.log`, `manifest_raw.json`, `detections_refined.json`). This leaves run directories incomplete:
1. `.pipeline_state.json` is not included in the run folder, making runs impossible to inspect or resume individually.
2. Rendered highlight clips are placed in the output root rather than in the run folder, preventing each run from being an independent, self-contained snapshot.
3. The manifest filename `manifest_raw.json` is clunky and inconsistent with the cleaner naming of other artifacts.
4. Resuming an existing run requires passing flags without a clean way to specify or continue a particular prior run folder or automatically restore prior execution arguments.

Making run directories self-contained and enhancing `--resume` ensures every execution is an isolated, reproducible snapshot that can be resumed or audited effortlessly.

---

## 2. Requirements & Acceptance Criteria

- [x] Each run creates and executes within a self-contained workspace under `<output_dir>/runs/<run_id>/`:
  - `pipeline_state.json` (visible state file containing full run metadata and step/item progress).
  - `manifest.json` (renamed from `manifest_raw.json`).
  - `detections_refined.json`.
  - `run.log`.
  - `highlights/` (rendered MP4 highlight clips).
- [x] A convenience symlink `<output_dir>/latest` points to the active or most recently completed run directory.
- [x] `PipelineState` records full run metadata: `run_id`, `input_dir`, `output_dir`, `config_path`, CLI arguments, timestamps, step completion statuses, and item progress.
- [x] Renaming `manifest_raw.json` to `manifest.json`:
  - `00_ingest_media.py`, `02_scan_footage.py`, and `dive_clip.py` default to `manifest.json`.
  - Backward compatibility: if `manifest.json` does not exist but `manifest_raw.json` is present in a legacy directory, it is recognized as a fallback.
- [x] Enhanced `--resume` CLI option:
  - `dive-clip scan --resume`: Automatically locates and resumes the latest run in `<output_dir>/runs/`.
  - `dive-clip scan --resume <run_id_or_path>`: Resumes the specified run folder (resolving relative paths, absolute paths, or run directory names like `run-20261031-120000`).
  - Restores `input_dir`, `config`, and step progress from `pipeline_state.json` when omitted from CLI.
- [x] Background renderer in `dive_clip.py` outputs directly to `<run_dir>/highlights/`.
- [x] Automated tests cover run folder layout, artifact naming, symlink creation, and resume behavior.
- [x] Documentation (`README.md`, `AGENTS.md`) is updated to reflect new layout and resume usage.

---

## 3. Change Log and Release Impact

- **Change category**: Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**:
  - Add self-contained run snapshots in `<output>/runs/<run_id>/` storing `pipeline_state.json`, `manifest.json`, `detections_refined.json`, `run.log`, and `highlights/` clips with an `<output>/latest` symlink.
  - Support resuming previous runs via `dive-clip scan --resume [run_id_or_path]`, automatically continuing the latest run or targeting a specific run folder while restoring run parameters.
  - Rename manifest artifact from `manifest_raw.json` to `manifest.json` with backward-compatible fallback.

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Create `tests/test_run_resume.py` testing:
     - New run directory layout contains `pipeline_state.json`, `manifest.json`, `detections_refined.json`, `highlights/`, and `run.log`.
     - `PipelineState` serialization and deserialization of run context (`run_id`, `input_dir`, `output_dir`, `config_path`).
     - `--resume` without arguments resolving the latest run.
     - `--resume <run_id>` and `--resume <path>` resuming specified run folders.
     - `<output>/latest` symlink creation and updates.
   - Update existing tests in `tests/test_dive_clip.py`, `tests/test_checkpoint.py`, and `tests/test_scan_footage.py` for `manifest.json` and `pipeline_state.json`.
2. **Green (Implementation)**:
   - Update `utils/checkpoint.py`: add run context metadata methods to `PipelineState`.
   - Update `utils/run_archive.py`: add run resolution helpers (`find_latest_run_dir`, `resolve_run_dir`, `update_latest_symlink`).
   - Update `scripts/dive_clip.py`: configure run directory as primary workspace, support `--resume [run_id_or_path]`, point background renderer and Step 3 to `run_dir / "highlights"`, update `latest` symlink.
   - Update `scripts/00_ingest_media.py` and `scripts/02_scan_footage.py` to use `manifest.json` with fallback.
3. **Refactor & Quality**:
   - Run focused test suite: `uv run python -m pytest tests/test_run_resume.py tests/test_dive_clip.py tests/test_checkpoint.py`
   - Run full test suite: `uv run python -m pytest`
   - Run Ruff lint and formatting checks: `uv run ruff check .` and `uv run ruff format --check .`

---

## 5. Verification & Rollout Checklist

- [x] Tests pass: `uv run python -m pytest`
- [x] Changed Python files pass Ruff lint and formatting checks
- [x] `CHANGELOG.md` updated under `[Unreleased]`
- [x] PR has `semver:minor` label
- [x] PR submitted with `Closes #24` and plan linked
