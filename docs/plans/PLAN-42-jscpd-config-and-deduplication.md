# Architecture Plan: [Issue #42] Add jscpd configuration and deduplicate pipeline scripts

- **Issue**: [#42](https://github.com/tiwu/dive-clip-tui/issues/42)
- **Status**: Plan Review
- **Author**: Autonomous Agent
- **Date**: 2026-11-01
- **Target Branch**: `feature/issue-42-jscpd-config-and-deduplication`

---

## 1. Problem Statement & Motivation

Code clone analysis using `jscpd` revealed 12 duplicate code clones (3.56% duplication across 28 Python files). Significant hotspots include:
1. `setup_signal_handlers()` copy-pasted across 6 separate scripts (`00_ingest_media.py`, `02_scan_footage.py`, `05_render_clips.py`, `06_assemble_video.py`, `07_create_aftermovie.py`, and `dive_clip.py`).
2. `concat_clips()` FFmpeg concatenation demuxer routine identically duplicated in `06_assemble_video.py` and `07_create_aftermovie.py`.
3. Assembly artifact path resolution and validation logic duplicated between `06_assemble_video.py` and `07_create_aftermovie.py`.
4. Absence of a standard root `.jscpd.json` configuration file, requiring manual CLI path parameters and flags.

Consolidating these into existing or focused modules in `utils/` eliminates fragile copy-paste maintenance while maintaining backward compatibility and test coverage.

---

## 2. Requirements & Acceptance Criteria

- [ ] **Acceptance Criterion 1 (.jscpd.json)**: Add `.jscpd.json` at repository root configured for Python with ignored directories (`.venv`, `outputs`, caches) so `npx jscpd` runs out of the box across `scripts/`, `utils/`, and `tui/`.
- [ ] **Acceptance Criterion 2 (Signal Handling)**: Consolidate `setup_signal_handlers()` into a shared utility function (e.g., in `utils/logging_setup.py` or `utils/process.py`) and import it in all 6 scripts.
- [ ] **Acceptance Criterion 3 (FFmpeg Concat & Assembly Helpers)**: Consolidate `concat_clips()` into a shared utility (such as `utils/ffmpeg_streamer.py` or `utils/run_archive.py`) with unified return code checking and error logging, referenced by both `06_assemble_video.py` and `07_create_aftermovie.py`.
- [ ] **Acceptance Criterion 4 (Artifact Resolution)**: Streamline detections and clip path verification helper in `utils/run_archive.py` to remove duplicate boilerplate in `06_assemble_video.py` and `07_create_aftermovie.py`.
- [ ] **Acceptance Criterion 5 (Verification & Non-regression)**: All 342+ unit tests pass without regressions (`uv run python -m pytest`), and Ruff lint/format checks pass.
- [ ] **Acceptance Criterion 6 (Duplication Reduction)**: Re-running `npx jscpd` demonstrates a measurable drop in clone count and duplication percentage.

---

## 3. Change Log and Release Impact

- **Change category**: Changed
- **Proposed SemVer impact**: `semver:patch`
- **Draft changelog entry**:
  - `Added`: Add root `.jscpd.json` configuration for repository clone detection.
  - `Changed`: Consolidate duplicated signal handling, FFmpeg concatenation, and artifact resolution into shared `utils/` helpers.

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Add unit tests for the shared signal handler helper in `tests/test_logging_setup.py` (or new `tests/test_signals.py`).
   - Add unit tests for the extracted `concat_clips` and artifact verification helpers in `tests/test_assemble_video.py` and `tests/test_aftermovie.py`.
2. **Green (Implementation)**:
   - Create root `.jscpd.json`.
   - Implement extracted functions in `utils/`.
   - Update `00_ingest_media.py`, `02_scan_footage.py`, `05_render_clips.py`, `06_assemble_video.py`, `07_create_aftermovie.py`, and `dive_clip.py`.
3. **Refactor & Quality**:
   - Run `uv run python -m pytest` to confirm all existing and new tests pass.
   - Run `uv run ruff check .` and `uv run ruff format .` on all changed files.
   - Run `npx jscpd` to verify duplication reduction.

---

## 5. Verification & Rollout Checklist

- [ ] Tests pass: `uv run python -m pytest`
- [ ] Changed Python files pass Ruff lint and formatting checks
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR created with label `semver:patch`, referencing `Closes #42`
- [ ] Plan embedded and linked in issue #42
