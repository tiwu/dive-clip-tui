# Architecture Plan: [Issue #40] Add end-to-end highlight assembly to CLI (`dive-clip scan --assemble`, `assemble`, and `aftermovie`)

- **Issue**: [#40](https://github.com/tiwu/dive-clip-tui/issues/40)
- **Status**: Approved / In Progress
- **Author**: Autonomous Agent / Benevolent Product Owner
- **Date**: 2026-11-01
- **Target Branch**: `feature/issue-40-end-to-end-cli-assembly`

---

## 1. Problem Statement & Motivation

The central mission of **Dive Clip TUI** is:
> *Point at a folder of raw dive or motorcycle footage, let local AI find the interesting moments, and get back a watchable highlight video — without cloud subscriptions, mandatory DaVinci Resolve workflows, or manual clipping fatigue.*

Currently, the primary entry point `dive-clip scan` stops after Phase 1 Step 1.3: it ingests, scans, and renders individual highlight clips into `<output>/highlights/`, leaving the user with dozens of unstitched short MP4 files. To get a finished video or aftermovie, users must:
1. Discover standalone scripts (`scripts/06_assemble_video.py` or `scripts/07_create_aftermovie.py`).
2. Manually look up and supply input paths, clip folders, and configuration flags.
3. Cope with the fact that these standalone scripts do not automatically resolve run archives (`outputs/latest`).

This breaks the "one-command watchable video" promise. Users need:
1. An `--assemble` flag on `dive-clip scan` that runs hands-off from raw SD card ingestion all the way through to a final music-backed aftermovie.
2. First-class `dive-clip assemble` and `dive-clip aftermovie` subcommands that auto-detect the latest run archive if paths are omitted.

---

## 2. Requirements & Acceptance Criteria

### Functional Requirements

1. **`dive-clip scan --assemble` integration:**
   - Add `--assemble` flag to `dive-clip scan` CLI parser.
   - Support optional `--music <path>` for background music track.
   - Support `--top-n <int>` (default: 10, or read from config) to control how many highlights are included.
   - Support `--assemble-mode {aftermovie,simple}` (default: `aftermovie` with crossfades and music ducking; `simple` for quick concatenation).
   - Execute Step 4/4 ("Assemble aftermovie" or "Assemble highlight video") after Step 3/3 ("Render clips").
   - Support `--resume`: if the `assemble` step is already marked completed in `pipeline_state.json`, skip it on resume unless reset.
   - Save the assembled video to `<output>/aftermovie.mp4` (or `highlight_reel.mp4` in simple mode), archive a copy to `<run_dir>/`, and update the `latest` snapshot.
   - Prominently display the final video file path in the Rich terminal summary panel and log summary.

2. **`dive-clip assemble` subcommand:**
   - Direct CLI subcommand to concatenate rendered clips.
   - Arguments:
     - `--input`: Path to `detections_refined.json` (defaults to `<output>/latest/detections_refined.json` or `<output>/detections_refined.json`).
     - `--clips-dir`: Path to rendered highlights (defaults to `<output>/highlights`).
     - `--output`: Path to output MP4 file (defaults to `<output>/highlight_reel.mp4`).
     - `--top-n`: Number of clips to assemble (default: 10).
     - `--config`: Path to config YAML (default: `config.yaml`).
   - Reuses `scripts/06_assemble_video.py` logic.

3. **`dive-clip aftermovie` subcommand:**
   - Direct CLI subcommand to generate an aftermovie with music and crossfades.
   - Arguments:
     - `--input`: Path to `detections_refined.json` (defaults to `<output>/latest/detections_refined.json` or `<output>/detections_refined.json`).
     - `--clips-dir`: Path to rendered highlights (defaults to `<output>/highlights`).
     - `--output`: Path to output MP4 file (defaults to `<output>/aftermovie.mp4`).
     - `--music`: Path to audio file.
     - `--music-volume`: Music ducking level (default: 0.15).
     - `--fade-duration`: Transition crossfade duration in seconds (default: 1.0).
     - `--top-n`: Number of clips (default: 10).
     - `--config`: Path to config YAML (default: `config.yaml`).
   - Reuses `scripts/07_create_aftermovie.py` logic.

4. **Roadmap alignment:**
   - Update `docs/roadmap/simple_plan.md` to reflect Option 1 implementation and confirm Option 3 (social vertical 9:16 reframing) remains on the roadmap for future execution.

### Non-Functional Requirements
- Maintain strictly local-first processing (FFmpeg only).
- Preserve existing checkpoint and resume guarantees in `utils/checkpoint.py`.
- Graceful error handling: if assembly fails (e.g., FFmpeg timeout or no clips found), scan returns an informative error code without corrupting prior scan/render outputs.
- Comprehensive test coverage with unit and CLI mocks.

---

## 3. Change Log and Release Impact

- **Change category**: Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**:
  - Add end-to-end highlight assembly to `dive-clip scan` via `--assemble` and introduce first-class `assemble` and `aftermovie` CLI subcommands.

---

## 4. Proposed Architecture & Data Flow

```text
[Raw Footage]
      │
      ▼
00_ingest_media ──────────► manifest.json
      │
      ▼
02_scan_footage ──────────► detections_refined.json
      │
      ▼
05_render_clips ──────────► <output>/highlights/*.mp4
      │
      ▼ (if --assemble or dive-clip assemble/aftermovie)
06_assemble_video / 07_create_aftermovie
      │
      ├───► <run_dir>/aftermovie.mp4
      ├───► <output>/aftermovie.mp4
      └───► Updated Rich Summary Panel with final video path
```

### Component Details
1. **`scripts/dive_clip.py`**:
   - `build_parser()`: Register `assemble` and `aftermovie` subparsers; add `--assemble`, `--music`, `--top-n`, and `--assemble-mode` to `scan`.
   - `cmd_scan()`: Extend pipeline loop with Step 4/4; track `assemble_duration_seconds` in `PipelineState` and `step_durations`; display final video path in the summary panel.
   - `cmd_assemble()` and `cmd_aftermovie()`: Resolve latest run directories when omitted and invoke assembly handlers.
2. **`utils/run_archive.py`**:
   - Add `aftermovie_path` parameter to `archive_run_artifacts()` so generated aftermovies are archived in per-run snapshots alongside `run_stats.json` and manifests.
3. **`docs/roadmap/simple_plan.md`**:
   - Document the completed CLI assembly workflow while preserving social 9:16 and TUI review as upcoming roadmap milestones.

---

## 5. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - In `tests/test_dive_clip.py`:
     - Test `dive-clip scan --assemble` adds assembly step and calls aftermovie creation.
     - Test `dive-clip scan --assemble --resume` skips assembly when already completed in state.
     - Test `dive-clip assemble` subcommand arguments, path resolution, and execution.
     - Test `dive-clip aftermovie` subcommand arguments, path resolution, and execution.
     - Test summary panel output includes aftermovie path when `--assemble` is used.
   - In `tests/test_run_archive.py`:
     - Verify `archive_run_artifacts` safely handles and archives `aftermovie_path`.
2. **Green (Implementation)**:
   - Implement CLI flags and subcommand handlers in `scripts/dive_clip.py`.
   - Support `aftermovie_path` in `archive_run_artifacts`.
   - Run focused tests until all pass.
3. **Refactor & Quality**:
   - Run full pytest test suite (`uv run python -m pytest`).
   - Check linting and formatting on changed files (`uv run ruff check` & `uv run ruff format --check`).

---

## 6. Verification & Rollout Checklist

- [x] Issue [#40](https://github.com/tiwu/dive-clip-tui/issues/40) updated with plan and awaiting approval
- [x] Tests pass: `uv run python -m pytest`
- [x] Changed Python files pass Ruff lint and formatting checks
- [x] `CHANGELOG.md` updated under `[Unreleased]`
- [x] PR created with `Closes #40` and SemVer label `semver:minor`
