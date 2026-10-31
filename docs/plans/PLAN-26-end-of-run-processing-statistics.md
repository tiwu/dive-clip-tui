# Architecture Plan: [Issue #26] Add end-of-run processing statistics and summary report

- **Issue**: [#26](https://github.com/tiwu/dive-clip-tui/issues/26)
- **Status**: In Progress
- **Author**: Autonomous Agent / Contributor
- **Date**: 2026-10-31
- **Target Branch**: `feature/issue-26-end-of-run-processing-statistics`

---

## 1. Problem Statement & Motivation

When processing video batches, users lack visibility into execution totals upon completion: how many videos were processed, total source duration, how many candidate windows Phase 1 found and their duration, how many Phase 2 refined detections and highlights were produced and their duration, step-by-step elapsed time, and processing throughput (e.g. speed ratio vs realtime).

Providing a clean Rich summary panel in the terminal, logging it to `run.log`, and persisting structured data as `run_stats.json` gives immediate operational feedback and records verifiable performance history across runs.

---

## 2. Requirements & Acceptance Criteria

- [x] Core statistics model and calculations implemented in `utils/stats.py`:
  - Total videos processed and total video duration (from manifest).
  - Phase 1 coarse detection candidates count, candidate duration, frame counts, Ollama stats, elapsed time.
  - Phase 2 fine detection count, duration, frame counts, Ollama stats, elapsed time.
  - Highlights rendered count, skipped count, failed count, total highlight duration.
  - Overall pipeline elapsed time, per-step timing breakdown, and processing speed ratios (e.g. realtime ratio, fps).
- [x] Visual terminal presentation using Rich table/panel at the end of execution.
- [x] Structured JSON export saved to `<output>/runs/<run_id>/run_stats.json`.
- [x] Standalone `scripts/02_scan_footage.py` produces scanning statistics when invoked independently.
- [x] Main orchestrator `scripts/dive_clip.py` produces full end-to-end statistics across Ingest, Scan, and Render.
- [x] Resumed runs calculate cumulative statistics across all items.
- [x] `run_stats.json` included in run artifact archiving.
- [x] Automated unit and integration tests cover models, calculations, formatting, and CLI execution.

---

## 3. Change Log and Release Impact

- **Change category**: Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**:
  - Add end-of-run processing statistics, Rich summary table, and `run_stats.json` artifact for full pipeline and standalone scanning.

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Create `tests/test_stats.py` testing `Phase1Stats`, `Phase2Stats`, `RenderStats`, `PipelineStats`, formatting helpers, speed ratios, and JSON serialization.
   - Add test cases in `tests/test_scan_footage.py` verifying scan statistics calculation and emission.
   - Add test cases in `tests/test_dive_clip.py` verifying `run_stats.json` creation and summary table display.
2. **Green (Implementation)**:
   - Implement `utils/stats.py` with Pydantic models, helper functions, and Rich table formatting.
   - Update `scripts/02_scan_footage.py` to record candidate window durations, refined detection durations, and scan stats.
   - Update `scripts/dive_clip.py` to compute aggregate pipeline stats, display summary table, and write `run_stats.json`.
   - Update `utils/run_archive.py` to archive `run_stats.json`.
3. **Refactor & Quality**:
   - Run `uv run python -m pytest tests/test_stats.py tests/test_dive_clip.py tests/test_scan_footage.py`.
   - Run full test suite `uv run python -m pytest`.
   - Verify linting and formatting with `uv run ruff check .` and `uv run ruff format --check .`.

---

## 5. Verification & Rollout Checklist

- [x] Tests pass: `uv run python -m pytest`
- [x] Changed Python files pass Ruff lint and formatting checks
- [x] `CHANGELOG.md` updated under `[Unreleased]`
- [x] PR has no conflicting SemVer labels (`semver:minor`)
- [x] PR submitted with `Closes #26` and plan linked
