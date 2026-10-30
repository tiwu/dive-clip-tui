# Architecture Plan: [Issue #15] Optimize scan performance with frame downscaling, streaming persistence, and concurrent clip rendering

- **Issue**: [#15](https://github.com/tiwu/dive-clip-tui/issues/15)
- **Status**: In Progress
- **Author**: Autonomous Agent / Contributor
- **Date**: 2026-10-30
- **Target Branch**: `feature/issue-15-perf-scaling-pipeline`

---

## 1. Problem Statement & Motivation

Processing large batches of video footage (e.g. 127+ videos / ~180GB) presents three primary bottlenecks in the current architecture:
1. **VLM Inference Latency**: Coarse scanning extracts native resolution frames (often 1080p or 4K), generating maximum vision tokens in Qwen2.5-VL and high base64 encoding/HTTP transport overhead for simple presence detection.
2. **Delayed Persistence**: `02_scan_footage.py` only writes `detections_refined.json` at the very end after all clips complete. If an interruption occurs mid-batch, prior scanning work is lost from the detections file.
3. **Sequential Pipeline Lockstep**: `dive-clip scan` runs ingest -> scan all clips -> render all clips sequentially. Highlight rendering (CPU/encoder bound) sits idle during hours of VLM scanning instead of rendering highlight clips as soon as each video's fine scan finishes.

---

## 2. Requirements & Acceptance Criteria

- [x] `config.yaml`: Support configurable `coarse_scale` (default `"640:-2"`) and `fine_scale` (default `null` for native) under `scan_parameters`.
- [x] `scripts/02_scan_footage.py`: Pass `coarse_scale` and `fine_scale` to `extract_frames()`.
- [x] `scripts/02_scan_footage.py`: Persist detections atomically to `detections_refined.json` after each clip completes.
- [x] `scripts/02_scan_footage.py`: Support resuming already-scanned clips by pre-loading existing detections from `detections_refined.json` and skipping completed clips.
- [x] `scripts/05_render_clips.py`: Provide a modular `render_detections()` function that renders highlight clips for a list of detections without re-rendering existing clips.
- [x] `scripts/dive_clip.py`: Stream completed detections to a background render worker so highlight clips are rendered concurrently while subsequent clips are being scanned.
- [x] Automated tests cover downscaling configuration, atomic per-clip saving, resume detection preservation, and concurrent/incremental clip rendering.
- [x] CHANGELOG.md updated with user-facing entry under `[Unreleased]`.

---

## 3. Change Log and Release Impact

- **Change category**: Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**: Add frame downscaling for faster coarse scanning, atomic per-clip detection persistence with resume, and concurrent highlight clip rendering during scan batches.

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Add tests in `tests/test_scan_footage.py` verifying `coarse_scale` and `fine_scale` are passed to `extract_frames`.
   - Add tests in `tests/test_scan_footage.py` verifying `detections_refined.json` is updated incrementally after each clip.
   - Add tests in `tests/test_scan_footage.py` verifying `--resume` skips already-completed clips and preserves prior detections.
   - Add tests in `tests/test_render_clips.py` verifying incremental rendering skips already-rendered files.
   - Add tests in `tests/test_dive_clip.py` verifying concurrent/pipelined scan-and-render execution.
2. **Green (Implementation)**:
   - Update `config.yaml` with `coarse_scale` and `fine_scale`.
   - Update `scripts/02_scan_footage.py` to pass scale parameters, atomically save per clip, and load existing detections on resume.
   - Refactor `scripts/05_render_clips.py` to expose `render_detections()` and avoid duplicate renders.
   - Update `scripts/dive_clip.py` to coordinate pipelined rendering of completed clips.
3. **Refactor & Quality**:
   - Run focused tests: `uv run python -m pytest tests/test_scan_footage.py tests/test_render_clips.py tests/test_dive_clip.py`
   - Run full test suite: `uv run python -m pytest`
   - Check lint and formatting: `uv run ruff check .` and `uv run ruff format --check .`

---

## 5. Verification & Rollout Checklist

- [x] Tests pass: `uv run python -m pytest`
- [x] Changed Python files pass Ruff lint and formatting checks
- [x] `CHANGELOG.md` updated under `[Unreleased]`
- [x] PR has `semver:minor` label
- [ ] PR submitted with `Closes #15` and plan linked
