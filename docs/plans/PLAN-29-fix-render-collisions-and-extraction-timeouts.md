# Architecture Plan: [Issue #29] Fix highlight clip collisions, frame extraction timeouts, and detection fragmentation in long runs

- **Issue**: [#29](https://github.com/tiwu/dive-clip-tui/issues/29)
- **Status**: In Progress
- **Author**: Autonomous Agent / Contributor
- **Date**: 2026-11-01
- **Target Branch**: `feature/issue-29-fix-render-collisions-and-extraction-timeouts`

---

## 1. Problem Statement & Motivation

During the first large-batch production run (`run-20261031-230736`, comprising 111 4K 50fps HEVC clips and 3h 13m of underwater footage from Nosy Be), analysis revealed four critical pipeline issues:

1. **Highlight Filename Collision in `05_render_clips.py`**:
   The output filename template is `f"{clip_id}_{subject}_q{quality}.mp4"`. Because it lacks time bounds, multiple detections of the same subject in a video (e.g., 58 turtle detections in `GX010824.MP4`) map to the exact same file. The first detection renders; all subsequent detections are skipped as "already rendered". Out of 900 detections, 739 were skipped and only 161 files were written.
2. **Fixed 300s FFmpeg Extraction Timeout in `utils/ffmpeg_streamer.py`**:
   Extracting frames from long 4K HEVC clips (>4.5 minutes) on CPU exceeded the hardcoded 300-second subprocess timeout. Four long clips totaling 39.3 minutes (`GX010874.MP4`, `GX020874.MP4`, `GX010832.MP4`, `GX010825.MP4`) timed out and returned 0 frames, wasting 20 minutes of CPU time and missing all footage.
3. **Short Clip Frame Starvation (< 1/fps)**:
   Clips shorter than the sampling interval (e.g. `GX010886.MP4` of 0.96s with coarse fps of 0.5) yielded 0 frames because FFmpeg fps filter skips fractional initial windows.
4. **Fine Detection Window Fragmentation**:
   Fine detections are isolated 2-second windows around positive frames (`[timestamp - 1, timestamp + 1]`). A continuous 30-second encounter with marine life is fragmented into disjoint 2-second slices rather than merged into a continuous highlight sequence.
5. **Symlink Permission Errors on exFAT External Drives**:
   Creating the `latest` symlink in `utils/run_archive.py` fails on exFAT external filesystems with `[Errno 1] Operation not permitted`.

---

## 2. Requirements & Acceptance Criteria

- [x] **Unique Highlight Filenames**:
  - `05_render_clips.py` filename template includes start and end timestamps: `{clip_id}_{start_time:.1f}s-{end_time:.1f}s_{subject}_q{quality}.mp4`.
  - Also ensure backward compatibility or migration if downstream tools look for rendered highlights.
- [x] **Dynamic / Scalable FFmpeg Timeout**:
  - `utils/ffmpeg_streamer.py` calculates timeout based on video duration: `timeout = max(300, int(duration * 2.0))` (or configurable), ensuring long 4K files have sufficient decoding budget.
  - Failures and timeouts logged with `logger.warning` / `logger.error` instead of unformatted `print()`.
- [x] **Guaranteed Frame Sampling for Short Clips**:
  - In `utils/ffmpeg_streamer.py`, if a clip has valid duration > 0 and the `fps` filter yields 0 frames, fall back to extracting a single frame at timestamp 0.0.
- [x] **Contiguous Detection Merging**:
  - In `scripts/02_scan_footage.py`, merge adjacent/overlapping detections of the same subject (within a small tolerance/padding window) into a single cohesive highlight window preserving the highest quality score and combined duration.
- [x] **Graceful exFAT Symlink Handling**:
  - In `utils/run_archive.py`, catch `OSError` (such as `EPERM` / `EOPNOTSUPP`) when creating the `latest` symlink on filesystems that do not support symlinks, write a `latest.txt` fallback file, and log at `debug` instead of `warning`.
- [x] **Automated Tests**:
  - Unit and integration tests covering:
    - Highlight rendering with multiple detections of identical subject and quality to unique files.
    - Dynamic timeout computation and short clip fallback extraction.
    - Detection deduplication and contiguous window merging.
    - Symlink creation fallback on non-symlink environments.

---

## 3. Change Log and Release Impact

- **Change category**: Fixed
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**:
  - Fixed highlight clip collisions by including timestamp ranges in output filenames.
  - Fixed FFmpeg extraction timeouts on long 4K video files by dynamically scaling extraction timeouts to video duration.
  - Added guaranteed single-frame extraction fallback for clips shorter than the sampling interval.
  - Merged consecutive detection windows of the same subject into continuous highlight clips.
  - Added graceful fallback for `latest` symlink creation on exFAT and non-symlink filesystems.

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Add tests in `tests/test_render_clips.py` verifying that detections with identical `clip_id`, `subject`, and `quality` but different timestamps generate distinct files and both render successfully without colliding.
   - Add tests in `tests/test_ffmpeg_streamer.py` verifying timeout calculation and short-video fallback frame extraction.
   - Add tests in `tests/test_scan_footage.py` verifying merging of contiguous detections of the same subject into continuous windows.
   - Add tests in `tests/test_run_archive.py` verifying graceful fallback when symlink creation raises `OSError(EPERM)`.
2. **Green (Implementation)**:
   - Update `scripts/05_render_clips.py` output naming logic.
   - Update `utils/ffmpeg_streamer.py` timeout and short clip handling.
   - Update `scripts/02_scan_footage.py` detection merging logic.
   - Update `utils/run_archive.py` symlink fallback logic.
3. **Refactor & Quality**:
   - Run focused pytest suites: `uv run python -m pytest tests/test_render_clips.py tests/test_ffmpeg_streamer.py tests/test_scan_footage.py tests/test_run_archive.py`.
   - Run full test suite: `uv run python -m pytest`.
   - Check linting and formatting: `uv run ruff check .` and `uv run ruff format --check .`.

---

## 5. Verification & Rollout Checklist

- [x] Tests pass: `uv run python -m pytest`
- [x] Changed Python files pass Ruff lint and formatting checks
- [x] `CHANGELOG.md` updated under `[Unreleased]`
- [x] Issue #29 labeled `agent:approved` before implementation
- [ ] PR created with `Closes #29` and plan linked
