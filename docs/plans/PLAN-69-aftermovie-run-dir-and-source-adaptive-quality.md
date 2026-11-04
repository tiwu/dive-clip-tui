# Architecture Plan: [Issue #69] fix: aftermovie output placement in run_dir, source-adaptive video encoding quality, and clip duplication

- **Issue**: [#69](https://github.com/tiwu/dive-clip-tui/issues/69)
- **Status**: Plan Review
- **Author**: Antigravity
- **Date**: 2026-11-04
- **Target Branch**: `feature/issue-69-aftermovie-run-dir-and-source-adaptive-quality`

---

## 1. Problem Statement & Motivation

During review of run `run-20261104-024634`, three core issues and one architectural enhancement were identified:

1. **Duplicate Clips Rendered in Aftermovie (Filename Collision)**:
   - In `scripts/07_create_aftermovie.py`, when applying fade transitions, each clip is rendered to a temporary path named:
     ```python
     faded = tmp_dir / f"{clip_id}_faded.mp4"
     ```
   - In Dive Clip's data model, `clip_id` is the hash of the *source video file* (e.g. `clip_152836c2cf4cd5ff`), **not** a unique detection identifier. Multiple highlight windows from the same source file share the exact same `clip_id`.
   - As a result, subsequent detections from the same source file overwrite `tmp_dir / f"{clip_id}_faded.mp4"`.
   - Because `clip_files.append(faded)` appends the same path each time, `concat_clips()` concatenates that single overwritten clip multiple times in the final aftermovie.

2. **Artifact Location Inconsistency**:
   - In `scripts/dive_clip.py`, the assembled highlight reel or aftermovie is rendered to `output_folder / "aftermovie.mp4"` (the top-level output directory), rather than directly inside the active run directory `run_dir` (`<output_dir>/runs/run-<timestamp>/`).
   - Although `archive_run_artifacts()` afterwards duplicates the file into `run_dir`, placing the primary target in `output_folder` causes successive runs to overwrite `out/aftermovie.mp4` and wastes I/O performing multi-gigabyte redundant copies on external SSDs.
   - CLI commands `dive-clip assemble` and `dive-clip aftermovie` also default to `<output_dir>/aftermovie.mp4` instead of the latest run folder.

3. **Severe Bitrate Starvation & Video Degradation ("Grainy 4K")**:
   - The original camera footage (e.g. GoPro 4K 50fps HEVC 10-bit) records at ~120 Mbps.
   - In `utils/encoder.py`, the hardware encoder `h264_nvenc` was invoked with only `["-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-preset", "p4"]` without specifying rate control (`-rc`), constant quality (`-cq`), or target bitrate (`-b:v`).
   - FFmpeg's `h264_nvenc` defaults to 2,000 kbps (2.0 Mbps) when no bitrate or CQ is provided. This collapses 4K 50fps underwater video down to 0.005 bits/pixel (a ~50x bitrate reduction), resulting in extreme macroblocking, banding, and noise that destroys high-frequency details.
   - CPU encoding (`libx264`) similarly lacked an explicit `-crf` argument.
   - **Generation Loss**: `scripts/07_create_aftermovie.py` then takes the already-degraded 2.4 Mbps clips from `highlights/` and re-encodes them a second time through `h264_nvenc` to apply fade transitions, compounding compression artifacts.

4. **VLM Recognition Detail & Scale**:
   - Frame sampling downscaling (`coarse_scale: 426:-2` [240p], `fine_scale: 854:-2` [480p]) causes small or camouflaged macro subjects (seahorses, eels in crevices, nudibranchs) to be shrunk down to 10–25 pixels wide, blending into reef backgrounds and causing generic or hallucinated classifications.
   - Upgrading `fine_scale` to `1280:-2` (720p) gives the VLM sufficient pixel density to identify diagnostic subject anatomy while keeping inference fast.

---

## 2. Requirements & Acceptance Criteria

- [ ] **Fix Clip Duplication in Aftermovie**:
  - In `scripts/07_create_aftermovie.py`, unique temporary filenames must be used for each highlight window (incorporating detection index, timestamp range, or `detection_id`, e.g. `f"{clip_id}_{i:03d}_{start_time:.1f}s-{end_time:.1f}s_faded.mp4"`).
  - Every selected highlight clip in `selection_trace.json` must produce a distinct video file during assembly without overwriting peers.
- [ ] **Run Directory Placement**:
  - `cmd_scan` in `scripts/dive_clip.py` writes `aftermovie.mp4` (or `highlight_reel.mp4`) directly inside `run_dir`.
  - `cmd_aftermovie` and `cmd_assemble` default output to `<latest_run>/aftermovie.mp4` when an explicit `--output` is not specified.
  - `archive_run_artifacts` detects when `aftermovie_path` already resides in `run_dir` and avoids duplicate file copying.
  - A relative symlink (or fallback entry in `latest.txt` on filesystems without symlink support) at `<output_dir>/aftermovie.mp4` points to the latest run's aftermovie for user convenience.
- [ ] **Source-Adaptive High-Fidelity Rate Control**:
  - `utils/encoder.py` configures `h264_nvenc` with Constant Quality VBR: `-preset p5 -tune hq -rc:v vbr -cq:v 19 -b:v 0 -spatial-aq 1 -temporal-aq 1`.
  - `utils/encoder.py` configures `libx264` with Constant Rate Factor: `-crf 19 -preset veryfast`.
  - Quality thresholds are configurable via `config.yaml` under `scan_parameters.video_quality` (or `scan_parameters.encoder_cq` / `scan_parameters.encoder_crf`) with sensible defaults (`cq: 19`, `crf: 19`).
- [ ] **Single-Generation Direct Source Assembly in Aftermovie**:
  - In `scripts/07_create_aftermovie.py`, when applying fade transitions, the script cuts and fades directly from the original source file (`det["file_path"]`) in a single encode pass whenever the source file is accessible.
  - If the original source file is unavailable, it gracefully falls back to the pre-rendered clip in `clips_dir`.
- [ ] **Configurable VLM Fine Scale**:
  - Upgrade default `fine_scale` in `config.yaml` to `"1280:-2"` (720p) to provide sufficient detail for macro subjects, while keeping `coarse_scale` at `"426:-2"` for rapid initial screening.
- [ ] **Automated Tests**:
  - Unit tests verify distinct temporary filenames in `07_create_aftermovie.py` when multiple detections share the same `clip_id`.
  - Unit tests verify NVENC and libx264 encoder argument generation with quality parameters.
  - Unit tests verify output path resolution directs aftermovie and reel outputs into `run_dir`.

---

## 3. Change Log and Release Impact

- **Change category**: Fixed
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**:
  - `Fixed`: Prevent duplicate clips in aftermovie assembly by using unique per-detection temporary filenames during fade rendering.
  - `Fixed`: Ensure rendered aftermovie and highlight reel are written directly to `<run_dir>` rather than top-level output root.
  - `Fixed`: Add Constant Quality rate control (`-cq 19` for NVENC, `-crf 19` for libx264) and adaptive quantization to eliminate severe bitrate starvation and macroblocking in 4K renders.
  - `Changed`: Update aftermovie generation to cut and apply transitions directly from source footage in a single pass to eliminate multi-generation re-encoding loss.
  - `Changed`: Update default `fine_scale` to `1280:-2` (720p) for improved VLM recognition of macro marine life.

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Add a test in `tests/test_aftermovie.py` where multiple detections share the same `clip_id` to assert that distinct intermediate clips are rendered and concatenated without duplicates.
   - Update `tests/test_encoder.py` to assert that `get_video_encoder()` returns correct rate control flags (`-cq`, `-rc`, `-b:v 0` for NVENC, `-crf` for libx264).
   - Update `tests/test_run_archive.py` and `tests/test_dive_clip.py` to assert that aftermovie assembly targets `run_dir` rather than `output_folder`.
2. **Green (Implementation)**:
   - Fix filename collision in `scripts/07_create_aftermovie.py` using unique detection identifiers.
   - Update `scripts/07_create_aftermovie.py` to render directly from source when available with source-adaptive quality settings.
   - Enhance `utils/encoder.py` with quality parameter support and sensible defaults.
   - Update `scripts/dive_clip.py` assembly commands to target `run_dir`.
   - Update `config.yaml` fine_scale to `1280:-2`.
3. **Refactor & Quality**:
   - Run focused pytest suites (`tests/test_aftermovie.py`, `tests/test_encoder.py`, `tests/test_run_archive.py`).
   - Run the full test suite (`uv run python -m pytest`).
   - Run Ruff lint and formatting checks on changed files.

---

## 5. Verification & Rollout Checklist

- [ ] Tests pass: `uv run python -m pytest`
- [ ] Changed Python files pass Ruff lint and formatting checks: `uv run ruff check .` and `uv run ruff format --check .`
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR created linking `Closes #69` with `semver:minor` label
