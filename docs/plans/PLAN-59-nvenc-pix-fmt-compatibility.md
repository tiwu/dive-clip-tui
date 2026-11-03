# Architecture Plan: [Issue #59] Ensure 8-bit Pixel Format Compatibility for NVENC and Track pix_fmt in Manifest

- **Issue**: [#59](https://github.com/tiwu/dive-clip-tui/issues/59)
- **Status**: Completed
- **Author**: Autonomous Agent / Contributor
- **Date**: 2026-11-03
- **Target Branch**: `feature/issue-59-nvenc-pix-fmt-compatibility`

---

## 1. Problem Statement & Motivation

When users process action camera footage (such as 10-bit HEVC `yuv420p10le` from GoPro Hero 11/12/13), `probe_nvenc()` detects NVIDIA NVENC availability using a synthetic 8-bit test source, but subsequent highlight clip rendering with `h264_nvenc` fails with:
```text
[h264_nvenc] 10 bit encode not supported
[h264_nvenc] No capable devices found
```
Because `-pix_fmt yuv420p` is omitted from the encoder arguments in `utils/encoder.py`, FFmpeg attempts to pass raw 10-bit frames directly to `h264_nvenc`, which only supports 8-bit inputs.

This triggers an automatic fallback to CPU software encoding (`libx264`). Furthermore, background render tasks suppress console warnings via `ExcludeNoConsoleFilter`, so users see that NVENC was detected as available, but observe high CPU load (near 100%) in `btop` due to silent CPU encoding.

Additionally, `scripts/00_ingest_media.py` extracts resolution, FPS, and codec from source files using `ffprobe`, but omits the pixel format (`pix_fmt`), leaving the pipeline without visibility into whether source clips are 8-bit or 10-bit.

---

## 2. Requirements & Acceptance Criteria

- [x] `utils/encoder.py`: `get_video_encoder()` and `get_fallback_encoder()` include `"-pix_fmt", "yuv420p"` for `h264_nvenc` and `libx264` argument lists to ensure consistent 8-bit YUV 4:2:0 output across all source formats.
- [x] `utils/models.py`: Update `ClipManifest` to include `pix_fmt: str = ""` (optional/default empty), maintaining backward compatibility with existing manifests.
- [x] `scripts/00_ingest_media.py`: Include `stream=pix_fmt` in `ffprobe` query and map it into the generated manifest metadata.
- [x] Automated tests cover:
  - `tests/test_encoder.py`: Assert that `-pix_fmt yuv420p` is present in generated arguments for `h264_nvenc`, `libx264`, and fallback configs.
  - `tests/test_ingest_media.py`: Assert that `pix_fmt` is properly parsed from `ffprobe` JSON output and populated in `manifest.json`.
  - All existing tests in `tests/test_encoder.py`, `tests/test_render_clips.py`, and `tests/test_aftermovie.py` pass.
- [x] Documentation and comments updated to reflect pixel format handling and requirements.

---

## 3. Change Log and Release Impact

- **Change category**: Fixed / Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**:
  ```markdown
  ### Fixed
  - Fixed NVENC hardware encoding failure on 10-bit source footage (`yuv420p10le`) by explicitly setting `-pix_fmt yuv420p` in encoder arguments, preventing fallback to CPU `libx264`.

  ### Added
  - Probed and recorded source video pixel format (`pix_fmt`) in `manifest.json` during media ingestion.
  ```

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Update `tests/test_encoder.py` to test that `get_video_encoder()` and `get_fallback_encoder()` return `"-pix_fmt", "yuv420p"`.
   - Update `tests/test_ingest.py` to verify `pix_fmt` extraction from ffprobe data.
   - Run focused pytest to confirm failure before code edits.
2. **Green (Implementation)**:
   - Update `utils/encoder.py` to add `"-pix_fmt", "yuv420p"` to encoder argument lists.
   - Update `utils/models.py` (`VideoMetadata`) to add `pix_fmt: str = "unknown"`.
   - Update `scripts/00_ingest_media.py` to request `stream=pix_fmt` from `ffprobe` and set `pix_fmt`.
   - Run tests until green.
3. **Refactor & Quality**:
   - Run `uv run python -m pytest` across the entire test suite.
   - Run `uv run ruff check .` and `uv run ruff format --check .` on changed files.

---

## 5. Verification & Rollout Checklist

- [ ] Tests pass: `uv run python -m pytest`
- [ ] Changed Python files pass Ruff lint and formatting checks
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR labeled with `semver:minor`
- [ ] PR submitted with `Closes #59` and plan linked
