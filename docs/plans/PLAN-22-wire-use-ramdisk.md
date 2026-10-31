# Architecture Plan: Issue #22 - Wire up use_ramdisk configuration setting and document benefits

- **Issue**: [#22](https://github.com/tiwu/dive-clip-tui/issues/22)
- **Status**: Completed
- **Author**: Autonomous Agent
- **Date**: 2026-10-31
- **Target Branch**: `feature/issue-22-wire-use-ramdisk`

---

## 1. Problem Statement & Motivation

The `use_ramdisk` setting exists under the `processing` section of `config.yaml` and related configuration presets, but it is currently dormant:
- `utils/ffmpeg_streamer.py:get_work_dir()` auto-detects `/dev/shm` without accepting or respecting a user-supplied flag.
- `scripts/02_scan_footage.py` invokes `get_work_dir()` without passing `config.get("processing", {}).get("use_ramdisk")`.

When processing long video footage, frame extraction creates thousands of short-lived JPEG frames during coarse and fine scans. Using a RAM disk (`/dev/shm` on Linux) eliminates disk I/O bottlenecks and avoids write cycles on SSD drives. However, on low-memory environments, containers with limited `/dev/shm`, or when users prefer standard disk storage, users should be able to disable RAM-disk usage (`use_ramdisk: false`). The codebase must wire this setting end-to-end and document its purpose and benefits clearly.

---

## 2. Requirements & Acceptance Criteria

- [x] `utils/ffmpeg_streamer.py:get_work_dir(use_ramdisk: bool = True) -> Path` accepts a boolean flag.
  - When `use_ramdisk=True` and `/dev/shm` exists and is writable, it creates a private temporary directory under `/dev/shm`.
  - When `use_ramdisk=False` or `/dev/shm` is unavailable/unwritable, it falls back to standard temporary storage (`tempfile.mkdtemp()`).
- [x] `scripts/02_scan_footage.py` retrieves `use_ramdisk` from `config.get("scan_parameters", {}).get("use_ramdisk", ...)` and passes it to `get_work_dir()`.
- [x] Automated tests cover both `use_ramdisk=True` and `use_ramdisk=False` behaviors in `tests/test_ffmpeg_streamer.py` and verify `02_scan_footage.py` honors the config in `tests/test_scan_footage.py`.
- [x] Documentation (`README.md` and `config.yaml`) clearly describes the RAM-disk feature, Linux `/dev/shm` usage, OS fallback behavior, and performance/SSD durability benefits.

---

## 3. Change Log and Release Impact

- **Change category**: Changed
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**:
  - Wire up `scan_parameters.use_ramdisk` configuration setting in `scripts/02_scan_footage.py` and `utils/ffmpeg_streamer.py`, allowing users to explicitly control RAM-disk usage (`/dev/shm`) and documenting its I/O performance and SSD wear reduction benefits.

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Add unit tests in `tests/test_ffmpeg_streamer.py`:
     - Test `get_work_dir(use_ramdisk=True)` uses `/dev/shm` if available.
     - Test `get_work_dir(use_ramdisk=False)` uses standard temp directory even if `/dev/shm` is available.
   - Add test in `tests/test_scan_footage.py` verifying `scan_footage` passes `use_ramdisk` from configuration to `get_work_dir`.
2. **Green (Implementation)**:
   - Update `utils/ffmpeg_streamer.py:get_work_dir(use_ramdisk: bool = True)`.
   - Update `scripts/02_scan_footage.py` to extract `scan_parameters.use_ramdisk` and call `get_work_dir(use_ramdisk=use_ramdisk)`.
   - Update `config.yaml` comments and `README.md` to document the feature and benefits.
3. **Refactor & Quality**:
   - Run tests: `uv run python -m pytest tests/test_ffmpeg_streamer.py tests/test_scan_footage.py`
   - Run full test suite: `uv run python -m pytest`
   - Run Ruff: `uv run ruff check .` and `uv run ruff format --check .`

---

## 5. Verification & Rollout Checklist

- [x] Tests pass: `uv run python -m pytest`
- [x] Changed Python files pass Ruff lint and formatting checks
- [x] `CHANGELOG.md` updated under `[Unreleased]`
- [x] PR labeled with `semver:minor`
- [x] PR submitted with `Closes #22` and plan linked
