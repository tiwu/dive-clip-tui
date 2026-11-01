# Architecture Plan: [Issue #33] Resume long footage scans with intra-clip progress checkpointing

- **Issue**: [#33](https://github.com/tiwu/dive-clip-tui/issues/33)
- **Status**: Completed
- **Author**: Autonomous Agent / Contributor
- **Date**: 2026-11-01
- **Target Branch**: `feature/issue-33-intra-clip-scan-checkpointing`

---

## 1. Problem Statement & Motivation

Long footage files (such as 1–2+ hour action camera or dive videos) require extensive Vision Language Model (VLM) inference time during the scan phase (often 1–3+ hours per video file at 0.5 coarse FPS).

Currently, pipeline checkpointing (`PipelineState` and `detections_refined.json`) only records progress at the whole-clip boundary. While intermediate frames are extracted into temporary storage (`/dev/shm` or temporary directories), individual frame analysis results and candidate highlight windows remain in memory in `process_clip()`.

If a scan is interrupted (user pressing `Ctrl+C`, machine reboot, power interruption, or process termination) while a long video is in Pass 1 or Pass 2:
1. All frame evaluations completed up to that point are lost.
2. When the user resumes execution with `--resume`, the scanner treats the clip as completely unprocessed and starts over from frame 1 (0%).
3. Users lose hours of compute time and cannot cleanly pause or reboot during long single-file scans.

Adding persistent intra-clip checkpointing ensures progress is committed every few frames, allowing interrupted scans to resume exactly where they left off.

---

## 2. Requirements & Acceptance Criteria

- [x] **Persistent On-Disk Checkpoint Storage**:
  - Intra-clip checkpoints must be written to disk within the active run directory: `<run_dir>/checkpoints/<clip_id>.json`.
  - Checkpoints must live in persistent storage (e.g. output directory / drive), never in RAM-disk `/dev/shm`.
- [x] **Pass 1 (Coarse Scan) State Persistence**:
  - Checkpoint records analyzed coarse frames: timestamp, Ollama screening evaluation (`detected`, `confidence`, `quality_score`, `subject`), and whether it triggered a candidate window.
  - Checkpoint tracks accumulated candidate windows (`candidate_windows`) and Phase 1 stats.
- [x] **Pass 2 (Fine Scan) State Persistence**:
  - If Pass 1 is complete, candidate windows are saved to the checkpoint so Pass 1 does not re-run.
  - Checkpoint records analyzed fine frames, completed window indices, and accumulated `RefinedDetection` items.
- [x] **Periodic & Signal-Driven Flushes**:
  - Intermediate state is flushed to disk periodically every $N$ frames (default: 20 frames, configurable via `scan_parameters.checkpoint_interval_frames`).
  - Signal handling (`SIGINT` / `Ctrl+C`) triggers an immediate state flush in `process_clip()` before shutdown so zero completed frame evaluations are lost.
- [x] **Atomic Writes & Corruption Resistance**:
  - Checkpoint updates use atomic writes (writing to a temporary file in the same directory followed by atomic replacement) to prevent corruption if power or execution is abruptly cut.
  - If an invalid or corrupted checkpoint file is encountered, the scanner logs a warning and falls back gracefully to a clean scan for that clip without crashing.
- [x] **Resume Flow**:
  - When `--resume` is supplied:
    - If a valid checkpoint exists for an in-progress clip, coarse frame extraction verifies/extracts frames, existing frame evaluations are loaded, and Ollama inference skips all evaluated frames, resuming at frame $K+1$.
    - If Pass 1 was already completed in the checkpoint, coarse inference is skipped entirely and Pass 2 resumes from the last completed fine window.
- [x] **Completion & Cleanup**:
  - Once a clip successfully completes both passes and its detections are merged into `detections_refined.json`, the intra-clip checkpoint is marked completed or cleaned up.
- [x] **Automated Tests**:
  - Comprehensive unit and integration tests covering checkpoint creation, periodic flushes, interrupt handling, frame skipping on resume, Pass 2 resume, corrupted file handling, and cleanup upon completion.

---

## 3. Change Log and Release Impact

- **Change category**: Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**:
  - Add intra-clip progress checkpointing for long video scans, periodically saving Pass 1 and Pass 2 frame evaluations to disk so interrupted scans can resume without re-running completed frames.

---

## 4. Architecture & Implementation Design

### 4.1 Data Model: `ClipScanCheckpoint`

A new data model in `utils/checkpoint.py` (or `utils/models.py`):

```python
class CoarseFrameResult(BaseModel):
    timestamp: float
    detected: bool
    confidence: float
    quality_score: int
    subject: str = ""
    candidate_window: tuple[float, float] | None = None

class ClipScanCheckpoint(BaseModel):
    clip_id: str
    file_name: str
    pass_stage: Literal["coarse", "coarse_done", "fine", "completed"] = "coarse"
    coarse_fps: float
    coarse_frames_analyzed: int = 0
    coarse_results: dict[str, CoarseFrameResult] = Field(default_factory=dict)  # keyed by frame index or timestamp
    candidate_windows: list[tuple[float, float]] = Field(default_factory=list)
    fine_windows_completed: list[int] = Field(default_factory=list)
    refined_detections: list[RefinedDetection] = Field(default_factory=list)
    updated_at: str = Field(default_factory=_utc_now)
```

### 4.2 Storage Location

Inside `<run_dir>/checkpoints/<clip_id>.json`:
- Created at start of `process_clip()`.
- Flushed every `checkpoint_interval_frames` (default: 20) and on `KeyboardInterrupt`.
- Deleted or marked `completed` once `refined.detections` are updated and saved to `detections_refined.json`.

### 4.3 Execution Flow in `process_clip()`

1. **Initialization:**
   - Resolve `checkpoint_file = run_dir / "checkpoints" / f"{clip_id}.json"`.
   - If `resume=True` and `checkpoint_file.exists()`, load `ClipScanCheckpoint`.
2. **Pass 1 (Coarse Scan):**
   - If `checkpoint.pass_stage in ("coarse_done", "fine", "completed")`: skip Pass 1 VLM inference entirely, restoring `candidate_windows` from checkpoint.
   - Otherwise, extract coarse frames as usual (fast FFmpeg operation).
   - In frame loop:
     - If frame timestamp/index already in `checkpoint.coarse_results`: retrieve cached result, skip `call_ollama()`, update stats/progress bar.
     - Else: call Ollama, record result in checkpoint.
     - Every $N$ frames, call `checkpoint.save_atomic(checkpoint_file)`.
   - Mark `checkpoint.pass_stage = "coarse_done"`.
3. **Pass 2 (Fine Scan):**
   - For each window in `merged`:
     - If window index in `checkpoint.fine_windows_completed`: skip window inference.
     - Else: extract fine frames, analyze, record detections.
     - Save checkpoint after each window or every $N$ frames.
4. **Finalization:**
   - On clip completion, return detections to `scan_manifest()`.
   - Once saved to `detections_refined.json` and `state.add_progress("scan", clip_name)` is called, remove or archive `checkpoint_file`.

---

## 5. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Create `tests/test_intra_clip_checkpoint.py`:
     - Test `ClipScanCheckpoint` serialization, deserialization, and atomic file writes.
     - Test coarse pass resume: mock Ollama, run Pass 1 for 10 frames, interrupt, verify checkpoint saved. Run again with resume, verify only remaining frames invoke Ollama.
     - Test fine pass resume: verify completed coarse pass skips Pass 1 entirely on resume.
     - Test corrupt checkpoint fallback: corrupted JSON triggers warning and starts clip fresh without crashing.
     - Test cleanup: verify checkpoint file is removed when clip finishes normally.
2. **Green (Implementation)**:
   - Add `ClipScanCheckpoint` in `utils/checkpoint.py` with atomic write helper.
   - Update `scripts/02_scan_footage.py`:
     - Pass `run_dir` / `checkpoints` directory to `process_clip()`.
     - Wire checkpoint loading, frame skipping, periodic flushing, and signal handling into Pass 1 and Pass 2.
     - Clean up checkpoint after successful clip detection commit.
   - Add `checkpoint_interval_frames: 20` to `config.yaml` scan_parameters with sensible default fallback.
3. **Refactor & Quality**:
   - Run focused tests: `uv run pytest tests/test_intra_clip_checkpoint.py`
   - Run full test suite: `uv run pytest`
   - Check Ruff lint and formatting on changed files: `uv run ruff check` & `uv run ruff format --check`

---

## 6. Verification & Rollout Checklist

- [x] Tests pass: `uv run python -m pytest`
- [x] Changed Python files pass Ruff lint and formatting checks: `uv run ruff check .`
- [x] `CHANGELOG.md` updated under `[Unreleased]` with `Added` entry
- [ ] PR linked with `Closes #33`
- [ ] PR review and merge

