# Architecture Plan: [Issue #75] Pipeline Stage Reordering: Prescreen Early Exit before Pass 0 and Pass 1

- **Issue**: [#75](https://github.com/tiwu/dive-clip-tui/issues/75)
- **Status**: In Progress
- **Author**: Autonomous Agent / Contributor
- **Date**: 2026-11-05
- **Target Branch**: `feature/issue-75-reorder-pipeline-prescreen`

---

## 1. Problem Statement & Motivation

In `scripts/02_scan_footage.py`, video clips are currently processed in an inverted order:
1. **Pass 0**: Domain classification samples frames and queries Ollama VLM.
2. **Pass 1 Setup**: Coarse frame extraction begins, logging `[Pass 1] Coarse scan...` and extracting frames.
3. **Prescreen**: Runs telemetry decoding and programmatic gates on candidate windows.
4. **Pass 1 Inference**: Evaluates frames against VLM and logs `[Pass 1] Summary...`.
5. **Pass 2**: Fine scan.

This ordering introduces multiple issues:
- **Wasted VLM calls & frame extractions**: A clip that is completely unusable (e.g. pitch-black, inside a bag, camera tumbling, extreme blur, or upside-down) still undergoes Pass 0 VLM classification and coarse frame extraction before Prescreen ever runs.
- **Confusing log order**: Prescreen log lines appear in the middle of Pass 1, between coarse frame extraction and coarse VLM summary.
- **Contact sheet leakage**: When `prescreen.contact_sheet: true`, frames are tiled into contact sheets before individual window prescreen rejections are evaluated in the frame loop.

By reordering the pipeline into a true **cost funnel**, cheapest/fastest zero-VLM gates run first, eliminating unnecessary computation and VLM inference before deeper passes run.

---

## 2. Requirements & Acceptance Criteria

- [ ] **Stage 1 (Prescreen) Runs First**: `_prescreen_clip()` executes at the start of `process_clip()`, before Pass 0 frame extraction and before Pass 1 coarse frame extraction.
- [ ] **Whole-Clip Early Exit on Rejection**: When `prescreen.enabled: true` and `prescreen.gate_mode == "enabled"`, if all candidate windows in a clip are rejected (`action == "reject"`), the clip early-exits immediately without extracting Pass 0 frames, invoking Pass 0 VLM, or extracting coarse frames.
- [ ] **Auditable Trace & Checkpoint on Early Exit**: When a clip is dropped by Prescreen, `prescreen_trace.json` records the decision, `pass0_collector` logs disposition `skipped_prescreen`, and `ClipScanCheckpoint` marks `pass_stage = "completed"`.
- [ ] **Contact Sheet Filter Alignment**: If contact sheets are enabled in Pass 1, frames falling inside rejected prescreen intervals are excluded prior to contact sheet mosaic creation.
- [ ] **Clean and Predictable Log Hierarchy**: Logs follow a clean sequence:
  `[Prescreen] ...` -> `[Pass 0] ...` -> `[Pass 1] ...` -> `[Pass 2] ...`.
- [ ] **Backward Compatibility**: Existing configs (`prescreen.enabled: false`, `gate_mode: "advisory"`) retain unchanged behavior.
- [ ] Automated tests cover early rejection, trace persistence, log order, and contact sheet filtering.

---

## 3. Architecture & Data Flow

```text
                  Incoming Video Clip
                          │
                          ▼
             ┌─────────────────────────┐
             │ Stage 1: Pre-screen     │  (Zero VLM calls)
             │ - Container metadata    │  (Fast FFmpeg 64x36 rgb24)
             │ - GPMF/IMU Telemetry    │  (Fast audio astats)
             │ - Blur/Luma/Orientation │
             └────────────┬────────────┘
                          │
       All windows        │ Surviving or deferred
       rejected?          │ windows
     ┌────────────────────┴────────────────────┐
     │ YES (gate_mode: enabled)                │ NO / advisory
     ▼                                         ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│ Early Exit:             │       │ Stage 2: Pass 0 Domain  │  (1 VLM call)
│ - Log rejection         │       │ - Sample 3 frames       │  (Only if domain
│ - Flush prescreen trace │       │ - Classify domain       │   is not fixed)
│ - Save checkpoint       │       └────────────┬────────────┘
│ - Return []             │                    │ Usable domain?
└─────────────────────────┘          ┌─────────┴─────────┐
                                     │ NO                │ YES
                                     ▼                   ▼
                          ┌─────────────────────┐   ┌─────────────────────────┐
                          │ Early Exit:         │   │ Stage 3: Pass 1 Coarse  │
                          │ Non-target/unusable │   │ - Extract coarse frames │
                          │ Return []           │   │ - Filter rejected spans │
                          └─────────────────────┘   │ - Coarse VLM / sheets   │
                                                    └────────────┬────────────┘
                                                                 │ Candidate hits?
                                                     ┌───────────┴───────────┐
                                                     │ NO                    │ YES
                                                     ▼                       ▼
                                          ┌─────────────────────┐   ┌─────────────────────────┐
                                          │ Finish:             │   │ Stage 4: Pass 2 Fine    │
                                          │ Return detections   │   │ - Window fine scan      │
                                          └─────────────────────┘   │ - Quality score & MMR   │
                                                                    └─────────────────────────┘
```

---

## 4. Change Log and Release Impact

- **Change category**: Changed
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**: Reorder footage scanning pipeline to run pre-VLM screening and whole-clip gate rejection before Pass 0 domain classification and Pass 1 coarse frame extraction.

---

## 5. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - In `tests/test_scan_footage.py`, add `test_process_clip_prescreen_early_exit_skips_pass0_and_pass1`:
     Verify that when all windows are rejected with `gate_mode="enabled"`, neither Pass 0 frame extraction/classification nor Pass 1 coarse frame extraction/inference are called.
   - Add `test_process_clip_prescreen_runs_before_pass0`:
     Verify execution order through mock call inspection (`_prescreen_clip` called before `extract_classification_frames` / `classify_clip_domain`).
   - Add `test_process_clip_contact_sheets_skips_prescreen_rejected_frames`:
     Verify that frames inside rejected intervals are not included in `compose_frame_mosaic`.
2. **Green (Implementation)**:
   - Move `prescreen_config` initialization and `_prescreen_clip()` call to the top of `process_clip()`.
   - Add whole-clip rejection check: if `prescreen_config.enabled` and `prescreen_config.gate_mode == "enabled"` and `len(prescreen_result.records) > 0 and len(prescreen_result.rejected_intervals) == len(prescreen_result.records)`:
     - Log warning and print console notification.
     - Record in `pass0_collector` with `disposition="skipped_prescreen"`, `usable=False`.
     - Update checkpoint `pass_stage = "completed"`, `detected_domain = "unusable"`.
     - Flush trace and return `[]`.
   - Align Pass 1 coarse scan: log `[Pass 1] Coarse scan...` only when entering Pass 1.
   - Filter `coarse_frames` passed to `_analyze_contact_sheets` to exclude timestamps covered by `prescreen_result.rejected_intervals`.
3. **Refactor & Quality**:
   - Run `uv run python -m pytest tests/test_scan_footage.py tests/test_prescreen.py`.
   - Run full pytest test suite.
   - Run Ruff checks: `uv run ruff check .` and `uv run ruff format --check .`.

---

## 6. Verification & Rollout Checklist

- [ ] Tests pass: `uv run python -m pytest`
- [ ] Changed Python files pass Ruff lint and formatting checks
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Issue #75 updated with plan and labeled `agent:plan-review`
- [ ] PR created with `Closes #75` once plan is approved
