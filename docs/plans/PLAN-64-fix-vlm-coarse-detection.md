# Architecture Plan: [Issue #64] Fix VLM Coarse Detection Regressions & Usability Rejections

- **Issue**: [#64](https://github.com/tiwu/dive-clip-tui/issues/64)
- **Status**: In Progress
- **Author**: Autonomous Agent / Contributor
- **Date**: 2026-11-03
- **Target Branch**: `feature/issue-64-fix-vlm-coarse-detection`

---

## 1. Problem Statement & Motivation

Following release v0.27.0, pipeline runs on real dive footage and motorcycle footage produced 0 candidate windows and 0 rendered clips:
1. `qwen2.5vl:3b` was biased by the newly added `reject_reasons` field and aggressive usability wording in coarse prompts, causing it to hallucinate camera defects (`"camera orientation is incorrect"`, `"accidental"`) on almost every frame.
2. Even when the VLM identified valid subjects (e.g. `"stonefish"`, `"scenic road vista"`) with high confidence (0.95), quality score >= 7, and empty reject reasons (`[]`), it returned `"detected": false`. The strict `detected` gate in `02_scan_footage.py` dropped all candidates.
3. In Pass 0, single-frame misidentifications (e.g. blue water misclassified as "lens covered") caused valid clips (such as `GX010830-turtle.MP4`) to be classified as `domain: "unusable"` and skipped entirely.
4. Vertical (9:16) footage in motorcycle clips was rejected because the pavement in the bottom half of the portrait frame was interpreted as pointing at the ground.

---

## 2. Requirements & Acceptance Criteria

- [x] In `utils/prompt_templates.py`:
  - `normalize_vlm_frame_response`: When `raw` contains a valid subject (not empty, "none", or "unknown"), `confidence >= 0.5`, `quality_score >= 4`, and `reject_reasons` is empty, infer `detected = True`.
  - Coarse prompts across all domains (`underwater`, `motorcycle`, `wildlife`, `generic`): soften the usability instructions to prevent 3B models from defaulting to hallucinated rejections; clarify that `reject_reasons` should only be returned if `detected` is false, and instruct defaulting `detected: true` when target subjects/actions are visible.
  - Motorcycle coarse prompt: handle portrait / vertical camera orientations without false "pointing at ground" rejections.
- [x] In `scripts/02_scan_footage.py`:
  - `aggregate_frame_classifications`: Loosen unusable threshold so a clip isn't discarded unless genuine persistent unusable state is detected (e.g., require `unusable_ratio > 0.66` AND mean quality score <= 2.0, or check if any valid domain classification was made with high confidence).
  - Incrementally flush stage trace files (`classifications.json`, `pass1_coarse_trace.json`, `pass2_fine_trace.json`) to disk as clips and scan stages finish processing instead of only at scan completion.
- [x] Automated tests cover `normalize_vlm_frame_response` defensive fallback, domain prompt formatting, aggregate frame classifications, and incremental trace flushes.
- [x] `uv run python -m pytest` passes.
- [x] Ruff lint and format checks pass.

---

## 3. Change Log and Release Impact

- **Change category**: Fixed / Added
- **Proposed SemVer impact**: `semver:patch`
- **Draft changelog entry**:
  - Fix VLM coarse detection regression where prompt schema biased small models toward hallucinated orientation rejections and dropped valid subject candidates.
  - Add defensive `detected` normalization when valid subjects are identified with high confidence and no rejection reasons.
  - Prevent premature Pass 0 unusable clip skipping on valid action camera footage.
  - Incrementally flush stage traceability files (`classifications.json`, `pass1_coarse_trace.json`, `pass2_fine_trace.json`) to disk as clips and scan stages finish processing instead of only at scan completion.

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Add unit tests in `tests/test_prompt_templates.py` and `tests/test_classify.py` validating defensive normalization and relaxed unusable aggregation.
2. **Green (Implementation)**:
   - Update `utils/prompt_templates.py` and `scripts/02_scan_footage.py`.
3. **Refactor & Quality**:
   - Run `uv run python -m pytest tests/test_prompt_templates.py tests/test_classify.py tests/test_scan_footage.py`.
   - Run full test suite: `uv run python -m pytest`.
   - Verify formatting and linting: `uv run ruff check .` and `uv run ruff format .`.

---

## 5. Verification & Rollout Checklist

- [ ] Tests pass: `uv run python -m pytest`
- [ ] Changed Python files pass Ruff lint and formatting checks
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR created linking #64
