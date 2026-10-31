# Architecture Plan: [Issue #17] Refine VLM prompts and scoring to reject upside-down/dangling clips and evaluate lighting, focus, and stability

- **Issue**: [#17](https://github.com/tiwu/dive-clip-tui/issues/17)
- **Status**: In Progress
- **Author**: Autonomous Agent / Contributor
- **Date**: 2026-10-31
- **Target Branch**: `feature/issue-17-vlm-prompt-quality`

---

## 1. Problem Statement & Motivation

Underwater footage screening currently suffers from two main quality issues:
1. **Pass 1 False Positives on Inverted/Dangling Footage**: The coarse pass uses generic detection prompts that do not check camera orientation or intentionality. Footage filmed while the camera is dangling, tumbling, spinning, or upside down is marked as `detected: true`, and Pass 1 checks only confidence, allowing unusable footage to trigger fine scans and enter the pipeline.
2. **Pass 2 Uncalibrated Quality Scoring**: The fine pass uses the identical prompt without specific guidance on lighting (murky/backscatter vs well-lit), focus clarity (crisp vs blurry), or camera stability (smooth vs shaky/motion-blurred/stuttering), leading to noisy EDL selections.

---

## 2. Requirements & Acceptance Criteria

- [ ] `utils/prompt_templates.py`: Update `PromptTemplate` dataclass to include `coarse_prompt` and `fine_prompt`. Maintain `user_prompt` as a backward-compatible alias.
- [ ] `utils/prompt_templates.py`: Update all prompt templates (`PELAGICS_TEMPLATE`, `MACRO_TEMPLATE`, `DIVER_TEMPLATE`, `BROAD_MARINE_TEMPLATE`):
  - `coarse_prompt`: Explicitly instruct VLM to check camera orientation and intentional framing; mark `detected: false` and `quality_score: 1` if footage is upside-down, tumbling, dangling, or accidental.
  - `fine_prompt`: Provide a calibrated 1-10 quality rubric assessing lighting, focus sharpness, camera stability, and composition.
- [ ] `config.yaml`: Add `coarse_quality_threshold: 4` under `scan_parameters`.
- [ ] `scripts/02_scan_footage.py`:
  - Pass 1 uses `template.coarse_prompt` (or fallback).
  - Pass 1 filters candidates using `detected and confidence >= coarse_threshold and quality_score >= coarse_quality_threshold`.
  - Pass 2 uses `template.fine_prompt` (or fallback).
- [ ] `tests/test_prompt_templates.py` and `tests/test_scan_footage.py`: Updated tests covering coarse/fine prompt usage, orientation rejection, and coarse quality threshold filtering.
- [ ] `CHANGELOG.md`: Entry added under `[Unreleased]` with `semver:minor` impact.

---

## 3. Change Log and Release Impact

- **Change category**: Added / Changed
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**: Add coarse and fine VLM prompt separation, rejecting upside-down or dangling camera footage in Pass 1 and calibrating lighting, focus, and stability scoring in Pass 2.

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Update `tests/test_prompt_templates.py` to assert `coarse_prompt` and `fine_prompt` existence, criteria keywords (upside down, dangling, lighting, focus, stability), and `user_prompt` backward compatibility.
   - Update `tests/test_scan_footage.py` to assert coarse pass uses `coarse_prompt` and checks `coarse_quality_threshold`, while fine pass uses `fine_prompt`.
2. **Green (Implementation)**:
   - Implement `coarse_prompt` and `fine_prompt` in `utils/prompt_templates.py`.
   - Update `scripts/02_scan_footage.py` to use appropriate prompts and enforce `coarse_quality_threshold`.
   - Update `config.yaml` with `coarse_quality_threshold: 4`.
3. **Refactor & Verification**:
   - Run pytest and ruff checks.
