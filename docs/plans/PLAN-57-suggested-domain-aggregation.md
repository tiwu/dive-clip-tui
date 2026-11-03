# Architecture Plan: [Issue #57] Add suggested domain aggregation for generic clips and refine generic screening prompt

- **Issue**: [#57](https://github.com/tiwu/dive-clip-tui/issues/57)
- **Status**: In Progress
- **Author**: Antigravity
- **Date**: 2026-11-03
- **Target Branch**: `feature/issue-57-suggested-domain-aggregation`

---

## 1. Problem Statement & Motivation

When classifying clips into domains, footage that falls outside the built-in target domains (`underwater`, `motorcycle`, `wildlife`) is classified as `generic`. The classification prompt asks the VLM to suggest a specific domain in `suggested_domain` (such as `mountain_biking`, `skiing`, or `cityscape`).

Previously:
1. `ClipClassification` lacked aggregation across sample frames for `suggested_domain`, improperly referencing the loop-scoped `result` variable in `classify_clip`. This caused crashes when responses were unusable/timeout and produced non-deterministic results when frames differed.
2. The generic coarse prompt and fine prompt were modernized with clearer usability criteria and JSON structures, but `test_universal_coarse_orientation_rejection` required update to recognize the refined rejection rules.

---

## 2. Requirements & Acceptance Criteria

- [x] Add `suggested_domain` field to `ClipClassification` in `utils/models.py`.
- [x] Implement robust cross-frame aggregation for `suggested_domain` in `scripts/02_scan_footage.py` (`aggregate_suggested_domain`) ranking candidates by frequency and confidence.
- [x] Prevent `AttributeError` or loop-leak crashes in `classify_clip` when Ollama calls fail or return no usable responses.
- [x] Update `test_universal_coarse_orientation_rejection` in `tests/test_prompt_templates.py` to match the refined prompt wording.
- [x] Add automated unit test for `suggested_domain` aggregation in `tests/test_classify.py`.
- [x] All 427 tests pass and Ruff checks pass on changed files.

---

## 3. Change Log and Release Impact

- **Change category**: Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**:
  - Add `suggested_domain` aggregation across sample frames for generic clip classifications and refine generic screening prompt criteria.

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Identify failure in `test_universal_coarse_orientation_rejection` and add `test_classify_clip_aggregates_suggested_domain`.
2. **Green (Implementation)**:
   - Implement `aggregate_suggested_domain` in `scripts/02_scan_footage.py` and assign in `ClipClassification`.
   - Update rejection rules assertion in `tests/test_prompt_templates.py`.
3. **Refactor & Quality**:
   - Run `uv run pytest` (427 passed).
   - Verify formatting and linting via `uv run ruff check` and `uv run ruff format`.

---

## 5. Verification & Rollout Checklist

- [x] Tests pass: `uv run python -m pytest`
- [x] Changed Python files pass Ruff lint and formatting checks
- [x] `CHANGELOG.md` updated under `[Unreleased]`
- [x] PR labeled `semver:minor`
- [x] PR submitted with `Closes #57` and plan linked
