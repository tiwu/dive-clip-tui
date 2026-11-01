# Architecture Plan: [Issue #1] Use broad marine-life prompt for untagged clips

- **Issue**: [#1](https://github.com/tiwu/dive-clip-tui/issues/1)
- **Status**: Plan Review
- **Author**: Coding Agent
- **Date**: 2026-10-30
- **Target Branch**: `feature/issue-1-broad-marine-life-prompt`

---

## 1. Problem Statement & Motivation

Currently, when a clip has no matching content tags, the scan pipeline falls back to a prompt that may assume pelagic (open-water) marine life. This is too narrow and misses common benthic (reef/floor) and cryptic species. A broad marine-biology fallback prompt should consider the full spectrum of marine life — common and rare, pelagic and benthic — so untagged clips are scanned with broad coverage.

---

## 2. Requirements & Acceptance Criteria

- [ ] Untagged or unmatched clips use the broad marine-life prompt in both coarse and fine scan passes
- [ ] Existing matched macro, pelagics, and diver tags retain their current specialized behavior
- [ ] The broad prompt asks for structured detection, confidence, quality, description, and timestamp output
- [ ] The broad prompt includes representative examples: corals, worms, nudibranchs, fish, octopus, moray eels, sharks, turtles, whales, frogfish, crocodilefish, seahorses, and stonefish
- [ ] Focused prompt-template and scanner tests pass

---

## 3. Change Log and Release Impact

- **Change category**: Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**: Add broad marine-life fallback prompt for untagged clips, covering corals, worms, nudibranchs, fish, octopus, moray eels, sharks, turtles, whales, frogfish, crocodilefish, seahorses, and stonefish

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Add or update focused tests in `tests/test_prompt_templates.py` for the broad fallback prompt content and structure
   - Add or update focused tests in `tests/test_scan_footage.py` for fallback selection logic in both coarse and fine passes

2. **Green (Implementation)**:
   - Update `utils/prompt_templates.py`: define the broad fallback prompt with full marine-life coverage
   - Update `scripts/02_scan_footage.py`: use the broad template as fallback in both coarse and fine scan passes

3. **Refactor & Quality**:
   - Run the focused tests, then the full test suite (`uv run python -m pytest`)
   - Check lint and formatting on changed Python files with Ruff (`uv run ruff check .` and `uv run ruff format --check .`)

---

## 5. Verification & Rollout Checklist

- [ ] Tests pass: `uv run python -m pytest`
- [ ] Changed Python files pass Ruff lint and formatting checks
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR has no conflicting SemVer labels (unlabeled defaults to minor)
- [ ] PR submitted with `Closes #1` and plan linked