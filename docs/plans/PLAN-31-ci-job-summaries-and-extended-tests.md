# Architecture Plan: [Issue #31] Add GitHub job summaries in CI pipeline and extend test coverage

- **Issue**: [#31](https://github.com/tiwu/dive-clip-tui/issues/31)
- **Status**: In Progress
- **Author**: Autonomous Agent / Contributor
- **Date**: 2026-11-01
- **Target Branch**: `feature/issue-31-ci-job-summaries-and-extended-tests`

---

## 1. Problem Statement & Motivation

1. **Lack of Visual CI Feedback**:
   Currently, `.github/workflows/ci.yml` runs test execution, Ruff checks, and PR validation as plain terminal output. Contributors and reviewers must dig into log folds to see test counts, pass/fail status, or code quality details. Adding GitHub Step Summaries (`$GITHUB_STEP_SUMMARY`) surfaces test results, code coverage, lint/format status, and SemVer release impact directly on the PR workflow summary page.
2. **Opportunities to Extend Test Coverage**:
   A coverage analysis across `scripts/`, `utils/`, and `tui/` revealed several completely untested or low-coverage areas:
   - `scripts/check_ollama_connection.py` (0% coverage): No tests verify connection polling, timeout handling, or model availability checks.
   - `scripts/entrypoints.py` (0% coverage): No tests verify that console scripts correctly invoke their underlying functions.
   - `scripts/04_plan_storyline.py` (29% coverage): Storyline prompt generation, API calling, and JSON response parsing lack unit tests.
   - `tui/screens/` (`browse.py`, `pipeline.py`, `review.py`): Screen mounting and event handlers have minimal test coverage.

---

## 2. Requirements & Acceptance Criteria

- [x] **GitHub Job Summaries in CI Workflow**:
  - Update `.github/workflows/ci.yml` to write Markdown summaries to `$GITHUB_STEP_SUMMARY`:
    - **Test & Coverage Summary**: Total tests passed/failed/skipped, execution time, and package coverage table (`scripts/`, `utils/`, `tui/`).
    - **Code Quality Summary**: Changed Python files inspected, Ruff lint status, Ruff format status.
    - **Release Impact Summary**: PR labels, resolved SemVer bump (`semver:minor`, etc.), and extracted changelog entry.
- [x] **Coverage Tooling in Dev Dependencies**:
  - Add `pytest-cov>=5.0.0` to `[dependency-groups] dev` in `pyproject.toml` and update `uv.lock`.
  - Configure coverage settings in `pyproject.toml` (`[tool.coverage.run]`, `[tool.coverage.report]`).
- [x] **Extend Test Coverage**:
  - Add `tests/test_check_ollama_connection.py` testing successful connection, connection failure, model missing, and CLI entry point.
  - Add `tests/test_entrypoints.py` testing each registered console script entry point.
  - Add `tests/test_plan_storyline_extended.py` (or extend `tests/test_plan_storyline.py`) covering Claude/Gemini prompt formatting, token limits, and fallback logic.
  - Add `tests/test_tui_screens.py` testing screen composition, widget loading, and directory browsing.
- [x] **Verification**:
  - All tests pass locally and in CI: `uv run python -m pytest --cov`.
  - Ruff lint and formatting pass: `uv run ruff check .` and `uv run ruff format --check .`.
  - CI workflow emits step summary without breaking or failing.

---

## 3. Change Log and Release Impact

- **Change category**: Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**:
  - Add GitHub Actions step summaries to CI pipeline reporting test results, code coverage, Ruff lint status, and changelog/SemVer validation.
  - Add `pytest-cov` dev dependency and extend unit tests covering entrypoints, Ollama connection checks, storyline planning, and TUI screens.

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Add new test modules for untested components:
     - `tests/test_check_ollama_connection.py`
     - `tests/test_entrypoints.py`
     - `tests/test_tui_screens.py`
     - Extended tests in `tests/test_plan_storyline.py`
   - Verify newly targeted code paths are exercised and asserted.
2. **Green (Implementation)**:
   - Add `pytest-cov` to `pyproject.toml` and update `uv.lock`.
   - Update `.github/workflows/ci.yml` to collect coverage, test counts, lint results, and changelog info into `$GITHUB_STEP_SUMMARY`.
3. **Refactor & Quality**:
   - Run `uv run python -m pytest --cov=scripts --cov=utils --cov=tui`.
   - Verify overall coverage increases significantly.
   - Run `uv run ruff check .` and `uv run ruff format --check .`.

---

## 5. Verification & Rollout Checklist

- [x] Tests pass: `uv run python -m pytest --cov`
- [x] Changed Python files pass Ruff lint and formatting checks
- [x] `CHANGELOG.md` updated under `[Unreleased]`
- [x] Issue #31 approved with `agent:approved`
- [ ] PR submitted with `Closes #31` and plan linked
