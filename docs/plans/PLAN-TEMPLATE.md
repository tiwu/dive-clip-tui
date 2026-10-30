# Architecture Plan: [Issue #N] [Feature Title]

- **Issue**: [#N](https://github.com/tiwu/dive-clip-tui/issues/N)
- **Status**: Proposed | Plan Review | Approved | In Progress | Completed
- **Author**: Autonomous Agent / Contributor
- **Date**: YYYY-MM-DD
- **Target Branch**: `feature/issue-N-<slug>`

---

## 1. Problem Statement & Motivation

<!-- What problem does this solve? What user need does it address? -->

---

## 2. Requirements & Acceptance Criteria

<!-- Explicit functional and non-functional requirements -->
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Automated tests cover the acceptance criteria
- [ ] All relevant documentation has been updated

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Add or update focused tests in `tests/` for the acceptance criteria
2. **Green (Implementation)**:
   - Implement in the relevant existing package (`scripts/`, `utils/`, or `tui/`)
3. **Refactor & Quality**:
   - Run the focused tests, then the full test suite (`uv run pytest`)
   - Check lint and formatting (`uv run ruff check .` and `uv run ruff format --check .`)

---

## 5. Verification & Rollout Checklist

- [ ] Tests pass: `uv run pytest`
- [ ] Lint passes: `uv run ruff check .`
- [ ] Formatting passes: `uv run ruff format --check .`
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR submitted with `Closes #N` and plan linked
