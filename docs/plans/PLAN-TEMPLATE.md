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

## 3. Change Log and Release Impact

- **Change category**: Added | Changed | Deprecated | Removed | Fixed | Security | None
- **Proposed SemVer impact**: `semver:major` | `semver:minor` | `semver:patch` | `semver:none`
- **Draft changelog entry**: <!-- User-facing bullet to add under CHANGELOG.md [Unreleased] -->

---

## 4. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Add or update focused tests in `tests/` for the acceptance criteria
2. **Green (Implementation)**:
   - Implement in the relevant existing package (`scripts/`, `utils/`, or `tui/`)
3. **Refactor & Quality**:
   - Run the focused tests, then the full test suite (`uv run python -m pytest`)
   - Check lint and formatting on changed Python files with Ruff

---

## 5. Verification & Rollout Checklist

- [ ] Tests pass: `uv run python -m pytest`
- [ ] Changed Python files pass Ruff lint and formatting checks
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR has no conflicting SemVer labels (unlabeled defaults to minor)
- [ ] PR submitted with `Closes #N` and plan linked
