# Architecture Plan: [Issue #2] Changelog, SemVer, and automated releases

- **Issue**: [#2](https://github.com/tiwu/dive-clip-tui/issues/2)
- **Status**: In Progress
- **Author**: Copilot
- **Date**: 2026-10-30
- **Target Branch**: `feature/issue-2-changelog-semver-release`

---

## 1. Problem Statement & Motivation

The project has a package version but no standard changelog, PR quality gate, or
release process. Establish a Keep a Changelog 1.1.0 and SemVer process integrated
with the existing issue-plan and implementation-PR stages.

## 2. Requirements & Acceptance Criteria

- Add a Keep a Changelog `CHANGELOG.md` with an `[Unreleased]` section.
- Record proposed change category and SemVer impact in issue plans; PRs add the
  final user-facing changelog entry and may specify a SemVer label.
- Run the full test suite and Ruff lint/format checks on changed Python files for
  pull requests; require a new Unreleased bullet on releasable PRs.
- On pushes to `main` from merged PRs, group changes since the prior release,
  choose the highest SemVer bump, update package and lockfile versions, archive
  the changelog entries, and create a `vX.Y.Z` tag and GitHub Release.
- Default an unlabeled PR to minor; `semver:none` skips release creation.
- Use `0.1.0` as the initial baseline and let the release bot update `main`
  directly.

## 3. Change Log and Release Impact

- **Change category**: Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**: Add Keep a Changelog, SemVer, pull-request CI, and
  automated GitHub releases.

## 4. Test-Driven Development (TDD) Plan

1. Add focused tests for SemVer calculation, label validation, changelog
   validation/archive behavior, project version updates, and release preparation.
2. Implement the shared release helper and connect it to the PR and release
   workflows.
3. Run focused tests, the complete test suite, and Ruff checks on changed Python
   files; parse the workflow YAML and inspect the final diff.

## 5. Verification & Rollout Checklist

- [x] Tests cover SemVer labels and release/changelog transformations.
- [x] PR CI passes tests, changed-file lint/format checks, and changelog validation.
- [ ] Release workflow updates versions, changelog, tag, and GitHub Release.
- [ ] Repository settings allow Actions `contents: write` and bot updates to `main`.
- [ ] PR merged and release workflow observed.
