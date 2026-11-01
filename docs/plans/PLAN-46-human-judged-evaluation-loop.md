# Architecture Plan: [Issue #46] Review and record feedback on scan clips

- **Issue**: [#46](https://github.com/tiwu/dive-clip-tui/issues/46)
- **Status**: Plan Review
- **Author**: Contributor
- **Date**: 2026-11-01
- **Target Branch**: `feature/issue-46-run-clip-feedback`

---

## 1. Problem Statement & Motivation

Users need a quick way to judge clips from a scan and capture what the detector got right or wrong. That feedback should help users improve prompts manually. Today the CLI can render clips but has no simple review flow or place to save judgments.

## 2. Requirements & Acceptance Criteria

- [ ] Add `dive-clip review` to review rendered clips from the latest run or a chosen run ID.
- [ ] Open one clip at a time with the operating system's default video player; ask for feedback in the terminal.
- [ ] Offer `keep`, `reject`, and `wrong subject`. Ask for the corrected subject only for `wrong subject`. Allow an optional note.
- [ ] Support skip, go back, and quit controls. Save each answer immediately.
- [ ] Resume by skipping clips with saved feedback. Allow reviewing saved answers again to edit them.
- [ ] Save feedback as a separate JSON file in the run folder. Do not edit detections or prompts.
- [ ] Summarize feedback by decision and corrected subject when review ends. Users use this summary to guide prompt edits manually.
- [ ] If opening a clip fails or a clip is missing, show a clear error and let the user skip or quit. Do not lose prior answers.
- [ ] The feature uses the existing CLI, run folders, and rendered clips. It does not extend the deprecated TUI.
- [ ] No scoring, cross-run matching, automatic prompt changes, or extra provenance system.

## 3. Proposed Architecture & UX

Add a small review flow to `scripts/dive_clip.py`. Reuse run discovery and the rendered clips in each run's `highlights/` folder. Keep feedback loading, saving, and review progression together; keep CLI setup and terminal display in the CLI entry point. Do not add a generic evaluation framework.

Command shape:

```text
dive-clip review [--run RUN_ID] [--all]
```

Without `--run`, review the latest run. By default, show clips without saved feedback. `--all` includes answered clips for corrections. Store answers in `<run>/feedback.json`; write after every answer so quitting or interruption does not lose work.

For each clip, show its filename, detected subject, domain, and time range. Open it with the platform's default file opener. Then ask for `keep`, `reject`, `wrong subject`, `skip`, `back`, or `quit`. Ask for corrected subject after `wrong subject`, and offer an optional note. Keep the interaction sequential and keyboard-friendly. Never modify model prompts or detection data.

After the last clip or quit, show counts for each choice and a list of corrected subjects and notes. This is the user-facing prompt-improvement evidence; prompt edits remain manual.

### Feedback record

Each record identifies the rendered clip and stores the user's choice, optional corrected subject, optional note, and review time. Include enough detection context to read feedback later: source filename, time range, detected subject, and domain. Keep it separate from `detections_refined.json`.

## 4. Change Log and Release Impact

- **Change category**: Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**: Add a CLI flow to review rendered clips and save keep, reject, and subject-correction feedback per run.

## 5. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - Test selecting latest and explicit runs.
   - Test unanswered-only review and `--all` editing behavior.
   - Test each answer, corrected-subject prompt, optional note, skip/back/quit, and immediate persistence.
   - Test missing clip and player-open errors preserve saved feedback and report clearly.
   - Test feedback summary includes decision counts and corrected subjects.
2. **Green (Implementation)**:
   - Add CLI parser and review command.
   - Add small helpers for opening clips, loading/saving feedback, and stepping through review.
   - Save feedback atomically after each answer.
   - Document the command in `README.md`.
3. **Refactor & Quality**:
   - Run focused review tests.
   - Run full test suite: `uv run python -m pytest`.
   - Run Ruff lint and format checks on changed Python files.

## 6. Verification & Rollout Checklist

- [ ] Review flow works with a normal run and a partially reviewed run.
- [ ] Feedback survives interruption and can be edited.
- [ ] Missing clips and player errors are explicit; saved feedback remains intact.
- [ ] Tests pass; changed Python files pass Ruff.
- [ ] `README.md` documents the review command.
- [ ] PR uses `semver:minor` and includes `Closes #46` with this plan linked.
