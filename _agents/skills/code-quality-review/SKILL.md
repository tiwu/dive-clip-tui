---
name: code-quality-review
description: >-
  Senior Python & CI code reviewer for Dive Clip TUI: reviews diffs and pull requests against repository standards (Python 3.14+, ruff, pytest), Keep a Changelog rules, SemVer policy, and architectural contracts. Use when the user asks for a code review, PR check, or lint/test verification.
---

# Code Quality Review — Dive Clip TUI

Use this skill when reviewing code diffs, branches, pull requests, or newly written modules.

## Reference & Persona

Full review guidelines, quality standards, and checklists are defined in:
[.github/agents/code-quality-review.agent.md](../../../.github/agents/code-quality-review.agent.md)

## Standards Baseline

- **Python Version**: `>=3.14`
- **Tooling**: `uv`, `ruff` (line length 100, py314 target), `pytest`
- **First-Party Packages**: `scripts` and `utils`
- **Shared Test Fixtures**: [`tests/conftest.py`](file:///home/twuytens/work/dive-clip-tui/tests/conftest.py) (`sample_manifest`, `sample_detections`, `tmp_output_dir`)
- **Logging**: Use [`utils/logging_setup.py`](file:///home/twuytens/work/dive-clip-tui/utils/logging_setup.py) rather than raw `print` statements in core modules.

## Two-Axis Review Protocol

Evaluate the changes along both axes:
1. **Axis 1 (Standards & Hygiene)**:
   - Ruff lint & format checks (`ruff check`, `ruff format --check`).
   - Clean types, Pydantic v2 schemas for pipeline models.
   - Resource cleanup: context managers (`with`) for files, buffers, and FFmpeg processes.
   - Specific exception handling (no bare `except:`).
2. **Axis 2 (Spec Alignment & Test Proof)**:
   - Verification against approved Stage 1 plan (`docs/plans/PLAN-<id>-<slug>.md`).
   - Behavioral tests in `tests/` covering new logic and edge cases.
   - [`CHANGELOG.md`](file:///home/twuytens/work/dive-clip-tui/CHANGELOG.md) entry under `[Unreleased]` with valid SemVer impact.

## Output Protocol

Format your review with:
1. Summary Assessment (Ready / Needs Minor Polish / Blocked)
2. Critical Issues (Blocking)
3. Suggestions & Improvements (Non-blocking)
4. Verification Checklist
