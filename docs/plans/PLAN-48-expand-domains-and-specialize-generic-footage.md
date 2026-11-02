# Architecture Plan: [Issue #48] Expand domains and specialize generic footage

- **Issue**: [#48](https://github.com/tiwu/dive-clip-tui/issues/48)
- **Status**: Plan Review
- **Author**: Autonomous Agent
- **Date**: 2026-11-02
- **Target Branch**: `feature/issue-48-expand-domains-and-specialize-generic-footage`

---

## 1. Problem Statement & Motivation

Dive Clip currently specializes video understanding for two primary domains: `underwater` (pelagics, macro, diver action) and `motorcycle` (dynamic riding, scenic vistas, group rides), with a catch-all `generic` domain.

Roadmap Item 4 directs the project to:
1. Expand domains based on real footage and evaluation examples (specifically adding `wildlife` as identified in the roadmap).
2. For footage that remains generic, infer likely context and create subject-specific coarse and fine prompt candidates.
3. Crucially, treat generated prompts as candidates to evaluate, not trusted prompts that change behavior automatically. Unreviewed model-generated prompts must not execute during live highlight scans.

## 2. Requirements & Acceptance Criteria

- [ ] Each supported domain has a clear label, description, and keywords (`underwater`, `motorcycle`, `wildlife`, `generic`).
- [ ] Add the `wildlife` domain with coarse prompt and specialist templates (`wildlife_animal`, `wildlife_bird`, `wildlife_action`) and fallback (`broad_wildlife`).
- [ ] Coarse prompts across all domains reject accidental, inverted, dangling, or unusable frames.
- [ ] Fine prompts provide an explicit 1-10 quality rubric assessing lighting, focus, camera stability, and subject composition.
- [ ] Representative validation clips covering `underwater`, `motorcycle`, `wildlife`, and `generic` are recorded in `docs/evaluation/seed-cases.yaml`.
- [ ] For clips classified as `generic`, the classification pipeline preserves extracted `environment` and `primary_subjects`.
- [ ] A candidate prompt generator synthesizes candidate coarse and fine prompts from inferred context and subjects for human evaluation.
- [ ] `dive-clip classify` supports `--specialize-generic` to generate candidate prompt files without altering scan behavior.
- [ ] `dive-clip scan` remains completely isolated from unverified generated prompt candidates.
- [ ] Existing `underwater` and `motorcycle` domain detections, templates, and tests continue to pass without regression.

## 3. Architecture & Data Flow

```text
Pass 0 / Classification:
  Sample frames -> VLM Domain Classification -> {domain, confidence, environment, primary_subjects}
    ├── Domain is 'underwater' | 'motorcycle' | 'wildlife'
    │     └── Resolved to specialized domain templates during scan
    └── Domain is 'generic'
          ├── Scan: Uses verified generic_highlight template (never dynamic unverified prompt)
          └── Classify (--specialize-generic):
                └── Inferred Context & Primary Subjects
                      └── Candidate Prompt Generator
                            └── Saved to prompt_candidates.json for evaluation
```

### Safety and Trust Invariant
Generated prompts are output purely as artifacts/files for user inspection and evaluation. The highlight scanning pipeline (`dive-clip scan`) only runs prompt templates that are explicitly registered in code or declared by the user in `config.yaml`.

## 4. Change Log and Release Impact

- **Change category**: Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**: Add built-in `wildlife` domain with specialized prompt templates, expand evaluation seed cases across all domains, and add context inference and candidate prompt generation for generic footage in `dive-clip classify`.

## 5. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - `test_wildlife_templates`: Assert `wildlife` domain, keywords, rubrics, and resolution in `tests/test_prompt_templates.py`.
   - `test_builtin_domains_includes_wildlife`: Verify `BUILTIN_DOMAINS` and `VIDEO_CLASSIFICATION_PROMPT` include `wildlife`.
   - `test_seed_cases_coverage`: Verify `docs/evaluation/seed-cases.yaml` includes cases for each supported domain.
   - `test_generic_context_inference_and_candidate_generation`: Assert context inference and candidate coarse/fine prompt generation produce valid rubrics and orientation checks.
   - `test_classify_specialize_generic_cli`: Assert `--specialize-generic` outputs candidate file when generic clips exist.
   - `test_scan_does_not_execute_untrusted_candidates`: Verify scan preserves standard generic template execution.
2. **Green (Implementation)**:
   - Add `wildlife` domain templates and update `BUILTIN_DOMAINS` in `utils/prompt_templates.py`.
   - Add `infer_generic_context` and `create_prompt_candidate_for_generic` in `utils/prompt_templates.py`.
   - Update `ClipClassification` in `utils/models.py` to preserve environment and primary subjects.
   - Add `--specialize-generic` and `--candidate-output` to `cmd_classify` in `scripts/dive_clip.py`.
   - Update `docs/evaluation/seed-cases.yaml` with representative cases.
3. **Refactor & Quality**:
   - Run focused tests and full test suite: `uv run python -m pytest`.
   - Run Ruff check and format.

## 6. Verification & Rollout Checklist

- [ ] All supported domains (`underwater`, `motorcycle`, `wildlife`, `generic`) have verified templates and keywords.
- [ ] Seed cases in `docs/evaluation/seed-cases.yaml` represent all domains.
- [ ] Generic candidate prompts follow standard quality rubrics and remain strictly candidate artifacts.
- [ ] Tests pass: `uv run python -m pytest`.
- [ ] Ruff lint and format clean.
- [ ] `CHANGELOG.md` updated under `[Unreleased]`.
- [ ] Linked PR opened with `Closes #48` and `semver:minor`.
