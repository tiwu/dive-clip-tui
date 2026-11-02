# TEST_READY: Domain Expansion & Generic Specialization (Issue #48)

## Executive Summary

The end-to-end requirement-driven test suite for Issue #48 Stage 2 ("Expand domains and specialize generic footage") is designed, implemented, and fully verified for compilation, collection, and lint compliance.

- **Status**: READY FOR M1 VERIFICATION (TDD Red Phase Complete)
- **Test File**: `tests/test_e2e_domain_expansion.py`
- **Total Test Cases**: 19 comprehensive, non-facade test cases across Tiers 1–4
- **Linter Status**: 0 Ruff violations (`ruff check` and `ruff format --check` passed)
- **Collection Status**: 100% clean test collection with zero syntax or import errors

---

## How to Run

```bash
# Execute the Issue #48 E2E test suite with verbose output
uv run python -m pytest tests/test_e2e_domain_expansion.py -v

# Execute the entire repository test suite to confirm non-regression
uv run python -m pytest

# Validate code formatting and linting
uv run ruff check tests/test_e2e_domain_expansion.py
uv run ruff format --check tests/test_e2e_domain_expansion.py
```

---

## 4-Tier Test Breakdown

### Tier 1: Feature Coverage (9 Tests)
- [x] `TestTier1FeatureCoverage.test_wildlife_domain_registration`: Registers `wildlife` in `BUILTIN_DOMAINS` with keywords, description, coarse prompt, specialist templates, and fallback.
- [x] `TestTier1FeatureCoverage.test_wildlife_classification_prompt_and_builder`: Verifies `VIDEO_CLASSIFICATION_PROMPT` and `build_classification_prompt` include `wildlife`.
- [x] `TestTier1FeatureCoverage.test_wildlife_specialist_templates_resolution`: Routes animal, bird, and action subjects to specialist templates (`wildlife_animal`, `wildlife_bird`, `wildlife_action`, `broad_wildlife`).
- [x] `TestTier1FeatureCoverage.test_universal_coarse_orientation_rejection`: Asserts coarse prompts across all 4 built-in domains reject inverted, dangling, and covered frames.
- [x] `TestTier1FeatureCoverage.test_standardized_fine_quality_rubric_hierarchy`: Asserts all fine prompts enforce a 1-10 quality rubric assessing lighting, focus, stability, and composition.
- [x] `TestTier1FeatureCoverage.test_clip_classification_context_fields`: Verifies schema preservation of `environment` and `primary_subjects` on `ClipClassification`.
- [x] `TestTier1FeatureCoverage.test_generic_context_inference_and_candidate_generation_schema`: Validates interface contracts for `infer_generic_context` and `create_prompt_candidate_for_generic`.
- [x] `TestTier1FeatureCoverage.test_dive_clip_classify_cli_flags`: Confirms CLI parser supports `--specialize-generic` and `--candidate-output`.
- [x] `TestTier1FeatureCoverage.test_scan_time_candidate_isolation`: Proves highlight scanning ignores candidate files and executes only registered domain configurations.

### Tier 2: Boundary & Corner Cases (5 Tests)
- [x] `TestTier2BoundaryAndCornerCases.test_context_inference_empty_and_none_fields`: Handles empty lists, empty strings, and None context fields without crashing.
- [x] `TestTier2BoundaryAndCornerCases.test_context_inference_conflicting_and_tie_rankings`: Resolves multi-clip tie rankings and frequency deduplication deterministically.
- [x] `TestTier2BoundaryAndCornerCases.test_candidate_generation_special_characters_and_escaping`: Sanitizes special characters in candidate template slugs and prompt strings.
- [x] `TestTier2BoundaryAndCornerCases.test_candidate_generation_empty_context`: Generates safe fallback candidate templates for blank contexts.
- [x] `TestTier2BoundaryAndCornerCases.test_cli_classify_flags_variations`: Tests CLI variations (omitted, default path, and explicit `--candidate-output`).

### Tier 3: Cross-Feature Combinations (2 Tests)
- [x] `TestTier3CrossFeatureCombinations.test_generic_pipeline_flow_to_candidate_and_scan_isolation`: Full workflow for generic footage from classification through candidate synthesis, verifying candidate isolation from scan execution.
- [x] `TestTier3CrossFeatureCombinations.test_wildlife_pipeline_flow_classification_to_fine_rubric`: Full workflow for wildlife footage from classification to specialist resolution and 1-10 rubric scoring.

### Tier 4: Real-World Scenarios (3 Tests)
- [x] `TestTier4RealWorldScenarios.test_multi_clip_mixed_domain_batch_classification`: Ingests and classifies a mixed 5-clip batch spanning underwater, motorcycle, wildlife, generic, and unusable footage.
- [x] `TestTier4RealWorldScenarios.test_end_to_end_classify_cli_with_specialize_generic`: End-to-end `cmd_classify` CLI execution producing `classifications.json` and `prompt_candidates.json`.
- [x] `TestTier4RealWorldScenarios.test_evaluation_seed_cases_schema_and_domain_coverage`: Validates schema and domain coverage in `docs/evaluation/seed-cases.yaml`.

---

## Target Feature Checklist for M1 (Core Implementation)

| Feature | Target Component | Acceptance Contract Verified by Test |
|---|---|---|
| Wildlife domain in `BUILTIN_DOMAINS` | `utils/prompt_templates.py` | `test_wildlife_domain_registration` |
| Wildlife in classification prompt | `utils/prompt_templates.py` | `test_wildlife_classification_prompt_and_builder` |
| Wildlife specialist template resolution | `utils/prompt_templates.py` | `test_wildlife_specialist_templates_resolution` |
| Orientation rejection across coarse prompts | `utils/prompt_templates.py` | `test_universal_coarse_orientation_rejection` |
| 1-10 quality rubric across fine prompts | `utils/prompt_templates.py` | `test_standardized_fine_quality_rubric_hierarchy` |
| `environment` & `primary_subjects` fields | `utils/models.py` | `test_clip_classification_context_fields` |
| `infer_generic_context` function | `utils/prompt_templates.py` | `test_generic_context_inference_and_candidate_generation_schema` |
| `create_prompt_candidate_for_generic` function | `utils/prompt_templates.py` | `test_candidate_generation_empty_context` |
| `--specialize-generic` CLI flags | `scripts/dive_clip.py` | `test_dive_clip_classify_cli_flags` |
| Candidate generation in `cmd_classify` | `scripts/dive_clip.py` | `test_end_to_end_classify_cli_with_specialize_generic` |
| Scan isolation invariant | `scripts/02_scan_footage.py` | `test_scan_time_candidate_isolation` |
| Seed cases expansion for 4 domains | `docs/evaluation/seed-cases.yaml` | `test_evaluation_seed_cases_schema_and_domain_coverage` |
