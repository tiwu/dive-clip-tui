# Test Infrastructure & Methodology: Domain Expansion & Generic Specialization (Issue #48)

## 1. Overview & Objective

This document defines the architecture, methodology, feature inventory, and execution strategy for the comprehensive, opaque-box end-to-end (E2E) test suite developed for Issue #48 ("Expand domains and specialize generic footage").

The suite is located in `tests/test_e2e_domain_expansion.py`. It provides progressive, requirements-driven validation derived strictly from:
- **Requirement R1**: Wildlife Domain & Specialized Templates
- **Requirement R2**: Generic Footage Context Inference & Candidate Generation
- **Requirement R3**: Evaluation Seed Cases & Non-Regression
- **Architecture Plan**: `docs/plans/PLAN-48-expand-domains-and-specialize-generic-footage.md`
- **Project Specification**: `.agents/teamwork/orchestrator/PROJECT.md`

In adherence to Test-Driven Development (TDD) principles, tests are written strictly against specifications prior to M1 implementation ("Red" phase), establishing executable contracts that will transition to "Green" as M1 delivers production code.

---

## 2. 4-Tier Testing Methodology

The test suite is structured into four progressive tiers to ensure thorough verification from atomic contracts to integrated real-world workflows:

```text
+-------------------------------------------------------------------------------+
| Tier 4: Real-World Scenarios                                                  |
| (Multi-clip mixed batches, E2E CLI execution, evaluation dataset coverage)   |
+-------------------------------------------------------------------------------+
                                      ▲
+-------------------------------------------------------------------------------+
| Tier 3: Cross-Feature Combinations (Pairwise Workflows)                       |
| (Generic context -> candidate synthesis -> scan isolation verification)       |
| (Wildlife classification -> specialist routing -> 1-10 quality rubric scoring)|
+-------------------------------------------------------------------------------+
                                      ▲
+-------------------------------------------------------------------------------+
| Tier 2: Boundary & Corner Cases                                               |
| (Empty/None fields, tie rankings, special characters & escaping, CLI flags)   |
+-------------------------------------------------------------------------------+
                                      ▲
+-------------------------------------------------------------------------------+
| Tier 1: Feature Coverage (Core Functional Contracts)                          |
| (BUILTIN_DOMAINS, specialist templates, orientation checks, 1-10 rubrics,     |
| ClipClassification schema, candidate functions, CLI parser, scan isolation)   |
+-------------------------------------------------------------------------------+
```

### Tier 1: Feature Coverage (Core Functional Contracts)
Validates baseline schemas, domain registration, prompt hierarchies, and scan isolation:
- `wildlife` domain registered in `BUILTIN_DOMAINS` and `VIDEO_CLASSIFICATION_PROMPT`.
- Specialist templates (`wildlife_animal`, `wildlife_bird`, `wildlife_action`) and fallback (`broad_wildlife`).
- Universal coarse orientation & usability rejection across all 4 built-in domains.
- Standardized 1-10 quality rubric assessing lighting/exposure, focus/clarity, stability, and composition.
- `ClipClassification` schema extension (`environment`, `primary_subjects`).
- `infer_generic_context` and `create_prompt_candidate_for_generic` execution and output schema.
- `dive-clip classify` flags (`--specialize-generic`, `--candidate-output`).
- Strict scan-time candidate isolation invariant.

### Tier 2: Boundary & Corner Cases
Tests edge conditions, sparse inputs, adversarial values, and CLI flag variations:
- Empty record lists and missing/None context fields.
- Conflicting environments and tie-breaking in primary subjects ranking.
- Special characters, HTML-like tags, punctuation, and escaping in candidate slugs and prompts.
- Completely empty context fallback generation.
- CLI argument variations (flag-only, flag with explicit path, flag omitted).

### Tier 3: Cross-Feature Combinations (Pairwise)
Validates end-to-end data flow between interconnected subsystems:
- **Generic Pipeline Flow**: Footage classified as generic -> context preserved -> candidate prompt synthesized -> validated against orientation/rubric contracts -> confirmed completely isolated from scan execution.
- **Wildlife Pipeline Flow**: Footage classified as wildlife -> routed to specialist templates -> fine highlight scored via 1-10 quality rubric -> refined detection validated.

### Tier 4: Real-World Scenarios
Exercises complex user-facing workflows and repository datasets:
- Multi-clip mixed domain batch classification (`underwater`, `motorcycle`, `wildlife`, `generic`, and unusable footage).
- End-to-end `dive-clip classify` CLI execution producing `classifications.json` and `prompt_candidates.json`.
- Comprehensive validation of `docs/evaluation/seed-cases.yaml` schema and 4-domain coverage.

---

## 3. Feature Inventory & Test Mapping Matrix

| Feature ID | Feature Description | Test Method | Tier | Source Contract |
|---|---|---|---|---|
| **F1** | Wildlife Domain Registration | `TestTier1FeatureCoverage.test_wildlife_domain_registration` | Tier 1 | R1, PLAN-48 §2 |
| **F2** | Wildlife Specialist Templates | `TestTier1FeatureCoverage.test_wildlife_specialist_templates_resolution` | Tier 1 | R1, PLAN-48 §2 |
| **F3** | Wildlife Fallback Template | `TestTier1FeatureCoverage.test_wildlife_domain_registration` | Tier 1 | R1, PLAN-48 §2 |
| **F4** | Wildlife Classification Prompt | `TestTier1FeatureCoverage.test_wildlife_classification_prompt_and_builder` | Tier 1 | R1, PLAN-48 §2 |
| **F5** | Classification Builder Update | `TestTier1FeatureCoverage.test_wildlife_classification_prompt_and_builder` | Tier 1 | R1, PROJECT.md |
| **F6** | Coarse Orientation Rejection | `TestTier1FeatureCoverage.test_universal_coarse_orientation_rejection` | Tier 1 | R1, PLAN-48 §2 |
| **F7** | Standardized 1-10 Quality Rubric | `TestTier1FeatureCoverage.test_standardized_fine_quality_rubric_hierarchy` | Tier 1 | R1, PLAN-48 §2 |
| **F8** | ClipClassification Context Schema | `TestTier1FeatureCoverage.test_clip_classification_context_fields` | Tier 1 | R2, PLAN-48 §2 |
| **F9** | Context Inference Contract | `TestTier1FeatureCoverage.test_generic_context_inference_and_candidate_generation_schema` | Tier 1 | R2, PROJECT.md |
| **F10** | Candidate Prompt Generation | `TestTier1FeatureCoverage.test_generic_context_inference_and_candidate_generation_schema` | Tier 1 | R2, PROJECT.md |
| **F11** | CLI Flags Registration | `TestTier1FeatureCoverage.test_dive_clip_classify_cli_flags` | Tier 1 | R2, PLAN-48 §2 |
| **F12** | Scan-Time Candidate Isolation | `TestTier1FeatureCoverage.test_scan_time_candidate_isolation` | Tier 1 | R2, PLAN-48 §3 |
| **F13** | Sparse Context & None Handling | `TestTier2BoundaryAndCornerCases.test_context_inference_empty_and_none_fields` | Tier 2 | R2, Edge Cases |
| **F14** | Tie Ranking & Deduplication | `TestTier2BoundaryAndCornerCases.test_context_inference_conflicting_and_tie_rankings` | Tier 2 | R2, Edge Cases |
| **F15** | Special Character Sanitization | `TestTier2BoundaryAndCornerCases.test_candidate_generation_special_characters_and_escaping` | Tier 2 | R2, Edge Cases |
| **F16** | Blank Context Fallback | `TestTier2BoundaryAndCornerCases.test_candidate_generation_empty_context` | Tier 2 | R2, Edge Cases |
| **F17** | CLI Flag Output Variations | `TestTier2BoundaryAndCornerCases.test_cli_classify_flags_variations` | Tier 2 | R2, PLAN-48 §2 |
| **F18** | Generic Lifecycle & Isolation | `TestTier3CrossFeatureCombinations.test_generic_pipeline_flow_to_candidate_and_scan_isolation` | Tier 3 | R2, Pairwise |
| **F19** | Wildlife Scoring Lifecycle | `TestTier3CrossFeatureCombinations.test_wildlife_pipeline_flow_classification_to_fine_rubric` | Tier 3 | R1, Pairwise |
| **F20** | Mixed Domain Batch Ingestion | `TestTier4RealWorldScenarios.test_multi_clip_mixed_domain_batch_classification` | Tier 4 | R1-R2, Integration |
| **F21** | End-to-End CLI Invocation | `TestTier4RealWorldScenarios.test_end_to_end_classify_cli_with_specialize_generic` | Tier 4 | R2, CLI Workflow |
| **F22** | Seed Cases Dataset Coverage | `TestTier4RealWorldScenarios.test_evaluation_seed_cases_schema_and_domain_coverage` | Tier 4 | R3, Dataset |

---

## 4. Execution & Verification

### Running the Test Suite
```bash
# Run the E2E domain expansion test suite
uv run python -m pytest tests/test_e2e_domain_expansion.py -v

# Run the entire repository test suite (verifying non-regression)
uv run python -m pytest

# Run code style and formatting checks
uv run ruff check tests/test_e2e_domain_expansion.py
uv run ruff format --check tests/test_e2e_domain_expansion.py
```

### Safety & Isolation Guarantee
The test suite explicitly tests that model-generated candidate prompts are purely human-facing evaluation artifacts. In `TestTier1FeatureCoverage.test_scan_time_candidate_isolation` and `TestTier3CrossFeatureCombinations.test_generic_pipeline_flow_to_candidate_and_scan_isolation`, the tests verify that highlight scanning (`scripts/02_scan_footage.py` and `cmd_scan`) loads only registered templates from `BUILTIN_DOMAINS` and `config.yaml`, ensuring that dynamic unreviewed prompts can never alter highlight extraction behavior.
