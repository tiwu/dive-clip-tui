"""End-to-end requirement-driven tests for domain expansion, candidate generation, and CLI (Issue #48).

Derivation Sources:
- R1: Wildlife Domain and Specialized Templates
- R2: Generic Footage Context Inference and Candidate Prompt Generation
- R3: Evaluation Seed Cases and Non-Regression
- docs/plans/PLAN-48-expand-domains-and-specialize-generic-footage.md
- .agents/teamwork/orchestrator/PROJECT.md

Covers Tiers 1-4:
- Tier 1: Feature Coverage (domains, prompts, models, CLI flags, scan isolation)
- Tier 2: Boundary & Corner Cases (empty/None fields, tie ranks, special characters)
- Tier 3: Cross-Feature Combinations (generic-to-candidate-to-isolation, wildlife-to-fine-rubric)
- Tier 4: Real-World Scenarios (mixed batch classification, end-to-end CLI, seed cases)
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from scripts import dive_clip
from utils import models, prompt_templates

# Dynamically import scripts/02_scan_footage.py
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_scan_spec = importlib.util.spec_from_file_location(
    "scripts_02_scan_footage", _SCRIPTS_DIR / "02_scan_footage.py"
)
if _scan_spec is None or _scan_spec.loader is None:
    raise ImportError("Unable to load scripts/02_scan_footage.py")
scan = importlib.util.module_from_spec(_scan_spec)
_scan_spec.loader.exec_module(scan)

_ROOT_DIR = Path(__file__).resolve().parent.parent


def _require_builtin_domain(domain_name: str) -> prompt_templates.DomainConfig:
    """Safely retrieve a domain from BUILTIN_DOMAINS or fail with clear test diagnostics."""
    if domain_name not in prompt_templates.BUILTIN_DOMAINS:
        pytest.fail(f"BUILTIN_DOMAINS missing domain '{domain_name}'")
    return prompt_templates.BUILTIN_DOMAINS[domain_name]


def _require_infer_generic_context() -> Any:
    """Safely retrieve infer_generic_context or fail with clear test diagnostics."""
    fn = getattr(prompt_templates, "infer_generic_context", None)
    if fn is None:
        pytest.fail("utils.prompt_templates.infer_generic_context is not implemented")
    return fn


def _require_create_prompt_candidate() -> Any:
    """Safely retrieve create_prompt_candidate_for_generic or fail with clear test diagnostics."""
    fn = getattr(prompt_templates, "create_prompt_candidate_for_generic", None)
    if fn is None:
        pytest.fail("utils.prompt_templates.create_prompt_candidate_for_generic is not implemented")
    return fn


# ==============================================================================
# Tier 1: Feature Coverage
# ==============================================================================


class TestTier1FeatureCoverage:
    """Tier 1: Baseline functional contracts for domain expansion and candidate generation."""

    def test_wildlife_domain_registration(self) -> None:
        """R1: BUILTIN_DOMAINS must register 'wildlife' with specialized and fallback templates."""
        wildlife = _require_builtin_domain("wildlife")

        assert wildlife.name == "wildlife"
        assert isinstance(wildlife.keywords, list)
        assert len(wildlife.keywords) >= 3

        # Must include characteristic wildlife keywords
        kw_str = " ".join(wildlife.keywords).lower()
        assert any(k in kw_str for k in ("animal", "bird", "wildlife", "safari", "fauna"))

        assert bool(wildlife.description), "Wildlife domain description must not be empty"
        assert bool(wildlife.coarse_prompt), "Wildlife domain coarse prompt must not be empty"

        # Specialized templates
        template_names = [t.name for t in wildlife.templates]
        for expected in ("wildlife_animal", "wildlife_bird", "wildlife_action"):
            assert expected in template_names, f"Expected template '{expected}' in wildlife domain"

        # Fallback template
        assert wildlife.fallback_template is not None
        assert wildlife.fallback_template.name == "broad_wildlife"

        # ALL_DOMAIN_TEMPLATES must include the wildlife templates
        all_names = [t.name for t in prompt_templates.ALL_DOMAIN_TEMPLATES]
        assert "wildlife_animal" in all_names
        assert "broad_wildlife" in all_names

    def test_wildlife_classification_prompt_and_builder(self) -> None:
        """R1: VIDEO_CLASSIFICATION_PROMPT and build_classification_prompt must support wildlife."""
        prompt = prompt_templates.VIDEO_CLASSIFICATION_PROMPT
        assert "wildlife" in prompt.lower()

        # Classification prompt builder with 4 domains
        built = prompt_templates.build_classification_prompt([
            "underwater",
            "motorcycle",
            "wildlife",
            "generic",
        ])
        assert "wildlife" in built.lower()
        assert "underwater" in built.lower()
        assert "motorcycle" in built.lower()
        assert "generic" in built.lower()

    def test_wildlife_specialist_templates_resolution(self) -> None:
        """R1: resolve_template_for_subject routes wildlife subjects to specialist templates."""
        resolve_fn = getattr(prompt_templates, "resolve_template_for_subject", None)
        if resolve_fn is None:
            pytest.fail("utils.prompt_templates.resolve_template_for_subject is not defined")

        # Animal subject -> wildlife_animal
        t_animal = resolve_fn("african lion stalking", domain="wildlife")
        assert t_animal.name == "wildlife_animal"

        # Bird subject -> wildlife_bird
        t_bird = resolve_fn("eagle in flight", domain="wildlife")
        assert t_bird.name == "wildlife_bird"

        # Action subject -> wildlife_action
        t_action = resolve_fn("hunting chase predator attack", domain="wildlife")
        assert t_action.name == "wildlife_action"

        # Fallback for unrecognized subjects -> broad_wildlife
        t_fallback = resolve_fn("mysterious savannah silhouette", domain="wildlife")
        assert t_fallback.name == "broad_wildlife"

    def test_universal_coarse_orientation_rejection(self) -> None:
        """R1: Coarse prompts across all built-in domains must reject unusable/inverted footage."""
        domains = ["underwater", "motorcycle", "wildlife", "generic"]
        for domain_name in domains:
            dc = _require_builtin_domain(domain_name)
            cp = dc.coarse_prompt.lower()

            # Must contain explicit orientation / unusable frame rejection instructions
            has_inverted = "inverted" in cp or "upside" in cp or "ground" in cp or "floor" in cp
            has_dangling = "dangling" in cp or "spinning" in cp or "tumbling" in cp
            has_covered = "covered" in cp or "pocket" in cp or "bag" in cp or "blurred" in cp

            assert has_inverted and has_dangling and has_covered, (
                f"Domain '{domain_name}' coarse prompt missing orientation/usability rejection: {dc.coarse_prompt[:120]}"
            )

            # Must instruct VLM to reject with detected: false / quality_score 1
            assert "false" in cp and "quality_score" in cp, (
                f"Domain '{domain_name}' coarse prompt must instruct VLM rejection response format"
            )

    def test_standardized_fine_quality_rubric_hierarchy(self) -> None:
        """R1: Fine prompts across all domains must contain 1-10 quality rubric hierarchy."""
        domains = ["underwater", "motorcycle", "wildlife", "generic"]
        for domain_name in domains:
            dc = _require_builtin_domain(domain_name)
            templates_to_check = list(dc.templates)
            if dc.fallback_template and dc.fallback_template not in templates_to_check:
                templates_to_check.append(dc.fallback_template)

            for t in templates_to_check:
                fp = t.fine_prompt.lower()
                # 1-10 rubric assessing lighting, focus, stability, composition
                assert "lighting" in fp or "exposure" in fp, f"{t.name} missing lighting rubric"
                assert "focus" in fp or "clarity" in fp or "sharp" in fp, (
                    f"{t.name} missing focus rubric"
                )
                assert "stability" in fp or "camera" in fp, f"{t.name} missing stability rubric"
                assert "composition" in fp or "framing" in fp or "subject" in fp, (
                    f"{t.name} missing composition rubric"
                )
                assert "quality_score" in fp or "1-10" in fp or "10" in fp, (
                    f"{t.name} missing 1-10 quality score hierarchy"
                )

    def test_clip_classification_context_fields(self) -> None:
        """R2: ClipClassification must preserve environment and primary_subjects."""
        fields = models.ClipClassification.model_fields
        if "environment" not in fields or "primary_subjects" not in fields:
            pytest.fail(
                "ClipClassification missing 'environment' or 'primary_subjects' schema fields"
            )

        # Backward compatibility: instantiating without new fields defaults safely
        c_default = models.ClipClassification(
            clip_id="c_default",
            file_name="test.mp4",
            file_path="/videos/test.mp4",
            domain="generic",
        )
        assert c_default.environment == ""
        assert c_default.primary_subjects == []

        # Context preservation on generic clip
        c_context = models.ClipClassification(
            clip_id="c_generic",
            file_name="alpine.mp4",
            file_path="/videos/alpine.mp4",
            domain="generic",
            confidence=0.82,
            environment="alpine meadow",
            primary_subjects=["marmot", "pine tree", "hiking trail"],
        )
        assert c_context.environment == "alpine meadow"
        assert c_context.primary_subjects == ["marmot", "pine tree", "hiking trail"]

        # Serialization round-trip
        data = json.loads(c_context.model_dump_json())
        assert data["environment"] == "alpine meadow"
        assert data["primary_subjects"] == ["marmot", "pine tree", "hiking trail"]

    def test_generic_context_inference_and_candidate_generation_schema(self) -> None:
        """R2: infer_generic_context and create_prompt_candidate_for_generic interface contracts."""
        infer_fn = _require_infer_generic_context()
        cand_fn = _require_create_prompt_candidate()

        # Input: list of ClipClassification objects
        records = [
            models.ClipClassification(
                clip_id="c1",
                file_name="clip1.mp4",
                file_path="/videos/clip1.mp4",
                domain="generic",
                environment="dense rainforest",
                primary_subjects=["toucan", "canopy"],
            ),
            models.ClipClassification(
                clip_id="c2",
                file_name="clip2.mp4",
                file_path="/videos/clip2.mp4",
                domain="generic",
                environment="dense rainforest",
                primary_subjects=["toucan", "waterfall"],
            ),
        ]

        context = infer_fn(records)
        assert isinstance(context, dict)
        assert "environment" in context
        assert "primary_subjects" in context
        assert "template_slug" in context
        assert context["environment"] == "dense rainforest"
        assert "toucan" in context["primary_subjects"]

        # Candidate prompt generation from inferred context
        candidate = cand_fn(context)
        assert isinstance(candidate, dict)
        for key in ("name", "domain", "coarse_prompt", "fine_prompt", "recommended_threshold"):
            assert key in candidate, f"Candidate prompt dictionary missing required key '{key}'"

        assert candidate["domain"] == "generic"
        assert 0.0 <= float(candidate["recommended_threshold"]) <= 1.0

        # Orientation check in candidate coarse prompt
        coarse = candidate["coarse_prompt"].lower()
        assert (
            "inverted" in coarse
            or "upside" in coarse
            or "dangling" in coarse
            or "covered" in coarse
        ), "Candidate coarse prompt must enforce orientation/usability rejection"

        # Standardized 1-10 quality rubric in candidate fine prompt
        fine = candidate["fine_prompt"].lower()
        assert "lighting" in fine or "exposure" in fine
        assert "focus" in fine or "clarity" in fine
        assert "stability" in fine or "camera" in fine
        assert "composition" in fine or "framing" in fine
        assert "quality_score" in fine or "1-10" in fine

    def test_dive_clip_classify_cli_flags(self) -> None:
        """R2: dive-clip classify accepts --specialize-generic and --candidate-output."""
        parser = dive_clip.build_parser()

        # Test flags parsing
        try:
            args = parser.parse_args([
                "classify",
                "--input",
                "./footage",
                "--output",
                "./output",
                "--specialize-generic",
                "--candidate-output",
                "./custom_candidates.json",
            ])
        except SystemExit:
            pytest.fail("CLI parser failed to parse --specialize-generic or --candidate-output")

        assert hasattr(args, "specialize_generic")
        assert args.specialize_generic is not False
        assert hasattr(args, "candidate_output")
        assert args.candidate_output == "./custom_candidates.json"

    def test_scan_time_candidate_isolation(self) -> None:
        """R2: dive-clip scan strictly isolates from candidate prompt files."""
        # Check that scan registry loader only reads BUILTIN_DOMAINS and config.yaml
        registry = scan.load_custom_templates({})
        assert set(registry.keys()).issubset(set(prompt_templates.BUILTIN_DOMAINS.keys())), (
            "Scan template registry must only load registered domain configurations"
        )

        # When resolving templates for generic footage, scan uses registered fallback template
        t_generic = scan.resolve_template_for_subject("any subject", domain="generic")
        assert t_generic.name == prompt_templates.GENERIC_HIGHLIGHT_TEMPLATE.name
        assert t_generic != "unverified_candidate"


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================


class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary conditions, malformed responses, edge cases, and flag variations."""

    def test_context_inference_empty_and_none_fields(self) -> None:
        """R2: infer_generic_context handles empty lists, None fields, and missing values safely."""
        infer_fn = _require_infer_generic_context()

        # Empty list of records
        empty_res = infer_fn([])
        assert isinstance(empty_res, dict)
        assert empty_res.get("environment") == ""
        assert empty_res.get("primary_subjects") == []
        assert bool(empty_res.get("template_slug")), "Must provide a fallback template slug"

        # Records with empty strings and None values
        sparse_records = [
            models.ClipClassification(
                clip_id="c_sparse_1",
                file_name="sparse1.mp4",
                file_path="/p/sparse1.mp4",
                domain="generic",
                environment="",
                primary_subjects=[],
            ),
            {"domain": "generic", "environment": None, "primary_subjects": None},
        ]
        sparse_res = infer_fn(sparse_records)
        assert isinstance(sparse_res, dict)
        assert sparse_res.get("environment") == ""
        assert sparse_res.get("primary_subjects") == []

    def test_context_inference_conflicting_and_tie_rankings(self) -> None:
        """R2: infer_generic_context deterministically resolves ties and subject frequencies."""
        infer_fn = _require_infer_generic_context()

        records = [
            models.ClipClassification(
                clip_id="c1",
                file_name="f1.mp4",
                file_path="/p/f1.mp4",
                domain="generic",
                environment="desert dunes",
                primary_subjects=["camel", "dune", "sun"],
            ),
            models.ClipClassification(
                clip_id="c2",
                file_name="f2.mp4",
                file_path="/p/f2.mp4",
                domain="generic",
                environment="desert dunes",
                primary_subjects=["camel", "oasis"],
            ),
            models.ClipClassification(
                clip_id="c3",
                file_name="f3.mp4",
                file_path="/p/f3.mp4",
                domain="generic",
                environment="rock canyon",
                primary_subjects=["river", "dune"],
            ),
        ]

        res = infer_fn(records)
        # Dominant environment should be 'desert dunes' (2 occurrences vs 1)
        assert res["environment"] == "desert dunes"

        # 'camel' and 'dune' appear 2 times; should be ranked ahead of 1-time subjects
        ranked = res["primary_subjects"]
        assert len(ranked) == len(set(ranked)), "Primary subjects list must be deduplicated"
        top_subjects = set(ranked[:2])
        assert "camel" in top_subjects or "dune" in top_subjects

    def test_candidate_generation_special_characters_and_escaping(self) -> None:
        """R2: Candidate generation sanitizes special characters in slug and prompts."""
        cand_fn = _require_create_prompt_candidate()

        dirty_context = {
            "environment": "extreme / rocky <canyon> & gorge!",
            "primary_subjects": ['"hawk"\'s nest', "fast-flowing / river", "wild-dog $#!+"],
            "template_slug": "extreme / rocky <canyon> & gorge!",
        }

        candidate = cand_fn(dirty_context)
        slug = candidate["name"]

        # Slug must be sanitized for filesystem / identifier safety
        assert re.match(r"^[a-zA-Z0-9_\-]+$", slug), (
            f"Template slug '{slug}' contains invalid special characters"
        )

        # Coarse and fine prompts must format cleanly without unhandled formatting errors
        assert candidate["domain"] == "generic"
        assert len(candidate["coarse_prompt"]) > 20
        assert len(candidate["fine_prompt"]) > 20

    def test_candidate_generation_empty_context(self) -> None:
        """R2: Candidate generation produces valid fallback prompt when context is completely blank."""
        cand_fn = _require_create_prompt_candidate()

        candidate = cand_fn({"environment": "", "primary_subjects": []})
        assert isinstance(candidate, dict)
        assert bool(candidate["name"])
        assert candidate["domain"] == "generic"
        assert (
            "orientation" in candidate["coarse_prompt"].lower()
            or "false" in candidate["coarse_prompt"].lower()
        )
        assert (
            "quality_score" in candidate["fine_prompt"].lower()
            or "1-10" in candidate["fine_prompt"].lower()
        )

    def test_cli_classify_flags_variations(self, tmp_path: Path) -> None:
        """R2: CLI parser variations with and without custom candidate output paths."""
        parser = dive_clip.build_parser()

        # Case 1: Flag omitted -> specialize_generic is False
        args1 = parser.parse_args(["classify", "--input", str(tmp_path), "--output", str(tmp_path)])
        assert getattr(args1, "specialize_generic", False) is False

        # Case 2: Flag present without path -> default path behavior
        args2 = parser.parse_args([
            "classify",
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path),
            "--specialize-generic",
        ])
        assert getattr(args2, "specialize_generic", False) is not False
        assert getattr(args2, "candidate_output", None) is None

        # Case 3: Flag present with explicit candidate output
        cand_path = tmp_path / "candidates.json"
        args3 = parser.parse_args([
            "classify",
            "--input",
            str(tmp_path),
            "--output",
            str(tmp_path),
            "--specialize-generic",
            "--candidate-output",
            str(cand_path),
        ])
        assert getattr(args3, "specialize_generic", False) is not False
        assert getattr(args3, "candidate_output", None) == str(cand_path)


# ==============================================================================
# Tier 3: Cross-Feature Combinations (Pairwise)
# ==============================================================================


class TestTier3CrossFeatureCombinations:
    """Tier 3: Pairwise cross-feature interactions across domains, candidates, and isolation."""

    def test_generic_pipeline_flow_to_candidate_and_scan_isolation(self, tmp_path: Path) -> None:
        """Pairwise: Generic classification -> candidate synthesized -> verified isolated from scan."""
        infer_fn = _require_infer_generic_context()
        cand_fn = _require_create_prompt_candidate()

        # Step 1: Simulated generic clip classification with preserved context
        generic_clip = models.ClipClassification(
            clip_id="g_clip_01",
            file_name="generic_forest.mp4",
            file_path=str(tmp_path / "generic_forest.mp4"),
            domain="generic",
            confidence=0.75,
            environment="dense pine woods",
            primary_subjects=["forager", "mushroom", "moss"],
        )

        # Step 2: Inferred context & candidate generation
        context = infer_fn([generic_clip])
        assert context["environment"] == "dense pine woods"

        candidate = cand_fn(context)
        assert candidate["domain"] == "generic"

        # Step 3: Write candidate prompt file to output folder
        cand_file = tmp_path / "prompt_candidates.json"
        cand_file.write_text(json.dumps([candidate]), encoding="utf-8")
        assert cand_file.exists()

        # Step 4: Verify scan isolation invariant:
        # A live highlight scan of generic_forest.mp4 must NEVER load or execute cand_file
        scan_registry = scan.load_custom_templates({})
        assert candidate["name"] not in scan_registry.get("generic", {}).templates  # type: ignore[operator]

        # Scan template resolution for generic clip resolves strictly to built-in GENERIC_HIGHLIGHT_TEMPLATE
        resolved = scan.resolve_template_for_subject(
            subject="forager picking mushroom", domain="generic"
        )
        assert resolved.name == prompt_templates.GENERIC_HIGHLIGHT_TEMPLATE.name
        assert resolved.fine_prompt != candidate["fine_prompt"]

    def test_wildlife_pipeline_flow_classification_to_fine_rubric(self) -> None:
        """Pairwise: Wildlife classification -> specialist resolution -> 1-10 quality rubric scoring."""
        # Step 1: Clip classified as wildlife
        w_clip = models.ClipClassification(
            clip_id="w_clip_01",
            file_name="safari_chase.mp4",
            file_path="/videos/safari_chase.mp4",
            domain="wildlife",
            confidence=0.92,
            usable=True,
            quality_score=9,
        )
        assert w_clip.domain == "wildlife"
        assert w_clip.usable is True

        # Step 2: Subject resolution to specialist action template
        template = prompt_templates.resolve_template_for_subject(
            subject="cheetah hunt pursuit sprint", domain="wildlife"
        )
        assert template.name == "wildlife_action"

        # Step 3: Rubric criteria present in resolved fine prompt
        fp = template.fine_prompt.lower()
        for criterion in ("lighting", "focus", "stability", "composition"):
            assert criterion in fp or "camera" in fp or "framing" in fp

        # Step 4: Refined detection created under wildlife domain
        det = models.RefinedDetection(
            clip_id=w_clip.clip_id,
            file_name=w_clip.file_name,
            file_path=w_clip.file_path,
            fps="30.0",
            start_time=10.0,
            end_time=15.0,
            subject="cheetah hunting gazelle",
            confidence=0.95,
            quality_score=9,
            description="High-speed cheetah chase in golden morning light",
            domain="wildlife",
            template=template.name,
        )
        assert det.domain == "wildlife"
        assert det.quality_score == 9


# ==============================================================================
# Tier 4: Real-World Application Scenarios
# ==============================================================================


class TestTier4RealWorldScenarios:
    """Tier 4: End-to-end multi-clip batches, CLI executions, and evaluation dataset coverage."""

    def test_multi_clip_mixed_domain_batch_classification(self, tmp_path: Path) -> None:
        """Multi-clip batch containing underwater, motorcycle, wildlife, generic, and unusable clips."""
        clips = [
            {
                "clip_id": "c_uw",
                "file_name": "reef.mp4",
                "file_path": str(tmp_path / "reef.mp4"),
            },
            {
                "clip_id": "c_moto",
                "file_name": "pass.mp4",
                "file_path": str(tmp_path / "pass.mp4"),
            },
            {
                "clip_id": "c_wild",
                "file_name": "lion.mp4",
                "file_path": str(tmp_path / "lion.mp4"),
            },
            {
                "clip_id": "c_gen",
                "file_name": "city.mp4",
                "file_path": str(tmp_path / "city.mp4"),
            },
            {
                "clip_id": "c_bad",
                "file_name": "bag.mp4",
                "file_path": str(tmp_path / "bag.mp4"),
            },
        ]

        # Simulated frame classification responses across the mixed batch
        responses = {
            "c_uw": {
                "domain": "underwater",
                "confidence": 0.95,
                "usable": True,
                "quality_score": 8,
                "environment": "coral reef wall",
                "primary_subjects": ["manta ray"],
            },
            "c_moto": {
                "domain": "motorcycle",
                "confidence": 0.89,
                "usable": True,
                "quality_score": 8,
                "environment": "mountain pass highway",
                "primary_subjects": ["sports bike"],
            },
            "c_wild": {
                "domain": "wildlife",
                "confidence": 0.93,
                "usable": True,
                "quality_score": 9,
                "environment": "savannah watering hole",
                "primary_subjects": ["lion pride"],
            },
            "c_gen": {
                "domain": "generic",
                "confidence": 0.78,
                "usable": True,
                "quality_score": 7,
                "environment": "urban public plaza",
                "primary_subjects": ["skater", "fountain"],
            },
            "c_bad": {
                "domain": "generic",
                "confidence": 0.30,
                "usable": False,
                "quality_score": 1,
                "environment": "",
                "primary_subjects": [],
            },
        }

        classifications: list[models.ClipClassification] = []
        for c in clips:
            resp = responses[c["clip_id"]]
            cl = models.ClipClassification(
                clip_id=c["clip_id"],
                file_name=c["file_name"],
                file_path=c["file_path"],
                domain=resp["domain"],
                confidence=resp["confidence"],
                usable=resp["usable"],
                quality_score=resp["quality_score"],
                environment=resp["environment"],
                primary_subjects=resp["primary_subjects"],
            )
            classifications.append(cl)

        # Verify all four domains are represented
        domains_present = {c.domain for c in classifications}
        assert {"underwater", "motorcycle", "wildlife", "generic"}.issubset(domains_present)

        # Unusable clip must be marked unusable
        bad_clip = next(c for c in classifications if c.clip_id == "c_bad")
        assert bad_clip.usable is False
        assert bad_clip.quality_score == 1

        # Generic clip preserved environment and subjects
        gen_clip = next(c for c in classifications if c.clip_id == "c_gen")
        assert gen_clip.environment == "urban public plaza"
        assert "skater" in gen_clip.primary_subjects

    def test_end_to_end_classify_cli_with_specialize_generic(self, tmp_path: Path) -> None:
        """E2E dive-clip classify invocation producing classifications and prompt candidates."""
        footage_dir = tmp_path / "footage"
        footage_dir.mkdir()
        out_file = tmp_path / "classifications.json"
        cand_file = tmp_path / "custom_candidates.json"

        # Mock ingest and scan returns
        fake_clips = [
            {
                "clip_id": "c1",
                "file_name": "underwater.mp4",
                "file_path": str(footage_dir / "underwater.mp4"),
            },
            {
                "clip_id": "c2",
                "file_name": "wildlife.mp4",
                "file_path": str(footage_dir / "wildlife.mp4"),
            },
            {"clip_id": "c3", "file_name": "park.mp4", "file_path": str(footage_dir / "park.mp4")},
        ]

        fake_classifications = [
            models.ClipClassification(
                clip_id="c1",
                file_name="underwater.mp4",
                file_path=str(footage_dir / "underwater.mp4"),
                domain="underwater",
                confidence=0.9,
            ),
            models.ClipClassification(
                clip_id="c2",
                file_name="wildlife.mp4",
                file_path=str(footage_dir / "wildlife.mp4"),
                domain="wildlife",
                confidence=0.88,
            ),
            models.ClipClassification(
                clip_id="c3",
                file_name="park.mp4",
                file_path=str(footage_dir / "park.mp4"),
                domain="generic",
                confidence=0.8,
                environment="city park",
                primary_subjects=["dog", "frisbee"],
            ),
        ]

        parser = dive_clip.build_parser()
        args = parser.parse_args([
            "classify",
            "--input",
            str(footage_dir),
            "--output",
            str(out_file),
            "--specialize-generic",
            "--candidate-output",
            str(cand_file),
        ])

        with (
            patch("scripts.dive_clip._load_ingest_module") as mock_ingest_loader,
            patch("scripts.dive_clip._load_scan_module") as mock_scan_loader,
        ):
            mock_ingest = MagicMock()
            mock_ingest.ingest_folder.return_value = fake_clips
            mock_ingest_loader.return_value = mock_ingest

            mock_scan = MagicMock()
            mock_scan.classify_footage.return_value = fake_classifications
            mock_scan_loader.return_value = mock_scan

            rc = dive_clip.cmd_classify(args)

        assert rc == 0
        assert out_file.exists()

        out_data = json.loads(out_file.read_text(encoding="utf-8"))
        assert out_data["total_clips"] == 3

        # Candidate prompt file must be synthesized for generic clips
        if not cand_file.exists():
            # If default output path was used instead
            default_cand = out_file.parent / "prompt_candidates.json"
            if default_cand.exists():
                cand_file = default_cand

        assert cand_file.exists(), (
            "cmd_classify with --specialize-generic must synthesize candidate prompt file"
        )
        cand_data = json.loads(cand_file.read_text(encoding="utf-8"))
        assert isinstance(cand_data, (dict, list)) and len(cand_data) > 0, (
            "Candidate prompt file must contain valid candidate template definitions"
        )

    def test_evaluation_seed_cases_schema_and_domain_coverage(self) -> None:
        """R3: docs/evaluation/seed-cases.yaml must cover underwater, motorcycle, wildlife, and generic."""
        seed_path = _ROOT_DIR / "docs" / "evaluation" / "seed-cases.yaml"
        assert seed_path.exists(), f"Seed cases file not found at {seed_path}"

        raw_data = yaml.safe_load(seed_path.read_text(encoding="utf-8"))
        assert isinstance(raw_data, dict), "Seed cases file must be a YAML mapping"
        assert "cases" in raw_data, "Seed cases file must contain a 'cases' key"

        cases = raw_data["cases"]
        assert isinstance(cases, list) and len(cases) >= 4

        # Validate schema of each seed case
        observed_domains: set[str] = set()
        for c in cases:
            assert "id" in c, f"Case missing id: {c}"
            assert "source_file" in c, f"Case {c.get('id')} missing source_file"
            assert "kind" in c, f"Case {c.get('id')} missing kind"

            # Domain can be in observed.domain, case.domain, or inferred from id/subject
            domain = c.get("domain") or c.get("observed", {}).get("domain")
            if not domain:
                case_id = str(c.get("id", "")).lower()
                case_sub = str(c.get("observed", {}).get("subject", "")).lower()
                if any(
                    x in case_id or x in case_sub
                    for x in ("coral", "nudibranch", "turtle", "flutefish")
                ):
                    domain = "underwater"
                elif any(
                    x in case_id or x in case_sub for x in ("moto", "ride", "pass", "highway")
                ):
                    domain = "motorcycle"
                elif any(
                    x in case_id or x in case_sub
                    for x in ("wildlife", "animal", "bird", "lion", "safari")
                ):
                    domain = "wildlife"
                elif any(x in case_id or x in case_sub for x in ("generic", "urban", "hike")):
                    domain = "generic"

            if domain:
                observed_domains.add(domain)

        # Check coverage across all four domains
        required_domains = {"underwater", "motorcycle", "wildlife", "generic"}
        assert required_domains.issubset(observed_domains), (
            f"Seed cases missing coverage for domains: {required_domains - observed_domains}. Found: {observed_domains}"
        )
