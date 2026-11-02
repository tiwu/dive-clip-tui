"""Adversarial stress-testing suite for Issue #48 Stage 2 domain expansion and prompts.

Created by Challenger 1 (critic / empirical challenger) to stress-test:
1. Template resolution (resolve_template_for_subject) across wildlife edge cases and substring collisions.
2. Universal coarse orientation rejection phrasing across all 4 domains.
3. Standardized 1-10 fine quality rubrics adhering strictly to the 4 dimensions.
4. Generic candidate generation and scan isolation invariants.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from utils.models import ClipClassification
from utils.prompt_templates import (
    ALL_DOMAIN_TEMPLATES,
    BUILTIN_DOMAINS,
    create_prompt_candidate_for_generic,
    infer_generic_context,
    resolve_template_for_subject,
)


class TestAdversarialWildlifeTemplateResolution:
    """Adversarially probe resolve_template_for_subject with noisy, edge-case, and boundary inputs."""

    def test_wildlife_standard_routing(self) -> None:
        """Standard clear subjects route to their specialist templates."""
        assert (
            resolve_template_for_subject(subject="african lion", domain="wildlife").name
            == "wildlife_animal"
        )
        assert (
            resolve_template_for_subject(subject="soaring hawk", domain="wildlife").name
            == "wildlife_bird"
        )
        assert (
            resolve_template_for_subject(subject="cheetah hunt", domain="wildlife").name
            == "wildlife_action"
        )
        assert (
            resolve_template_for_subject(subject="savannah tree landscape", domain="wildlife").name
            == "broad_wildlife"
        )

    def test_wildlife_action_priority_over_animal_and_bird(self) -> None:
        """Action keywords should take precedence when an animal or bird is performing dynamic action."""
        # lion is animal, hunt is action -> should be wildlife_action
        assert (
            resolve_template_for_subject(subject="lion hunting zebra", domain="wildlife").name
            == "wildlife_action"
        )
        # eagle is bird, attack is action -> should be wildlife_action
        assert (
            resolve_template_for_subject(subject="eagle attack on fish", domain="wildlife").name
            == "wildlife_action"
        )
        # leopard is animal, chase is action -> should be wildlife_action
        assert (
            resolve_template_for_subject(subject="leopard chase", domain="wildlife").name
            == "wildlife_action"
        )

    def test_wildlife_precedence_hierarchy(self) -> None:
        """Explicit tags > explicit category > subject keywords > fallback."""
        # Category overrides subject
        res = resolve_template_for_subject(
            subject="lion", category="wildlife_bird", domain="wildlife"
        )
        assert res.name == "wildlife_bird"

        # Tags override category
        res = resolve_template_for_subject(
            subject="lion", category="wildlife_bird", tags=["hunt"], domain="wildlife"
        )
        assert res.name == "wildlife_action"

        # Explicit fallback category
        res = resolve_template_for_subject(
            subject="lion", category="broad_wildlife", domain="wildlife"
        )
        assert res.name == "broad_wildlife"

    def test_wildlife_boundary_and_noisy_inputs(self) -> None:
        """Test empty strings, whitespace, punctuation, uppercase, and unicode."""
        # Empty string should fallback
        assert resolve_template_for_subject(subject="", domain="wildlife").name == "broad_wildlife"
        assert (
            resolve_template_for_subject(subject="   ", domain="wildlife").name == "broad_wildlife"
        )

        # Uppercase and whitespace padding
        assert (
            resolve_template_for_subject(subject="  LION  ", domain="wildlife").name
            == "wildlife_animal"
        )
        assert (
            resolve_template_for_subject(subject="\n\tEAGLE\n", domain="wildlife").name
            == "wildlife_bird"
        )

        # Punctuation and symbols
        assert (
            resolve_template_for_subject(subject="lion!!!", domain="wildlife").name
            == "wildlife_animal"
        )
        assert (
            resolve_template_for_subject(subject="[eagle] - flight (high)", domain="wildlife").name
            == "wildlife_bird"
        )

        # Non-matching noisy text
        assert (
            resolve_template_for_subject(subject="12345 @#$%^&*", domain="wildlife").name
            == "broad_wildlife"
        )

    def test_none_input_resilience(self) -> None:
        """Probe behavior when None is passed for subject or category."""
        # If None is passed, check if resolve_template_for_subject crashes or handles gracefully
        try:
            res = resolve_template_for_subject(subject=None, category=None, domain="wildlife")  # type: ignore[arg-type]
            assert res.name == "broad_wildlife"
        except AttributeError as exc:
            pytest.fail(
                f"resolve_template_for_subject crashed on subject=None with AttributeError: {exc}"
            )

    def test_substring_word_boundary_collisions(self) -> None:
        """Empirical vulnerability probe: substring matching causing cross-template misclassifications.

        Keywords like 'owl', 'kill', and 'nest' match as substrings inside unrelated words
        like 'growling', 'howling', 'howler', 'bowl', 'skilled', and 'honest'.
        Because WILDLIFE_BIRD_TEMPLATE is evaluated before WILDLIFE_ANIMAL_TEMPLATE,
        mammals exhibiting howling, growling, or near a bowl are routed to wildlife_bird.
        """
        # Howling wolf -> contains 'wolf' (animal). But 'howling' contains 'owl' (bird).
        res_wolf = resolve_template_for_subject(subject="howling wolf", domain="wildlife")
        assert res_wolf.name == "wildlife_animal", (
            f"'howling wolf' misclassified as {res_wolf.name} due to 'owl' substring in 'howling'"
        )

        # Growling bear -> contains 'bear' (animal). But 'growling' contains 'owl' (bird).
        res_bear = resolve_template_for_subject(subject="growling bear", domain="wildlife")
        assert res_bear.name == "wildlife_animal", (
            f"'growling bear' misclassified as {res_bear.name} due to 'owl' substring in 'growling'"
        )

        # Howler monkey -> contains 'monkey' (animal). But 'howler' contains 'owl' (bird).
        res_monkey = resolve_template_for_subject(subject="howler monkey", domain="wildlife")
        assert res_monkey.name == "wildlife_animal", (
            f"'howler monkey' misclassified as {res_monkey.name} due to 'owl' substring in 'howler'"
        )

        # Lion at water bowl -> contains 'lion' (animal). But 'bowl' contains 'owl' (bird).
        res_bowl = resolve_template_for_subject(subject="lion at water bowl", domain="wildlife")
        assert res_bowl.name == "wildlife_animal", (
            f"'lion at water bowl' misclassified as {res_bowl.name} due to 'owl' substring in 'bowl'"
        )


class TestAdversarialCoarseOrientationRejection:
    """Verify that all coarse prompts across all domains strictly mandate rejection of unusable footage."""

    REQUIRED_KEYWORDS: ClassVar[list[str]] = ["accidental", "inverted", "dangling", "unusable"]

    def test_all_builtin_domains_coarse_prompt_keywords(self) -> None:
        """Every built-in domain's coarse_prompt must contain all four required rejection keywords."""
        for domain_name, domain_cfg in BUILTIN_DOMAINS.items():
            cp_lower = domain_cfg.coarse_prompt.lower()
            for kw in self.REQUIRED_KEYWORDS:
                assert kw in cp_lower, (
                    f"Domain '{domain_name}' coarse_prompt missing required rejection keyword '{kw}'"
                )

    def test_all_templates_coarse_prompt_rejection_directives(self) -> None:
        """Every template coarse prompt must explicitly instruct setting detected=false, confidence=0.0, quality_score=1."""
        for tpl in ALL_DOMAIN_TEMPLATES:
            cp_lower = tpl.coarse_prompt.lower()
            for kw in self.REQUIRED_KEYWORDS:
                assert kw in cp_lower, (
                    f"Template '{tpl.name}' ({tpl.domain}) coarse_prompt missing required rejection keyword '{kw}'"
                )
            assert "detected" in cp_lower and "false" in cp_lower, (
                f"Template '{tpl.name}' coarse prompt missing detected: false directive"
            )
            assert "0.0" in cp_lower, (
                f"Template '{tpl.name}' coarse prompt missing confidence: 0.0 directive"
            )
            assert "quality_score" in cp_lower and "1" in cp_lower, (
                f"Template '{tpl.name}' coarse prompt missing quality_score: 1 directive"
            )

    def test_candidate_prompt_generator_coarse_rejection(self) -> None:
        """Synthesized generic candidate coarse prompts must also contain all four required rejection keywords."""
        cand = create_prompt_candidate_for_generic({
            "environment": "park",
            "primary_subjects": ["runner"],
        })
        cp_lower = cand["coarse_prompt"].lower()
        for kw in self.REQUIRED_KEYWORDS:
            assert kw in cp_lower, (
                f"Candidate coarse_prompt missing required rejection keyword '{kw}'"
            )
        assert "0.0" in cp_lower
        assert "quality_score" in cp_lower


class TestAdversarialFineQualityRubrics:
    """Verify that all fine prompts strictly enforce the 4 dimensions and 1-10 scoring hierarchy."""

    def test_all_templates_four_dimensions(self) -> None:
        """Every template must evaluate Lighting, Focus & Clarity, Camera Stability, and Subject Composition."""
        for tpl in ALL_DOMAIN_TEMPLATES:
            fp = tpl.fine_prompt.lower()
            assert "quality evaluation rubric (1-10)" in fp, (
                f"Template '{tpl.name}' missing standard Quality Evaluation Rubric header"
            )
            assert "lighting" in fp or "light" in fp, (
                f"Template '{tpl.name}' missing Lighting dimension"
            )
            assert "focus" in fp, f"Template '{tpl.name}' missing Focus dimension"
            assert "stability" in fp or "steady" in fp or "shake" in fp, (
                f"Template '{tpl.name}' missing Camera Stability dimension"
            )
            assert "composition" in fp or "framing" in fp, (
                f"Template '{tpl.name}' missing Subject Composition dimension"
            )

    def test_all_templates_scoring_scale_tiers(self) -> None:
        """Every template fine prompt must define 9-10 (Exceptional), 7-8 (Strong Highlight), 5-6 (Acceptable), 1-4 (Reject)."""
        for tpl in ALL_DOMAIN_TEMPLATES:
            fp = tpl.fine_prompt
            assert "9-10" in fp, f"Template '{tpl.name}' missing 9-10 tier"
            assert "7-8" in fp, f"Template '{tpl.name}' missing 7-8 tier"
            assert "5-6" in fp, f"Template '{tpl.name}' missing 5-6 tier"
            assert "1-4" in fp, f"Template '{tpl.name}' missing 1-4 tier"
            assert "Reject" in fp, f"Template '{tpl.name}' missing Reject label on 1-4 tier"

    def test_candidate_prompt_fine_rubric_adherence(self) -> None:
        """Synthesized candidate fine prompt must adhere to the 4 dimensions and scoring scale."""
        cand = create_prompt_candidate_for_generic({
            "environment": "canyon",
            "primary_subjects": ["climber"],
        })
        fp = cand["fine_prompt"]
        assert "QUALITY EVALUATION RUBRIC (1-10)" in fp
        assert "Lighting" in fp
        assert "Focus" in fp
        assert "Stability" in fp
        assert "Composition" in fp
        assert "9-10" in fp
        assert "1-4" in fp
        assert "Reject" in fp


class TestAdversarialGenericInferenceAndScanIsolation:
    """Stress-test generic context aggregation, candidate synthesis, and scan isolation."""

    def test_infer_generic_context_empty_and_corrupt_records(self) -> None:
        """infer_generic_context should safely handle empty lists, None, and records without expected fields."""
        ctx_none = infer_generic_context(None)
        assert ctx_none["environment"] == ""
        assert ctx_none["primary_subjects"] == []
        assert ctx_none["template_slug"] == "generic_highlight"

        ctx_corrupt = infer_generic_context([
            {"corrupt": True},
            ClipClassification(clip_id="x", file_name="f.mp4", file_path="/p", domain="generic"),
        ])
        assert isinstance(ctx_corrupt["primary_subjects"], list)
        assert isinstance(ctx_corrupt["environment"], str)

    def test_infer_generic_context_frequency_sorting(self) -> None:
        """Subjects must be ordered by descending frequency."""
        records = [
            {"domain": "generic", "environment": "street", "primary_subjects": ["car", "bicycle"]},
            {
                "domain": "generic",
                "environment": "street",
                "primary_subjects": ["bicycle", "pedestrian"],
            },
            {"domain": "generic", "environment": "park", "primary_subjects": ["bicycle"]},
        ]
        ctx = infer_generic_context(records)
        assert ctx["environment"] == "street"
        assert ctx["primary_subjects"][0] == "bicycle"  # frequency 3
        assert "car" in ctx["primary_subjects"]
        assert "pedestrian" in ctx["primary_subjects"]

    def test_scan_isolation_invariant(self) -> None:
        """Verify dive-clip scan and scan_footage do not read candidate prompt files."""
        # Check source code of scripts/02_scan_footage.py
        with open("scripts/02_scan_footage.py", encoding="utf-8") as f:
            content = f.read()
        assert "prompt_candidates.json" not in content, (
            "scripts/02_scan_footage.py must not reference prompt_candidates.json"
        )


class TestAdversarialIteration2Verification:
    """Stress-test Worker It2 fixes for resolve_template_for_subject (Challenger It2-1)."""

    @pytest.mark.parametrize("subject", ["", "   ", None])
    @pytest.mark.parametrize("category", ["", "   ", None])
    def test_subject_and_category_empty_none_permutations(
        self, subject: str | None, category: str | None
    ) -> None:
        """Permutations of empty, whitespace, and None for subject and category."""
        tpl_wildlife = resolve_template_for_subject(
            subject=subject,
            category=category,
            domain="wildlife",  # type: ignore[arg-type]
        )
        assert tpl_wildlife.name == "broad_wildlife"

        tpl_underwater = resolve_template_for_subject(
            subject=subject,
            category=category,
            domain="underwater",  # type: ignore[arg-type]
        )
        assert tpl_underwater.name == "broad_marine"

        tpl_motorcycle = resolve_template_for_subject(
            subject=subject,
            category=category,
            domain="motorcycle",  # type: ignore[arg-type]
        )
        assert tpl_motorcycle.name == "broad_motorcycle"

        tpl_generic = resolve_template_for_subject(
            subject=subject,
            category=category,
            domain="generic",  # type: ignore[arg-type]
        )
        assert tpl_generic.name == "generic_highlight"

        tpl_none_domain = resolve_template_for_subject(
            subject=subject,
            category=category,
            domain=None,  # type: ignore[arg-type]
        )
        assert tpl_none_domain.name == "broad_marine"

    @pytest.mark.parametrize(
        ("subject", "expected_template"),
        [
            ("howling coyote", "broad_wildlife"),
            ("growling cougar", "broad_wildlife"),
            ("screech owl", "wildlife_bird"),
            ("barn owl", "wildlife_bird"),
            ("great horned owl", "wildlife_bird"),
            ("water bowl", "broad_wildlife"),
            ("fish eagle", "wildlife_bird"),
            ("hunting leopard", "wildlife_action"),
        ],
    )
    def test_dispatch_edge_case_keywords(self, subject: str, expected_template: str) -> None:
        """Verify explicit edge cases from dispatch instructions."""
        tpl = resolve_template_for_subject(subject=subject, domain="wildlife")
        assert tpl.name == expected_template

    @pytest.mark.parametrize(
        "subject",
        ["lion", "wolf", "bear", "monkey"],
    )
    def test_wildlife_animals_resolve_to_animal(self, subject: str) -> None:
        """Lions, wolves, bears, monkeys must resolve to wildlife_animal."""
        tpl = resolve_template_for_subject(subject=subject, domain="wildlife")
        assert tpl.name == "wildlife_animal"

    @pytest.mark.parametrize(
        "subject",
        [
            "hunting lion",
            "lion hunting zebra",
            "wolf attack",
            "wolf attack on deer",
            "bear fighting salmon",
            "monkey chase through trees",
        ],
    )
    def test_wildlife_animals_acting_resolve_to_action(self, subject: str) -> None:
        """Lions, wolves, bears, monkeys performing action must resolve to wildlife_action."""
        tpl = resolve_template_for_subject(subject=subject, domain="wildlife")
        assert tpl.name == "wildlife_action"

    @pytest.mark.parametrize(
        "subject",
        [
            "screech owl",
            "barn owl",
            "great horned owl",
            "snowy owl",
            "bald eagle",
            "golden eagle",
            "fish eagle",
            "perched eagle",
            "hawk in flight",
            "peregrine falcon",
        ],
    )
    def test_owls_and_eagles_resolve_to_bird(self, subject: str) -> None:
        """Owls, eagles, hawks, falcons must resolve to wildlife_bird."""
        tpl = resolve_template_for_subject(subject=subject, domain="wildlife")
        assert tpl.name == "wildlife_bird"

    def test_non_string_types_resilience(self) -> None:
        """Ensure non-string types do not raise unhandled exceptions."""
        tpl1 = resolve_template_for_subject(subject=12345, category=False, domain="wildlife")  # type: ignore[arg-type]
        assert tpl1.name == "broad_wildlife"

        tpl2 = resolve_template_for_subject(subject=["lion"], domain="wildlife")  # type: ignore[arg-type]
        assert tpl2.name == "wildlife_animal"
