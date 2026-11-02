"""Adversarial stress-testing suite by Challenger It2-2 for Issue #48 Stage 2.

Covers:
1. Multi-word keyword matching and whole-phrase boundary semantics in _keyword_matches.
2. Hyphenated, tabbed, multi-space, and punctuated phrase boundaries in resolve_template_for_subject.
3. Strict scan isolation invariants ensuring candidate prompt files are never loaded during scanning.
4. Cross-domain template resolution isolation and resilience.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from scripts import dive_clip
from utils import models
from utils.prompt_templates import (
    GENERIC_HIGHLIGHT_TEMPLATE,
    _keyword_matches,
    create_prompt_candidate_for_generic,
    infer_generic_context,
    resolve_template_for_subject,
)

# Dynamically import scripts/02_scan_footage.py
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_scan_spec = importlib.util.spec_from_file_location(
    "scripts_02_scan_footage", _SCRIPTS_DIR / "02_scan_footage.py"
)
if _scan_spec is None or _scan_spec.loader is None:
    raise ImportError("Unable to load scripts/02_scan_footage.py")
scan = importlib.util.module_from_spec(_scan_spec)
_scan_spec.loader.exec_module(scan)


class TestAdversarialMultiWordKeywords:
    """Stress-test _keyword_matches and resolve_template_for_subject on multi-word keywords."""

    def test_multi_word_exact_and_spaced_matching(self) -> None:
        """Verify multi-word keywords match with single space, multiple spaces, tabs, and newlines."""
        multi_words = [
            ("mountain road", "riding on a scenic mountain road"),
            ("mountain road", "mountain   road"),
            ("mountain road", "mountain\troad"),
            ("mountain road", "mountain\nroad"),
            ("mountain road", "mountain-road"),
            ("open road", "cruising the open road"),
            ("open road", "open   road"),
            ("open road", "open-road"),
            ("open road", "open\troad"),
            ("scenic lookout", "stopped at scenic lookout"),
            ("scenic lookout", "scenic-lookout"),
            ("scenic lookout", "scenic   lookout"),
            ("hunting prey", "lion hunting prey in the grass"),
            ("hunting prey", "hunting-prey"),
            ("bird of prey", "soaring bird of prey"),
            ("bird of prey", "bird-of-prey"),
            ("bird of prey", "bird   of   prey"),
            ("bird of prey", "bird\tof\tprey"),
        ]
        for kw, text in multi_words:
            assert _keyword_matches(kw, text) is True, (
                f"_keyword_matches('{kw}', '{text}') expected True but got False"
            )

    def test_multi_word_partial_substring_rejection(self) -> None:
        """Multi-word phrases should not match when words are sub-parts of larger words."""
        rejections = [
            ("mountain road", "catmountain road"),
            ("mountain road", "mountain roadmap"),
            ("open road", "reopen road"),
            ("open road", "open roadway"),
            ("scenic lookout", "unscenic lookout"),
            ("scenic lookout", "scenic lookouts"),
            ("hunting prey", "prehunting prey"),
            ("hunting prey", "hunting preyed"),
            ("bird of prey", "songbird of prey"),
            ("bird of prey", "bird of preying"),
        ]
        for kw, text in rejections:
            assert _keyword_matches(kw, text) is False, (
                f"_keyword_matches('{kw}', '{text}') expected False but got True"
            )

    def test_multi_word_in_motorcycle_domain(self) -> None:
        """'open road' in MOTO_SCENIC_TEMPLATE routes accurately without substring collisions."""
        # Exact and variations
        assert (
            resolve_template_for_subject(subject="open road cruising", domain="motorcycle").name
            == "moto_scenic"
        )
        assert (
            resolve_template_for_subject(subject="open-road cruising", domain="motorcycle").name
            == "moto_scenic"
        )
        assert (
            resolve_template_for_subject(subject="open\troad", domain="motorcycle").name
            == "moto_scenic"
        )
        # Substring in larger word should NOT trigger moto_scenic
        # 'reopen road' has no matching keywords -> falls back to broad_motorcycle
        assert (
            resolve_template_for_subject(subject="reopen road", domain="motorcycle").name
            == "broad_motorcycle"
        )
        assert (
            resolve_template_for_subject(subject="open roadway", domain="motorcycle").name
            == "broad_motorcycle"
        )


class TestAdversarialPunctuationAndHyphenation:
    """Stress-test hyphenated, punctuated, and noisy strings."""

    def test_hyphenated_action_and_animal_separation(self) -> None:
        """Hyphens act as word boundaries: 'lion-hunting' routes to action, 'mountain-lion' to animal."""
        # 'lion-hunting' -> 'hunting' is an action keyword -> wildlife_action
        assert (
            resolve_template_for_subject(subject="lion-hunting", domain="wildlife").name
            == "wildlife_action"
        )
        # 'lion - hunting' -> wildlife_action
        assert (
            resolve_template_for_subject(subject="lion - hunting", domain="wildlife").name
            == "wildlife_action"
        )
        # 'lion-chase' -> wildlife_action
        assert (
            resolve_template_for_subject(subject="lion-chase", domain="wildlife").name
            == "wildlife_action"
        )
        # 'mountain-lion' -> 'lion' is animal keyword, no action keyword -> wildlife_animal
        assert (
            resolve_template_for_subject(subject="mountain-lion", domain="wildlife").name
            == "wildlife_animal"
        )

    def test_comma_and_punctuation_boundaries(self) -> None:
        """Punctuation like commas, brackets, periods act as boundaries."""
        # 'wolf, howling' -> wolf matches wildlife_animal; howling does not match owl
        assert (
            resolve_template_for_subject(subject="wolf, howling", domain="wildlife").name
            == "wildlife_animal"
        )
        assert (
            resolve_template_for_subject(subject="howling, wolf!", domain="wildlife").name
            == "wildlife_animal"
        )
        assert (
            resolve_template_for_subject(subject="(bear) - growling...", domain="wildlife").name
            == "wildlife_animal"
        )
        assert (
            resolve_template_for_subject(subject="lion @ water bowl", domain="wildlife").name
            == "wildlife_animal"
        )

    def test_clean_visor_non_collision_with_lean(self) -> None:
        """'clean-visor' contains substring 'lean', but must not match moto_riding."""
        # Motorcycle domain
        assert (
            resolve_template_for_subject(subject="clean-visor", domain="motorcycle").name
            == "broad_motorcycle"
        )
        assert (
            resolve_template_for_subject(subject="clean visor", domain="motorcycle").name
            == "broad_motorcycle"
        )
        assert (
            resolve_template_for_subject(subject="cleaning visor", domain="motorcycle").name
            == "broad_motorcycle"
        )
        assert (
            resolve_template_for_subject(subject="cleanly polished", domain="motorcycle").name
            == "broad_motorcycle"
        )

        # Genuine 'lean' keywords must route to moto_riding
        assert (
            resolve_template_for_subject(subject="deep lean angle", domain="motorcycle").name
            == "moto_riding"
        )
        assert (
            resolve_template_for_subject(subject="aggressive lean-in", domain="motorcycle").name
            == "moto_riding"
        )
        assert (
            resolve_template_for_subject(subject="lean, fast corner", domain="motorcycle").name
            == "moto_riding"
        )

    def test_underwater_hyphen_and_punctuation(self) -> None:
        """Ensure underwater domain handles hyphens and punctuation cleanly."""
        assert (
            resolve_template_for_subject(subject="sea-turtle swimming", domain="underwater").name
            == "pelagics"
        )
        assert (
            resolve_template_for_subject(subject="manta-ray glide", domain="underwater").name
            == "pelagics"
        )
        assert (
            resolve_template_for_subject(
                subject="tiny [nudibranch] macro", domain="underwater"
            ).name
            == "macro"
        )
        assert (
            resolve_template_for_subject(subject="diver / buddy team", domain="underwater").name
            == "diver"
        )


class TestAdversarialScanIsolation:
    """Stress-test scan isolation from unverified candidate prompts."""

    def test_scan_registry_never_loads_candidate_files(self, tmp_path: Path) -> None:
        """Simulate malicious/unverified candidate files and prove scan registry ignores them."""
        # Create unverified candidate file in output directory
        fake_candidate = {
            "name": "malicious_injected_candidate",
            "domain": "generic",
            "coarse_prompt": "UNVERIFIED OVERRIDE",
            "fine_prompt": "UNVERIFIED OVERRIDE",
            "recommended_threshold": 0.99,
        }
        cand_file = tmp_path / "prompt_candidates.json"
        cand_file.write_text(json.dumps([fake_candidate]), encoding="utf-8")

        # Load scan templates with empty config or standard config
        scan_reg = scan.load_custom_templates({})

        # 1. Candidate template slug must NOT be in scan registry
        for dom_cfg in scan_reg.values():
            for tpl in dom_cfg.templates:
                assert tpl.name != "malicious_injected_candidate"
            assert dom_cfg.fallback_template.name != "malicious_injected_candidate"

        # 2. Template resolution in scan must ignore candidate file
        res = scan.resolve_template_for_subject(
            subject="malicious_injected_candidate",
            domain="generic",
            registry=scan_reg,
        )
        assert res.name == GENERIC_HIGHLIGHT_TEMPLATE.name
        assert "UNVERIFIED OVERRIDE" not in res.fine_prompt

    def test_scan_code_has_no_candidate_file_references(self) -> None:
        """Static assertion that scripts/02_scan_footage.py does not read candidate files."""
        scan_source = Path("scripts/02_scan_footage.py").read_text(encoding="utf-8")
        assert "prompt_candidates.json" not in scan_source
        assert "specialize_generic" not in scan_source
        assert "create_prompt_candidate_for_generic" not in scan_source

    def test_cli_classify_produces_candidates_without_polluting_scan(self, tmp_path: Path) -> None:
        """dive-clip classify --specialize-generic writes candidates, scan remains unaffected."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Build simulated generic classification
        records = [
            models.ClipClassification(
                clip_id="c1",
                file_name="clip1.mp4",
                file_path=str(input_dir / "clip1.mp4"),
                domain="generic",
                environment="bouldering gym",
                primary_subjects=["climber", "chalk bag"],
            )
        ]
        ctx = infer_generic_context(records)
        cand = create_prompt_candidate_for_generic(ctx)

        cand_output = output_dir / "prompt_candidates.json"
        cand_output.write_text(json.dumps([cand], indent=2), encoding="utf-8")
        assert cand_output.exists()

        # Now verify scan pipeline resolves strictly using built-in generic fallback
        scan_tpl = scan.resolve_template_for_subject(
            subject="climber",
            category="bouldering_gym",
            domain="generic",
        )
        assert scan_tpl.name == GENERIC_HIGHLIGHT_TEMPLATE.name
        assert scan_tpl.fine_prompt != cand["fine_prompt"]

    def test_cli_scan_subcommand_rejects_candidate_flags(self) -> None:
        """dive-clip scan CLI parser does not have candidate generation flags."""
        parser = dive_clip.build_parser()
        # Parse standard scan arguments
        args = parser.parse_args(["scan", "--input", "./in", "--output", "./out"])
        assert not hasattr(args, "specialize_generic")
        assert not hasattr(args, "candidate_output")


class TestAdversarialCrossDomainResolution:
    """Stress-test keyword resolution when domain is unspecified or ambiguous."""

    def test_cross_domain_fallback_when_domain_none(self) -> None:
        """When domain=None, keyword matching checks across domains."""
        # Wildlife subject
        assert resolve_template_for_subject(subject="tiger").name == "wildlife_animal"
        assert resolve_template_for_subject(subject="hunting gazelle").name == "wildlife_action"
        assert (
            resolve_template_for_subject(subject="eagle attack on fish").name == "wildlife_action"
        )
        assert resolve_template_for_subject(subject="eagle in flight").name == "wildlife_bird"

        # Motorcycle subject
        assert resolve_template_for_subject(subject="open road cruise").name == "moto_scenic"
        assert resolve_template_for_subject(subject="sharp cornering").name == "moto_riding"

        # Underwater subject
        assert resolve_template_for_subject(subject="manta ray").name == "pelagics"
        assert resolve_template_for_subject(subject="tiny seahorse").name == "macro"

    def test_empty_and_whitespace_cross_domain(self) -> None:
        """Empty or unknown subject with domain=None defaults cleanly to underwater broad_marine."""
        assert resolve_template_for_subject(subject="").name == "broad_marine"
        assert resolve_template_for_subject(subject="   ").name == "broad_marine"
        assert resolve_template_for_subject(subject="completely_unknown_xyz").name == "broad_marine"
