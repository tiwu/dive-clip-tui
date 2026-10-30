"""Tests for utils/prompt_templates.py."""

from __future__ import annotations

from utils.prompt_templates import (
    ALL_TEMPLATES,
    MACRO_TEMPLATE,
    PELAGICS_TEMPLATE,
    DIVER_TEMPLATE,
    PromptTemplate,
    get_template_for_keywords,
)


def test_prompt_template_creation() -> None:
    """PromptTemplate should be constructable with all fields."""
    template = PromptTemplate(
        name="test",
        keywords=["kw1", "kw2"],
        system_prompt="System prompt",
        user_prompt="User prompt",
        color="Blue",
    )
    assert template.name == "test"
    assert template.keywords == ["kw1", "kw2"]
    assert template.system_prompt == "System prompt"
    assert template.user_prompt == "User prompt"
    assert template.color == "Blue"


def test_pelagics_template() -> None:
    """PELAGICS_TEMPLATE should have correct properties."""
    assert PELAGICS_TEMPLATE.name == "pelagics"
    assert PELAGICS_TEMPLATE.color == "Cyan"
    assert "turtle" in PELAGICS_TEMPLATE.keywords
    assert "shark" in PELAGICS_TEMPLATE.keywords
    assert "ray" in PELAGICS_TEMPLATE.keywords
    assert "manta" in PELAGICS_TEMPLATE.keywords


def test_macro_template() -> None:
    """MACRO_TEMPLATE should have correct properties."""
    assert MACRO_TEMPLATE.name == "macro"
    assert MACRO_TEMPLATE.color == "Yellow"
    assert "nudibranch" in MACRO_TEMPLATE.keywords
    assert "shrimp" in MACRO_TEMPLATE.keywords
    assert "coral" in MACRO_TEMPLATE.keywords


def test_diver_template() -> None:
    """DIVER_TEMPLATE should have correct properties."""
    assert DIVER_TEMPLATE.name == "diver"
    assert DIVER_TEMPLATE.color == "Green"
    assert "diver" in DIVER_TEMPLATE.keywords
    assert "composition" in DIVER_TEMPLATE.keywords
    assert "action" in DIVER_TEMPLATE.keywords


def test_all_templates() -> None:
    """ALL_TEMPLATES should contain all three templates."""
    assert len(ALL_TEMPLATES) == 3
    assert PELAGICS_TEMPLATE in ALL_TEMPLATES
    assert MACRO_TEMPLATE in ALL_TEMPLATES
    assert DIVER_TEMPLATE in ALL_TEMPLATES


def test_get_template_for_keywords_pelagics() -> None:
    """get_template_for_keywords should match pelagic keywords."""
    template = get_template_for_keywords(["turtle", "underwater"])
    assert template is not None
    assert template.name == "pelagics"
    assert template.color == "Cyan"


def test_get_template_for_keywords_macro() -> None:
    """get_template_for_keywords should match macro keywords."""
    template = get_template_for_keywords(["nudibranch", "small"])
    assert template is not None
    assert template.name == "macro"
    assert template.color == "Yellow"


def test_get_template_for_keywords_diver() -> None:
    """get_template_for_keywords should match diver keywords."""
    template = get_template_for_keywords(["diver", "composition"])
    assert template is not None
    assert template.name == "diver"
    assert template.color == "Green"


def test_get_template_for_keywords_case_insensitive() -> None:
    """get_template_for_keywords should match case-insensitively."""
    template = get_template_for_keywords(["TURTLE", "Shark"])
    assert template is not None
    assert template.name == "pelagics"


def test_get_template_for_keywords_no_match() -> None:
    """get_template_for_keywords should return None for unknown keywords."""
    template = get_template_for_keywords(["landscape", "mountain", "forest"])
    assert template is None


def test_get_template_for_keywords_empty() -> None:
    """get_template_for_keywords should return None for empty keyword list."""
    template = get_template_for_keywords([])
    assert template is None
