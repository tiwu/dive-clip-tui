"""Dynamic prompt templates for Qwen2.5-VL based on keyword categories."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptTemplate:
    name: str
    keywords: list[str]
    system_prompt: str
    user_prompt: str
    color: str


# Cyan: Pelagics / large sea life (turtles, sharks, rays)
PELAGICS_TEMPLATE = PromptTemplate(
    name="pelagics",
    keywords=[
        "turtle",
        "schildpad",
        "shark",
        "haai",
        "ray",
        "rog",
        "pelagic",
        "pelagisch",
        "manta",
    ],
    system_prompt=(
        "You are an underwater video analyst specializing in large sea life."
    ),
    user_prompt=(
        "Analyze this underwater frame. Look for pelagic species: "
        "turtles, sharks, rays, mantas.\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is anything relevant?)\n"
        "- subject: string (species or 'unknown')\n"
        "- confidence: float (0-1)\n"
        "- quality_score: int (1-10, sharpness and composition)\n"
        "- description: string (short description of the action)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    color="Cyan",
)

# Yellow: Macro (nudibranchs, shrimp, corals)
MACRO_TEMPLATE = PromptTemplate(
    name="macro",
    keywords=[
        "macro",
        "nudibranch",
        "naaktslak",
        "shrimp",
        "garnal",
        "coral",
        "koraal",
        "small",
        "klein",
    ],
    system_prompt=(
        "You are an underwater video analyst specializing in macro underwater life."
    ),
    user_prompt=(
        "Analyze this underwater frame for macro subjects: nudibranchs, "
        "shrimp, corals, small sea creatures.\n"
        "Return a JSON object with:\n"
        "- detected: boolean\n"
        "- subject: string\n"
        "- confidence: float (0-1)\n"
        "- quality_score: int (1-10)\n"
        "- description: string\n"
        "- timestamp: float\n"
    ),
    color="Yellow",
)

# Green: Diver action / Good composition
DIVER_TEMPLATE = PromptTemplate(
    name="diver",
    keywords=[
        "diver",
        "duiker",
        "composition",
        "compositie",
        "action",
        "actie",
        "person",
        "people",
        "mensen",
    ],
    system_prompt=(
        "You are an underwater video analyst specializing in diver action "
        "and composition."
    ),
    user_prompt=(
        "Analyze this underwater frame for diver action and good composition.\n"
        "Return a JSON object with:\n"
        "- detected: boolean\n"
        "- subject: string\n"
        "- confidence: float (0-1)\n"
        "- quality_score: int (1-10)\n"
        "- description: string\n"
        "- timestamp: float\n"
    ),
    color="Green",
)


ALL_TEMPLATES = [PELAGICS_TEMPLATE, MACRO_TEMPLATE, DIVER_TEMPLATE]


def get_template_for_keywords(keywords: list[str]) -> PromptTemplate | None:
    """Find the best matching prompt template for a list of keywords."""
    keywords_lower = [k.lower() for k in keywords]
    for template in ALL_TEMPLATES:
        for kw in template.keywords:
            if kw.lower() in keywords_lower:
                return template
    return None
