"""Dynamic prompt templates and domain registry for multi-stage VLM video analysis."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from utils.models import VALID_STORY_ROLES

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    name: str
    keywords: list[str]
    system_prompt: str
    coarse_prompt: str = ""
    fine_prompt: str = ""
    color: str = ""
    domain: str = "underwater"

    def __init__(
        self,
        name: str,
        keywords: list[str],
        system_prompt: str,
        coarse_prompt: str = "",
        fine_prompt: str = "",
        color: str = "",
        domain: str = "underwater",
        user_prompt: str | None = None,
    ) -> None:
        self.name = name
        self.keywords = keywords
        self.system_prompt = system_prompt
        self.color = color
        self.domain = domain
        if user_prompt is not None:
            if not fine_prompt:
                fine_prompt = user_prompt
            if not coarse_prompt:
                coarse_prompt = user_prompt
        self.coarse_prompt = coarse_prompt
        self.fine_prompt = fine_prompt

    @property
    def user_prompt(self) -> str:
        """Backward-compatible alias for fine_prompt."""
        return self.fine_prompt

    @user_prompt.setter
    def user_prompt(self, value: str) -> None:
        self.fine_prompt = value


@dataclass
class DomainConfig:
    """Configuration for a video domain (e.g. underwater, motorcycle, generic)."""

    name: str
    keywords: list[str]
    description: str
    coarse_prompt: str
    fallback_template: PromptTemplate
    templates: list[PromptTemplate] = field(default_factory=list)


# =============================================================================
# Stage 0: Video Domain Classification Prompt
# =============================================================================

VIDEO_CLASSIFICATION_SYSTEM_PROMPT = (
    "You are an expert video analyst specializing in action camera and smartphone "
    "footage classification and technical quality assessment."
)

VIDEO_CLASSIFICATION_PROMPT = (
    "Analyze this video frame from an action camera or smartphone. Your goal is to classify "
    "the domain and rigorously assess the technical usability of the footage for downstream processing.\n\n"
    "Target Domains:\n"
    "- 'underwater': Scuba diving, snorkeling, free diving, marine life, coral reefs, "
    "wrecks, aquatic environments.\n"
    "- 'motorcycle': Motorbike, scooter, riding, cockpit/handlebar POV, pavement, track, "
    "curves, group rides.\n"
    "- 'wildlife': Terrestrial animals, safari game, birds, predators, reptiles, tracking, "
    "natural habitats.\n"
    "- 'generic': Any other domain not listed above. If selected, you must provide a specific "
    "categorization in the `suggested_domain` field.\n Don't limit your categorization to the listed domains.\n"
    "Usability & Quality Criteria:\n"
    "Evaluate the frame for lighting, stability (indicated by motion blur), and clear framing.\n"
    "Set 'usable': false and 'quality_score': 1-3 if ANY of these apply:\n"
    "- Camera is in a bag/pocket, lens is covered (e.g., by fingers/dirt), or pointing at empty darkness/floor.\n"
    "- Extreme motion blur indicating severe camera shake, making subjects unrecognizable.\n"
    "- Severe overexposure or underexposure.\n"
    "- The camera is clearly tumbling, upside-down (unintentionally), or dangling chaotically.\n\n"
    "Otherwise, set 'usable': true and assign a 'quality_score' (4-10) based on sharpness, "
    "lighting balance, and framing.\n\n"
    "Return ONLY a valid JSON object with the exact following structure:\n"
    "{\n"
    '  "domain": "string (strictly ONE of: underwater | motorcycle | wildlife | generic)",\n'
    '  "suggested_domain": "string (required when domain is \'generic\', suggest a specific category like '
    "'mountain_biking', 'skiing', 'cityscape', or what you see.)\",\n"
    '  "confidence": "float (0.0 to 1.0)",\n'
    '  "usable": "boolean",\n'
    "  \"reject_reasons\": [\"list of strings (e.g., 'heavy motion blur', 'underexposed', 'lens covered') "
    'or empty list if usable"],\n'
    '  "quality_score": "integer (1-10)",\n'
    '  "technical_details": {\n'
    '    "lighting": "string (good | overexposed | underexposed | harsh shadows)",\n'
    '    "sharpness": "string (sharp | slight blur | heavy motion blur)"\n'
    "  },\n"
    "  \"environment\": \"string (brief specific setting, e.g., 'coral reef wall', 'mountain pass road')\",\n"
    '  "primary_subjects": ["list of strings (main visible elements)"],\n'
    '  "scene_description": "string (a concise 1-2 sentence visual summary of the action and context)"\n'
    "}\n"
)


def build_classification_prompt(
    domains: list[str] | dict[str, DomainConfig] | None = None,
) -> str:
    """Build a classification prompt listing active domains."""
    if not domains:
        return VIDEO_CLASSIFICATION_PROMPT

    domain_names: list[str] = list(domains.keys()) if isinstance(domains, dict) else list(domains)
    if set(domain_names) == {"underwater", "motorcycle", "wildlife", "generic"} or set(
        domain_names
    ) == set(BUILTIN_DOMAINS.keys()):
        return VIDEO_CLASSIFICATION_PROMPT

    lines = [
        "Analyze this video frame and classify the primary setting, domain, and footage usability.\n",
        "Target domains:",
    ]
    for d in domain_names:
        if isinstance(domains, dict) and d in domains:
            lines.append(f"- '{d}': {domains[d].description}")
        else:
            lines.append(f"- '{d}'")

    valid_domains = ", ".join(f"'{d}'" for d in domain_names)
    lines.extend([
        "\nUsability & Camera State:",
        "- If the camera is inside a bag or pocket, lens is covered/black, upside down, tumbling, "
        "dangling on a clip/lanyard, or showing only floor/empty darkness with no scene, "
        "set 'usable': false and 'quality_score': 1.",
        "- Otherwise set 'usable': true with a quality_score from 4 to 10.\n",
        "Return a JSON object with:",
        f"- domain: string (one of {valid_domains})",
        "- confidence: float (0.0 to 1.0)",
        "- usable: boolean (true if footage is normal/usable; false if in-bag, covered, upside-down, or dangling)",
        "- quality_score: integer (1-10; 1-3 for in-bag/dangling/chaotic, 4+ for usable footage)",
        "- environment: string (brief note on setting)",
        "- primary_subjects: list of strings (main visible subjects/elements)",
    ])
    return "\n".join(lines)


def normalize_vlm_frame_response(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Sanitize and normalize raw VLM frame response to the locked schema."""
    if not isinstance(raw, dict):
        return {
            "detected": False,
            "subject": "none",
            "category": "none",
            "story_role": "none",
            "confidence": 0.0,
            "quality_score": 0,
            "reject_reasons": ["no valid response"],
            "description": "",
            "timestamp": 0.0,
        }

    raw_detected = bool(raw.get("detected", False))
    raw_subject = str(raw.get("subject", "") or "").strip()

    raw_reasons = raw.get("reject_reasons", [])
    if isinstance(raw_reasons, list):
        reject_reasons = [
            str(r).strip()
            for r in raw_reasons
            if str(r).strip()
            and str(r).strip().lower() not in {"none", "n/a", "no", "nil", "empty", "[]"}
        ]
    elif isinstance(raw_reasons, str) and raw_reasons.strip():
        val = raw_reasons.strip()
        reject_reasons = (
            [val] if val.lower() not in {"none", "n/a", "no", "nil", "empty", "[]"} else []
        )
    else:
        reject_reasons = []

    try:
        raw_conf = float(raw.get("confidence", 0.0) or 0.0)
        confidence = max(0.0, min(1.0, raw_conf))
    except ValueError, TypeError:
        confidence = 0.0

    try:
        raw_q = int(raw.get("quality_score", 0) or 0)
        raw_quality = max(0, min(10, raw_q))
    except ValueError, TypeError:
        raw_quality = 0

    # Defensive recovery: if VLM identified a clear subject with confidence >= 0.5,
    # quality >= 4, and no reject reasons, but returned detected=False, treat as detected.
    has_valid_subject = bool(
        raw_subject and raw_subject.lower() not in {"none", "unknown", "no", "false", ""}
    )
    if (
        not raw_detected
        and has_valid_subject
        and not reject_reasons
        and confidence >= 0.5
        and raw_quality >= 4
    ):
        detected = True
    else:
        detected = raw_detected

    subject = raw_subject if raw_subject else ("unknown" if detected else "none")

    raw_category = str(raw.get("category", "") or "").strip()
    category = raw_category if raw_category else ("generic" if detected else "none")

    raw_story_role = str(raw.get("story_role", "") or "").strip().lower()
    if raw_story_role in VALID_STORY_ROLES:
        story_role = raw_story_role
    else:
        story_role = "hero_subject" if detected else "none"

    quality_score = max(1, raw_quality) if detected else raw_quality
    description = str(raw.get("description", "") or "").strip()

    try:
        timestamp = float(raw.get("timestamp", 0.0) or 0.0)
    except ValueError, TypeError:
        timestamp = 0.0

    return {
        "detected": detected,
        "subject": subject,
        "category": category,
        "story_role": story_role,
        "confidence": confidence,
        "quality_score": quality_score,
        "reject_reasons": reject_reasons,
        "description": description,
        "timestamp": timestamp,
    }


# =============================================================================
# Domain: Underwater / Dive Footage
# =============================================================================

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
        "whale",
        "walvis",
        "dolphin",
        "dolfijn",
    ],
    system_prompt="You are an underwater video analyst specializing in large sea life.",
    coarse_prompt=(
        "Analyze this underwater frame for candidate screening.\n"
        "Target subjects: pelagic species (turtles, sharks, rays, mantas, whales, dolphins) or compelling story moments (establishing underwater vistas, boat/gear prep, water entry, buddy interactions, unexpected encounters).\n\n"
        "SCREENING INSTRUCTIONS:\n"
        "- If target pelagic species, marine life, or story moments are visible in usable footage, mark 'detected': true and assign a quality_score (4-10).\n"
        "- You MUST reject accidental, inverted, dangling, or unusable footage (inside bag/pocket, lens completely blocked/covered, inverted/dangling chaotically). If no target subjects are present or the footage is unusable, reject the frame.\n"
        "- If rejected as accidental, inverted, dangling, or unusable, set 'detected': false, 'confidence': 0.0, 'quality_score': 1, and list specific reject_reasons.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (true if target pelagic or story moment is visible in usable footage, false otherwise)\n"
        "- subject: string (species name, action, or 'none')\n"
        "- category: string ('pelagics')\n"
        "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10; 1-3 for unusable/chaotic, 4+ for usable footage)\n"
        "- reject_reasons: list of strings (list issues ONLY if detected is false; empty list [] if detected is true)\n"
        "- description: string (short note on subject or rejection reason)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    fine_prompt=(
        "Analyze this underwater frame for pelagic species (turtles, sharks, rays, mantas) or story moments.\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both subject presence and shot execution:\n"
        "1. Lighting: Well-lit with natural or video light, good contrast, minimal backscatter/murk.\n"
        "2. Focus & Clarity: Pelagic subject is in sharp focus, not blurry or soft.\n"
        "3. Camera Stability: Smooth camera movement without excessive shake, rapid tumbling, or heavy motion blur.\n"
        "4. Subject Composition: Upright, intentional composition centering the animal.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Crisp focus on animal, clear water/lighting, smooth stable camera.\n"
        "- 7-8 (Strong Highlight): Good lighting, animal in clear focus, minimal camera shake.\n"
        "- 5-6 (Acceptable): Passable lighting, slight blur or distance, minor camera movement.\n"
        "- 1-4 (Reject): Murky/dark, out of focus, heavy motion blur, upside down, or dangling.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is a pelagic species or story moment clearly visible?)\n"
        "- subject: string (species or 'unknown')\n"
        "- category: string ('pelagics')\n"
        "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
        "- reject_reasons: list of strings (or empty list if detected is true)\n"
        "- description: string (details on species, behavior, lighting, and stability)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    color="Cyan",
    domain="underwater",
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
        "frogfish",
        "seahorse",
        "zeepaardje",
        "critter",
    ],
    system_prompt="You are an underwater video analyst specializing in macro underwater life.",
    coarse_prompt=(
        "Analyze this underwater frame for candidate screening.\n"
        "Target subjects: macro subjects (nudibranchs, shrimp, corals, small sea creatures) or interesting story details.\n\n"
        "SCREENING INSTRUCTIONS:\n"
        "- If macro subjects, corals, critters, or marine life are visible in usable footage, mark 'detected': true and assign a quality_score (4-10).\n"
        "- You MUST reject accidental, inverted, dangling, or unusable footage (inside bag/pocket, lens completely covered/blocked, inverted/dangling chaotically). If no macro subjects are present or the footage is unusable, reject the frame.\n"
        "- If rejected as accidental, inverted, dangling, or unusable, set 'detected': false, 'confidence': 0.0, 'quality_score': 1, and list specific reject_reasons.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (true if macro subject, coral, or marine life is visible in usable footage, false otherwise)\n"
        "- subject: string (species/subject name or 'none')\n"
        "- category: string ('macro')\n"
        "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10; 1-3 for unusable/chaotic, 4+ for usable footage)\n"
        "- reject_reasons: list of strings (list issues ONLY if detected is false; empty list [] if detected is true)\n"
        "- description: string (short note on subject or rejection reason)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    fine_prompt=(
        "Analyze this underwater frame for macro subjects (nudibranchs, shrimp, corals, small creatures) or story details.\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both subject presence and shot execution:\n"
        "1. Lighting: Proper lighting illuminating macro details, balanced exposure, no blown-out highlights.\n"
        "2. Focus & Clarity: Macro subject is in crisp, sharp focus (shallow depth of field placed correctly).\n"
        "3. Camera Stability: Rock-steady camera without shake, surge drift, or motion blur.\n"
        "4. Subject Composition: Upright, clean macro composition.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Razor-sharp focus on subject, beautiful lighting, steady motionless camera.\n"
        "- 7-8 (Strong Highlight): Sharp focus on macro subject, good lighting, minimal camera motion.\n"
        "- 5-6 (Acceptable): Soft focus or slight camera movement, acceptable lighting.\n"
        "- 1-4 (Reject): Blurry, missed focus, murky/dark, severe shake, upside down, or dangling.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is a macro subject or story moment clearly visible?)\n"
        "- subject: string (subject name or 'unknown')\n"
        "- category: string ('macro')\n"
        "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
        "- reject_reasons: list of strings (or empty list if detected is true)\n"
        "- description: string (details on creature, lighting, focus, and stability)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    color="Yellow",
    domain="underwater",
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
        "buddy",
        "swimthrough",
        "wreck",
        "wrak",
    ],
    system_prompt="You are an underwater video analyst specializing in diver action and composition.",
    coarse_prompt=(
        "Analyze this underwater frame for candidate screening.\n"
        "Target subjects: diver action, buddy interaction, swim-throughs, intentional composition, and water entries/exits.\n\n"
        "SCREENING INSTRUCTIONS:\n"
        "- If intentional diver action, buddy interactions, or water entry/exit is visible in usable footage, mark 'detected': true and assign a quality_score (4-10).\n"
        "- You MUST reject accidental, inverted, dangling, or unusable footage (inside bag/pocket, lens completely covered/blocked, inverted/dangling chaotically). If no diver action is present or the footage is unusable, reject the frame.\n"
        "- If rejected as accidental, inverted, dangling, or unusable, set 'detected': false, 'confidence': 0.0, 'quality_score': 1, and list specific reject_reasons.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (true if diver action or story moment is visible in usable footage, false otherwise)\n"
        "- subject: string (diver activity, story moment, or 'none')\n"
        "- category: string ('diver')\n"
        "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10; 1-3 for unusable/chaotic, 4+ for usable footage)\n"
        "- reject_reasons: list of strings (list issues ONLY if detected is false; empty list [] if detected is true)\n"
        "- description: string (short note on diver action or rejection reason)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    fine_prompt=(
        "Analyze this underwater frame for diver action and composition.\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both subject action and shot execution:\n"
        "1. Lighting: Good visibility of diver and background, balanced exposure.\n"
        "2. Focus & Clarity: Diver and gear in sharp focus, clear facial mask/expressions or silhouette.\n"
        "3. Camera Stability: Smooth camera gliding or steady hold, free of jerky movements or motion blur.\n"
        "4. Subject Composition: Upright, cinematic composition with good rule of thirds or leading lines.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Crisp focus, dramatic lighting/silhouette, smooth stable glide, compelling action.\n"
        "- 7-8 (Strong Highlight): Good lighting, sharp diver, stable camera framing.\n"
        "- 5-6 (Acceptable): Adequate lighting, slight camera wobble or distance.\n"
        "- 1-4 (Reject): Dark/murky, out of focus, erratic camera shake, upside down, or dangling.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is intentional diver action or story moment visible?)\n"
        "- subject: string (e.g. 'diver exploring reef', 'diver entering swimthrough')\n"
        "- category: string ('diver')\n"
        "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
        "- reject_reasons: list of strings (or empty list if detected is true)\n"
        "- description: string (details on action, composition, lighting, and stability)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    color="Green",
    domain="underwater",
)

# Broad marine-life fallback template
BROAD_MARINE_TEMPLATE = PromptTemplate(
    name="broad_marine",
    keywords=[],
    system_prompt=(
        "You are an underwater video analyst specializing in comprehensive marine life identification."
    ),
    coarse_prompt=(
        "Analyze this underwater frame for candidate screening.\n"
        "Target subjects: any marine life: corals, worms, nudibranchs, fish, octopus, moray eels, "
        "sharks, turtles, whales, frogfish, crocodilefish, seahorses, stonefish, diver action, sea creatures, or story moments.\n\n"
        "SCREENING INSTRUCTIONS:\n"
        "- If marine life, coral formations, fish, diver action, or reef setting is visible in usable footage, mark 'detected': true and assign a quality_score (4-10).\n"
        "- You MUST reject accidental, inverted, dangling, or unusable footage (inside bag/pocket, lens completely covered/blocked, inverted/dangling chaotically). If no marine life or action is present, reject the frame.\n"
        "- If rejected as accidental, inverted, dangling, or unusable, set 'detected': false, 'confidence': 0.0, 'quality_score': 1, and list specific reject_reasons.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (true if marine life, corals, or action is visible in usable footage, false otherwise)\n"
        "- subject: string (species name, diver activity, or 'none')\n"
        "- category: string (one of 'pelagics', 'macro', 'diver', 'broad_marine')\n"
        "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10; 1-3 for unusable/chaotic, 4+ for usable footage)\n"
        "- reject_reasons: list of strings (list issues ONLY if detected is false; empty list [] if detected is true)\n"
        "- description: string (short note on subject or rejection reason)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    fine_prompt=(
        "Analyze this underwater frame in detail for marine life: corals, worms, nudibranchs, "
        "fish, octopus, moray eels, sharks, turtles, whales, frogfish, crocodilefish, seahorses, "
        "stonefish, and other sea creatures.\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both subject presence and shot execution:\n"
        "1. Lighting: Proper underwater lighting, natural contrast, clear visibility with minimal backscatter.\n"
        "2. Focus & Clarity: Marine subject is in crisp, sharp focus.\n"
        "3. Camera Stability: Smooth camera tracking or steady hold, free of jerky shake or heavy motion blur.\n"
        "4. Subject Composition: Upright, intentional composition highlighting the subject.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Razor-sharp focus, beautiful lighting, clear water, smooth stable camera.\n"
        "- 7-8 (Strong Highlight): Good lighting, sharp subject, stable camera with minimal motion blur.\n"
        "- 5-6 (Acceptable): Passable lighting, slight softness or distance, mild camera movement.\n"
        "- 1-4 (Reject): Murky/dark, out of focus, severe camera shake, upside down, or dangling.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is marine life clearly visible?)\n"
        "- subject: string (species or 'unknown')\n"
        "- category: string ('broad_marine')\n"
        "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
        "- reject_reasons: list of strings (or empty list if detected is true)\n"
        "- description: string (details on species, behavior, lighting, and camera stability)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    color="Cyan",
    domain="underwater",
)


# =============================================================================
# Domain: Motorcycle Footage
# =============================================================================

MOTORCYCLE_COARSE_PROMPT = (
    "Analyze this motorcycle footage frame for candidate screening.\n"
    "Target subjects: dynamic motorcycle riding (cornering, lean, acceleration, overtaking), "
    "scenic road vistas, twisties, hairpins, fellow riders / group ride action, roadside prep, or trip story moments.\n\n"
    "SCREENING INSTRUCTIONS:\n"
    "- Footage may be landscape (16:9) or vertical (9:16 portrait). A forward view of the road, street, or scenery in vertical framing is normal riding footage, NOT pointed at the ground.\n"
    "- If motorcycle riding action, forward road view, scenic street/highway, rider, or travel story moment is visible in usable footage, mark 'detected': true and assign a quality_score (4-10).\n"
    "- You MUST reject accidental, inverted, dangling, or unusable footage (inside bag/pocket/helmet, lens completely blocked/covered, inverted/dangling chaotically, or completely stationary with no movement). If no riding action or road view is present, reject the frame.\n"
    "- If rejected as accidental, inverted, dangling, or unusable, set 'detected': false, 'confidence': 0.0, 'quality_score': 1, and list specific reject_reasons.\n\n"
    "Return a JSON object with:\n"
    "- detected: boolean (true if riding action, road view, scenic vista, or story moment is visible in usable footage, false otherwise)\n"
    "- subject: string (e.g. 'cornering through hairpin', 'scenic road vista', 'riding with buddy', or 'none')\n"
    "- category: string (one of 'moto_riding', 'moto_scenic', 'moto_group', 'broad_motorcycle')\n"
    "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
    "- confidence: float (0.0 to 1.0)\n"
    "- quality_score: int (1-10; 1-3 for unusable/chaotic, 4+ for usable footage)\n"
    "- reject_reasons: list of strings (list issues ONLY if detected is false; empty list [] if detected is true)\n"
    "- description: string (short note on riding action or rejection reason)\n"
    "- timestamp: float (seconds in the clip)\n"
)

# Orange: Dynamic motorcycle riding (cornering, lean, overtaking)
MOTO_RIDING_TEMPLATE = PromptTemplate(
    name="moto_riding",
    keywords=[
        "riding",
        "rijden",
        "cornering",
        "bocht",
        "lean",
        "twisties",
        "curve",
        "hairpin",
        "overtake",
        "acceleration",
        "track",
        "apex",
        "speed",
        "sport",
    ],
    system_prompt="You are a motorcycle action video analyst specializing in dynamic riding and cornering.",
    coarse_prompt=MOTORCYCLE_COARSE_PROMPT,
    fine_prompt=(
        "Analyze this motorcycle frame for dynamic riding action and cornering technique.\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both riding action and shot execution:\n"
        "1. Lighting: Good natural daylight or ambient road lighting, balanced exposure without blinding glare.\n"
        "2. Focus & Clarity: Clear forward road view, crisp cockpit or rider silhouette, minimal motion smear.\n"
        "3. Camera Stability: Stable horizon or smooth lean without violent engine vibration or camera shake.\n"
        "4. Subject Composition: Upright composition with leading lines following the curve of the road and dynamic riding action.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Dramatic lean angle, razor-sharp view of the apex, rock-steady footage, thrilling speed.\n"
        "- 7-8 (Strong Highlight): Good cornering action, sharp road view, minimal vibration.\n"
        "- 5-6 (Acceptable): Moderate riding action, slight engine buzz or soft focus.\n"
        "- 1-4 (Reject): Pointed at ground/tank, stationary, heavy vibration blur, upside down, or covered.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is compelling motorcycle riding action visible?)\n"
        "- subject: string (e.g. 'deep lean right hairpin', 'accelerating out of curve')\n"
        "- category: string ('moto_riding')\n"
        "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
        "- reject_reasons: list of strings (or empty list if detected is true)\n"
        "- description: string (details on lean angle, speed, road condition, and stability)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    color="Orange",
    domain="motorcycle",
)

# Blue: Scenic road / POV vistas
MOTO_SCENIC_TEMPLATE = PromptTemplate(
    name="moto_scenic",
    keywords=[
        "scenic",
        "mountain",
        "pass",
        "coastal",
        "landscape",
        "vista",
        "sunset",
        "sunrise",
        "bridge",
        "canyon",
        "open road",
        "bergpas",
        "uitzicht",
        "panorama",
    ],
    system_prompt="You are a video analyst specializing in scenic travel and motorcycle route vistas.",
    coarse_prompt=MOTORCYCLE_COARSE_PROMPT,
    fine_prompt=(
        "Analyze this motorcycle frame for scenic landscapes, vistas, and road cinematography.\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both landscape visual appeal and shot execution:\n"
        "1. Scenery & Lighting: Beautiful natural lighting (golden hour, clear skies, dramatic clouds), scenic vista.\n"
        "2. Focus & Clarity: Clean lens/visor without water droplets or glare obstructing the landscape.\n"
        "3. Camera Stability: Smooth cruising motion, level horizon, no erratic jitter.\n"
        "4. Composition: Cinematic rule-of-thirds, road sweeping into the landscape, open vista.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Breathtaking landscape, golden lighting, crystal clear lens, cinematic sweep.\n"
        "- 7-8 (Strong Highlight): Beautiful scenery, good exposure, stable horizon.\n"
        "- 5-6 (Acceptable): Pleasant landscape, slight lens smear or flat lighting.\n"
        "- 1-4 (Reject): Dark, washed out, blocked view, heavy vibration, or upside down.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is a scenic road vista clearly visible?)\n"
        "- subject: string (e.g. 'mountain pass vista', 'coastal road sunset')\n"
        "- category: string ('moto_scenic')\n"
        "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
        "- reject_reasons: list of strings (or empty list if detected is true)\n"
        "- description: string (details on landscape, road framing, lighting, and camera stability)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    color="Blue",
    domain="motorcycle",
)

# Green: Group ride & rider interaction
MOTO_GROUP_TEMPLATE = PromptTemplate(
    name="moto_group",
    keywords=[
        "rider",
        "riders",
        "group",
        "formation",
        "buddy",
        "biker",
        "bikers",
        "convoy",
        "wave",
        "motorrijder",
        "groep",
    ],
    system_prompt="You are a motorcycle video analyst specializing in group riding and rider interactions.",
    coarse_prompt=MOTORCYCLE_COARSE_PROMPT,
    fine_prompt=(
        "Analyze this motorcycle frame for group riding, fellow riders, and rider interaction.\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both rider presence and shot execution:\n"
        "1. Lighting: Clear ambient road lighting, well-exposed riders and road surroundings.\n"
        "2. Focus & Clarity: Crisp focus on fellow riders and bikes, clear formation details.\n"
        "3. Camera Stability: Smooth following distance without erratic braking or jitter.\n"
        "4. Subject Composition: Intentional framing and composition showing the relationship between riders and the road.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Perfectly framed formation, crisp focus on riders, great lighting and stability.\n"
        "- 7-8 (Strong Highlight): Clear view of fellow riders, good spacing and stable camera.\n"
        "- 5-6 (Acceptable): Riders visible but distant or slightly obscured.\n"
        "- 1-4 (Reject): Too distant, out of focus, shaky, or no fellow riders present.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (are fellow riders in frame with intentional composition?)\n"
        "- subject: string (e.g. 'two riders in formation', 'biker wave on pass')\n"
        "- category: string ('moto_group')\n"
        "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
        "- reject_reasons: list of strings (or empty list if detected is true)\n"
        "- description: string (details on riders, formation, and camera stability)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    color="Green",
    domain="motorcycle",
)

# Broad motorcycle fallback template
BROAD_MOTORCYCLE_TEMPLATE = PromptTemplate(
    name="broad_motorcycle",
    keywords=[],
    system_prompt="You are a motorcycle video analyst evaluating motorcycle footage highlights.",
    coarse_prompt=MOTORCYCLE_COARSE_PROMPT,
    fine_prompt=(
        "Analyze this motorcycle frame for noteworthy highlights: riding action, scenic road views, "
        "or group riding.\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both riding highlight and shot execution:\n"
        "1. Lighting: Balanced road lighting and exposure, clear visibility of scene.\n"
        "2. Focus & Clarity: Clear road and subject view, minimal motion smear.\n"
        "3. Camera Stability: Smooth cruising motion without engine vibration shake.\n"
        "4. Subject Composition: Upright, intentional road framing and subject composition.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Dynamic riding or gorgeous scenery, razor-sharp focus, rock-steady hold.\n"
        "- 7-8 (Strong Highlight): Good road view, clear lighting, stable camera.\n"
        "- 5-6 (Acceptable): Adequate riding or view, minor vibration.\n"
        "- 1-4 (Reject): Stationary at light, pointed at ground/tank, blurry, or covered.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is this a highlight moment?)\n"
        "- subject: string (description of highlight or 'none')\n"
        "- category: string ('broad_motorcycle')\n"
        "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
        "- reject_reasons: list of strings (or empty list if detected is true)\n"
        "- description: string (details on action, view, and stability)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    color="Orange",
    domain="motorcycle",
)


# =============================================================================
# Domain: Generic Action / Highlight
# =============================================================================

GENERIC_COARSE_PROMPT = (
    "Analyze this high-resolution video frame for candidate screening and highlight detection. "
    "Your objective is to identify compelling action, a clear central subject, scenic views, or notable events.\n\n"
    "SCREENING INSTRUCTIONS:\n"
    "- If a clear central subject, intentional action, landscape, or story moment is visible in usable footage, mark 'detected': true and assign a quality_score (4-10).\n"
    "- You MUST reject accidental, inverted, dangling, or unusable footage (inside a pocket/bag, lens completely covered, inverted/dangling chaotically, or severe unrecoverable blur). If no highlight is present, reject the frame.\n"
    "- If rejected as accidental, inverted, dangling, or unusable, set 'detected': false, 'confidence': 0.0, 'quality_score': 1, and list specific reject_reasons.\n\n"
    "Return ONLY a valid JSON object with the exact following structure:\n"
    "{\n"
    '  "detected": "boolean (true if an intentional highlight or subject is present in usable footage, false otherwise)",\n'
    '  "subject": "string (specific name of the main subject/action, or \'none\')",\n'
    "  \"category\": \"string (use 'generic_highlight' or a more specific term like 'mountain_biking_highlight')\",\n"
    '  "story_role": "string (establishing | entry_transition | hero_subject | human_moment | unexpected_event | none)",\n'
    '  "confidence": "float (0.0 to 1.0)",\n'
    '  "reject_reasons": ["list of strings (list issues ONLY if detected=false; empty list [] if detected=true)"],\n'
    '  "quality_score": "integer (1-10; 1-3 for unusable/chaotic, 4-6 for average, 7-10 for great composition/action)",\n'
    '  "technical_details": {\n'
    '    "lighting": "string (good | overexposed | underexposed)",\n'
    '    "sharpness": "string (sharp | slight blur | heavy motion blur)",\n'
    '    "framing": "string (well-framed | subject partially cut off | empty scene)"\n'
    "  },\n"
    '  "description": "string (short 1-sentence description of the subject or the reason for rejection)",\n'
    '  "timestamp": "float (seconds in the clip, if provided or estimated, otherwise 0.0)"\n'
    "}\n"
)

GENERIC_HIGHLIGHT_TEMPLATE = PromptTemplate(
    name="generic_highlight",
    keywords=[],
    system_prompt="You are an expert video analyst evaluating general highlight moments for final selection.",
    coarse_prompt=GENERIC_COARSE_PROMPT,
    fine_prompt=(
        "Analyze this high-resolution video frame in detail to determine its final highlight quality.\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both subject presence and shot execution based on these criteria:\n"
        "1. Lighting & Exposure: Well-exposed, balanced contrast, good natural or artificial light.\n"
        "2. Focus & Clarity: Subject is in crisp, sharp focus, not soft or heavily compressed.\n"
        "3. Stability (Motion Blur): Minimal motion blur on the subject and background, indicating a steady camera.\n"
        "4. Composition: Intentional, upright framing. Subject is well-placed and not awkwardly cut off.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Crisp focus on a compelling subject, beautiful lighting, excellent composition, zero blur.\n"
        "- 7-8 (Strong Highlight): Good lighting, sharp subject, well-framed, minimal motion blur.\n"
        "- 5-6 (Acceptable): Adequate lighting, visible subject, but slight softness, minor blur, or average framing.\n"
        "- 1-4 (Reject): Underexposed/overexposed, severe motion blur, upside down, covered lens, or empty scene.\n\n"
        "Return ONLY a valid JSON object with the exact following structure:\n"
        "{\n"
        '  "detected": "boolean (true ONLY if it is a clear highlight scoring 5 or above)",\n'
        '  "subject": "string (specific main subject/action, or \'none\')",\n'
        '  "category": "string (\'generic_highlight\')",\n'
        '  "story_role": "string (establishing | entry_transition | hero_subject | human_moment | unexpected_event | none)",\n'
        '  "confidence": "float (0.0 to 1.0)",\n'
        '  "quality_score": "integer (1-10 based on the rubric above)",\n'
        "  \"reject_reasons\": [\"list of strings (e.g., 'severe motion blur', 'poor composition') or empty list if score is 5+\"],\n"
        '  "technical_details": {\n'
        '    "lighting": "string (exceptional | good | acceptable | poor)",\n'
        '    "sharpness": "string (crisp | slight softness | blurry)",\n'
        '    "composition": "string (well-framed | average | cut-off)"\n'
        "  },\n"
        '  "description": "string (detailed assessment of the action, lighting, focus, and overall aesthetic)",\n'
        '  "timestamp": "float (seconds in the clip, if provided or estimated, otherwise 0.0)"\n'
        "}\n"
    ),
    color="Magenta",
    domain="generic",
)


# =============================================================================
# Domain: Wildlife Footage
# =============================================================================

WILDLIFE_COARSE_PROMPT = (
    "Analyze this wildlife footage frame for candidate screening.\n"
    "Target subjects: wild animals, mammals, safari game, predators, birds, raptors, "
    "dynamic animal behavior, natural habitats, or expedition story moments.\n\n"
    "SCREENING INSTRUCTIONS:\n"
    "- If wild animals, birds, dynamic animal action, or scenic natural habitats are visible in usable footage, mark 'detected': true and assign a quality_score (4-10).\n"
    "- You MUST reject accidental, inverted, dangling, or unusable footage (inside bag/pocket, lens completely covered/blocked, inverted/dangling chaotically). If no wildlife or habitat is present, reject the frame.\n"
    "- If rejected as accidental, inverted, dangling, or unusable, set 'detected': false, 'confidence': 0.0, 'quality_score': 1, and list specific reject_reasons.\n\n"
    "Return a JSON object with:\n"
    "- detected: boolean (true if wildlife subject, action, or habitat is visible in usable footage, false otherwise)\n"
    "- subject: string (species or behavior name, e.g. 'lion pride', 'perched eagle', or 'none')\n"
    "- category: string (one of 'wildlife_animal', 'wildlife_bird', 'wildlife_action', 'broad_wildlife')\n"
    "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
    "- confidence: float (0.0 to 1.0)\n"
    "- quality_score: int (1-10; 1-3 for unusable/chaotic, 4+ for usable footage)\n"
    "- reject_reasons: list of strings (list issues ONLY if detected is false; empty list [] if detected is true)\n"
    "- description: string (short note on wildlife subject or rejection reason)\n"
    "- timestamp: float (seconds in the clip)\n"
)

# Yellow: Wild animals, mammals, predators, safari game
WILDLIFE_ANIMAL_TEMPLATE = PromptTemplate(
    name="wildlife_animal",
    keywords=[
        "animal",
        "mammal",
        "predator",
        "safari",
        "lion",
        "elephant",
        "giraffe",
        "zebra",
        "leopard",
        "cheetah",
        "tiger",
        "bear",
        "deer",
        "monkey",
        "wolf",
        "rhino",
        "hippo",
        "dier",
        "zoogdier",
        "leeuw",
        "olifant",
        "giraf",
        "luipaard",
        "tijger",
        "beer",
        "hert",
        "aap",
        "neushoorn",
        "nijlpaard",
    ],
    system_prompt="You are a wildlife video analyst specializing in wild mammals, predators, and safari animals.",
    coarse_prompt=WILDLIFE_COARSE_PROMPT,
    fine_prompt=(
        "Analyze this wildlife frame for wild mammals, predators, and safari animals.\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both subject presence and shot execution:\n"
        "1. Lighting: Natural exposure, good contrast, clear subject definition against foliage/savannah.\n"
        "2. Focus & Clarity: Crisp focus on animal eyes, head, or coat texture, minimal motion blur.\n"
        "3. Camera Stability: Steady tracking or motionless framing without violent vehicle jitter.\n"
        "4. Subject Composition: Upright composition with rule-of-thirds, clean line of sight.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Razor-sharp focus on animal, beautiful lighting, rock-steady hold, compelling composition.\n"
        "- 7-8 (Strong Highlight): Good lighting, animal in clear focus, minimal camera vibration.\n"
        "- 5-6 (Acceptable): Adequate lighting, slight softness or distant subject, minor camera movement.\n"
        "- 1-4 (Reject): Dark/murky, out of focus, severe shake, accidental, inverted, dangling, or unusable.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is a wild animal clearly visible?)\n"
        "- subject: string (species name or 'unknown')\n"
        "- category: string ('wildlife_animal')\n"
        "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
        "- reject_reasons: list of strings (or empty list if detected is true)\n"
        "- description: string (details on species, behavior, lighting, and camera stability)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    color="Yellow",
    domain="wildlife",
)

# Cyan: Birds, raptors, flight, nesting, perching
WILDLIFE_BIRD_TEMPLATE = PromptTemplate(
    name="wildlife_bird",
    keywords=[
        "bird",
        "vogel",
        "raptor",
        "eagle",
        "hawk",
        "owl",
        "falcon",
        "parrot",
        "flight",
        "vliegen",
        "nest",
        "nesting",
        "perch",
        "heron",
        "crane",
        "seabird",
        "waterfowl",
        "duck",
        "goose",
        "arend",
        "uil",
        "valk",
        "reiger",
    ],
    system_prompt="You are a wildlife video analyst specializing in avian species and birds in flight or perching.",
    coarse_prompt=WILDLIFE_COARSE_PROMPT,
    fine_prompt=(
        "Analyze this wildlife frame for birds, raptors, flight dynamics, and perching.\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both subject presence and shot execution:\n"
        "1. Lighting: Plumage illumination, balanced exposure against bright sky or dense canopy.\n"
        "2. Focus & Clarity: Sharp eye or feather detail, minimal motion blur on wings in flight.\n"
        "3. Camera Stability: Smooth flight tracking or motionless perch lock without erratic shake.\n"
        "4. Subject Composition: Intentional framing with leading space in direction of flight or gaze.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Crisp eye focus, dramatic flight or perch, gorgeous lighting, steady tracking.\n"
        "- 7-8 (Strong Highlight): Good lighting, bird in clear focus, minimal tracking shake.\n"
        "- 5-6 (Acceptable): Passable lighting, small in frame or slight softness, minor camera wobble.\n"
        "- 1-4 (Reject): Underexposed silhouette without detail, blurry, erratic shake, accidental, inverted, dangling, or unusable.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is a bird clearly visible?)\n"
        "- subject: string (species name or 'unknown')\n"
        "- category: string ('wildlife_bird')\n"
        "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
        "- reject_reasons: list of strings (or empty list if detected is true)\n"
        "- description: string (details on species, behavior/flight, lighting, and camera stability)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    color="Cyan",
    domain="wildlife",
)

# Orange: Dynamic animal behaviors (hunting, chasing, feeding, fights)
WILDLIFE_ACTION_TEMPLATE = PromptTemplate(
    name="wildlife_action",
    keywords=[
        "action",
        "actie",
        "hunt",
        "hunting",
        "jacht",
        "jagen",
        "chase",
        "kill",
        "feeding",
        "eating",
        "fight",
        "fighting",
        "mating",
        "running",
        "sprint",
        "predator",
        "predation",
        "attack",
        "pursuit",
    ],
    system_prompt="You are a wildlife video analyst specializing in dynamic animal behaviors and predator-prey interactions.",
    coarse_prompt=WILDLIFE_COARSE_PROMPT,
    fine_prompt=(
        "Analyze this wildlife frame for dynamic animal behavior (hunting, chasing, feeding, fighting).\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both behavioral action and shot execution:\n"
        "1. Lighting: Clear visibility of the action, balanced exposure throughout dynamic movement.\n"
        "2. Focus & Clarity: Core behavioral interaction maintained in usable focus, sharp subject definition.\n"
        "3. Camera Stability: Controlled panning following the action without disorienting shake.\n"
        "4. Subject Composition: Intentional framing containing the critical animal interaction or chase inside frame.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Dramatic behavioral moment, crisp focus on action, smooth panning, compelling framing.\n"
        "- 7-8 (Strong Highlight): Clear action, good lighting, stable tracking.\n"
        "- 5-6 (Acceptable): Distant action, moderate camera movement or partial obstruction by vegetation.\n"
        "- 1-4 (Reject): Missed action, severe motion smear, heavy shake, accidental, inverted, dangling, or unusable.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is dynamic wildlife action visible?)\n"
        "- subject: string (description of action e.g. 'cheetah chasing gazelle', 'lions feeding')\n"
        "- category: string ('wildlife_action')\n"
        "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
        "- reject_reasons: list of strings (or empty list if detected is true)\n"
        "- description: string (details on action, interaction, lighting, and camera stability)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    color="Orange",
    domain="wildlife",
)

# Broad wildlife fallback template
BROAD_WILDLIFE_TEMPLATE = PromptTemplate(
    name="broad_wildlife",
    keywords=[],
    system_prompt="You are a wildlife video analyst evaluating general nature and wildlife footage highlights.",
    coarse_prompt=WILDLIFE_COARSE_PROMPT,
    fine_prompt=(
        "Analyze this wildlife frame for general wildlife, nature, and fauna highlights.\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both subject presence and shot execution:\n"
        "1. Lighting: Natural lighting, balanced exposure across wildlife subject and natural habitat.\n"
        "2. Focus & Clarity: Crisp focus on animals, birds, or nature elements, sharp details.\n"
        "3. Camera Stability: Smooth camera tracking or steady hold, free of jerky shake or heavy motion blur.\n"
        "4. Subject Composition: Upright, intentional nature composition framing the subject clearly.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Razor-sharp focus, beautiful lighting, rock-steady hold, compelling wildlife composition.\n"
        "- 7-8 (Strong Highlight): Good lighting, sharp subject, minimal camera shake, well-framed.\n"
        "- 5-6 (Acceptable): Adequate lighting, slight softness or distant subject, minor camera movement.\n"
        "- 1-4 (Reject): Dark/murky, out of focus, severe camera shake, accidental, inverted, dangling, or unusable.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is wildlife or nature highlight clearly visible?)\n"
        "- subject: string (species, nature element, or 'unknown')\n"
        "- category: string ('broad_wildlife')\n"
        "- story_role: string ('establishing' | 'entry_transition' | 'hero_subject' | 'human_moment' | 'unexpected_event' | 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
        "- reject_reasons: list of strings (or empty list if detected is true)\n"
        "- description: string (details on subject, habitat, lighting, and camera stability)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    color="Yellow",
    domain="wildlife",
)

WILDLIFE_DOMAIN = DomainConfig(
    name="wildlife",
    keywords=[
        "wildlife",
        "animal",
        "safari",
        "bird",
        "nature",
        "fauna",
        "mammal",
        "wilderness",
        "dier",
        "natuur",
    ],
    description="Wild animals, birds, safari game, predators, dynamic animal behavior, and natural habitat scenes",
    coarse_prompt=WILDLIFE_COARSE_PROMPT,
    fallback_template=BROAD_WILDLIFE_TEMPLATE,
    templates=[WILDLIFE_ACTION_TEMPLATE, WILDLIFE_ANIMAL_TEMPLATE, WILDLIFE_BIRD_TEMPLATE],
)


# =============================================================================
# Domain Registry & Helpers
# =============================================================================

UNDERWATER_DOMAIN = DomainConfig(
    name="underwater",
    keywords=[
        "underwater",
        "dive",
        "diving",
        "scuba",
        "snorkel",
        "reef",
        "ocean",
        "marine",
        "duik",
        "onderwater",
    ],
    description=(
        "Scuba diving, snorkeling, marine life, coral reefs, underwater wrecks, and diver action"
    ),
    coarse_prompt=BROAD_MARINE_TEMPLATE.coarse_prompt,
    fallback_template=BROAD_MARINE_TEMPLATE,
    templates=[PELAGICS_TEMPLATE, MACRO_TEMPLATE, DIVER_TEMPLATE],
)

MOTORCYCLE_DOMAIN = DomainConfig(
    name="motorcycle",
    keywords=[
        "motorcycle",
        "motorbike",
        "moto",
        "biker",
        "riding",
        "ride",
        "motorrijder",
        "motor",
        "twisties",
        "cornering",
    ],
    description="Motorcycle riding, cornering, scenic road vistas, POV handlebar view, group rides",
    coarse_prompt=MOTORCYCLE_COARSE_PROMPT,
    fallback_template=BROAD_MOTORCYCLE_TEMPLATE,
    templates=[MOTO_RIDING_TEMPLATE, MOTO_SCENIC_TEMPLATE, MOTO_GROUP_TEMPLATE],
)

GENERIC_DOMAIN = DomainConfig(
    name="generic",
    keywords=["action", "travel", "landscape", "sports", "general", "video"],
    description="General action camera, travel, sports, landscape, or family video highlights",
    coarse_prompt=GENERIC_COARSE_PROMPT,
    fallback_template=GENERIC_HIGHLIGHT_TEMPLATE,
    templates=[GENERIC_HIGHLIGHT_TEMPLATE],
)

BUILTIN_DOMAINS: dict[str, DomainConfig] = {
    "underwater": UNDERWATER_DOMAIN,
    "motorcycle": MOTORCYCLE_DOMAIN,
    "wildlife": WILDLIFE_DOMAIN,
    "generic": GENERIC_DOMAIN,
}

# Backward compatibility list:
ALL_TEMPLATES: list[PromptTemplate] = [PELAGICS_TEMPLATE, MACRO_TEMPLATE, DIVER_TEMPLATE]
MOTORCYCLE_TEMPLATES: list[PromptTemplate] = [
    MOTO_RIDING_TEMPLATE,
    MOTO_SCENIC_TEMPLATE,
    MOTO_GROUP_TEMPLATE,
]
WILDLIFE_TEMPLATES: list[PromptTemplate] = [
    WILDLIFE_ACTION_TEMPLATE,
    WILDLIFE_ANIMAL_TEMPLATE,
    WILDLIFE_BIRD_TEMPLATE,
]
ALL_DOMAIN_TEMPLATES: list[PromptTemplate] = [
    *UNDERWATER_DOMAIN.templates,
    UNDERWATER_DOMAIN.fallback_template,
    *MOTORCYCLE_DOMAIN.templates,
    MOTORCYCLE_DOMAIN.fallback_template,
    *WILDLIFE_DOMAIN.templates,
    WILDLIFE_DOMAIN.fallback_template,
    *GENERIC_DOMAIN.templates,
]


def load_custom_templates(config: dict[str, Any]) -> dict[str, DomainConfig]:
    """Merge custom domain and template configurations from config.yaml into the registry.

    Example config.yaml section:
        prompts:
          domains:
            cycling:
              description: "Bicycle and mountain biking footage"
              keywords: ["bicycle", "cycling", "mtb", "gravel"]
              coarse_prompt: "..."
              fallback_color: "Green"
              templates:
                - name: "mtb_descent"
                  keywords: ["trail", "downhill", "jump"]
                  color: "Red"
                  fine_prompt: "..."
          templates:
            - name: "custom_wreck"
              domain: "underwater"
              keywords: ["wreck", "shipwreck"]
              color: "Green"
              fine_prompt: "..."
    """
    registry: dict[str, DomainConfig] = {
        name: DomainConfig(
            name=dc.name,
            keywords=list(dc.keywords),
            description=dc.description,
            coarse_prompt=dc.coarse_prompt,
            fallback_template=dc.fallback_template,
            templates=list(dc.templates),
        )
        for name, dc in BUILTIN_DOMAINS.items()
    }

    prompts_cfg = config.get("prompts", {})
    if not isinstance(prompts_cfg, dict):
        return registry

    # 1. Custom domains
    domains_cfg = prompts_cfg.get("domains", {})
    if isinstance(domains_cfg, dict):
        for dom_name, dom_info in domains_cfg.items():
            if not isinstance(dom_info, dict):
                continue
            desc = dom_info.get("description", f"Custom domain: {dom_name}")
            keywords = dom_info.get("keywords", [dom_name])
            coarse = dom_info.get("coarse_prompt", GENERIC_COARSE_PROMPT)
            fallback_color = dom_info.get("fallback_color", "Magenta")

            fallback_tpl = PromptTemplate(
                name=f"broad_{dom_name}",
                keywords=[],
                system_prompt=f"You are a video analyst specializing in {dom_name} footage.",
                coarse_prompt=coarse,
                fine_prompt=dom_info.get(
                    "fallback_fine_prompt", GENERIC_HIGHLIGHT_TEMPLATE.fine_prompt
                ),
                color=fallback_color,
                domain=dom_name,
            )

            custom_tpls: list[PromptTemplate] = []
            for tpl_data in dom_info.get("templates", []):
                if not isinstance(tpl_data, dict) or "name" not in tpl_data:
                    continue
                custom_tpls.append(
                    PromptTemplate(
                        name=tpl_data["name"],
                        keywords=tpl_data.get("keywords", []),
                        system_prompt=tpl_data.get("system_prompt", fallback_tpl.system_prompt),
                        coarse_prompt=tpl_data.get("coarse_prompt", coarse),
                        fine_prompt=tpl_data.get("fine_prompt", fallback_tpl.fine_prompt),
                        color=tpl_data.get("color", fallback_color),
                        domain=dom_name,
                    )
                )

            registry[dom_name] = DomainConfig(
                name=dom_name,
                keywords=keywords,
                description=desc,
                coarse_prompt=coarse,
                fallback_template=fallback_tpl,
                templates=custom_tpls,
            )

    # 2. Individual custom templates added to existing domains
    extra_templates = prompts_cfg.get("templates", [])
    if isinstance(extra_templates, list):
        for tpl_data in extra_templates:
            if not isinstance(tpl_data, dict) or "name" not in tpl_data:
                continue
            target_domain = tpl_data.get("domain", "generic")
            if target_domain not in registry:
                registry[target_domain] = DomainConfig(
                    name=target_domain,
                    keywords=[target_domain],
                    description=f"Domain for {target_domain}",
                    coarse_prompt=GENERIC_COARSE_PROMPT,
                    fallback_template=GENERIC_HIGHLIGHT_TEMPLATE,
                    templates=[],
                )
            dom = registry[target_domain]
            new_tpl = PromptTemplate(
                name=tpl_data["name"],
                keywords=tpl_data.get("keywords", []),
                system_prompt=tpl_data.get("system_prompt", dom.fallback_template.system_prompt),
                coarse_prompt=tpl_data.get("coarse_prompt", dom.coarse_prompt),
                fine_prompt=tpl_data.get("fine_prompt", dom.fallback_template.fine_prompt),
                color=tpl_data.get("color", dom.fallback_template.color),
                domain=target_domain,
            )
            # Replace existing template with same name or append
            dom.templates = [t for t in dom.templates if t.name != new_tpl.name] + [new_tpl]

    return registry


def resolve_domain_from_keywords(
    keywords: list[str],
    registry: dict[str, DomainConfig] | None = None,
) -> str | None:
    """Infer the video domain from a list of keywords or tags."""
    if not keywords:
        return None
    reg = registry or BUILTIN_DOMAINS
    keywords_lower = [k.lower().strip() for k in keywords if k]

    # First check domain-level keywords
    for dom_name, dom_config in reg.items():
        for kw in dom_config.keywords:
            if kw.lower() in keywords_lower:
                return dom_name

    # Next check template-level keywords
    for dom_name, dom_config in reg.items():
        for tpl in dom_config.templates:
            for kw in tpl.keywords:
                if kw.lower() in keywords_lower:
                    return dom_name

    return None


def _keyword_matches(kw: str, text: str) -> bool:
    """Check if keyword matches text as a whole word or phrase, ignoring case."""
    kw_clean = kw.strip()
    if not kw_clean:
        return False
    parts = [re.escape(p) for p in re.split(r"[\s\-_]+", kw_clean) if p]
    if not parts:
        return False
    pattern = r"\b" + r"[\s\-_]+".join(parts) + r"\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


def resolve_template_for_subject(
    subject: str = "",
    category: str = "",
    domain: str | None = None,
    tags: list[str] | None = None,
    registry: dict[str, DomainConfig] | None = None,
) -> PromptTemplate:
    """Find the best fine prompt template for a candidate window.

    Resolution precedence:
    1. Tags explicitly matching a template in the domain (or all templates).
    2. Category explicitly matching a template name in the domain.
    3. Subject text matching any template keywords in the domain.
    4. Subject text matching template keywords across any domain.
    5. Fallback template for the domain (or BROAD_MARINE_TEMPLATE).
    """
    reg = registry or BUILTIN_DOMAINS
    dom_key = domain if (domain and domain in reg) else "underwater"
    dom_config = reg[dom_key]

    # 1. Check explicit tags
    if tags:
        tags_lower = [t.lower().strip() for t in tags if t]
        # Check domain templates first
        for tpl in dom_config.templates:
            for kw in tpl.keywords:
                if kw.lower() in tags_lower:
                    return tpl
        # Check all templates in registry
        for other_dom in reg.values():
            for tpl in other_dom.templates:
                for kw in tpl.keywords:
                    if kw.lower() in tags_lower:
                        return tpl

    # 2. Check category name match
    cat_clean = (
        category.strip().lower()
        if isinstance(category, str)
        else ("" if category is None else str(category).strip().lower())
    )
    if cat_clean:
        for tpl in dom_config.templates:
            if tpl.name.lower() == cat_clean:
                return tpl
        if dom_config.fallback_template.name.lower() == cat_clean:
            return dom_config.fallback_template

    # 3. Check subject string against templates in current domain
    subj_clean = (
        subject.strip().lower()
        if isinstance(subject, str)
        else ("" if subject is None else str(subject).strip().lower())
    )
    if subj_clean:
        for tpl in dom_config.templates:
            for kw in tpl.keywords:
                if _keyword_matches(kw, subj_clean):
                    return tpl

        # 4. Check subject string across other domains only if domain was not specified
        if not domain:
            for other_key, other_dom in reg.items():
                if other_key == dom_key:
                    continue
                for tpl in other_dom.templates:
                    for kw in tpl.keywords:
                        if _keyword_matches(kw, subj_clean):
                            return tpl

    # 5. Return fallback template for domain
    return dom_config.fallback_template


def get_template_for_keywords(
    keywords: list[str],
    templates: list[PromptTemplate] | None = None,
) -> PromptTemplate | None:
    """Backward-compatible helper: find the best matching prompt template for keywords."""
    target_templates = templates if templates is not None else ALL_TEMPLATES
    keywords_lower = [k.lower() for k in keywords]
    for template in target_templates:
        for kw in template.keywords:
            if kw.lower() in keywords_lower:
                return template
    return None


def _slugify(text: str) -> str:
    """Helper to convert text into a clean snake_case slug identifier."""
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return clean or "highlight"


_CANDIDATE_COLOR = "Magenta"
_CANDIDATE_THRESHOLD = 0.65  # Default mid-range confidence; tune per candidate during review.
_MAX_CONTEXT_CHARS = 60
_MAX_CONTEXT_SUBJECTS = 5


def _sanitize_context_text(value: Any) -> str:
    """Neutralize model-derived text before interpolating it into prompts (injection guard)."""
    text = re.sub(r"[^A-Za-z0-9 ,'/&-]+", " ", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()[:_MAX_CONTEXT_CHARS]


def infer_generic_context(
    records: list[Any] | Any,
) -> dict[str, Any]:
    """Extract dominant environment and ranked primary subjects from generic clip classifications.

    Args:
        records: A single ClipClassification, a dict, or a list of ClipClassification/dicts from Pass 0.

    Returns:
        dict with:
            - 'environment': str (dominant physical setting/environment, or "" if none)
            - 'primary_subjects': list[str] (ranked unique visible subjects)
            - 'template_slug': str (clean slug for template naming)
            - 'environments': list[str] (all unique non-empty environments seen)
            - 'clip_count': int (number of generic clips inspected)
    """
    if records is None:
        records_list: list[Any] = []
    elif isinstance(records, list):
        records_list = records
    else:
        records_list = [records]

    # Filter for generic domain clips if available
    generic_records: list[Any] = []
    for r in records_list:
        domain_val = (
            getattr(r, "domain", None)
            if hasattr(r, "domain")
            else (r.get("domain") if isinstance(r, dict) else None)
        )
        if domain_val == "generic" or domain_val is None or domain_val == "":
            generic_records.append(r)

    if not generic_records:
        generic_records = list(records_list)

    # 1. Aggregate environment
    env_counts: dict[str, int] = {}
    for r in generic_records:
        env_val = (
            getattr(r, "environment", "")
            if hasattr(r, "environment")
            else (r.get("environment", "") if isinstance(r, dict) else "")
        )
        if isinstance(env_val, str) and env_val.strip():
            clean_env = env_val.strip()
            env_counts[clean_env] = env_counts.get(clean_env, 0) + 1

    dominant_env = max(env_counts, key=env_counts.get) if env_counts else ""
    all_envs = sorted(env_counts.keys())

    # 2. Aggregate primary subjects
    subj_counts: dict[str, int] = {}
    subj_canonical: dict[str, str] = {}
    first_seen_keys: list[str] = []
    for r in generic_records:
        raw_subjs = (
            getattr(r, "primary_subjects", [])
            if hasattr(r, "primary_subjects")
            else (r.get("primary_subjects", []) if isinstance(r, dict) else [])
        )
        if isinstance(raw_subjs, str):
            raw_subjs = [s.strip() for s in raw_subjs.split(",") if s.strip()]
        elif not isinstance(raw_subjs, list):
            raw_subjs = []

        seen_in_record: set[str] = set()
        for s in raw_subjs:
            if isinstance(s, str) and s.strip():
                clean_s = s.strip().rstrip(".,;")
                if clean_s:
                    key = clean_s.lower()
                    if key not in seen_in_record:
                        seen_in_record.add(key)
                        subj_counts[key] = subj_counts.get(key, 0) + 1
                        if key not in subj_canonical:
                            subj_canonical[key] = clean_s
                            first_seen_keys.append(key)

    # Sort unique subjects by frequency (descending), preserving first-seen order on ties
    sorted_subj_keys = sorted(first_seen_keys, key=lambda k: -subj_counts[k])
    ranked_subjects = [subj_canonical[k] for k in sorted_subj_keys]

    # 3. Derive template slug
    if ranked_subjects:
        top_subj_slug = _slugify(ranked_subjects[0])
        template_slug = (
            f"generic_{top_subj_slug}"
            if not top_subj_slug.startswith("generic_")
            else top_subj_slug
        )
    elif dominant_env:
        env_slug = _slugify(dominant_env)
        template_slug = f"generic_{env_slug}" if not env_slug.startswith("generic_") else env_slug
    else:
        template_slug = "generic_highlight"

    return {
        "environment": dominant_env,
        "primary_subjects": ranked_subjects,
        "template_slug": template_slug,
        "environments": all_envs,
        "clip_count": len(generic_records),
    }


def create_prompt_candidate_for_generic(
    context_or_record: dict[str, Any] | Any,
    candidate_name: str | None = None,
) -> dict[str, Any]:
    """Generate candidate coarse and fine prompt strings with orientation checks and 1-10 quality rubric.

    Args:
        context_or_record: Context dict from infer_generic_context, or a ClipClassification/dict/list.
        candidate_name: Optional explicit name override for the candidate template.

    Returns:
        dict candidate template with:
            - 'name': str
            - 'domain': 'generic'
            - 'coarse_prompt': str
            - 'fine_prompt': str
            - 'recommended_threshold': float (0.65)
            - 'keywords': list[str]
            - 'system_prompt': str
            - 'color': str ('Magenta')
            - 'status': str ('candidate')
            - 'environment': str
            - 'primary_subjects': list[str]
    """
    if isinstance(context_or_record, dict) and (
        "template_slug" in context_or_record or "environment" in context_or_record
    ):
        context = dict(context_or_record)
        if "template_slug" not in context:
            context["template_slug"] = infer_generic_context(context).get(
                "template_slug", "generic_highlight"
            )
    else:
        context = infer_generic_context(context_or_record)

    environment: str = _sanitize_context_text(context.get("environment"))
    raw_subjects = context.get("primary_subjects") or []
    if isinstance(raw_subjects, str):
        raw_subjects = raw_subjects.split(",")
    elif not isinstance(raw_subjects, list):
        raw_subjects = []
    primary_subjects = []
    for raw in raw_subjects:
        cleaned = _sanitize_context_text(raw)
        if cleaned and cleaned not in primary_subjects:
            primary_subjects.append(cleaned)
    primary_subjects = primary_subjects[:_MAX_CONTEXT_SUBJECTS]

    raw_slug = candidate_name or context.get("template_slug") or "generic_highlight"
    name = _slugify(str(raw_slug))
    if not name.startswith("generic"):
        name = f"generic_{name}"
    if name in {t.name for d in BUILTIN_DOMAINS.values() for t in d.templates}:
        name = f"{name}_candidate"

    if primary_subjects:
        subjects_str = ", ".join(primary_subjects[:3])
        primary_subject_label = primary_subjects[0]
    else:
        subjects_str = "highlight action and central subjects"
        primary_subject_label = "highlight"

    env_phrase = f"in {environment} setting" if environment else "in general footage"

    keywords_list = list(primary_subjects)
    if environment and environment not in keywords_list:
        keywords_list.append(environment)

    system_prompt = (
        f"You are a video analyst specializing in {environment} footage and evaluating {subjects_str} highlights."
        if environment
        else f"You are a video analyst evaluating {subjects_str} highlights."
    )

    coarse_prompt = (
        f"Analyze this video frame for candidate screening {env_phrase}.\n"
        f"Target subjects: {subjects_str}.\n\n"
        "CAMERA ORIENTATION & USABILITY CHECK:\n"
        "- The camera must be intentionally aimed and upright (right-side up).\n"
        "- You MUST explicitly reject accidental, inverted, dangling, or unusable footage:\n"
        "  If the camera is inside a pocket/bag, pointed accidentally at the floor/ground, covered,\n"
        "  upside down, inverted, dangling on a clip/strap, spinning uncontrollably, or completely blurred,\n"
        "  you MUST reject the frame:\n"
        "  Set 'detected': false, 'confidence': 0.0, 'quality_score': 1.\n"
        "- Accidental, inverted, dangling, or unusable frames must always be rejected even if a subject is partially visible.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (true ONLY if target subject is present AND camera is upright and usable)\n"
        f"- subject: string (name of detected subject, e.g. '{primary_subject_label}' or 'none')\n"
        f"- category: string ('{name}')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10; 1-3 for accidental/inverted/dangling/unusable/chaotic, 4+ for usable)\n"
        "- description: string (short note on subject or rejection reason)\n"
        "- timestamp: float (seconds in the clip)\n"
    )

    fine_prompt = (
        f"Analyze this video frame in detail for {subjects_str} {env_phrase}.\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both subject presence and shot execution:\n"
        "1. Lighting & Exposure: Well-exposed, balanced natural or artificial lighting, clear subject contrast.\n"
        f"2. Focus & Clarity: {subjects_str.capitalize()} is in crisp, sharp focus, free of motion blur.\n"
        "3. Camera Stability: Smooth camera motion or steady hold without jerky shake or rapid vibration.\n"
        "4. Subject Composition: Intentional upright framing and composition centering or highlighting the subject.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Razor-sharp focus, beautiful lighting, rock-steady camera, compelling subject composition.\n"
        "- 7-8 (Strong Highlight): Good lighting, sharp subject, minimal camera shake, well-framed composition.\n"
        "- 5-6 (Acceptable): Adequate lighting, slight softness or distance, minor camera wobble.\n"
        "- 1-4 (Reject): Dark/murky, out of focus, severe camera shake, accidental, inverted, dangling, or unusable.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is this a clear highlight moment?)\n"
        "- subject: string (subject name or 'unknown')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
        "- description: string (details on subject, lighting, focus, and stability)\n"
        "- timestamp: float (seconds in the clip)\n"
    )

    return {
        "name": name,
        "domain": "generic",
        "coarse_prompt": coarse_prompt,
        "fine_prompt": fine_prompt,
        "recommended_threshold": _CANDIDATE_THRESHOLD,
        "keywords": keywords_list,
        "system_prompt": system_prompt,
        "color": _CANDIDATE_COLOR,
        "status": "candidate",
        "environment": environment,
        "primary_subjects": primary_subjects,
    }
