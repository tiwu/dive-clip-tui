"""Dynamic prompt templates and domain registry for multi-stage VLM video analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

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

VIDEO_CLASSIFICATION_PROMPT = (
    "Analyze this video frame and classify the primary setting, domain, and footage usability.\n\n"
    "Target domains:\n"
    "- 'underwater': Scuba diving, snorkeling, free diving, marine life, coral reefs, "
    "underwater wrecks, diver action, oceanic or aquatic environments.\n"
    "- 'motorcycle': Motorcycle, motorbike, scooter, riding, cockpit or handlebar POV, "
    "pavement, track, twisties, curves, group ride, highway or scenic road ride.\n"
    "- 'generic': Any other domain (e.g. action sports, hiking, travel, city, family, landscape).\n\n"
    "Usability & Camera State:\n"
    "- If the camera is inside a bag or pocket, lens is covered/black, upside down, tumbling, "
    "dangling on a clip/lanyard, or showing only floor/empty darkness with no scene, "
    "set 'usable': false and 'quality_score': 1.\n"
    "- Otherwise set 'usable': true with a quality_score from 4 to 10.\n\n"
    "Return a JSON object with:\n"
    "- domain: string (one of 'underwater', 'motorcycle', 'generic')\n"
    "- confidence: float (0.0 to 1.0)\n"
    "- usable: boolean (true if footage is normal/usable; false if in-bag, covered, upside-down, or dangling)\n"
    "- quality_score: integer (1-10; 1-3 for in-bag/dangling/chaotic, 4+ for usable footage)\n"
    "- environment: string (brief note on setting, e.g. 'coral reef wall', 'mountain pass road')\n"
    "- primary_subjects: list of strings (main visible subjects/elements)\n"
)


def build_classification_prompt(
    domains: list[str] | dict[str, DomainConfig] | None = None,
) -> str:
    """Build a classification prompt listing active domains."""
    if not domains:
        return VIDEO_CLASSIFICATION_PROMPT

    domain_names: list[str] = list(domains.keys()) if isinstance(domains, dict) else list(domains)
    if set(domain_names) == {"underwater", "motorcycle", "generic"}:
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
        "Target subjects: pelagic species (turtles, sharks, rays, mantas, whales, dolphins).\n\n"
        "CAMERA ORIENTATION & USABILITY CHECK:\n"
        "- The camera must be intentionally aimed and upright (right-side up).\n"
        "- If the camera is upside down, tumbling, dangling on a clip/lanyard, spinning, "
        "or pointed accidentally at the seabed/boat/diver gear, you MUST reject the frame:\n"
        "  Set 'detected': false, 'confidence': 0.0, 'quality_score': 1.\n"
        "- Even if a turtle or shark is partially in view, if the camera is dangling or inverted, "
        "mark 'detected': false.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (true ONLY if target pelagic is present AND camera is upright/stable)\n"
        "- subject: string (species name or 'none')\n"
        "- category: string ('pelagics')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10; 1-3 for inverted/dangling/chaotic, 4+ for usable)\n"
        "- description: string (short note on subject or rejection reason)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    fine_prompt=(
        "Analyze this underwater frame for pelagic species (turtles, sharks, rays, mantas).\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both subject presence and shot execution:\n"
        "1. Lighting: Well-lit with natural or video light, good contrast, minimal backscatter/murk.\n"
        "2. Focus & Clarity: Pelagic subject is in sharp focus, not blurry or soft.\n"
        "3. Camera Stability: Smooth camera movement without excessive shake, rapid tumbling, or heavy motion blur.\n"
        "4. Framing & Orientation: Upright, intentional composition centering the animal.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Crisp focus on animal, clear water/lighting, smooth stable camera.\n"
        "- 7-8 (Strong Highlight): Good lighting, animal in clear focus, minimal camera shake.\n"
        "- 5-6 (Acceptable): Passable lighting, slight blur or distance, minor camera movement.\n"
        "- 1-4 (Reject): Murky/dark, out of focus, heavy motion blur, upside down, or dangling.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is a pelagic species clearly visible?)\n"
        "- subject: string (species or 'unknown')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
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
        "Target subjects: macro subjects (nudibranchs, shrimp, corals, small sea creatures).\n\n"
        "CAMERA ORIENTATION & USABILITY CHECK:\n"
        "- The camera must be intentionally aimed and upright (right-side up).\n"
        "- If the camera is upside down, tumbling, dangling on a clip/lanyard, spinning, "
        "or pointed accidentally at the reef/seabed, you MUST reject the frame:\n"
        "  Set 'detected': false, 'confidence': 0.0, 'quality_score': 1.\n"
        "- Even if coral is visible, if the camera is dangling or inverted, mark 'detected': false.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (true ONLY if macro subject is present AND camera is upright/stable)\n"
        "- subject: string (species/subject name or 'none')\n"
        "- category: string ('macro')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10; 1-3 for inverted/dangling/chaotic, 4+ for usable)\n"
        "- description: string (short note on subject or rejection reason)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    fine_prompt=(
        "Analyze this underwater frame for macro subjects (nudibranchs, shrimp, corals, small creatures).\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both subject presence and shot execution:\n"
        "1. Lighting: Proper lighting illuminating macro details, balanced exposure, no blown-out highlights.\n"
        "2. Focus & Clarity: Macro subject is in crisp, sharp focus (shallow depth of field placed correctly).\n"
        "3. Camera Stability: Rock-steady camera without shake, surge drift, or motion blur.\n"
        "4. Framing & Orientation: Upright, clean macro composition.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Razor-sharp focus on subject, beautiful lighting, steady motionless camera.\n"
        "- 7-8 (Strong Highlight): Sharp focus on macro subject, good lighting, minimal camera motion.\n"
        "- 5-6 (Acceptable): Soft focus or slight camera movement, acceptable lighting.\n"
        "- 1-4 (Reject): Blurry, missed focus, murky/dark, severe shake, upside down, or dangling.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is a macro subject clearly visible?)\n"
        "- subject: string (subject name or 'unknown')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
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
        "Target subjects: diver action, buddy interaction, swim-throughs, and intentional composition.\n\n"
        "CAMERA ORIENTATION & USABILITY CHECK:\n"
        "- The camera must be intentionally operated and upright (right-side up).\n"
        "- If the camera is upside down, tumbling, dangling on a BCD lanyard, spinning, "
        "or filming accidentally (e.g. fins/boat ladder/empty blue while dangling), you MUST reject the frame:\n"
        "  Set 'detected': false, 'confidence': 0.0, 'quality_score': 1.\n"
        "- Accidental footage of divers where the camera is dangling or inverted is NOT diver action: mark 'detected': false.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (true ONLY if intentional diver action is present AND camera is upright)\n"
        "- subject: string (diver activity or 'none')\n"
        "- category: string ('diver')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10; 1-3 for inverted/dangling/chaotic, 4+ for usable)\n"
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
        "4. Framing & Orientation: Upright, cinematic composition with good rule of thirds or leading lines.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Crisp focus, dramatic lighting/silhouette, smooth stable glide, compelling action.\n"
        "- 7-8 (Strong Highlight): Good lighting, sharp diver, stable camera framing.\n"
        "- 5-6 (Acceptable): Adequate lighting, slight camera wobble or distance.\n"
        "- 1-4 (Reject): Dark/murky, out of focus, erratic camera shake, upside down, or dangling.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is intentional diver action visible?)\n"
        "- subject: string (e.g. 'diver exploring reef', 'diver entering swimthrough')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
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
        "sharks, turtles, whales, frogfish, crocodilefish, seahorses, stonefish, diver action, and sea creatures.\n\n"
        "CAMERA ORIENTATION & USABILITY CHECK:\n"
        "- The camera must be intentionally aimed and upright (right-side up).\n"
        "- If the camera is upside down, tumbling, dangling on a clip/lanyard, spinning, "
        "or pointed accidentally at the seabed/hull/diver gear, you MUST reject the frame:\n"
        "  Set 'detected': false, 'confidence': 0.0, 'quality_score': 1.\n"
        "- Even if background corals or fish are present, if the camera is dangling or inverted, "
        "mark 'detected': false.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (true ONLY if marine life or diver action is present AND camera is upright and intentionally operated)\n"
        "- subject: string (species name, diver activity, or 'none')\n"
        "- category: string (one of 'pelagics', 'macro', 'diver', 'broad_marine')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10; 1-3 for inverted/dangling/chaotic, 4+ for usable)\n"
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
        "4. Framing & Orientation: Upright, intentional composition highlighting the subject.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Razor-sharp focus, beautiful lighting, clear water, smooth stable camera.\n"
        "- 7-8 (Strong Highlight): Good lighting, sharp subject, stable camera with minimal motion blur.\n"
        "- 5-6 (Acceptable): Passable lighting, slight softness or distance, mild camera movement.\n"
        "- 1-4 (Reject): Murky/dark, out of focus, severe camera shake, upside down, or dangling.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is marine life clearly visible?)\n"
        "- subject: string (species or 'unknown')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
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
    "scenic road vistas, twisties, hairpins, and fellow riders / group ride action.\n\n"
    "CAMERA ORIENTATION & USABILITY CHECK:\n"
    "- The camera must be intentionally aimed at the road, horizon, bike cockpit/handlebar, rider, or scenery.\n"
    "- If the camera is pointed accidentally at the pavement/ground, fuel tank, inside a helmet, "
    "blocked/covered, upside down, dangling, or stationary with zero movement/action (e.g. stopped at a light), "
    "you MUST reject the frame:\n"
    "  Set 'detected': false, 'confidence': 0.0, 'quality_score': 1.\n\n"
    "Return a JSON object with:\n"
    "- detected: boolean (true ONLY if motorcycle highlight action, scenic road, or rider interaction is present AND camera is usable)\n"
    "- subject: string (e.g. 'cornering through hairpin', 'scenic mountain pass', 'riding with buddy', or 'none')\n"
    "- category: string (one of 'moto_riding', 'moto_scenic', 'moto_group', 'broad_motorcycle')\n"
    "- confidence: float (0.0 to 1.0)\n"
    "- quality_score: int (1-10; 1-3 for ground/blocked/stationary, 4+ for usable)\n"
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
        "1. Riding Action: Visible lean angle, smooth apex tracking, dynamic road curve, or overtakes.\n"
        "2. Focus & Clarity: Clear forward road view, crisp cockpit or rider silhouette, minimal motion smear.\n"
        "3. Camera Stability: Stable horizon or smooth lean without violent engine vibration or camera shake.\n"
        "4. Framing & Orientation: Upright composition with leading lines following the curve of the road.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Dramatic lean angle, razor-sharp view of the apex, rock-steady footage, thrilling speed.\n"
        "- 7-8 (Strong Highlight): Good cornering action, sharp road view, minimal vibration.\n"
        "- 5-6 (Acceptable): Moderate riding action, slight engine buzz or soft focus.\n"
        "- 1-4 (Reject): Pointed at ground/tank, stationary, heavy vibration blur, upside down, or covered.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is compelling motorcycle riding action visible?)\n"
        "- subject: string (e.g. 'deep lean right hairpin', 'accelerating out of curve')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
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
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
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
        "1. Rider Interaction: Clear view of fellow riders, staggered formation, wave or gesture.\n"
        "2. Focus & Clarity: Crisp focus on the lead/following motorcycle and rider.\n"
        "3. Camera Stability: Smooth following distance without erratic braking or jitter.\n"
        "4. Framing: Intentional framing showing the relationship between riders and the road.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Perfectly framed formation, crisp focus on riders, great lighting and stability.\n"
        "- 7-8 (Strong Highlight): Clear view of fellow riders, good spacing and stable camera.\n"
        "- 5-6 (Acceptable): Riders visible but distant or slightly obscured.\n"
        "- 1-4 (Reject): Too distant, out of focus, shaky, or no fellow riders present.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (are fellow riders in frame with intentional composition?)\n"
        "- subject: string (e.g. 'two riders in formation', 'biker wave on pass')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
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
        "1. Subject & Action: Engaging riding action, scenery, or rider focus.\n"
        "2. Focus & Clarity: Clear road view, minimal motion smear.\n"
        "3. Camera Stability: Smooth cruising without engine vibration shake.\n"
        "4. Framing: Intentional upright road composition.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Dynamic riding or gorgeous scenery, razor-sharp focus, rock-steady hold.\n"
        "- 7-8 (Strong Highlight): Good road view, clear lighting, stable camera.\n"
        "- 5-6 (Acceptable): Adequate riding or view, minor vibration.\n"
        "- 1-4 (Reject): Stationary at light, pointed at ground/tank, blurry, or covered.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is this a highlight moment?)\n"
        "- subject: string (description of highlight or 'none')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
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
    "Analyze this video frame for candidate screening.\n"
    "Target subjects: compelling action, clear central subject, scenic view, or notable event.\n\n"
    "CAMERA ORIENTATION & USABILITY CHECK:\n"
    "- The camera must be intentionally aimed and upright.\n"
    "- If the camera is inside a pocket/bag, pointed accidentally at the floor/ground, covered, "
    "spinning uncontrollably, or completely blurred, you MUST reject the frame:\n"
    "  Set 'detected': false, 'confidence': 0.0, 'quality_score': 1.\n\n"
    "Return a JSON object with:\n"
    "- detected: boolean (true ONLY if an intentional highlight subject is present AND camera is usable)\n"
    "- subject: string (name of subject/action or 'none')\n"
    "- category: string ('generic_highlight')\n"
    "- confidence: float (0.0 to 1.0)\n"
    "- quality_score: int (1-10; 1-3 for pocket/floor/chaotic, 4+ for usable)\n"
    "- description: string (short note on subject or rejection reason)\n"
    "- timestamp: float (seconds in the clip)\n"
)

GENERIC_HIGHLIGHT_TEMPLATE = PromptTemplate(
    name="generic_highlight",
    keywords=[],
    system_prompt="You are a video analyst evaluating general highlight moments.",
    coarse_prompt=GENERIC_COARSE_PROMPT,
    fine_prompt=(
        "Analyze this video frame in detail for highlight quality.\n\n"
        "QUALITY EVALUATION RUBRIC (1-10):\n"
        "Assess both subject presence and shot execution:\n"
        "1. Subject & Action: Clear, intentional subject or dynamic action in view.\n"
        "2. Lighting & Exposure: Well-exposed, balanced contrast, good natural or artificial light.\n"
        "3. Focus & Clarity: Subject is in crisp, sharp focus, not blurry.\n"
        "4. Camera Stability: Smooth camera movement without jerky shake or heavy motion blur.\n\n"
        "Scoring Scale:\n"
        "- 9-10 (Exceptional): Crisp focus on compelling subject, beautiful lighting, rock-steady camera.\n"
        "- 7-8 (Strong Highlight): Good lighting, sharp subject, minimal camera shake.\n"
        "- 5-6 (Acceptable): Adequate lighting, slight softness or minor camera wobble.\n"
        "- 1-4 (Reject): Dark, out of focus, severe camera shake, upside down, or covered.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (is this a clear highlight moment?)\n"
        "- subject: string (subject name or 'unknown')\n"
        "- confidence: float (0.0 to 1.0)\n"
        "- quality_score: int (1-10 based on the rubric above)\n"
        "- description: string (details on subject, lighting, focus, and stability)\n"
        "- timestamp: float (seconds in the clip)\n"
    ),
    color="Magenta",
    domain="generic",
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
    "generic": GENERIC_DOMAIN,
}

# Backward compatibility list:
ALL_TEMPLATES: list[PromptTemplate] = [PELAGICS_TEMPLATE, MACRO_TEMPLATE, DIVER_TEMPLATE]
MOTORCYCLE_TEMPLATES: list[PromptTemplate] = [
    MOTO_RIDING_TEMPLATE,
    MOTO_SCENIC_TEMPLATE,
    MOTO_GROUP_TEMPLATE,
]
ALL_DOMAIN_TEMPLATES: list[PromptTemplate] = (
    UNDERWATER_DOMAIN.templates + MOTORCYCLE_DOMAIN.templates + GENERIC_DOMAIN.templates
)


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
    cat_clean = category.strip().lower()
    if cat_clean:
        for tpl in dom_config.templates:
            if tpl.name.lower() == cat_clean:
                return tpl
        if dom_config.fallback_template.name.lower() == cat_clean:
            return dom_config.fallback_template

    # 3. Check subject string against templates in current domain
    subj_clean = subject.strip().lower()
    if subj_clean:
        for tpl in dom_config.templates:
            for kw in tpl.keywords:
                if kw.lower() in subj_clean:
                    return tpl

        # 4. Check subject string across other domains only if domain was not specified
        if not domain:
            for other_key, other_dom in reg.items():
                if other_key == dom_key:
                    continue
                for tpl in other_dom.templates:
                    for kw in tpl.keywords:
                        if kw.lower() in subj_clean:
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
