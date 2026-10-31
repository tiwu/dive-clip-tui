"""Dynamic prompt templates for Qwen2.5-VL based on keyword categories."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptTemplate:
    name: str
    keywords: list[str]
    system_prompt: str
    coarse_prompt: str
    fine_prompt: str
    color: str

    def __init__(
        self,
        name: str,
        keywords: list[str],
        system_prompt: str,
        coarse_prompt: str = "",
        fine_prompt: str = "",
        color: str = "",
        user_prompt: str | None = None,
    ) -> None:
        self.name = name
        self.keywords = keywords
        self.system_prompt = system_prompt
        self.color = color
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
    system_prompt=("You are an underwater video analyst specializing in large sea life."),
    coarse_prompt=(
        "Analyze this underwater frame for candidate screening.\n"
        "Target subjects: pelagic species (turtles, sharks, rays, mantas).\n\n"
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
    system_prompt=("You are an underwater video analyst specializing in macro underwater life."),
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
        "You are an underwater video analyst specializing in diver action and composition."
    ),
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
)

# Cyan: Broad marine-life fallback (corals, worms, nudibranchs, fish, octopus, moray eels, sharks, turtles, whales, frogfish, crocodilefish, seahorses, stonefish)
# Fallback-only template: not in ALL_TEMPLATES so it is never keyword-matched.
BROAD_MARINE_TEMPLATE = PromptTemplate(
    name="broad_marine",
    keywords=[],
    system_prompt=(
        "You are an underwater video analyst specializing in comprehensive "
        "marine life identification."
    ),
    coarse_prompt=(
        "Analyze this underwater frame for candidate screening.\n"
        "Target subjects: any marine life: corals, worms, nudibranchs, fish, octopus, moray eels, "
        "sharks, turtles, whales, frogfish, crocodilefish, seahorses, stonefish, and other sea creatures.\n\n"
        "CAMERA ORIENTATION & USABILITY CHECK:\n"
        "- The camera must be intentionally aimed and upright (right-side up).\n"
        "- If the camera is upside down, tumbling, dangling on a clip/lanyard, spinning, "
        "or pointed accidentally at the seabed/hull/diver gear, you MUST reject the frame:\n"
        "  Set 'detected': false, 'confidence': 0.0, 'quality_score': 1.\n"
        "- Even if background corals or fish are present, if the camera is dangling or inverted, "
        "mark 'detected': false.\n\n"
        "Return a JSON object with:\n"
        "- detected: boolean (true ONLY if marine life is present AND camera is upright and intentionally operated)\n"
        "- subject: string (species name or 'none')\n"
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
