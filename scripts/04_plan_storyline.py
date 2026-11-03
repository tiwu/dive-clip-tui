"""Cloud API storyline assembler.

Takes detections_refined.json (text only, few thousand tokens) and sends it
to a frontier LLM (Claude 3.5 Sonnet via Anthropic, or Gemini via Google GenAI)
to produce a professional edit narrative and shot sequence.

Output: outputs/storyline_edit_guide.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError

from utils.models import DetectionsRefined
from utils.selection import select_diverse_highlights


def build_prompt(detections: DetectionsRefined, target_duration: int = 4) -> str:
    """Build the prompt for the cloud LLM.

    Truncates detections to the highest-quality ones if the prompt would
    exceed a safe token budget (~8,000 tokens ≈ 32,000 chars).
    """
    MAX_DETECTIONS = 200
    det_list = detections.detections
    if len(det_list) > MAX_DETECTIONS:
        det_list = select_diverse_highlights(det_list, top_n=MAX_DETECTIONS)

    lines = [
        "You are a professional underwater video editor and storyteller.",
        "You get a list of AI-detected highlights from underwater dive footage (Madagascar).",
        "",
        f"Goal: create an edit plan of approximately {target_duration} minutes.",
        "",
        "Detections (per clip):",
    ]

    for i, det in enumerate(det_list, 1):
        lines.append(
            f"{i}. {det.file_name} | {det.start_time:.1f}s-{det.end_time:.1f}s | "
            f"{det.subject} | quality {det.quality_score}/10 | confidence {det.confidence:.2f} | "
            f"{det.description}"
        )

    lines.extend([
        "",
        "Requirements for the edit plan:",
        "1. Determine a logical narrative arc (Establishing -> Reef Scape -> Encounters -> Macro Treasures -> Safety Stop).",
        "2. Provide a recommended shot order with timecodes.",
        "3. Suggest music and color grading per segment.",
        "4. Mark which shots can serve as B-roll.",
        "5. Take into account the colored markers (Cyan=pelagic, Yellow=macro, Green=diver/composition).",
        "",
        "Return the result as markdown with a clear timeline order.",
    ])
    return "\n".join(lines)


def call_claude(api_key: str, model: str, prompt: str) -> str:
    """Call Anthropic Claude API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, timeout=120.0)
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text if response.content else ""


def call_gemini(api_key: str, model: str, prompt: str) -> str:
    """Call Google Gemini API."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model_instance = genai.GenerativeModel(model)
    response = model_instance.generate_content(prompt, request_options={"timeout": 120})
    return response.text or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud API storyline assembler (Claude / Gemini)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config.get("general", {}).get("output_dir", "./outputs"))
    detections_path = output_dir / "detections_refined.json"

    if not detections_path.exists():
        detections_path = (
            Path(__file__).resolve().parent.parent / "outputs" / "detections_refined.json"
        )

    if not detections_path.exists():
        print(f"ERROR: detections_refined.json not found at {detections_path}")
        print("Run 02_scan_footage.py first.")
        sys.exit(1)

    with open(detections_path) as f:
        data = json.load(f)

    try:
        detections = DetectionsRefined(**data)
    except ValidationError as exc:
        print(f"ERROR: Invalid detections_refined.json: {exc}")
        sys.exit(1)

    cloud_cfg = config.get("cloud_storyline", {})
    provider = cloud_cfg.get("provider", "claude")
    model = cloud_cfg.get("model", "claude-3-5-sonnet-latest")
    target_duration = cloud_cfg.get("target_edit_duration_minutes", 4)

    prompt = build_prompt(detections, target_duration)

    if provider == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
            sys.exit(1)
        result = call_claude(api_key, model, prompt)
    elif provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            print("ERROR: GEMINI_API_KEY environment variable not set.")
            sys.exit(1)
        result = call_gemini(api_key, model, prompt)
    else:
        print(f"ERROR: Unknown provider: {provider}")
        sys.exit(1)

    output_path = output_dir / "storyline_edit_guide.md"
    output_path.write_text(result, encoding="utf-8")
    print(f"Storyline edit guide written: {output_path}")


if __name__ == "__main__":
    main()
