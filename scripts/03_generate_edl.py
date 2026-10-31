"""CMX 3600 EDL & CSV marker generator.

Converts refined AI detections into a DaVinci Resolve-importable
CMX 3600 EDL with colored markers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError

from utils.config import ConfigError, load_config
from utils.models import DetectionsRefined
from utils.timecode_math import Timecode

COLOR_MAP = {
    "Cyan": "Cyan",
    "Yellow": "Yellow",
    "Green": "Green",
}


def generate_edl(detections: DetectionsRefined, output_path: Path) -> None:
    """Generate CMX 3600 EDL file from refined detections."""
    lines = [
        "TITLE: MADAGASCAR_AI_MARKERS",
        "FCM: NON-DROP FRAME",
        "",
    ]

    edit_number = 1
    for det in detections.detections:
        fps = float(det.fps) if det.fps else 25.0
        start_tc = Timecode.from_seconds(det.start_time, fps)
        end_tc = Timecode.from_seconds(det.end_time, fps)

        color = COLOR_MAP.get(det.color, "Green")
        source_name = det.file_name.upper().replace(".", "_")[:8]

        lines.append(
            f"{edit_number:03d}  AX       V     C        {start_tc} {end_tc} "
            f"01:00:00:00 01:00:00:00"
        )
        lines.append(f"* FROM CLIP NAME: {source_name}")
        lines.append(f"* LOC: {start_tc} {color} {det.subject} - {det.description}")
        lines.append("")
        edit_number += 1

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"EDL written: {output_path} ({edit_number - 1} events)")


def generate_csv(detections: DetectionsRefined, output_path: Path) -> None:
    """Generate CSV marker file from refined detections."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Clip Name",
            "Start Timecode",
            "End Timecode",
            "Color",
            "Subject",
            "Description",
            "Confidence",
            "Quality Score",
        ])

        def _sanitize_csv_cell(val: object) -> object:
            if isinstance(val, str) and val and val[0] in ("=", "+", "-", "@", "\t", "\r"):
                return f"'{val}"
            return val

        for det in detections.detections:
            fps = float(det.fps) if det.fps else 25.0
            start_tc = str(Timecode.from_seconds(det.start_time, fps))
            end_tc = str(Timecode.from_seconds(det.end_time, fps))
            writer.writerow([
                _sanitize_csv_cell(det.file_name),
                start_tc,
                end_tc,
                det.color,
                _sanitize_csv_cell(det.subject),
                _sanitize_csv_cell(det.description),
                det.confidence,
                det.quality_score,
            ])

    print(f"CSV written: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CMX 3600 EDL and CSV markers from detections"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

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

    edl_path = output_dir / "markers_import.edl"
    csv_path = output_dir / "markers_import.csv"

    generate_edl(detections, edl_path)
    generate_csv(detections, csv_path)

    print(
        "Done. Import markers_import.edl in DaVinci Resolve "
        "(Timeline -> Import -> Timeline Markers from EDL)."
    )


if __name__ == "__main__":
    main()
