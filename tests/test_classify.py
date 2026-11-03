"""Tests for standalone Pass 0 classification and scan reuse (Issue #47)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from scripts.dive_clip import build_parser, cmd_classify, cmd_scan
from utils.models import ClassificationOutput, ClipClassification

# Dynamically load scripts/02_scan_footage.py
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_scan_spec = importlib.util.spec_from_file_location(
    "scripts_02_scan_footage", _SCRIPTS_DIR / "02_scan_footage.py"
)
scan = importlib.util.module_from_spec(_scan_spec)
_scan_spec.loader.exec_module(scan)


def test_select_representative_frames_bounded() -> None:
    """select_representative_frames should boundedly pick evenly spaced frames."""
    frames = [{"frame_number": i, "timestamp": float(i)} for i in range(10)]

    # Exactly 3 frames from 10 frames
    selected = scan.select_representative_frames(frames, count=3)
    assert len(selected) == 3
    # With step = 10 / 4 = 2.5 -> indices [2, 5, 7]
    assert [f["frame_number"] for f in selected] == [2, 5, 7]

    # Fewer frames than requested count returns all frames
    small_frames = [{"frame_number": 1}, {"frame_number": 2}]
    assert scan.select_representative_frames(small_frames, count=5) == small_frames

    # Count <= 0 returns empty list
    assert scan.select_representative_frames(frames, count=0) == []
    assert scan.select_representative_frames([], count=3) == []


def test_aggregate_frame_classifications_no_usable_responses() -> None:
    """No usable frame results must result in uncertain generic classification with None confidence."""
    domain, conf, uncertain, reason = scan.aggregate_frame_classifications([])
    assert domain == "generic"
    assert conf is None
    assert uncertain is True
    assert reason is not None
    assert "no usable responses" in reason.lower()


def test_aggregate_frame_classifications_confident_agreement() -> None:
    """Sampled frames agreeing with mean confidence >= 0.6 should be confident."""
    frame_results = [
        {"domain": "underwater", "confidence": 0.8},
        {"domain": "underwater", "confidence": 0.9},
    ]
    domain, conf, uncertain, reason = scan.aggregate_frame_classifications(frame_results)
    assert domain == "underwater"
    assert conf == 0.85
    assert uncertain is False
    assert reason is None


def test_aggregate_frame_classifications_low_confidence() -> None:
    """Sampled frames agreeing with mean confidence < 0.6 must be marked uncertain."""
    frame_results = [
        {"domain": "underwater", "confidence": 0.4},
        {"domain": "underwater", "confidence": 0.5},
    ]
    domain, conf, uncertain, reason = scan.aggregate_frame_classifications(frame_results)
    assert domain == "underwater"
    assert conf == 0.45
    assert uncertain is True
    assert reason is not None
    assert "low confidence" in reason.lower()


def test_aggregate_frame_classifications_disagreement() -> None:
    """Sampled frames disagreeing on domain must be marked uncertain."""
    frame_results = [
        {"domain": "underwater", "confidence": 0.9},
        {"domain": "motorcycle", "confidence": 0.7},
    ]
    domain, conf, uncertain, reason = scan.aggregate_frame_classifications(frame_results)
    assert uncertain is True
    assert reason is not None
    assert "disagree" in reason.lower()
    # Winning domain is the one with higher confidence/count
    assert domain == "underwater"
    assert conf == 0.9


def test_aggregate_frame_classifications_preserves_clip_with_usable_high_quality() -> None:
    """A clip with at least one high-quality usable frame must not be discarded as unusable."""
    # Simulates GX010830-turtle: 1 good underwater frame, 2 overexposed/flared frames
    frame_results = [
        {"domain": "underwater", "confidence": 0.95, "usable": True, "quality_score": 7},
        {"domain": "underwater", "confidence": 0.90, "usable": False, "quality_score": 1},
        {"domain": "generic", "confidence": 0.85, "usable": False, "quality_score": 1},
    ]
    domain, _conf, _uncertain, _reason = scan.aggregate_frame_classifications(frame_results)
    assert domain != "unusable"
    assert domain == "underwater"


def test_classify_clip_calls_only_classify_phase() -> None:
    """classify_clip must call Ollama with phase='classify' only, never coarse or fine."""
    clip = {
        "clip_id": "dive_01",
        "file_name": "dive_01.mp4",
        "file_path": "/videos/dive_01.mp4",
    }
    sample_frames = [
        {"frame_path": "/tmp/f1.jpg", "timestamp": 1.0, "frame_number": 1},
        {"frame_path": "/tmp/f2.jpg", "timestamp": 5.0, "frame_number": 2},
    ]
    registry = scan.load_custom_templates({})

    phases_called = []

    def mock_call(host, model, img, prompt, timeout, client=None, phase="unspecified", **kwargs):
        phases_called.append(phase)
        return {"domain": "underwater", "confidence": 0.85}

    client = MagicMock()
    with patch.object(scan, "call_ollama", side_effect=mock_call):
        res = scan.classify_clip(
            clip=clip,
            sample_frames=sample_frames,
            client=client,
            ollama_host="http://localhost:11434",
            ollama_model="qwen2.5-vl:7b",
            ollama_timeout=90,
            registry=registry,
        )

    assert isinstance(res, ClipClassification)
    assert res.clip_id == "dive_01"
    assert res.domain == "underwater"
    assert res.confidence == 0.85
    assert res.uncertain is False
    assert set(phases_called) == {"classify"}
    assert "coarse" not in phases_called
    assert "fine" not in phases_called


def test_classify_clip_handles_unusable_responses() -> None:
    """Missing or unparseable responses must become uncertain rather than confident generic."""
    clip = {
        "clip_id": "unknown_01",
        "file_name": "unknown_01.mp4",
        "file_path": "/videos/unknown_01.mp4",
    }
    sample_frames = [{"frame_path": "/tmp/f1.jpg", "timestamp": 1.0, "frame_number": 1}]
    registry = scan.load_custom_templates({})

    # Ollama returns None (timeout/failure)
    with patch.object(scan, "call_ollama", return_value=None):
        res = scan.classify_clip(
            clip=clip,
            sample_frames=sample_frames,
            client=MagicMock(),
            ollama_host="http://localhost:11434",
            ollama_model="qwen2.5-vl:7b",
            ollama_timeout=90,
            registry=registry,
        )

    assert res.domain == "generic"
    assert res.suggested_domain is None
    assert res.confidence is None
    assert res.uncertain is True
    assert res.reason is not None


def test_classify_clip_aggregates_suggested_domain() -> None:
    """Generic clips should aggregate suggested_domain by frequency and confidence."""
    clip = {
        "clip_id": "gen_01",
        "file_name": "gen_01.mp4",
        "file_path": "/videos/gen_01.mp4",
    }
    sample_frames = [
        {"frame_path": "/tmp/f1.jpg", "timestamp": 1.0, "frame_number": 1},
        {"frame_path": "/tmp/f2.jpg", "timestamp": 2.0, "frame_number": 2},
        {"frame_path": "/tmp/f3.jpg", "timestamp": 3.0, "frame_number": 3},
    ]
    registry = scan.load_custom_templates({})

    mock_responses = [
        {
            "domain": "generic",
            "suggested_domain": "mountain_biking",
            "confidence": 0.8,
            "usable": True,
        },
        {
            "domain": "generic",
            "suggested_domain": "mountain_biking",
            "confidence": 0.9,
            "usable": True,
        },
        {"domain": "generic", "suggested_domain": "hiking", "confidence": 0.7, "usable": True},
    ]

    with patch.object(scan, "call_ollama", side_effect=mock_responses):
        res = scan.classify_clip(
            clip=clip,
            sample_frames=sample_frames,
            client=MagicMock(),
            ollama_host="http://localhost:11434",
            ollama_model="qwen2.5-vl:7b",
            ollama_timeout=90,
            registry=registry,
        )

    assert res.domain == "generic"
    assert res.suggested_domain == "mountain_biking"
    assert res.confidence == 0.8
    assert res.uncertain is False


def test_load_and_validate_classifications(tmp_path: Path) -> None:
    """load_and_validate_classifications should validate JSON schema and reject malformed files."""
    valid_file = tmp_path / "valid.json"
    valid_payload = {
        "total_clips": 1,
        "classifications": [
            {
                "clip_id": "c1",
                "file_name": "c1.mp4",
                "file_path": "/videos/c1.mp4",
                "domain": "motorcycle",
                "confidence": 0.95,
                "uncertain": False,
            }
        ],
    }
    valid_file.write_text(json.dumps(valid_payload), encoding="utf-8")

    loaded = scan.load_and_validate_classifications(valid_file)
    assert len(loaded) == 1
    assert loaded[0].clip_id == "c1"
    assert loaded[0].domain == "motorcycle"

    # Malformed JSON (syntax error)
    bad_syntax_file = tmp_path / "bad_syntax.json"
    bad_syntax_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        scan.load_and_validate_classifications(bad_syntax_file)

    # Malformed Schema (missing required fields)
    bad_schema_file = tmp_path / "bad_schema.json"
    bad_schema_file.write_text(
        json.dumps({"classifications": [{"some_field": 123}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        scan.load_and_validate_classifications(bad_schema_file)


def test_merge_classifications_into_manifest() -> None:
    """Confident labels should be merged, uncertain ignored, and unmatched reported."""
    clips = [
        {"clip_id": "c1", "file_name": "c1.mp4", "file_path": "/v/c1.mp4"},
        {"clip_id": "c2", "file_name": "c2.mp4", "file_path": "/v/c2.mp4"},
        {"clip_id": "c3", "file_name": "c3.mp4", "file_path": "/v/c3.mp4"},
    ]
    classifications = [
        # Confident match
        ClipClassification(
            clip_id="c1",
            file_name="c1.mp4",
            file_path="/v/c1.mp4",
            domain="underwater",
            confidence=0.9,
            uncertain=False,
        ),
        # Uncertain match - should be ignored
        ClipClassification(
            clip_id="c2",
            file_name="c2.mp4",
            file_path="/v/c2.mp4",
            domain="generic",
            confidence=0.4,
            uncertain=True,
            reason="low confidence",
        ),
        # Unmatched record
        ClipClassification(
            clip_id="c99",
            file_name="c99.mp4",
            file_path="/v/c99.mp4",
            domain="motorcycle",
            confidence=0.95,
            uncertain=False,
        ),
    ]

    reused, ignored, unmatched = scan.merge_classifications_into_manifest(clips, classifications)
    assert reused == 1
    assert ignored == 1
    assert unmatched == ["c99"]

    # c1 got domain set
    assert clips[0]["domain"] == "underwater"
    # c2 domain was NOT set (left for scan Pass 0)
    assert "domain" not in clips[1]
    # c3 domain untouched
    assert "domain" not in clips[2]


def test_build_parser_classify() -> None:
    """build_parser should configure the classify subcommand and scan --classifications flag."""
    parser = build_parser()
    args = parser.parse_args(["classify", "--input", "./footage", "--output", "./out.json"])
    assert args.command == "classify"
    assert args.input == "./footage"
    assert args.output == "./out.json"

    scan_args = parser.parse_args(["scan", "--classifications", "./saved.json"])
    assert scan_args.classifications == "./saved.json"


def test_cmd_classify_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """cmd_classify should ingest, classify with Pass 0, write validated JSON, and print summary."""
    footage_dir = tmp_path / "footage"
    footage_dir.mkdir()
    (footage_dir / "vid1.mp4").touch()
    (footage_dir / "vid2.mp4").touch()

    output_json = tmp_path / "classifications.json"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"""
general:
  output_dir: "{tmp_path / "outputs"}"
scan_parameters:
  classify_frames_count: 2
  coarse_scale: "426:-2"
ollama:
  host: "http://localhost:11434"
  model: "qwen2.5-vl:7b"
""",
        encoding="utf-8",
    )

    fake_clips = [
        {"file_name": "vid1.mp4", "file_path": str(footage_dir / "vid1.mp4"), "duration": 10.0},
        {"file_name": "vid2.mp4", "file_path": str(footage_dir / "vid2.mp4"), "duration": 10.0},
    ]

    mock_classification = [
        ClipClassification(
            clip_id="vid1",
            file_name="vid1.mp4",
            file_path=str(footage_dir / "vid1.mp4"),
            domain="underwater",
            confidence=0.88,
            uncertain=False,
        ),
        ClipClassification(
            clip_id="vid2",
            file_name="vid2.mp4",
            file_path=str(footage_dir / "vid2.mp4"),
            domain="generic",
            confidence=0.45,
            uncertain=True,
            reason="low confidence",
        ),
    ]

    parser = build_parser()
    args = parser.parse_args([
        "classify",
        "--input",
        str(footage_dir),
        "--output",
        str(output_json),
        "--config",
        str(config_file),
    ])

    with (
        patch("scripts.dive_clip._load_ingest_module") as mock_ingest_loader,
        patch("scripts.dive_clip._load_scan_module") as mock_scan_loader,
    ):
        mock_ingest = MagicMock()
        mock_ingest.ingest_folder.return_value = fake_clips
        mock_ingest_loader.return_value = mock_ingest

        mock_scan = MagicMock()
        mock_scan.classify_footage.return_value = mock_classification
        mock_scan_loader.return_value = mock_scan

        rc = cmd_classify(args)

    assert rc == 0
    assert output_json.exists()
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["total_clips"] == 2
    assert len(data["classifications"]) == 2
    assert data["classifications"][0]["domain"] == "underwater"
    assert data["classifications"][1]["uncertain"] is True


def test_cmd_classify_empty_folder(tmp_path: Path) -> None:
    """cmd_classify should save empty classifications and return 0 if no video files found."""
    footage_dir = tmp_path / "empty_footage"
    footage_dir.mkdir()
    output_json = tmp_path / "empty_out.json"

    config_file = tmp_path / "config.yaml"
    config_file.write_text("general:\n  output_dir: ./out\n", encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args([
        "classify",
        "--input",
        str(footage_dir),
        "--output",
        str(output_json),
        "--config",
        str(config_file),
    ])

    with patch("scripts.dive_clip._load_ingest_module") as mock_ingest_loader:
        mock_ingest = MagicMock()
        mock_ingest.ingest_folder.return_value = []
        mock_ingest_loader.return_value = mock_ingest

        rc = cmd_classify(args)

    assert rc == 0
    assert output_json.exists()
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["total_clips"] == 0
    assert data["classifications"] == []


def test_cmd_classify_missing_input(tmp_path: Path) -> None:
    """cmd_classify should return 2 if no input folder is provided."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("general: {}\n", encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(["classify", "--config", str(config_file)])
    assert cmd_classify(args) == 2


def test_cmd_scan_with_classifications_argument(tmp_path: Path) -> None:
    """cmd_scan should pass --classifications down to 02_scan_footage.py."""
    footage_dir = tmp_path / "footage"
    footage_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    class_file = tmp_path / "classifications.json"
    class_file.write_text('{"total_clips": 0, "classifications": []}', encoding="utf-8")

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"general:\n  input_dir: '{footage_dir}'\n  output_dir: '{output_dir}'\n",
        encoding="utf-8",
    )

    parser = build_parser()
    args = parser.parse_args([
        "scan",
        "--config",
        str(config_file),
        "--classifications",
        str(class_file),
    ])

    run_commands = []

    def mock_run(cmd, description):
        run_commands.append(cmd)
        # Pretend ingest and scan succeed
        if "00_ingest_media.py" in cmd[1]:
            # Create dummy manifest
            manifest_file = output_dir / "runs" / Path(cmd[cmd.index("--output") + 1]).name
            manifest_file.parent.mkdir(parents=True, exist_ok=True)
            manifest_file.write_text('{"total_clips": 0, "clips": []}', encoding="utf-8")
        if "02_scan_footage.py" in cmd[1]:
            # Create dummy detections
            det_file = Path(cmd[cmd.index("--run-dir") + 1]) / "detections_refined.json"
            det_file.parent.mkdir(parents=True, exist_ok=True)
            det_file.write_text(
                '{"total_clips": 0, "total_detections": 0, "detections": []}', encoding="utf-8"
            )
        return 0

    with (
        patch("scripts.dive_clip.run", side_effect=mock_run),
        patch("scripts.dive_clip._run_background_render_worker"),
    ):
        rc = cmd_scan(args)

    assert rc == 0
    # Verify scan_cmd included --classifications
    scan_call = [cmd for cmd in run_commands if any("02_scan_footage.py" in arg for arg in cmd)]
    assert len(scan_call) == 1
    assert "--classifications" in scan_call[0]
    idx = scan_call[0].index("--classifications")
    assert scan_call[0][idx + 1] == str(class_file.resolve())


def test_cmd_scan_with_missing_classifications_file(tmp_path: Path) -> None:
    """cmd_scan should fail early with exit code 1 if --classifications file does not exist."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "general:\n  input_dir: '/tmp'\n  output_dir: '/tmp'\n", encoding="utf-8"
    )

    parser = build_parser()
    args = parser.parse_args([
        "scan",
        "--config",
        str(config_file),
        "--classifications",
        str(tmp_path / "non_existent.json"),
    ])
    assert cmd_scan(args) == 1


def test_scan_footage_main_with_classifications_reuse(tmp_path: Path) -> None:
    """02_scan_footage.py main should merge confident classifications and handle scan."""
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(
        json.dumps({
            "project_name": "test",
            "total_clips": 1,
            "clips": [
                {
                    "clip_id": "clip_01",
                    "file_name": "clip_01.mp4",
                    "file_path": str(tmp_path / "clip_01.mp4"),
                    "duration": "10.0",
                }
            ],
        }),
        encoding="utf-8",
    )
    (tmp_path / "clip_01.mp4").touch()

    class_file = tmp_path / "classifications.json"
    class_file.write_text(
        json.dumps({
            "total_clips": 1,
            "classifications": [
                {
                    "clip_id": "clip_01",
                    "file_name": "clip_01.mp4",
                    "file_path": str(tmp_path / "clip_01.mp4"),
                    "domain": "motorcycle",
                    "confidence": 0.9,
                    "uncertain": False,
                }
            ],
        }),
        encoding="utf-8",
    )

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
general:
  output_dir: "."
scan_parameters:
  domain: "auto"
  classify_clips: "auto"
  classify_frames_count: 2
ollama:
  host: "http://localhost:11434"
  model: "qwen2.5-vl:7b"
""",
        encoding="utf-8",
    )

    out_dir = tmp_path / "output"
    out_dir.mkdir()

    with (
        patch(
            "sys.argv",
            [
                "02_scan_footage.py",
                "--config",
                str(config_file),
                "--manifest",
                str(manifest_file),
                "--output-dir",
                str(out_dir),
                "--classifications",
                str(class_file),
            ],
        ),
        patch.object(scan, "process_clip", return_value=[]) as mock_process,
    ):
        scan.main()

    # process_clip should receive clip with reused domain="motorcycle"
    assert mock_process.call_count == 1
    passed_clip = mock_process.call_args[0][0]
    assert passed_clip["domain"] == "motorcycle"


def test_cmd_classify_specialize_generic(tmp_path: Path) -> None:
    """classify with --specialize-generic should generate prompt_candidates.json for generic clips."""
    footage_dir = tmp_path / "footage"
    footage_dir.mkdir()
    out_file = tmp_path / "classifications.json"
    cand_file = tmp_path / "custom_candidates.json"

    fake_clips = [
        {"clip_id": "c1", "file_name": "park.mp4", "file_path": str(footage_dir / "park.mp4")},
    ]
    fake_classifications = [
        ClipClassification(
            clip_id="c1",
            file_name="park.mp4",
            file_path=str(footage_dir / "park.mp4"),
            domain="generic",
            confidence=0.8,
            environment="city park",
            primary_subjects=["dog", "frisbee"],
        ),
    ]

    parser = build_parser()
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

        rc = cmd_classify(args)

    assert rc == 0
    assert out_file.exists()
    assert cand_file.exists()

    cand_data = json.loads(cand_file.read_text(encoding="utf-8"))
    assert cand_data["name"].startswith("generic_")
    assert cand_data["inferred_context"]["environment"] == "city park"
    assert "dog" in cand_data["inferred_context"]["primary_subjects"]
    assert len(cand_data["candidates"]) == 1


def test_cmd_classify_candidate_output_requires_specialize_generic(tmp_path: Path) -> None:
    """--candidate-output without --specialize-generic should be rejected."""
    parser = build_parser()
    args = parser.parse_args([
        "classify",
        "--input",
        str(tmp_path),
        "--candidate-output",
        str(tmp_path / "cand.json"),
    ])
    assert cmd_classify(args) == 1


def test_scan_candidate_isolation() -> None:
    """Highlight scanning must isolate against candidate templates and only run registered templates."""
    from utils.prompt_templates import BUILTIN_DOMAINS, GENERIC_HIGHLIGHT_TEMPLATE

    registry = scan.load_custom_templates({})
    assert set(registry.keys()).issubset(set(BUILTIN_DOMAINS.keys()))
    t_generic = scan.resolve_template_for_subject("any subject", domain="generic")
    assert t_generic.name == GENERIC_HIGHLIGHT_TEMPLATE.name


def test_build_parser_classify_resume() -> None:
    """build_parser should accept --resume for classify subcommand."""
    parser = build_parser()
    args1 = parser.parse_args(["classify", "--resume"])
    assert args1.resume is True

    args2 = parser.parse_args(["classify", "--resume", "run-20261102-120000"])
    assert args2.resume == "run-20261102-120000"


def test_classify_footage_incremental_save(tmp_path: Path) -> None:
    """classify_footage should atomically write intermediate results after each clip."""
    clips = [
        {"file_name": "vid1.mp4", "file_path": str(tmp_path / "vid1.mp4"), "duration": 10.0},
        {"file_name": "vid2.mp4", "file_path": str(tmp_path / "vid2.mp4"), "duration": 10.0},
    ]
    for c in clips:
        Path(c["file_path"]).touch()

    out_file = tmp_path / "classifications.json"
    saved_snapshots: list[int] = []

    mock_classifications = [
        ClipClassification(
            clip_id="vid1",
            file_name="vid1.mp4",
            file_path=str(tmp_path / "vid1.mp4"),
            domain="underwater",
            confidence=0.9,
            uncertain=False,
        ),
        ClipClassification(
            clip_id="vid2",
            file_name="vid2.mp4",
            file_path=str(tmp_path / "vid2.mp4"),
            domain="generic",
            confidence=0.8,
            uncertain=False,
        ),
    ]

    with (
        patch.object(scan, "extract_classification_frames") as mock_extract,
        patch.object(scan, "classify_clip") as mock_classify,
    ):
        mock_extract.return_value = ["frame1.jpg"]

        def side_effect_classify(clip: Any, **kwargs: Any) -> ClipClassification:
            c = (
                mock_classifications[0]
                if clip["file_name"] == "vid1.mp4"
                else mock_classifications[1]
            )
            return c

        mock_classify.side_effect = side_effect_classify

        # Record whenever out_file is written
        orig_save = getattr(scan, "save_classifications_atomic", None)

        def spy_save(path: Any, output_obj: Any) -> None:
            if orig_save:
                orig_save(path, output_obj)
            else:
                Path(path).write_text(output_obj.model_dump_json(indent=2), encoding="utf-8")
            if Path(path) == out_file:
                saved_snapshots.append(output_obj.total_clips)

        with patch.object(scan, "save_classifications_atomic", side_effect=spy_save):
            results = scan.classify_footage(clips, {}, output_file=out_file)

    assert len(results) == 2
    assert saved_snapshots == [1, 2]
    assert out_file.exists()
    final_data = json.loads(out_file.read_text(encoding="utf-8"))
    assert final_data["total_clips"] == 2
    assert len(final_data["classifications"]) == 2


def test_classify_footage_resume(tmp_path: Path) -> None:
    """classify_footage should skip already-classified clips when resume=True."""
    clips = [
        {"file_name": "vid1.mp4", "file_path": str(tmp_path / "vid1.mp4"), "duration": 10.0},
        {"file_name": "vid2.mp4", "file_path": str(tmp_path / "vid2.mp4"), "duration": 10.0},
    ]
    for c in clips:
        Path(c["file_path"]).touch()

    out_file = tmp_path / "classifications.json"
    initial_output = ClassificationOutput(
        total_clips=1,
        classifications=[
            ClipClassification(
                clip_id="vid1",
                file_name="vid1.mp4",
                file_path=str(tmp_path / "vid1.mp4"),
                domain="underwater",
                confidence=0.95,
                uncertain=False,
            )
        ],
    )
    out_file.write_text(initial_output.model_dump_json(indent=2), encoding="utf-8")

    with (
        patch.object(scan, "extract_classification_frames") as mock_extract,
        patch.object(scan, "classify_clip") as mock_classify,
    ):
        mock_extract.return_value = ["frame1.jpg"]
        mock_classify.return_value = ClipClassification(
            clip_id="vid2",
            file_name="vid2.mp4",
            file_path=str(tmp_path / "vid2.mp4"),
            domain="motorcycle",
            confidence=0.85,
            uncertain=False,
        )

        results = scan.classify_footage(clips, {}, output_file=out_file, resume=True)

    # vid1 was resumed, so mock_classify was only called once (for vid2)
    assert mock_classify.call_count == 1
    assert len(results) == 2
    assert results[0].domain == "underwater"
    assert results[1].domain == "motorcycle"

    final_data = json.loads(out_file.read_text(encoding="utf-8"))
    assert final_data["total_clips"] == 2


def test_cmd_classify_creates_run_dir_and_symlink(tmp_path: Path) -> None:
    """cmd_classify should create runs/run-<timestamp> and update latest symlink."""
    footage_dir = tmp_path / "footage"
    footage_dir.mkdir()
    (footage_dir / "vid1.mp4").touch()

    output_dir = tmp_path / "outputs"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f'general:\n  output_dir: "{output_dir}"\n', encoding="utf-8")

    fake_clips = [{"file_name": "vid1.mp4", "file_path": str(footage_dir / "vid1.mp4")}]
    mock_classification = [
        ClipClassification(
            clip_id="vid1",
            file_name="vid1.mp4",
            file_path=str(footage_dir / "vid1.mp4"),
            domain="underwater",
            confidence=0.9,
            uncertain=False,
        )
    ]

    parser = build_parser()
    args = parser.parse_args([
        "classify",
        "--input",
        str(footage_dir),
        "--output",
        str(output_dir),
        "--config",
        str(config_file),
    ])

    with (
        patch("scripts.dive_clip._load_ingest_module") as mock_ingest_loader,
        patch("scripts.dive_clip._load_scan_module") as mock_scan_loader,
    ):
        mock_ingest = MagicMock()
        mock_ingest.ingest_folder.return_value = fake_clips
        mock_ingest_loader.return_value = mock_ingest

        mock_scan = MagicMock()
        mock_scan.classify_footage.return_value = mock_classification
        mock_scan_loader.return_value = mock_scan

        rc = cmd_classify(args)

    assert rc == 0
    runs_dir = output_dir / "runs"
    assert runs_dir.exists()
    run_dirs = list(runs_dir.glob("run-*"))
    assert len(run_dirs) == 1
    active_run = run_dirs[0]

    assert (active_run / "classifications.json").exists()
    assert (active_run / "manifest.json").exists()
    assert (active_run / "classify.log").exists()

    # latest symlink or latest.txt
    latest = output_dir / "latest"
    assert latest.exists()
    assert (latest / "classifications.json").exists()

    # root classifications.json copy
    assert (output_dir / "classifications.json").exists()


def test_cmd_classify_resume_existing_run(tmp_path: Path) -> None:
    """cmd_classify with --resume should resolve and resume in the existing run directory."""
    footage_dir = tmp_path / "footage"
    footage_dir.mkdir()
    (footage_dir / "vid1.mp4").touch()

    output_dir = tmp_path / "outputs"
    existing_run = output_dir / "runs" / "run-20261102-100000"
    existing_run.mkdir(parents=True)
    (existing_run / "classifications.json").write_text(
        '{"total_clips": 1, "classifications": []}', encoding="utf-8"
    )

    # Point latest to existing_run
    latest = output_dir / "latest"
    latest.symlink_to(Path("runs/run-20261102-100000"))

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f'general:\n  output_dir: "{output_dir}"\n', encoding="utf-8")

    fake_clips = [{"file_name": "vid1.mp4", "file_path": str(footage_dir / "vid1.mp4")}]

    parser = build_parser()
    args = parser.parse_args([
        "classify",
        "--input",
        str(footage_dir),
        "--output",
        str(output_dir),
        "--resume",
        "--config",
        str(config_file),
    ])

    with (
        patch("scripts.dive_clip._load_ingest_module") as mock_ingest_loader,
        patch("scripts.dive_clip._load_scan_module") as mock_scan_loader,
    ):
        mock_ingest = MagicMock()
        mock_ingest.ingest_folder.return_value = fake_clips
        mock_ingest_loader.return_value = mock_ingest

        mock_scan = MagicMock()
        mock_scan.classify_footage.return_value = []
        mock_scan_loader.return_value = mock_scan

        rc = cmd_classify(args)

    assert rc == 0
    # Did not create a new run directory, reused existing
    run_dirs = list((output_dir / "runs").glob("run-*"))
    assert len(run_dirs) == 1
    assert run_dirs[0].name == "run-20261102-100000"
    mock_scan.classify_footage.assert_called_once()
    _, kwargs = mock_scan.classify_footage.call_args
    assert kwargs.get("resume") is True


def test_classify_footage_uses_configured_vlm_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (#66): standalone classify must honor vlm.provider, not fall back to Ollama."""
    from utils.vlm_client import OpenRouterClient

    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    clip_path = tmp_path / "vid1.mp4"
    clip_path.touch()
    clips = [{"file_name": "vid1.mp4", "file_path": str(clip_path), "duration": 10.0}]
    classification = ClipClassification(
        clip_id="vid1",
        file_name="vid1.mp4",
        file_path=str(clip_path),
        domain="generic",
        confidence=0.8,
        uncertain=False,
    )
    seen: list[Any] = []

    def fake_classify(clip: Any, **kwargs: Any) -> ClipClassification:
        seen.append(kwargs.get("client"))
        return classification

    with (
        patch.object(scan, "extract_classification_frames", return_value=["f.jpg"]),
        patch.object(scan, "classify_clip", side_effect=fake_classify),
    ):
        scan.classify_footage(clips, {"vlm": {"provider": "openrouter"}})

    assert len(seen) == 1
    assert isinstance(seen[0], OpenRouterClient)
