"""Tests for scripts/04_plan_storyline.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Load scripts/04_plan_storyline.py directly (name starts with a digit)
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_storyline_spec = importlib.util.spec_from_file_location(
    "scripts_04_plan_storyline", _SCRIPTS_DIR / "04_plan_storyline.py"
)
storyline = importlib.util.module_from_spec(_storyline_spec)
_storyline_spec.loader.exec_module(storyline)


def test_build_prompt() -> None:
    """build_prompt should create a prompt with detections and target duration."""
    from utils.models import DetectionsRefined, RefinedDetection

    det1 = RefinedDetection(
        clip_id="clip001",
        file_name="dive1.mp4",
        file_path="/tmp/dive1.mp4",
        fps="25.0",
        start_time=0.0,
        end_time=5.0,
        subject="turtle",
        confidence=0.9,
        quality_score=8,
        description="Sea turtle swimming",
        color="Cyan",
        template="pelagics",
    )
    detections = DetectionsRefined(total_clips=1, total_detections=1, detections=[det1])

    prompt = storyline.build_prompt(detections, target_duration=4)

    assert "professional underwater video editor" in prompt
    assert "Madagascar" not in prompt
    assert "4 minutes" in prompt
    assert "dive1.mp4" in prompt
    assert "turtle" in prompt
    assert "8/10" in prompt


def test_build_prompt_empty_detections() -> None:
    """build_prompt should handle empty detections list."""
    from utils.models import DetectionsRefined

    detections = DetectionsRefined(total_clips=0, total_detections=0, detections=[])

    prompt = storyline.build_prompt(detections, target_duration=4)

    assert "professional underwater video editor" in prompt
    assert "Madagascar" not in prompt
    assert "4 minutes" in prompt


def test_build_prompt_truncates_over_200_detections() -> None:
    """build_prompt should truncate to top 200 quality detections when list exceeds 200."""
    from utils.models import DetectionsRefined, RefinedDetection

    dets = [
        RefinedDetection(
            clip_id=f"c{i}",
            file_name=f"dive{i}.mp4",
            file_path=f"/tmp/dive{i}.mp4",
            fps="25.0",
            start_time=0.0,
            end_time=5.0,
            subject=f"critter_{i}",
            confidence=0.9,
            quality_score=(i % 10) + 1,
            description="Sample",
        )
        for i in range(250)
    ]
    detections = DetectionsRefined(total_clips=250, total_detections=250, detections=dets)

    prompt = storyline.build_prompt(detections, target_duration=5)

    assert "Goal: create an edit plan of approximately 5 minutes." in prompt
    assert "200. " in prompt
    assert "201. " not in prompt


def test_call_claude(monkeypatch) -> None:
    """call_claude should invoke Anthropic messages API and return text."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_msg = MagicMock()
    mock_msg.text = "# Edit Plan\n1. Establishing shot"
    mock_resp.content = [mock_msg]
    mock_client.messages.create.return_value = mock_resp

    mock_anthropic_mod = MagicMock()
    mock_anthropic_mod.Anthropic.return_value = mock_client
    monkeypatch.setattr("anthropic.Anthropic", mock_anthropic_mod.Anthropic, raising=False)

    import sys

    sys.modules["anthropic"] = mock_anthropic_mod

    result = storyline.call_claude("fake-key", "claude-3-5-sonnet", "Prompt")
    assert "Establishing shot" in result


def test_call_gemini(monkeypatch) -> None:
    """call_gemini should invoke Google GenAI model and return text."""
    from unittest.mock import MagicMock

    mock_model_instance = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "# Gemini Storyline"
    mock_model_instance.generate_content.return_value = mock_resp

    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model_instance

    import sys

    sys.modules["google.generativeai"] = mock_genai

    result = storyline.call_gemini("fake-key", "gemini-1.5-pro", "Prompt")
    assert result == "# Gemini Storyline"


def test_main_missing_config(tmp_path: Path, monkeypatch) -> None:
    """main should exit with code 1 if config file is missing."""
    import sys

    monkeypatch.setattr(
        sys, "argv", ["04_plan_storyline.py", "--config", str(tmp_path / "nonexistent.yaml")]
    )
    exit_code = None

    def fake_exit(code=0):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", fake_exit)

    with pytest.raises(SystemExit):
        storyline.main()
    assert exit_code == 1


def test_main_missing_detections(tmp_path: Path, monkeypatch) -> None:
    """main should exit with code 1 if detections_refined.json is missing."""
    import sys

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"general:\n  output_dir: '{tmp_path / 'nonexistent_out'}'\n")

    monkeypatch.setattr(sys, "argv", ["04_plan_storyline.py", "--config", str(config_path)])
    exit_code = None

    def fake_exit(code=0):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", fake_exit)

    with pytest.raises(SystemExit):
        storyline.main()
    assert exit_code == 1


def test_main_invalid_detections(tmp_path: Path, monkeypatch) -> None:
    """main should exit with code 1 if detections_refined.json has invalid schema."""
    import sys

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"general:\n  output_dir: '{output_dir}'\n")
    (output_dir / "detections_refined.json").write_text('{"invalid": "data"}')

    monkeypatch.setattr(sys, "argv", ["04_plan_storyline.py", "--config", str(config_path)])
    exit_code = None

    def fake_exit(code=0):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", fake_exit)

    with pytest.raises(SystemExit):
        storyline.main()
    assert exit_code == 1


def test_main_missing_claude_api_key(tmp_path: Path, monkeypatch) -> None:
    """main should exit with code 1 when ANTHROPIC_API_KEY is not set."""
    import json
    import sys

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"general:\n  output_dir: '{output_dir}'\ncloud_storyline:\n  provider: 'claude'\n"
    )
    detections = {"total_clips": 0, "total_detections": 0, "detections": []}
    (output_dir / "detections_refined.json").write_text(json.dumps(detections))

    monkeypatch.setattr(sys, "argv", ["04_plan_storyline.py", "--config", str(config_path)])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    exit_code = None

    def fake_exit(code=0):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", fake_exit)

    with pytest.raises(SystemExit):
        storyline.main()
    assert exit_code == 1


def test_main_missing_gemini_api_key(tmp_path: Path, monkeypatch) -> None:
    """main should exit with code 1 when GEMINI_API_KEY is not set."""
    import json
    import sys

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"general:\n  output_dir: '{output_dir}'\ncloud_storyline:\n  provider: 'gemini'\n"
    )
    detections = {"total_clips": 0, "total_detections": 0, "detections": []}
    (output_dir / "detections_refined.json").write_text(json.dumps(detections))

    monkeypatch.setattr(sys, "argv", ["04_plan_storyline.py", "--config", str(config_path)])
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    exit_code = None

    def fake_exit(code=0):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", fake_exit)

    with pytest.raises(SystemExit):
        storyline.main()
    assert exit_code == 1


def test_main_unknown_provider(tmp_path: Path, monkeypatch) -> None:
    """main should exit with code 1 for unknown cloud provider."""
    import json
    import sys

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"general:\n  output_dir: '{output_dir}'\ncloud_storyline:\n  provider: 'unsupported_llm'\n"
    )
    detections = {"total_clips": 0, "total_detections": 0, "detections": []}
    (output_dir / "detections_refined.json").write_text(json.dumps(detections))

    monkeypatch.setattr(sys, "argv", ["04_plan_storyline.py", "--config", str(config_path)])
    exit_code = None

    def fake_exit(code=0):
        nonlocal exit_code
        exit_code = code
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", fake_exit)

    with pytest.raises(SystemExit):
        storyline.main()
    assert exit_code == 1


def test_main_success_claude(tmp_path: Path, monkeypatch) -> None:
    """main should read detections, call LLM, and write storyline_edit_guide.md."""
    import json
    import sys
    from unittest.mock import patch

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"general:\n  output_dir: '{output_dir}'\n"
        "cloud_storyline:\n  provider: 'claude'\n  model: 'test-model'\n"
    )

    detections = {
        "total_clips": 1,
        "total_detections": 1,
        "detections": [
            {
                "clip_id": "c1",
                "file_name": "dive1.mp4",
                "file_path": "/dive1.mp4",
                "start_time": 0.0,
                "end_time": 5.0,
                "subject": "turtle",
                "quality_score": 8,
                "confidence": 0.9,
                "description": "turtle",
            }
        ],
    }
    (output_dir / "detections_refined.json").write_text(json.dumps(detections))

    monkeypatch.setattr(sys, "argv", ["04_plan_storyline.py", "--config", str(config_path)])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    with patch.object(storyline, "call_claude", return_value="# Storyline Output"):
        storyline.main()

    guide_path = output_dir / "storyline_edit_guide.md"
    assert guide_path.exists()
    assert guide_path.read_text(encoding="utf-8") == "# Storyline Output"


def test_deterministic_only_multi_option_and_validation(tmp_path: Path, monkeypatch) -> None:
    """--deterministic-only creates validated storyline_plans.json and storyline_edit_guide.md."""
    import json
    import sys

    from utils.models import RefinedDetection, StorylineProposalSet
    from utils.storyline import validate_and_normalize_plan

    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run-20261119-155336"
    run_dir.mkdir(parents=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"general:\n  output_dir: '{output_dir}'\n")

    manifest = {
        "clips": [
            {
                "file_name": "GX010814.MP4",
                "creation_time": "2026-09-12T11:43:00Z",
                "fps": 50.0,
                "has_audio": True,
            },
            {
                "file_name": "GX010881.MP4",
                "creation_time": "2026-09-15T08:09:00Z",
                "fps": 50.0,
                "has_audio": True,
            },
            {
                "file_name": "GX010803.MP4",
                "creation_time": "2026-09-12T08:22:00Z",
                "fps": 50.0,
                "has_audio": True,
            },
        ]
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    detections = {
        "total_clips": 3,
        "total_detections": 3,
        "detections": [
            {
                "clip_id": "c_814",
                "detection_id": "c_814_00001.50_00005.00",
                "file_name": "GX010814.MP4",
                "file_path": "/GX010814.MP4",
                "start_time": 1.5,
                "end_time": 13.0,
                "domain": "generic",
                "story_role": "establishing",
                "subject": "dive boats",
                "quality_score": 8,
                "confidence": 0.95,
                "description": "dive boats anchored",
            },
            {
                "clip_id": "c_881",
                "detection_id": "c_881_00030.50_00034.00",
                "file_name": "GX010881.MP4",
                "file_path": "/GX010881.MP4",
                "start_time": 30.5,
                "end_time": 42.0,
                "domain": "underwater",
                "template": "macro",
                "story_role": "hero_subject",
                "subject": "seahorse",
                "quality_score": 8,
                "confidence": 0.95,
                "description": "seahorse clinging to orange coral",
            },
            {
                "clip_id": "c_803",
                "detection_id": "c_803_00361.50_00365.00",
                "file_name": "GX010803.MP4",
                "file_path": "/GX010803.MP4",
                "start_time": 361.5,
                "end_time": 370.0,
                "domain": "underwater",
                "template": "pelagics",
                "story_role": "hero_subject",
                "subject": "shark",
                "quality_score": 8,
                "confidence": 0.95,
                "description": "shark in deep blue water",
            },
        ],
    }
    (run_dir / "detections_refined.json").write_text(json.dumps(detections), encoding="utf-8")

    # Unit-test validate_and_normalize_plan clamping, unknown IDs, and max_per_source_file
    det_by_id = {d["detection_id"]: RefinedDetection(**d) for d in detections["detections"]}
    raw_plan = {
        "id": "teaser",
        "title": "Teaser",
        "target_duration_seconds": 55,
        "acts": [
            {
                "act_number": 1,
                "title": "Act 1",
                "shots": [
                    {"detection_id": "hallucinated_id", "in_time": 0, "out_time": 5},
                    {"detection_id": "c_881_00030.50_00034.00", "in_time": 10.0, "out_time": 50.0},
                    {"detection_id": "c_881_00030.50_00034.00", "in_time": 31.0, "out_time": 35.0},
                ],
            }
        ],
    }
    norm_plan, warns = validate_and_normalize_plan(raw_plan, det_by_id, max_per_source_file=1)
    assert norm_plan.clip_count == 1
    assert norm_plan.shots[0].in_time == 30.5
    assert norm_plan.shots[0].out_time == 42.0
    assert norm_plan.shots[0].transition_in == "fade"
    assert norm_plan.shots[0].transition_out == "fade"
    assert any("hallucinated_id" in w for w in warns)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "04_plan_storyline.py",
            "--config",
            str(config_path),
            "--run",
            "run-20261119-155336",
            "--presets",
            "teaser,highlight",
            "--deterministic-only",
        ],
    )
    storyline.main()

    plans_path = run_dir / "storyline_plans.json"
    guide_path = run_dir / "storyline_edit_guide.md"
    assert plans_path.exists()
    assert guide_path.exists()
    loaded = StorylineProposalSet.model_validate_json(plans_path.read_text(encoding="utf-8"))
    assert [p.id for p in loaded.proposals] == ["teaser", "highlight"]
    assert loaded.provider == "deterministic"


def test_main_ollama_json_proposals(tmp_path: Path, monkeypatch) -> None:
    """main with provider=ollama calls TextLLMClient, validates proposals, and writes JSON + guide."""
    import json
    import sys
    from unittest.mock import MagicMock, patch

    from utils.models import StorylineProposalSet

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"general:\n  output_dir: '{output_dir}'\nstoryline:\n  provider: 'ollama'\n"
    )
    detections = {
        "total_clips": 1,
        "total_detections": 1,
        "detections": [
            {
                "clip_id": "c1",
                "detection_id": "c1_00001.00_00006.00",
                "file_name": "GX010778.MP4",
                "file_path": "/GX010778.MP4",
                "start_time": 1.0,
                "end_time": 6.0,
                "domain": "underwater",
                "story_role": "hero_subject",
                "subject": "Yellowtail Fusiliers",
                "quality_score": 9,
                "confidence": 0.95,
                "description": "fusiliers school",
            }
        ],
    }
    (output_dir / "detections_refined.json").write_text(json.dumps(detections), encoding="utf-8")

    fake_llm = MagicMock()
    fake_llm.generate_json.return_value = {
        "proposals": [
            {
                "id": "teaser",
                "title": "Fast Teaser",
                "target_duration_seconds": 55,
                "acts": [
                    {
                        "act_number": 1,
                        "title": "Act 1: Reef",
                        "shots": [
                            {
                                "detection_id": "c1_00001.00_00006.00",
                                "in_time": 1.5,
                                "out_time": 4.5,
                                "rationale": "Strong hero opener",
                            }
                        ],
                    }
                ],
            }
        ]
    }
    monkeypatch.setattr(
        sys,
        "argv",
        ["04_plan_storyline.py", "--config", str(config_path), "--presets", "teaser"],
    )
    with patch.object(storyline, "create_text_llm_client", return_value=fake_llm):
        storyline.main()

    loaded = StorylineProposalSet.model_validate_json(
        (output_dir / "storyline_plans.json").read_text(encoding="utf-8")
    )
    assert loaded.provider == "ollama"
    assert len(loaded.proposals) == 1
    assert loaded.proposals[0].shots[0].duration == 3.0
    assert "Strong hero opener" in (output_dir / "storyline_edit_guide.md").read_text(
        encoding="utf-8"
    )
