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
