"""Tests for intra-clip scan checkpointing and resume support."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.checkpoint import (
    ClipScanCheckpoint,
    CoarseFrameResult,
    remove_clip_checkpoint,
)
from utils.models import RefinedDetection

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_scan_spec = importlib.util.spec_from_file_location(
    "scripts_02_scan_footage", _SCRIPTS_DIR / "02_scan_footage.py"
)
scan = importlib.util.module_from_spec(_scan_spec)
assert _scan_spec.loader is not None
_scan_spec.loader.exec_module(scan)


def test_clip_scan_checkpoint_save_atomic_and_load(tmp_path: Path) -> None:
    """ClipScanCheckpoint should serialize, save atomically, and load correctly."""
    ckpt_path = tmp_path / "checkpoints" / "clip_123.json"
    ckpt = ClipScanCheckpoint(
        clip_id="clip_123",
        file_name="dive_gopro.mp4",
        pass_stage="coarse",
        coarse_fps=0.5,
    )
    frame_res = CoarseFrameResult(
        timestamp=2.0,
        detected=True,
        confidence=0.92,
        quality_score=8,
        subject="sea_turtle",
        candidate_window=(0.0, 4.0),
        elapsed=1.2,
    )
    ckpt.coarse_results["2.000"] = frame_res
    ckpt.candidate_windows.append((0.0, 4.0))
    ckpt.coarse_frames_analyzed = 1
    ckpt.coarse_detected = 1
    ckpt.coarse_confidences.append(0.92)
    ckpt.detected_domain = "motorcycle"
    ckpt.window_routing = {"1": {"subject": "cornering", "category": "moto_riding"}}

    ckpt.save_atomic(ckpt_path)
    assert ckpt_path.exists()

    loaded = ClipScanCheckpoint.load(ckpt_path)
    assert loaded is not None
    assert loaded.clip_id == "clip_123"
    assert loaded.file_name == "dive_gopro.mp4"
    assert loaded.pass_stage == "coarse"
    assert loaded.detected_domain == "motorcycle"
    assert loaded.window_routing == {"1": {"subject": "cornering", "category": "moto_riding"}}
    assert "2.000" in loaded.coarse_results
    assert loaded.coarse_results["2.000"].detected is True
    assert loaded.coarse_results["2.000"].confidence == pytest.approx(0.92)
    assert loaded.coarse_results["2.000"].candidate_window == (0.0, 4.0)
    assert loaded.candidate_windows == [(0.0, 4.0)]


def test_clip_scan_checkpoint_corrupt_file_returns_none(tmp_path: Path) -> None:
    """A corrupted checkpoint file should return None and not raise."""
    ckpt_path = tmp_path / "checkpoints" / "bad.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_text("{corrupt: json [", encoding="utf-8")

    loaded = ClipScanCheckpoint.load(ckpt_path)
    assert loaded is None


def test_remove_clip_checkpoint(tmp_path: Path) -> None:
    """remove_clip_checkpoint should delete the file if it exists without raising if missing."""
    ckpt_path = tmp_path / "checkpoints" / "clip_123.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.touch()
    assert ckpt_path.exists()

    remove_clip_checkpoint(ckpt_path)
    assert not ckpt_path.exists()

    # Removing again should be a no-op
    remove_clip_checkpoint(ckpt_path)


def test_process_clip_periodic_flush_and_resume_skips_evaluated_frames(tmp_path: Path) -> None:
    """process_clip should periodically flush checkpoint and skip evaluated frames on resume."""
    work_dir = tmp_path / "work"
    checkpoints_dir = tmp_path / "checkpoints"
    video_path = tmp_path / "test.mp4"
    video_path.touch()

    clip = {
        "clip_id": "test_clip",
        "file_name": "test.mp4",
        "file_path": str(video_path),
        "fps": 25.0,
    }
    config = {
        "scan_parameters": {
            "coarse_fps": 0.5,
            "coarse_confidence_threshold": 0.6,
            "coarse_quality_threshold": 4,
            "checkpoint_interval_frames": 2,
        },
        "ollama": {"host": "localhost:11434", "model": "test"},
    }

    mock_frames = [
        {"frame_path": str(tmp_path / f"frame_{i}.jpg"), "timestamp": float(i * 2)}
        for i in range(4)
    ]
    for f in mock_frames:
        Path(f["frame_path"]).touch()

    # Mock extract_frames
    with (
        patch.object(scan, "extract_frames", return_value=mock_frames),
        patch.object(scan, "call_ollama") as mock_ollama,
    ):
        mock_ollama.side_effect = [
            {"detected": True, "confidence": 0.85, "quality_score": 7, "subject": "turtle"},
            {"detected": False, "confidence": 0.1, "quality_score": 5, "subject": ""},
            KeyboardInterrupt("User stopped"),
        ]

        # First run: interrupted on 3rd frame
        with pytest.raises(KeyboardInterrupt):
            scan.process_clip(
                clip=clip,
                config=config,
                work_dir=work_dir,
                checkpoints_dir=checkpoints_dir,
                resume=True,
            )

    # Verify checkpoint was written on interrupt and contains the 2 evaluated frames
    ckpt_file = checkpoints_dir / "test_clip.json"
    assert ckpt_file.exists()
    ckpt = ClipScanCheckpoint.load(ckpt_file)
    assert ckpt is not None
    assert len(ckpt.coarse_results) == 2
    assert "0.000" in ckpt.coarse_results
    assert "2.000" in ckpt.coarse_results

    # Second run: resume from checkpoint
    with (
        patch.object(scan, "extract_frames", return_value=mock_frames),
        patch.object(scan, "call_ollama") as mock_ollama_resumed,
    ):
        mock_ollama_resumed.return_value = {
            "detected": True,
            "confidence": 0.9,
            "quality_score": 8,
            "subject": "turtle",
        }

        detections = scan.process_clip(
            clip=clip,
            config=config,
            work_dir=work_dir,
            checkpoints_dir=checkpoints_dir,
            resume=True,
        )

        # In coarse pass, frames 0 and 1 were already cached, so mock_ollama_resumed
        # should only be called for frames 2 and 3 in coarse pass
        coarse_calls = [
            call
            for call in mock_ollama_resumed.call_args_list
            if call.kwargs.get("phase") == "coarse"
        ]
        assert len(coarse_calls) == 2
        assert len(detections) >= 1


def test_process_clip_resumes_completed_coarse_pass(tmp_path: Path) -> None:
    """When coarse pass was completed in checkpoint, resume skips coarse inference."""
    work_dir = tmp_path / "work"
    checkpoints_dir = tmp_path / "checkpoints"
    video_path = tmp_path / "test.mp4"
    video_path.touch()

    clip = {
        "clip_id": "test_clip",
        "file_name": "test.mp4",
        "file_path": str(video_path),
        "fps": 25.0,
    }
    config = {
        "scan_parameters": {
            "coarse_fps": 0.5,
            "coarse_confidence_threshold": 0.6,
            "coarse_quality_threshold": 4,
            "fine_padding_seconds": 2,
        },
        "ollama": {"host": "localhost:11434", "model": "test"},
    }

    # Pre-create a checkpoint with coarse_done
    ckpt_file = checkpoints_dir / "test_clip.json"
    ckpt = ClipScanCheckpoint(
        clip_id="test_clip",
        file_name="test.mp4",
        pass_stage="coarse_done",
        coarse_fps=0.5,
        candidate_windows=[(10.0, 14.0)],
    )
    ckpt.save_atomic(ckpt_file)

    fine_frames = [{"frame_path": str(tmp_path / "fine_0.jpg"), "timestamp": 12.0}]
    Path(fine_frames[0]["frame_path"]).touch()

    with (
        patch.object(scan, "extract_frames", return_value=fine_frames) as mock_extract,
        patch.object(scan, "call_ollama") as mock_ollama,
    ):
        mock_ollama.return_value = {
            "detected": True,
            "confidence": 0.95,
            "quality_score": 9,
            "subject": "shark",
            "description": "reef shark",
        }

        detections = scan.process_clip(
            clip=clip,
            config=config,
            work_dir=work_dir,
            checkpoints_dir=checkpoints_dir,
            resume=True,
        )

        assert len(detections) >= 1
        assert detections[0].subject == "shark"
        # Coarse extract_frames should not be called with coarse parameters
        coarse_extractions = [
            call
            for call in mock_extract.call_args_list
            if "coarse" in str(call.kwargs.get("output_dir", ""))
        ]
        assert len(coarse_extractions) == 0


def test_scan_main_cleans_up_clip_checkpoint_on_completion(tmp_path: Path) -> None:
    """scan.main should clean up intra-clip checkpoint when the clip finishes and commits."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    video_path = tmp_path / "clip.mp4"
    video_path.touch()

    manifest_data = {
        "clips": [
            {
                "file_name": "clip.mp4",
                "file_path": str(video_path),
                "fps": 25.0,
                "clip_id": "clip_abc",
            }
        ]
    }
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"general:\n  output_dir: {output_dir}\n", encoding="utf-8")

    det = RefinedDetection(
        clip_id="clip_abc",
        file_name="clip.mp4",
        file_path=str(video_path),
        start_time=1.0,
        end_time=3.0,
        subject="turtle",
        confidence=0.9,
        quality_score=8,
        description="turtle",
    )

    checkpoints_dir = output_dir / "checkpoints"
    clip_ckpt = checkpoints_dir / "clip_abc.json"

    def mock_process_clip(*args, **kwargs):
        # Simulate creating an in-progress checkpoint during processing
        ckpt = ClipScanCheckpoint(
            clip_id="clip_abc",
            file_name="clip.mp4",
            pass_stage="coarse",
        )
        ckpt.save_atomic(clip_ckpt)
        assert clip_ckpt.exists()
        return [det]

    with (
        patch.object(scan, "process_clip", side_effect=mock_process_clip),
        patch(
            "sys.argv",
            [
                "02_scan_footage.py",
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
            ],
        ),
    ):
        scan.main()

    # Once clip is committed to detections_refined.json, checkpoint should be cleaned up
    assert not clip_ckpt.exists()
    refined_path = output_dir / "detections_refined.json"
    assert refined_path.exists()
