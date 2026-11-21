"""Tests for utils/audio_music.py."""

from __future__ import annotations

import json
import urllib.error
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.audio_music import (
    SCUBA_FILTER_CHAIN,
    build_audio_filter_complex,
    mix_video_audio,
    normalize_atmosphere,
    probe_has_audio,
    resolve_atmosphere_music,
    resolve_music_config,
    synthesize_atmosphere_track,
)


def test_normalize_atmosphere_aliases_and_errors() -> None:
    """normalize_atmosphere maps canonical names, aliases, and rejects unknown values."""
    assert normalize_atmosphere("ocean_ambient") == "ocean_ambient"
    assert normalize_atmosphere("ambient") == "ocean_ambient"
    assert normalize_atmosphere("OCEAN") == "ocean_ambient"
    assert normalize_atmosphere("chill") == "chill"
    assert normalize_atmosphere("calm") == "chill"
    assert normalize_atmosphere("cinematic") == "cinematic"
    assert normalize_atmosphere("upbeat") == "upbeat"
    assert normalize_atmosphere("action") == "upbeat"
    assert normalize_atmosphere("none") == "none"
    assert normalize_atmosphere("off") == "none"
    assert normalize_atmosphere(None) == "ocean_ambient"

    with pytest.raises(ValueError, match="Unknown music atmosphere"):
        normalize_atmosphere("death_metal")


def test_resolve_music_config_precedence() -> None:
    """CLI args override music: section, which overrides aftermovie: section and defaults."""
    cfg = {
        "music": {
            "atmosphere": "chill",
            "music_volume": 0.70,
            "original_audio_volume": 0.40,
            "scuba_filter": True,
            "audio_fade_duration": 1.5,
            "clip_music": False,
            "download_enabled": False,
            "cache_dir": "custom_music",
        },
        "aftermovie": {
            "music_volume": 0.25,
        },
    }

    class Args:
        atmosphere = "upbeat"
        music = None
        music_volume = 0.55
        original_volume = 0.30
        scuba_filter = False
        clip_music = True

    resolved = resolve_music_config(cfg, Args())
    assert resolved.atmosphere == "upbeat"
    assert resolved.music_volume == 0.55
    assert resolved.original_volume == 0.30
    assert resolved.scuba_filter is False
    assert resolved.audio_fade_duration == 1.5
    assert resolved.clip_music is True
    assert resolved.download_enabled is False
    assert resolved.cache_dir == Path("custom_music")


@pytest.mark.parametrize("atmosphere", ["ocean_ambient", "chill", "cinematic", "upbeat"])
def test_synthesize_atmosphere_track_valid_wav(tmp_path: Path, atmosphere: str) -> None:
    """synthesize_atmosphere_track produces a valid stereo 16-bit PCM WAV of expected duration."""
    out = tmp_path / f"{atmosphere}.wav"
    result = synthesize_atmosphere_track(atmosphere, out, duration_seconds=2.0, sample_rate=22050)
    assert result == out
    assert out.exists()

    with wave.open(str(out), "rb") as wf:
        assert wf.getnchannels() == 2
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 22050
        assert wf.getnframes() == 44100

    # Existing valid WAV should be reused without overwriting
    mtime = out.stat().st_mtime_ns
    reused = synthesize_atmosphere_track(atmosphere, out, duration_seconds=2.0, sample_rate=22050)
    assert reused == out
    assert out.stat().st_mtime_ns == mtime


def test_synthesize_atmosphere_rejects_none(tmp_path: Path) -> None:
    """Synthesizing 'none' should raise ValueError."""
    with pytest.raises(ValueError, match="none"):
        synthesize_atmosphere_track("none", tmp_path / "none.wav")


def test_resolve_atmosphere_music_tiers(tmp_path: Path) -> None:
    """resolve_atmosphere_music follows Tier 1 (explicit/local), Tier 2 (CC0 cache), Tier 3 (synth)."""
    # Explicit path wins when present
    explicit = tmp_path / "custom.mp3"
    explicit.write_bytes(b"mp3data")
    assert resolve_atmosphere_music("chill", music_path=explicit, cache_dir=tmp_path) == explicit

    # "none" returns None
    assert resolve_atmosphere_music("none", cache_dir=tmp_path) is None

    # Local track in <cache_dir>/<atmosphere>/ wins over download/synth
    cache_root = tmp_path / "music_lib"
    chill_dir = cache_root / "chill"
    chill_dir.mkdir(parents=True)
    local_chill = chill_dir / "01_lofi.ogg"
    local_chill.write_bytes(b"OggS" + b"\x00" * 2048)
    assert (
        resolve_atmosphere_music("calm", cache_dir=cache_root, download_enabled=True) == local_chill
    )

    # Tier 2: download & cache CC0 track when local file is absent
    fake_payload = b"OggS" + b"\x01" * 2048
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = fake_payload
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False

    with patch("utils.audio_music.urllib.request.urlopen", return_value=mock_resp) as mock_open:
        downloaded = resolve_atmosphere_music(
            "cinematic",
            cache_dir=cache_root,
            download_enabled=True,
            atmosphere_urls={"cinematic": ["https://example.org/cc0/cinema.ogg"]},
        )
    assert mock_open.called
    assert downloaded is not None
    assert downloaded == cache_root / "cache" / "cinematic" / "cinema.ogg"
    assert downloaded.read_bytes() == fake_payload

    # Tier 3: offline fallback to procedural synthesis when download fails
    with patch(
        "utils.audio_music.urllib.request.urlopen",
        side_effect=urllib.error.URLError("offline"),
    ):
        synth = resolve_atmosphere_music(
            "upbeat",
            duration_seconds=2.0,
            cache_dir=cache_root,
            download_enabled=True,
            atmosphere_urls={"upbeat": ["https://example.org/cc0/upbeat.ogg"]},
        )
    assert synth is not None
    assert synth == cache_root / "synth" / "upbeat.wav"
    assert synth.exists()


def test_probe_has_audio_cases(tmp_path: Path) -> None:
    """probe_has_audio handles audio streams, silent videos, and non-JSON mock output."""
    vid = tmp_path / "clip.mp4"

    mock_audio = MagicMock(returncode=0, stdout=json.dumps({"streams": [{"codec_type": "audio"}]}))
    assert probe_has_audio(vid, run_fn=lambda *a, **kw: mock_audio) is True

    mock_silent = MagicMock(returncode=0, stdout=json.dumps({"streams": []}))
    assert probe_has_audio(vid, run_fn=lambda *a, **kw: mock_silent) is False

    # Non-JSON mock output falls back to True so existing tests keep standard [0:a] wiring
    mock_legacy = MagicMock(returncode=0, stdout="5.0\n")
    assert probe_has_audio(vid, run_fn=lambda *a, **kw: mock_legacy) is True


def test_build_audio_filter_complex_variants() -> None:
    """build_audio_filter_complex builds scuba filter, volume mix, and silent fallback chains."""
    # Source audio + music + scuba filter
    fc = build_audio_filter_complex(
        has_source_audio=True,
        has_music=True,
        original_volume=0.35,
        music_volume=0.65,
        scuba_filter=True,
        fade_duration=1.0,
        video_duration=10.0,
    )
    assert fc is not None
    assert SCUBA_FILTER_CHAIN in fc
    assert "[0:a]" in fc
    assert "volume=0.35" in fc
    assert "[1:a]volume=0.65[music]" in fc
    assert "amix=inputs=2:duration=first:dropout_transition=2:normalize=0" in fc
    assert "afade=t=in:st=0:d=1" in fc
    assert "afade=t=out:st=9:d=1" in fc

    # Silent video + music (no [0:a] reference)
    fc_silent = build_audio_filter_complex(
        has_source_audio=False,
        has_music=True,
        original_volume=0.35,
        music_volume=0.65,
        scuba_filter=True,
        fade_duration=1.0,
        video_duration=8.0,
    )
    assert fc_silent is not None
    assert "[0:a]" not in fc_silent
    assert "atrim=0:8,asetpts=PTS-STARTPTS" in fc_silent
    assert "volume=0.65" in fc_silent

    # No music, no filtering -> None
    assert (
        build_audio_filter_complex(
            has_source_audio=True,
            has_music=False,
            original_volume=None,
            scuba_filter=False,
            fade_duration=0.0,
        )
        is None
    )


def test_mix_video_audio_normalize_fallback(tmp_path: Path) -> None:
    """mix_video_audio retries without :normalize=0 if FFmpeg rejects it."""
    video = tmp_path / "in.mp4"
    video.touch()
    music = tmp_path / "bg.wav"
    music.write_bytes(b"RIFF" + b"\x00" * 100)
    out = tmp_path / "out.mp4"

    probe_res = MagicMock(returncode=0, stdout=json.dumps({"streams": [{"codec_type": "audio"}]}))
    fail_norm = MagicMock(returncode=1, stderr="Option 'normalize' not found")
    ok_res = MagicMock(returncode=0, stderr="")

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        calls.append(cmd)
        if cmd[0] == "ffprobe":
            return probe_res
        if len(calls) == 2:
            return fail_norm
        return ok_res

    assert (
        mix_video_audio(
            video,
            music,
            out,
            music_volume=0.65,
            original_volume=0.35,
            scuba_filter=True,
            run_fn=fake_run,
        )
        is True
    )
    assert len(calls) == 3
    assert "-stream_loop" in calls[1]
    assert ":normalize=0" in calls[1][calls[1].index("-filter_complex") + 1]
    assert ":normalize=0" not in calls[2][calls[2].index("-filter_complex") + 1]
