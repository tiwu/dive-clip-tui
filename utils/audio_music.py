"""Copyright-free music atmosphere resolution and scuba audio mixing helpers.

Provides three-tier background music resolution (explicit/local user tracks,
optional on-demand CC0 download with local caching, and pure-Python procedural
stereo WAV synthesis), plus FFmpeg audio probing and filter graphs that keep
original underwater audio audible (default 35%) with an optional regulator-hiss
filter under the selected music bed (default 65%).
"""

from __future__ import annotations

import array
import json
import logging
import math
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CANONICAL_ATMOSPHERES = ("ocean_ambient", "chill", "cinematic", "upbeat", "none")

ATMOSPHERE_ALIASES: dict[str, str] = {
    "ocean_ambient": "ocean_ambient",
    "ambient": "ocean_ambient",
    "ocean": "ocean_ambient",
    "chill": "chill",
    "calm": "chill",
    "cinematic": "cinematic",
    "upbeat": "upbeat",
    "action": "upbeat",
    "none": "none",
    "off": "none",
    "": "none",
}

VALID_ATMOSPHERE_CHOICES = [
    "ocean_ambient",
    "ambient",
    "chill",
    "calm",
    "cinematic",
    "upbeat",
    "action",
    "none",
]

AUDIO_EXTENSIONS = (".mp3", ".ogg", ".wav", ".flac", ".m4a")

# Verified Creative Commons Zero (CC0 1.0 Public Domain Dedication) tracks on
# Wikimedia Commons. Users may override or extend these via music.atmosphere_urls
# in config.yaml, or drop custom files into assets/music/<atmosphere>/.
DEFAULT_CC0_ATMOSPHERE_URLS: dict[str, list[str]] = {
    "ocean_ambient": [
        "https://upload.wikimedia.org/wikipedia/commons/6/60/Monplaisir_-_02_-_Rain.ogg",
        "https://upload.wikimedia.org/wikipedia/commons/f/fa/Komiku_-_13_-_Bleu.ogg",
    ],
    "chill": [
        "https://upload.wikimedia.org/wikipedia/commons/a/a0/Komiku_-_02_-_Chill_Out_Theme.ogg",
        "https://upload.wikimedia.org/wikipedia/commons/9/9f/Chill_Beat.ogg",
    ],
    "cinematic": [
        "https://upload.wikimedia.org/wikipedia/commons/3/33/Komiku_-_01_-_Balance.ogg",
        "https://upload.wikimedia.org/wikipedia/commons/9/97/Monplaisir_-_03_-_Three.ogg",
    ],
    "upbeat": [
        "https://upload.wikimedia.org/wikipedia/commons/d/d2/Komiku_-_05_-_Surfing.ogg",
        "https://upload.wikimedia.org/wikipedia/commons/b/b9/Monplaisir_-_03_-_Dance_of_the_electronic_fairies.ogg",
    ],
}

# Tames underwater hydrophone rumble (<80Hz), harsh scuba regulator inhalation
# hiss (~2.8kHz notch), and shrill bubble fizz (>4.5kHz) while preserving dive
# ambience.
SCUBA_FILTER_CHAIN = "highpass=f=80,equalizer=f=2800:t=q:w=1.2:g=-8,lowpass=f=4500"

_USER_AGENT = "dive-clip-tui/0.36 (https://github.com/tiwu/dive-clip-tui)"


def normalize_atmosphere(name: str | None, default: str = "ocean_ambient") -> str:
    """Normalize an atmosphere preset or alias to its canonical identifier."""
    raw = default if name is None else str(name)
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if key in ATMOSPHERE_ALIASES:
        return ATMOSPHERE_ALIASES[key]
    raise ValueError(
        f"Unknown music atmosphere: {name!r}. Valid choices: {', '.join(VALID_ATMOSPHERE_CHOICES)}"
    )


@dataclass(slots=True)
class MusicConfig:
    """Resolved music and scuba-audio mixing parameters."""

    atmosphere: str = "ocean_ambient"
    music_path: Path | None = None
    original_volume: float = 0.35
    music_volume: float = 0.65
    scuba_filter: bool = True
    audio_fade_duration: float = 1.0
    clip_music: bool = False
    download_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("assets/music"))
    atmosphere_urls: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_CC0_ATMOSPHERE_URLS.items()}
    )


def _coalesce(*values: Any, default: Any = None) -> Any:
    for val in values:
        if val is not None:
            return val
    return default


def resolve_music_config(
    config: dict[str, Any] | None = None,
    args: Any = None,
) -> MusicConfig:
    """Resolve ``MusicConfig`` from CLI args, ``music:``, and ``aftermovie:`` sections."""
    cfg = config if isinstance(config, dict) else {}
    music_cfg = cfg.get("music") if isinstance(cfg.get("music"), dict) else {}
    aftermovie_cfg = cfg.get("aftermovie") if isinstance(cfg.get("aftermovie"), dict) else {}

    raw_atmo = _coalesce(
        getattr(args, "atmosphere", None),
        music_cfg.get("atmosphere"),
        aftermovie_cfg.get("atmosphere"),
        default="ocean_ambient",
    )
    atmosphere = normalize_atmosphere(raw_atmo)

    raw_music = _coalesce(
        getattr(args, "music", None),
        music_cfg.get("music_path"),
        aftermovie_cfg.get("music_path"),
        default=None,
    )
    music_path = Path(raw_music) if raw_music else None

    raw_music_vol = _coalesce(
        getattr(args, "music_volume", None),
        music_cfg.get("music_volume"),
        aftermovie_cfg.get("music_volume"),
        default=0.65,
    )
    music_volume = max(0.0, float(raw_music_vol))

    raw_orig_vol = _coalesce(
        getattr(args, "original_volume", None),
        music_cfg.get("original_audio_volume"),
        music_cfg.get("original_volume"),
        aftermovie_cfg.get("original_audio_volume"),
        aftermovie_cfg.get("original_volume"),
        default=0.35,
    )
    original_volume = max(0.0, float(raw_orig_vol))

    raw_scuba = _coalesce(
        getattr(args, "scuba_filter", None),
        music_cfg.get("scuba_filter"),
        aftermovie_cfg.get("scuba_filter"),
        default=True,
    )
    scuba_filter = bool(raw_scuba)

    raw_fade = _coalesce(
        getattr(args, "audio_fade_duration", None),
        music_cfg.get("audio_fade_duration"),
        aftermovie_cfg.get("audio_fade_duration"),
        default=1.0,
    )
    audio_fade_duration = max(0.0, float(raw_fade))

    raw_clip_music = _coalesce(
        getattr(args, "clip_music", None),
        music_cfg.get("clip_music"),
        default=False,
    )
    clip_music = bool(raw_clip_music)

    download_enabled = bool(music_cfg.get("download_enabled", True))
    cache_dir = Path(music_cfg.get("cache_dir") or "assets/music")

    urls = {k: list(v) for k, v in DEFAULT_CC0_ATMOSPHERE_URLS.items()}
    custom_urls = music_cfg.get("atmosphere_urls")
    if isinstance(custom_urls, dict):
        for raw_key, val in custom_urls.items():
            try:
                norm_key = normalize_atmosphere(str(raw_key))
            except ValueError:
                continue
            if norm_key == "none":
                continue
            if isinstance(val, list):
                urls[norm_key] = [str(u) for u in val if u]
            elif isinstance(val, str) and val.strip():
                urls[norm_key] = [val.strip()]

    return MusicConfig(
        atmosphere=atmosphere,
        music_path=music_path,
        original_volume=original_volume,
        music_volume=music_volume,
        scuba_filter=scuba_filter,
        audio_fade_duration=audio_fade_duration,
        clip_music=clip_music,
        download_enabled=download_enabled,
        cache_dir=cache_dir,
        atmosphere_urls=urls,
    )


def _synth_ocean_ambient(n_frames: int, sample_rate: int, fade_samples: int) -> array.array:
    """Generate a warm, slow-evolving underwater harmonic pad (C2-G2-D3-E3-G3-B3)."""
    freqs = (65.41, 98.00, 146.83, 164.81, 196.00, 246.94)
    weights = (0.32, 0.24, 0.18, 0.12, 0.08, 0.06)
    two_pi = 2.0 * math.pi
    sin = math.sin
    steps_l = [two_pi * (f - 0.12) / sample_rate for f in freqs]
    steps_r = [two_pi * (f + 0.12) / sample_rate for f in freqs]
    lfo1_step = two_pi * 0.08 / sample_rate
    lfo2_step = two_pi * 0.13 / sample_rate

    buf = array.array("h")
    for i in range(n_frames):
        env = 1.0
        if fade_samples > 0:
            if i < fade_samples:
                env = i / fade_samples
            elif i > n_frames - fade_samples:
                env = (n_frames - i) / fade_samples
        swell = 0.72 + 0.18 * sin(i * lfo1_step) + 0.10 * sin(i * lfo2_step)
        l_val = 0.0
        r_val = 0.0
        for idx, w in enumerate(weights):
            l_val += w * sin(i * steps_l[idx])
            r_val += w * sin(i * steps_r[idx])
        scale = 22000.0 * env * swell
        buf.append(int(max(-32767, min(32767, l_val * scale))))
        buf.append(int(max(-32767, min(32767, r_val * scale))))
    return buf


def _synth_chord_progression(
    chords: list[tuple[float, ...]],
    chord_seconds: float,
    n_frames: int,
    sample_rate: int,
    fade_samples: int,
    *,
    pulse_bpm: float | None = None,
) -> array.array:
    """Generate a smooth multi-chord stereo bed with optional rhythmic pulse."""
    two_pi = 2.0 * math.pi
    sin = math.sin
    chord_frames = max(1, int(chord_seconds * sample_rate))
    xfade = min(int(0.15 * sample_rate), chord_frames // 4)
    lfo_step = two_pi * 0.25 / sample_rate
    beat_freq = (pulse_bpm / 60.0) if pulse_bpm else 0.0

    chord_steps_l = [[two_pi * (f - 0.18) / sample_rate for f in chord] for chord in chords]
    chord_steps_r = [[two_pi * (f + 0.18) / sample_rate for f in chord] for chord in chords]

    buf = array.array("h")
    n_chords = len(chords)
    for i in range(n_frames):
        env = 1.0
        if fade_samples > 0:
            if i < fade_samples:
                env = i / fade_samples
            elif i > n_frames - fade_samples:
                env = (n_frames - i) / fade_samples

        c_idx = (i // chord_frames) % n_chords
        pos = i % chord_frames
        steps_l = chord_steps_l[c_idx]
        steps_r = chord_steps_r[c_idx]

        # Gentle envelope at chord boundary for click-free transitions
        c_env = 1.0
        if xfade > 0:
            if pos < xfade:
                c_env = 0.55 + 0.45 * (pos / xfade)
            elif pos > chord_frames - xfade:
                c_env = 0.55 + 0.45 * ((chord_frames - pos) / xfade)

        if beat_freq > 0.0:
            beat_phase = ((i / sample_rate) * beat_freq) % 1.0
            rhythm = 0.55 + 0.45 * ((1.0 - beat_phase) ** 2)
        else:
            rhythm = 0.85 + 0.15 * sin(i * lfo_step)

        l_val = 0.0
        r_val = 0.0
        for st_l, st_r in zip(steps_l, steps_r, strict=True):
            l_val += sin(i * st_l) + 0.20 * sin(i * st_l * 2.0)
            r_val += sin(i * st_r) + 0.20 * sin(i * st_r * 2.0)

        norm = len(steps_l) * 1.2
        scale = (20000.0 / norm) * env * c_env * rhythm
        buf.append(int(max(-32767, min(32767, l_val * scale))))
        buf.append(int(max(-32767, min(32767, r_val * scale))))
    return buf


def synthesize_atmosphere_track(
    atmosphere: str,
    output_path: Path,
    *,
    duration_seconds: float = 32.0,
    sample_rate: int = 22050,
) -> Path:
    """Synthesize a copyright-free stereo 16-bit PCM WAV ambient loop offline."""
    canonical = normalize_atmosphere(atmosphere)
    if canonical == "none":
        raise ValueError("Cannot synthesize audio for atmosphere 'none'")

    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > 44:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.5, float(duration_seconds))
    n_frames = max(1, int(dur * sample_rate))
    fade_samples = min(int(1.5 * sample_rate), n_frames // 4)

    if canonical == "ocean_ambient":
        buf = _synth_ocean_ambient(n_frames, sample_rate, fade_samples)
    elif canonical == "chill":
        # Am7 -> Dm9 -> G13 -> Cmaj7
        chords = [
            (110.00, 220.00, 261.63, 329.63, 392.00),
            (146.83, 220.00, 261.63, 329.63, 440.00),
            (98.00, 196.00, 246.94, 329.63, 392.00),
            (130.81, 196.00, 246.94, 329.63, 392.00),
        ]
        buf = _synth_chord_progression(chords, 4.0, n_frames, sample_rate, fade_samples)
    elif canonical == "cinematic":
        # Dm(add9) -> Bbmaj7 -> F -> Csus4
        chords = [
            (73.42, 146.83, 220.00, 261.63, 329.63),
            (58.27, 116.54, 174.61, 220.00, 293.66),
            (87.31, 174.61, 220.00, 261.63, 349.23),
            (65.41, 130.81, 196.00, 261.63, 349.23),
        ]
        buf = _synth_chord_progression(chords, 4.0, n_frames, sample_rate, fade_samples)
    else:  # "upbeat"
        # Am -> F -> C -> G at 116 BPM
        chords = [
            (110.00, 220.00, 261.63, 329.63),
            (87.31, 174.61, 220.00, 261.63),
            (130.81, 196.00, 261.63, 329.63),
            (98.00, 196.00, 246.94, 293.66),
        ]
        buf = _synth_chord_progression(
            chords, 2.0, n_frames, sample_rate, fade_samples, pulse_bpm=116.0
        )

    tmp_path = output_path.with_name(f".tmp_{output_path.name}")
    try:
        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(buf.tobytes())
        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return output_path


def _find_local_audio(base_dir: Path, canonical: str) -> Path | None:
    """Find an existing user-supplied or cached audio file for ``canonical``."""
    candidates: list[Path] = []
    atmo_dir = base_dir / canonical
    if atmo_dir.is_dir():
        candidates.extend(
            sorted(
                p
                for p in atmo_dir.iterdir()
                if p.is_file()
                and not p.name.startswith(".")
                and p.suffix.lower() in AUDIO_EXTENSIONS
                and p.stat().st_size > 0
            )
        )
    if candidates:
        return candidates[0]

    for ext in AUDIO_EXTENSIONS:
        direct = base_dir / f"{canonical}{ext}"
        if direct.is_file() and direct.stat().st_size > 0:
            return direct

    cached_dir = base_dir / "cache" / canonical
    if cached_dir.is_dir():
        cached = sorted(
            p
            for p in cached_dir.iterdir()
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in AUDIO_EXTENSIONS
            and p.stat().st_size > 0
        )
        if cached:
            return cached[0]

    return None


def _download_cc0_track(
    canonical: str,
    cache_dir: Path,
    urls: list[str],
    timeout: float = 15.0,
) -> Path | None:
    """Try downloading a CC0 track into ``cache_dir/cache/<atmosphere>/``."""
    if not urls:
        return None

    dest_dir = cache_dir / "cache" / canonical
    for idx, url in enumerate(urls, start=1):
        parsed = urllib.parse.urlparse(url)
        suffix = Path(parsed.path).suffix.lower()
        if suffix not in AUDIO_EXTENSIONS:
            suffix = ".ogg"
        raw_name = Path(urllib.parse.unquote(parsed.path)).name
        dest_name = raw_name if raw_name and Path(raw_name).suffix else f"track_{idx:02d}{suffix}"
        dest_file = dest_dir / dest_name
        if dest_file.is_file() and dest_file.stat().st_size > 1024:
            return dest_file

        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", 200)
                if status is not None and int(status) >= 400:
                    continue
                payload = resp.read()
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            logger.debug("CC0 music download failed for %s (%s): %s", canonical, url, exc)
            continue

        if len(payload) <= 1024:
            logger.debug("Downloaded music payload too small (%d bytes) for %s", len(payload), url)
            continue
        head = payload[:32].lstrip().lower()
        if head.startswith((b"<!doctype", b"<html")):
            logger.debug("Downloaded HTML instead of audio for %s", url)
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = dest_dir / f".tmp_{dest_name}"
        try:
            tmp_file.write_bytes(payload)
            tmp_file.replace(dest_file)
            logger.info("Cached CC0 %s music track: %s", canonical, dest_file)
            return dest_file
        except OSError as exc:
            tmp_file.unlink(missing_ok=True)
            logger.debug("Failed saving cached music file %s: %s", dest_file, exc)

    return None


def resolve_atmosphere_music(
    atmosphere: str | None = "ocean_ambient",
    *,
    music_path: Path | str | None = None,
    duration_seconds: float = 32.0,
    cache_dir: Path | str = Path("assets/music"),
    download_enabled: bool = True,
    atmosphere_urls: dict[str, list[str]] | None = None,
) -> Path | None:
    """Resolve a copyright-free music track using three-tier resolution.

    Tier 1: Explicit ``music_path`` or local audio files in
            ``<cache_dir>/<atmosphere>/`` (and cached downloads).
    Tier 2: Optional on-demand download of curated CC0 tracks into
            ``<cache_dir>/cache/<atmosphere>/``.
    Tier 3: Offline procedural stereo WAV synthesis into
            ``<cache_dir>/synth/<atmosphere>.wav``.
    """
    if music_path is not None:
        explicit = Path(music_path)
        if explicit.is_file():
            return explicit
        logger.warning("Music track not found at %s; skipping music.", explicit)
        return None

    canonical = normalize_atmosphere(atmosphere)
    if canonical == "none":
        return None

    base_dir = Path(cache_dir)
    local_track = _find_local_audio(base_dir, canonical)
    if local_track is not None:
        return local_track

    if download_enabled:
        url_map = atmosphere_urls if atmosphere_urls is not None else DEFAULT_CC0_ATMOSPHERE_URLS
        downloaded = _download_cc0_track(canonical, base_dir, list(url_map.get(canonical, [])))
        if downloaded is not None:
            return downloaded

    synth_path = base_dir / "synth" / f"{canonical}.wav"
    return synthesize_atmosphere_track(
        canonical,
        synth_path,
        duration_seconds=max(16.0, float(duration_seconds)),
    )


def probe_has_audio(
    video_path: Path,
    *,
    timeout: float = 15.0,
    run_fn: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> bool:
    """Return True if ``video_path`` contains at least one audio stream.

    When ``ffprobe`` returns non-JSON output (for example in mocked tests),
    defaults to ``True`` to preserve standard ``[0:a]`` filter wiring.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        res = run_fn(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return True

    if getattr(res, "returncode", 1) != 0:
        return True

    stdout = getattr(res, "stdout", "") or ""
    if not isinstance(stdout, str) or not stdout.strip():
        return True

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return True

    if isinstance(data, dict) and "streams" in data:
        return bool(data["streams"])
    return True


def build_audio_filter_complex(
    *,
    has_source_audio: bool = True,
    has_music: bool = True,
    original_volume: float | None = 0.35,
    music_volume: float = 0.65,
    scuba_filter: bool = False,
    fade_duration: float = 0.0,
    video_duration: float | None = None,
    normalize_amix: bool = True,
) -> str | None:
    """Build an FFmpeg ``-filter_complex`` string for background music + scuba audio mix."""
    fade_parts: list[str] = []
    if fade_duration > 0:
        fade_in_d = (
            fade_duration
            if video_duration is None
            else min(fade_duration, max(0.0, video_duration / 2.0))
        )
        if fade_in_d > 0:
            fade_parts.append(f"afade=t=in:st=0:d={fade_in_d:g}")
        if video_duration is not None and video_duration > fade_in_d * 2:
            fade_parts.append(f"afade=t=out:st={video_duration - fade_in_d:g}:d={fade_in_d:g}")
    fade_suffix = f",{','.join(fade_parts)}" if fade_parts else ""

    if has_source_audio and has_music:
        if not scuba_filter and original_volume is None and not fade_parts:
            return (
                f"[1:a]volume={music_volume}[music];[0:a][music]amix=inputs=2:duration=first[aout]"
            )

        orig_parts: list[str] = []
        if scuba_filter:
            orig_parts.append(SCUBA_FILTER_CHAIN)
        orig_vol = 0.35 if original_volume is None else max(0.0, float(original_volume))
        orig_parts.append(f"volume={orig_vol:g}")
        orig_chain = f"[0:a]{','.join(orig_parts)}[orig]"

        music_vol = max(0.0, float(music_volume))
        music_chain = f"[1:a]volume={music_vol:g}[music]"
        norm_opt = ":normalize=0" if normalize_amix else ""
        mix_chain = (
            f"[orig][music]amix=inputs=2:duration=first:dropout_transition=2"
            f"{norm_opt}{fade_suffix}[aout]"
        )
        return f"{orig_chain};{music_chain};{mix_chain}"

    if not has_source_audio and has_music:
        music_vol = max(0.0, float(music_volume))
        trim_prefix = (
            f"atrim=0:{video_duration:g},asetpts=PTS-STARTPTS,"
            if video_duration is not None and video_duration > 0
            else ""
        )
        return f"[1:a]{trim_prefix}volume={music_vol:g}{fade_suffix}[aout]"

    if has_source_audio and not has_music:
        orig_parts = []
        if scuba_filter:
            orig_parts.append(SCUBA_FILTER_CHAIN)
        if original_volume is not None and original_volume != 1.0:
            orig_parts.append(f"volume={max(0.0, float(original_volume)):g}")
        all_parts = orig_parts + fade_parts
        if not all_parts:
            return None
        return f"[0:a]{','.join(all_parts)}[aout]"

    return None


def mix_video_audio(
    video_path: Path,
    music_path: Path | None,
    output_path: Path,
    *,
    music_volume: float = 0.15,
    original_volume: float | None = None,
    scuba_filter: bool = False,
    fade_duration: float = 0.0,
    video_duration: float | None = None,
    run_fn: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> bool:
    """Mix optional background music and scuba-audio attenuation onto ``video_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_music = music_path is not None and music_path.exists()
    needs_audio_filter = (
        has_music
        or scuba_filter
        or (original_volume is not None and original_volume != 1.0)
        or fade_duration > 0
    )
    has_audio = probe_has_audio(video_path, run_fn=run_fn) if needs_audio_filter else True

    filter_complex = build_audio_filter_complex(
        has_source_audio=has_audio,
        has_music=has_music,
        original_volume=original_volume,
        music_volume=music_volume,
        scuba_filter=scuba_filter,
        fade_duration=fade_duration,
        video_duration=video_duration,
        normalize_amix=True,
    )

    if filter_complex is None:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-c",
            "copy",
            "-y",
            str(output_path),
        ]
        try:
            result = run_fn(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg timeout copying %s", output_path.name)
            return False
        except Exception as exc:
            logger.warning("FFmpeg error: %s", exc)
            return False
        if result.returncode != 0:
            logger.warning("FFmpeg copy failed: %s", (result.stderr or "").strip()[:200])
            return False
        return True

    def _build_cmd(fc: str) -> list[str]:
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
        ]
        if has_music and music_path is not None:
            ffmpeg_cmd.extend(["-stream_loop", "-1", "-i", str(music_path)])
        ffmpeg_cmd.extend([
            "-filter_complex",
            fc,
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
            str(output_path),
        ])
        return ffmpeg_cmd

    cmd = _build_cmd(filter_complex)
    try:
        result = run_fn(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg timeout adding music to %s", output_path.name)
        return False
    except Exception as exc:
        logger.warning("FFmpeg error: %s", exc)
        return False

    if result.returncode != 0 and ":normalize=0" in filter_complex:
        fallback_fc = build_audio_filter_complex(
            has_source_audio=has_audio,
            has_music=has_music,
            original_volume=original_volume,
            music_volume=music_volume,
            scuba_filter=scuba_filter,
            fade_duration=fade_duration,
            video_duration=video_duration,
            normalize_amix=False,
        )
        if fallback_fc:
            try:
                result = run_fn(
                    _build_cmd(fallback_fc),
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg timeout adding music to %s", output_path.name)
                return False
            except Exception as exc:
                logger.warning("FFmpeg error: %s", exc)
                return False

    if result.returncode != 0:
        logger.warning("FFmpeg music mix failed: %s", (result.stderr or "").strip()[:200])
        return False

    return True
