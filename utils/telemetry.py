"""Action-camera sensor telemetry decoding and pre-VLM advisory scoring.

The module decodes GoPro GPMF (``gpmd``) telemetry into per-signal time series and
exposes small, dependency-free helpers used by the screening funnel:

* :func:`classify_telemetry_type` — detect the camera family from container stream tags.
* :func:`detect_telemetry_stream` — detect which stream index carries telemetry.
* :func:`load_telemetry` — decode a clip's telemetry track (fail-soft, never raises).
* :func:`aggregate_window` — collapse 200 Hz samples to one record per candidate window.
* :func:`compute_telemetry_score` — normalized 0.0-1.0 kinematic/interest score.
* :func:`evaluate_gates` — advisory (or explicitly enabled) rejection gates.
* :func:`telemetry_domain_prior` — ``SCEN`` based Pass 0 domain fast-path prior.

Specifications & Formats Reference
----------------------------------
* **GoPro GPMF Parser**: https://github.com/gopro/gpmf-parser
* **GPMF 32-bit KLV Alignment**: All data within GPMF is 32-bit aligned. Payloads that
  do not naturally end on a 32-bit boundary are padded with zeros to maintain alignment.
  Parsers step forward by the 32-bit padded length: ``((payload_length + 3) & ~3)``.
* **GoPro Body Coordinates**: +X right, +Y bottom, +Z optical axis (lens forward).
  When upright and level, gravity vector points down +Y (gy = 1.0, tilt = 0°).
  Inversion / dangling occurs when gy < 0 (tilt > 90°).
* **Research & Deep Dive**: ``docs/research/highlight-detection-and-telemetry.md``
"""

from __future__ import annotations

import json
import logging
import math
import struct
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Container stream tags that identify a camera family.
TELEMETRY_STREAM_TAGS: dict[str, str] = {
    "gpmd": "gopro",
    "djmd": "dji",
    "insta": "insta360",
}

GOPRO = "gopro"
DJI = "dji"
INSTA360 = "insta360"

# --- GPMF KLV ---------------------------------------------------------------

# GPMF payload type character -> (struct format, element size in bytes).
GPMF_TYPES: dict[str, tuple[str, int]] = {
    "b": ("b", 1),
    "B": ("B", 1),
    "h": ("h", 2),
    "H": ("H", 2),
    "s": ("h", 2),
    "S": ("H", 2),
    "i": ("i", 4),
    "I": ("I", 4),
    "l": ("i", 4),
    "L": ("I", 4),
    "j": ("q", 8),
    "J": ("Q", 8),
    "f": ("f", 4),
    "d": ("d", 8),
}

# Type bytes that mark a payload as a nested KLV container rather than data.
GPMF_CONTAINER_TYPES = {0x00, ord("?")}

# Known container keys, used when a firmware omits the complex type marker.
GPMF_CONTAINER_KEYS = {"DEVC", "DVID", "STRM"}

SCEN_DOMAIN_MAP: dict[str, str] = {
    "WATR": "underwater",
    "VEGE": "generic",
    "URB": "generic",
    "INDO": "generic",
    "SNOW": "generic",
    "BEAC": "generic",
}

# Advisory gate defaults (tunable, documented as candidates — not universal facts).
DEFAULT_GATE_THRESHOLDS: dict[str, float] = {
    "yavg_min": 15.0,
    "unif_max": 0.92,
    "gyro_max": 3.5,
    "tilt_max": 100.0,
}

DEFAULT_TELEMETRY_WEIGHTS: dict[str, float] = {
    "lean_speed": 0.35,
    "gyro_stability": 0.25,
    "face": 0.20,
    "hue_change": 0.20,
}

# Normalization ranges for the score components (documented, tunable).
SCORE_NORMALIZATION: dict[str, float] = {
    "speed_ms": 3.0,
    "lean_deg": 45.0,
    "gyro_rms": 3.5,
    "hue_changes_per_s": 0.5,
}

DEFAULT_DOMAIN_PRIOR_THRESHOLDS: dict[str, float] = {
    "min_confidence": 0.80,
    "min_coverage": 0.80,
}


@dataclass
class TelemetrySeries:
    """One named telemetry signal sampled at irregular timestamps."""

    name: str
    timestamps: list[float] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)

    def samples_between(self, start: float, end: float) -> list[Any]:
        """Return values whose timestamps fall inside ``[start, end)``.

        Windows are half-open so adjacent windows never double-count a boundary
        sample; the closing sample is still included when ``end`` reaches it.
        """
        if not self.timestamps:
            return []
        include_tail = end >= self.timestamps[-1] - 1e-9
        return [
            value
            for timestamp, value in zip(self.timestamps, self.values, strict=False)
            if start <= timestamp < end or (include_tail and abs(timestamp - end) <= 1e-9)
        ]

    def mean(self, start: float, end: float) -> float | None:
        """Mean of numeric samples in the window, or ``None`` when absent."""
        samples = [float(v) for v in self.samples_between(start, end) if isinstance(v, int | float)]
        return sum(samples) / len(samples) if samples else None

    def rms(self, start: float, end: float) -> float | None:
        """Root-mean-square of numeric samples in the window, or ``None``."""
        samples = [float(v) for v in self.samples_between(start, end) if isinstance(v, int | float)]
        if not samples:
            return None
        return (sum(s * s for s in samples) / len(samples)) ** 0.5

    def component_mean(self, index: int, start: float, end: float) -> float | None:
        """Mean of one component of tuple-valued samples, or ``None``."""
        samples = [
            float(value[index])
            for value in self.samples_between(start, end)
            if isinstance(value, tuple) and len(value) > index
        ]
        return sum(samples) / len(samples) if samples else None


@dataclass
class TelemetryTrack:
    """Decoded telemetry for a single clip."""

    kind: str
    decoder: str
    duration: float = 0.0
    signals: dict[str, TelemetrySeries] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        """Total number of decoded samples across all signals."""
        return sum(len(series.values) for series in self.signals.values())

    def series(self, name: str) -> TelemetrySeries | None:
        """Return a named signal series, if decoded."""
        return self.signals.get(name)


def classify_telemetry_type(stream_tags: Iterable[str]) -> str:
    """Map container stream codec tags to a camera family, or ``""`` when unknown."""
    lowered = {str(tag).strip().lower() for tag in stream_tags if tag}
    for tag, kind in TELEMETRY_STREAM_TAGS.items():
        if tag in lowered:
            return kind
    return ""


def _iter_klv(
    data: bytes, offset: int = 0, end: int | None = None
) -> Iterator[tuple[str, int, bytes]]:
    """Yield ``(key, type_byte, payload)`` triples for KLV records in ``data``."""
    limit = len(data) if end is None else min(end, len(data))
    position = offset
    while position + 8 <= limit:
        key_bytes = data[position : position + 4]
        key = key_bytes.decode("latin-1")
        type_byte = data[position + 4]
        item_size = data[position + 5]
        item_repeat = struct.unpack(">H", data[position + 6 : position + 8])[0]
        payload_length = item_size * item_repeat
        payload_start = position + 8
        payload_end = payload_start + payload_length
        padded_length = (payload_length + 3) & ~3
        next_position = payload_start + padded_length
        if payload_length == 0 or payload_end > limit:
            return
        yield key, type_byte, data[payload_start:payload_end]
        if next_position >= limit:
            break
        position = next_position


def compute_tilt_deg(gx: float, gy: float, gz: float) -> float:
    """Compute tilt angle in degrees relative to upright horizontal.

    In the GoPro body-fixed frame (+X right, +Y bottom, +Z optical axis):
    - When upright and level: gravity points straight down +Y (gy = 1.0, tilt = 0.0 deg).
    - When tilted down at reef: gy > 0 (tilt < 90.0 deg).
    - When inverted / dangling: gy < 0 (tilt > 90.0 deg).
    """
    mag = math.sqrt(gx * gx + gy * gy + gz * gz)
    norm_y = (gy / mag) if mag > 1e-6 else gy
    clamped_y = max(-1.0, min(1.0, norm_y))
    return float(math.acos(clamped_y) * (180.0 / math.pi))


def _decode_leaf_values(key: str, type_byte: int, payload: bytes) -> list[Any]:
    """Decode a GPMF leaf payload into a list of Python values."""
    type_char = chr(type_byte)
    if type_char in {"c", "U", "F"}:
        text = payload.decode("latin-1").rstrip("\x00")
        return [text] if text else []
    mapping = GPMF_TYPES.get(type_char)
    if mapping is None:
        return []
    fmt, size = mapping
    count = len(payload) // size
    if count == 0:
        return []
    values = struct.unpack(f">{count}{fmt}", payload[: count * size])
    return [values[0]] if count == 1 else list(values)


def _group_samples(values: list[Any], components: int) -> list[tuple[int, ...]]:
    """Group flat integer arrays into fixed-width per-sample tuples."""
    numbers = [int(v) for v in values if isinstance(v, int)]
    usable = len(numbers) - (len(numbers) % components)
    return [tuple(numbers[i : i + components]) for i in range(0, usable, components)]


def _gps_speed_ms(sample: tuple[int, ...]) -> float | None:
    """Convert the 2D-speed component (mm/s) of a GPS sample into m/s."""
    if len(sample) < 4:
        return None
    return round(abs(float(sample[3])) / 1000.0, 3)


GPS_COMPONENTS: dict[str, int] = {"GPS5": 5, "GPS9": 9, "GPS2": 2}

# Vector leaves keep their axes together so per-component access stays possible.
VECTOR_COMPONENTS: dict[str, int] = {"CORI": 3, "IORI": 3, "GRAV": 3}

SAMPLE_COMPONENTS: dict[str, int] = {**GPS_COMPONENTS, **VECTOR_COMPONENTS}

GPS_SIGNALS = set(GPS_COMPONENTS)


def _walk_container(
    payload: bytes,
    leaves: dict[str, list[tuple[float, Any]]],
    state: dict[str, Any],
) -> None:
    """Walk a GPMF container payload, attributing leaf samples to the stream clock.

    ``STMP`` records carry the sample clock in microseconds. The first value seen
    becomes the base offset; later values advance the clock used for subsequent
    leaves. Leaf samples inside one KLV record share that record's timestamp, which
    keeps 200 Hz signals auditable without inventing per-sample timing.
    """
    for key, type_byte, child in _iter_klv(payload):
        if key == "STMP":
            # Firmware variance: STMP records are 4- or 8-byte microsecond counts.
            if len(child) in (4, 8):
                value = float(int.from_bytes(child, "big"))
                if state.get("stmp_base_us") is None:
                    state["stmp_base_us"] = value
                state["stmp_us"] = value
            continue
        is_container = type_byte in GPMF_CONTAINER_TYPES or key in GPMF_CONTAINER_KEYS
        if is_container:
            stream_state = dict(state)
            if key == "STRM":
                stream_state["scal"] = None
            _walk_container(child, leaves, stream_state)
            continue
        values = _decode_leaf_values(key, type_byte, child)
        if not values:
            continue

        if key == "SCAL":
            state["scal"] = values
            continue

        if key in ("GYRO", "ACCL") and state.get("scal"):
            scal = state["scal"]
            scal_len = len(scal)
            values = [
                float(val) / float(scal[i % scal_len])
                if float(scal[i % scal_len]) != 0
                else float(val)
                for i, val in enumerate(values)
            ]

        seconds = 0.0
        if state.get("stmp_us") is not None and state.get("stmp_base_us") is not None:
            seconds = max(0.0, (float(state["stmp_us"]) - state["stmp_base_us"]) / 1_000_000.0)
        if key in SAMPLE_COMPONENTS:
            grouped = _group_samples(values, SAMPLE_COMPONENTS[key])
            leaves.setdefault(key, []).extend((seconds, sample) for sample in grouped)
        else:
            leaves.setdefault(key, []).extend((seconds, value) for value in values)


def parse_gpmf(data: bytes, duration: float = 0.0) -> TelemetryTrack | None:
    """Parse a raw GPMF ``gpmd`` payload into a :class:`TelemetryTrack`."""
    if not data:
        return None
    leaves: dict[str, list[tuple[float, Any]]] = {}
    state: dict[str, Any] = {"stmp_us": None, "stmp_base_us": None}
    try:
        _walk_container(data, leaves, state)
    except (struct.error, IndexError, ValueError) as exc:
        logger.debug("GPMF decode failed: %s", exc)
        return None

    signals: dict[str, TelemetrySeries] = {}
    for key, samples in leaves.items():
        series = TelemetrySeries(name=key)
        for timestamp, value in samples:
            series.timestamps.append(float(timestamp))
            series.values.append(_normalize_signal_value(key, value))
        if series.values:
            signals[key] = series

    if not signals:
        return None
    return TelemetryTrack(
        kind=GOPRO,
        decoder="gpmf",
        duration=duration,
        signals=signals,
        provenance={
            "kind": GOPRO,
            "decoder": "gpmf",
            "supported": True,
            "signals": sorted(signals),
            "sample_count": sum(len(s.values) for s in signals.values()),
        },
    )


def _normalize_signal_value(key: str, value: Any) -> Any:
    """Normalize raw GPMF values into trace-friendly Python types."""
    if key == "SCEN" and isinstance(value, str):
        return value.strip().upper()[:4] or None
    if key == "FACE" and isinstance(value, int | float):
        return int(value)
    if key == "GRAV" and isinstance(value, tuple) and len(value) == 3:
        return tuple(round(float(c) / 32767.0, 4) for c in value)
    return value


def gps_speed_samples(track: TelemetryTrack) -> list[tuple[float, float]]:
    """Return ``(timestamp, speed_ms)`` pairs for the clip's GPS signal."""
    for name in ("GPS5", "GPS9", "GPS2"):
        series = track.series(name)
        if series is None:
            continue
        samples: list[tuple[float, float]] = []
        for timestamp, value in zip(series.timestamps, series.values, strict=False):
            if isinstance(value, tuple):
                speed = _gps_speed_ms(value)
                if speed is not None:
                    samples.append((timestamp, speed))
        if samples:
            return samples
    return []


def _unsupported_track(kind: str, reason: str) -> TelemetryTrack | None:
    """Return ``None`` while logging the documented reason a decoder is unavailable."""
    logger.debug("Telemetry decoder for %s unavailable: %s", kind, reason)
    return None


def telemetry_unsupported_reason(kind: str) -> str:
    """Return the documented reason telemetry decoding is unsupported for a camera family."""
    if kind == DJI:
        return (
            "DJI djmd telemetry is a protobuf payload without a verified, license-compatible "
            "Python 3.14 decoder; clip falls back to the visual pipeline"
        )
    if kind == INSTA360:
        return (
            "Insta360 trailer telemetry layout varies by firmware and is not decoded; "
            "clip falls back to the visual pipeline"
        )
    return "no supported telemetry stream detected"


def detect_telemetry_stream(
    video_path: Path, streams: list[dict[str, Any]] | None = None
) -> tuple[str, int] | None:
    """Detect which stream index in ``video_path`` carries action-camera telemetry.

    Inspects container streams using ffprobe (or reuses pre-probed ``streams``).
    Matches known codec tags ('gpmd' for GoPro, 'djmd' for DJI, 'insta' for Insta360)
    and stream handler names ('GoPro MET', 'DJI.Meta').

    Returns ``(camera_family, stream_index)`` (e.g. ``('gopro', 3)``) or
    ``None`` when no telemetry stream is present.
    """
    stream_list: list[dict[str, Any]] = []
    if streams is not None:
        stream_list = streams
    else:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(video_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("ffprobe stream detection failed for %s: %s", video_path, exc)
            return None

        if result.returncode != 0 or not result.stdout:
            return None

        try:
            data = json.loads(result.stdout)
            stream_list = data.get("streams", []) if isinstance(data, dict) else []
        except OSError, ValueError:
            return None

    for s in stream_list:
        try:
            idx = int(s.get("index", 0))
        except ValueError, TypeError:
            continue
        tag = str(s.get("codec_tag_string") or "").strip().lower()
        tags_dict = s.get("tags") if isinstance(s.get("tags"), dict) else {}
        handler = str(tags_dict.get("handler_name") or "").strip().lower()

        if tag == "gpmd" or "gopro met" in handler:
            return (GOPRO, idx)
        if tag == "djmd" or "dji.meta" in handler:
            return (DJI, idx)
        if tag == "insta" or "insta360" in handler:
            return (INSTA360, idx)

    return None


def extract_telemetry_bytes(
    video_path: Path, stream_index: int | None = None, timeout: int = 120
) -> bytes | None:
    """Extract a raw telemetry stream (e.g. ``gpmd``) from a clip using ffmpeg.

    Note that ``stream_index`` must be the absolute container stream index
    (e.g. ``3`` as reported by ``ffprobe stream.index`` or :func:`detect_telemetry_stream`),
    not the data-stream ordinal (e.g. ``0:d:0``). When ``stream_index`` is omitted,
    the telemetry stream is detected automatically.
    """
    target_index = stream_index
    if target_index is None:
        detected = detect_telemetry_stream(video_path)
        if detected is None:
            logger.debug("No telemetry stream detected for %s", video_path)
            return None
        target_index = detected[1]

    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-map",
        f"0:{target_index}",
        "-c",
        "copy",
        "-f",
        "rawvideo",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Telemetry extraction failed for %s: %s", video_path, exc)
        return None
    if result.returncode != 0 or not result.stdout:
        logger.debug(
            "Telemetry extraction returned no data for %s: %s",
            video_path,
            result.stderr.decode("utf-8", "replace").strip()[:200],
        )
        return None
    return result.stdout


def load_telemetry(
    video_path: Path,
    telemetry_type: str = "",
    duration: float = 0.0,
    cache_path: Path | None = None,
    stream_index: int | None = None,
) -> TelemetryTrack | None:
    """Decode a clip's telemetry track, failing soft on every unsupported case."""
    kind = telemetry_type or ""
    target_stream = stream_index
    if not kind or target_stream is None:
        detected = detect_telemetry_stream(video_path)
        if detected is not None:
            if not kind:
                kind = detected[0]
            if target_stream is None:
                target_stream = detected[1]
        elif not kind:
            return None

    if kind in {DJI, INSTA360}:
        return _unsupported_track(kind, telemetry_unsupported_reason(kind))

    payload: bytes | None = None
    if cache_path is not None and cache_path.exists():
        try:
            payload = cache_path.read_bytes()
        except OSError as exc:
            logger.debug("Could not read cached telemetry %s: %s", cache_path, exc)
            payload = None
    if payload is None:
        payload = extract_telemetry_bytes(video_path, stream_index=target_stream)
        if payload is not None and cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(payload)
            except OSError as exc:
                logger.debug("Could not cache telemetry to %s: %s", cache_path, exc)
    if payload is None:
        return None
    return parse_gpmf(payload, duration=duration)


def aggregate_window(track: TelemetryTrack | None, start: float, end: float) -> dict[str, Any]:
    """Collapse a telemetry track to one auditable record for a candidate window."""
    record: dict[str, Any] = {
        "interval": [round(start, 3), round(end, 3)],
        "sample_count": 0,
        "scen_dominant": None,
        "scen_confidence": None,
        "yavg": None,
        "unif": None,
        "roll_deg": None,
        "pitch_deg": None,
        "tilt_deg": None,
        "gyro_rms": None,
        "gps_speed_ms": None,
        "face_detected": None,
    }
    if track is None:
        return record

    scen = track.series("SCEN")
    if scen is not None:
        samples = [s for s in scen.samples_between(start, end) if isinstance(s, str) and s]
        if samples:
            counts: dict[str, int] = {}
            for sample in samples:
                counts[sample] = counts.get(sample, 0) + 1
            dominant, count = max(counts.items(), key=lambda item: item[1])
            record["scen_dominant"] = dominant
            record["scen_confidence"] = round(count / len(samples), 3)

    yavg = track.series("YAVG")
    if yavg is not None:
        record["yavg"] = _round_or_none(yavg.mean(start, end))

    unif = track.series("UNIF")
    if unif is not None:
        record["unif"] = _round_or_none(unif.mean(start, end))

    # CORI/IORI are int16 triples in raw sensor units; these are documented proxies,
    # not calibrated angles (see README telemetry notes).
    cori = track.series("CORI")
    if cori is not None:
        record["roll_deg"] = _round_or_none(cori.component_mean(0, start, end))

    iori = track.series("IORI")
    if iori is not None:
        record["pitch_deg"] = _round_or_none(iori.component_mean(1, start, end))
    if record["pitch_deg"] is None and cori is not None:
        record["pitch_deg"] = _round_or_none(cori.component_mean(1, start, end))

    grav = track.series("GRAV")
    if grav is not None:
        grav_samples = [
            s for s in grav.samples_between(start, end) if isinstance(s, tuple) and len(s) == 3
        ]
        if grav_samples:
            gx = sum(s[0] for s in grav_samples) / len(grav_samples)
            gy = sum(s[1] for s in grav_samples) / len(grav_samples)
            gz = sum(s[2] for s in grav_samples) / len(grav_samples)
            record["tilt_deg"] = round(compute_tilt_deg(gx, gy, gz), 1)

    gyro = track.series("GYRO")
    if gyro is not None:
        record["gyro_rms"] = _round_or_none(gyro.rms(start, end))

    gps = track.series("GPS5") or track.series("GPS9") or track.series("GPS2")
    if gps is not None:
        speeds = [
            speed
            for value in gps.samples_between(start, end)
            if isinstance(value, tuple) and (speed := _gps_speed_ms(value)) is not None
        ]
        if speeds:
            record["gps_speed_ms"] = round(max(speeds), 3)

    face = track.series("FACE")
    if face is not None:
        samples = [v for v in face.samples_between(start, end) if isinstance(v, int | float)]
        if samples:
            record["face_detected"] = any(v > 0 for v in samples)

    record["sample_count"] = sum(
        len(series.samples_between(start, end)) for series in track.signals.values()
    )
    return record


def _round_or_none(value: float | None, digits: int = 3) -> float | None:
    """Round a value, preserving ``None`` for missing sensor data."""
    return None if value is None else round(float(value), digits)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


def compute_telemetry_score(
    signals: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute the normalized ``telemetry_score`` from available signals.

    Missing signals are dropped and the remaining weights are renormalized, so an
    absent sensor never contributes an implicit zero or penalty.
    """
    effective_weights = dict(weights or DEFAULT_TELEMETRY_WEIGHTS)
    components: dict[str, float] = {}

    speed = signals.get("gps_speed_ms")
    lean_candidates = [
        abs(float(signals[key]))
        for key in ("roll_deg", "pitch_deg")
        if signals.get(key) is not None
    ]
    lean_parts: list[float] = []
    if isinstance(speed, int | float):
        lean_parts.append(_clamp(float(speed) / SCORE_NORMALIZATION["speed_ms"]))
    if lean_candidates:
        lean_parts.append(_clamp(max(lean_candidates) / SCORE_NORMALIZATION["lean_deg"]))
    if lean_parts:
        components["lean_speed"] = round(sum(lean_parts) / len(lean_parts), 4)

    gyro = signals.get("gyro_rms")
    if isinstance(gyro, int | float):
        components["gyro_stability"] = round(
            1.0 - _clamp(float(gyro) / SCORE_NORMALIZATION["gyro_rms"]), 4
        )

    face = signals.get("face_detected")
    if isinstance(face, bool):
        components["face"] = 1.0 if face else 0.0

    hue_rate = signals.get("hue_changes_per_s")
    if isinstance(hue_rate, int | float):
        components["hue_change"] = round(
            _clamp(float(hue_rate) / SCORE_NORMALIZATION["hue_changes_per_s"]), 4
        )

    used_weights = {
        name: effective_weights[name] for name in components if name in effective_weights
    }
    weight_sum = sum(used_weights.values())
    if weight_sum <= 0:
        return {
            "score": None,
            "components": components,
            "weights": {},
            "normalization": dict(SCORE_NORMALIZATION),
            "available": sorted(components),
        }

    normalized_weights = {
        name: round(weight / weight_sum, 4) for name, weight in used_weights.items()
    }
    score = sum(components[name] * weight for name, weight in normalized_weights.items())
    return {
        "score": round(_clamp(score), 4),
        "components": components,
        "weights": normalized_weights,
        "normalization": dict(SCORE_NORMALIZATION),
        "available": sorted(components),
    }


def evaluate_gates(
    signals: dict[str, Any],
    thresholds: dict[str, float] | None = None,
    mode: str = "advisory",
) -> list[dict[str, Any]]:
    """Evaluate telemetry gates, recording input, threshold, outcome, and reason."""
    effective = {**DEFAULT_GATE_THRESHOLDS, **(thresholds or {})}
    gates: list[dict[str, Any]] = []

    gates.append(
        _gate_result(
            "dark",
            signals.get("yavg"),
            effective["yavg_min"],
            "below",
            mode,
            "exposure mean below visibility floor",
        )
    )
    gates.append(
        _gate_result(
            "featureless",
            signals.get("unif"),
            effective["unif_max"],
            "above",
            mode,
            "frame uniformity above detail floor",
        )
    )
    gates.append(
        _gate_result(
            "shake",
            signals.get("gyro_rms"),
            effective["gyro_max"],
            "above",
            mode,
            "gyro rate RMS above stability floor",
        )
    )
    gates.append(
        _gate_result(
            "orientation",
            signals.get("tilt_deg"),
            effective["tilt_max"],
            "above",
            mode,
            "camera tilted or inverted past stability threshold",
        )
    )
    return gates


def _gate_result(
    name: str,
    value: Any,
    threshold: float,
    direction: str,
    mode: str,
    reason: str,
) -> dict[str, Any]:
    """Build a single gate record, marking missing inputs as skipped."""
    if not isinstance(value, int | float):
        return {
            "gate": name,
            "input": None,
            "threshold": threshold,
            "direction": direction,
            "outcome": "skip",
            "reason": "signal unavailable",
            "mode": mode,
        }
    triggered = float(value) < threshold if direction == "below" else float(value) > threshold
    return {
        "gate": name,
        "input": round(float(value), 3),
        "threshold": threshold,
        "direction": direction,
        "outcome": "fail" if triggered else "pass",
        "reason": reason if triggered else "within threshold",
        "mode": mode,
    }


def gates_reject(gates: list[dict[str, Any]]) -> bool:
    """Return ``True`` when an explicitly enabled gate rejects the sample."""
    return any(gate["outcome"] == "fail" and gate.get("mode") == "enabled" for gate in gates)


def telemetry_domain_prior(
    track: TelemetryTrack | None,
    thresholds: dict[str, float] | None = None,
    duration: float = 0.0,
) -> dict[str, Any] | None:
    """Derive a Pass 0 domain prior from dominant ``SCEN`` codes.

    Returns ``None`` when there is no usable scene signal. The result always
    records confidence, coverage, thresholds, and the fallback reason so a caller
    can decide whether to skip the Pass 0 VLM call.
    """
    if track is None:
        return None
    scen = track.series("SCEN")
    if scen is None:
        return None
    samples = [s for s in scen.values if isinstance(s, str) and s]
    if not samples:
        return None

    effective = {**DEFAULT_DOMAIN_PRIOR_THRESHOLDS, **(thresholds or {})}
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample] = counts.get(sample, 0) + 1
    dominant, count = max(counts.items(), key=lambda item: item[1])
    confidence = round(count / len(samples), 3)
    domain = SCEN_DOMAIN_MAP.get(dominant, "generic")

    # SCEN samples arrive at ~1 Hz, so samples-per-second approximates coverage.
    coverage = 1.0 if duration <= 0 else round(min(1.0, len(samples) / duration), 3)

    fallback_reason = ""
    if dominant not in SCEN_DOMAIN_MAP:
        fallback_reason = f"unmapped SCEN code {dominant!r}; falling back to visual classification"
    elif confidence < effective["min_confidence"]:
        fallback_reason = (
            f"SCEN confidence {confidence} below threshold {effective['min_confidence']}"
        )
    elif coverage < effective["min_coverage"]:
        fallback_reason = f"SCEN coverage {coverage} below threshold {effective['min_coverage']}"

    return {
        "domain": domain,
        "scen_dominant": dominant,
        "confidence": confidence,
        "coverage": coverage,
        "sample_count": len(samples),
        "thresholds": effective,
        "fast_path": not fallback_reason,
        "fallback_reason": fallback_reason,
    }


def telemetry_provenance(track: TelemetryTrack | None, telemetry_type: str = "") -> dict[str, Any]:
    """Describe telemetry availability for trace/provenance records."""
    if track is not None:
        return dict(track.provenance)
    kind = telemetry_type or ""
    return {
        "kind": kind,
        "decoder": "",
        "supported": False,
        "reason": telemetry_unsupported_reason(kind),
        "sample_count": 0,
    }
