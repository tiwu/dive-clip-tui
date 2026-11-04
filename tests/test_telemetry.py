"""Tests for the action-camera telemetry decoder and advisory scoring helpers."""

import struct
from pathlib import Path

import pytest

from utils.telemetry import (
    DJI,
    GOPRO,
    INSTA360,
    aggregate_window,
    classify_telemetry_type,
    compute_telemetry_score,
    evaluate_gates,
    gates_reject,
    gps_speed_samples,
    load_telemetry,
    parse_gpmf,
    telemetry_domain_prior,
    telemetry_provenance,
    telemetry_unsupported_reason,
)

# --- synthetic GPMF builders ------------------------------------------------


def _klv(key: str, type_byte: int, payload: bytes, item_size: int | None = None) -> bytes:
    """Build a single GPMF KLV record."""
    if item_size is None:
        item_size = len(payload)
    repeat = len(payload) // item_size
    header = key.encode("latin-1") + bytes([type_byte, item_size]) + struct.pack(">H", repeat)
    return header + payload


def _container(key: str, payload: bytes) -> bytes:
    """Build a GPMF container record (type byte 0x00)."""
    return _klv(key, 0x00, payload, 1)


def _stmp(microseconds: int) -> bytes:
    """Build an ``STMP`` clock record."""
    return _klv("STMP", 0x00, struct.pack(">I", microseconds), 4)


def _u32(key: str, values: list[int]) -> bytes:
    """Build a uint32 leaf record."""
    return _klv(key, ord("L"), struct.pack(f">{len(values)}I", *values), 4)


def _i32(key: str, values: list[int]) -> bytes:
    """Build an int32 leaf record."""
    return _klv(key, ord("l"), struct.pack(f">{len(values)}i", *values), 4)


def _i16(key: str, values: list[int]) -> bytes:
    """Build an int16 leaf record."""
    return _klv(key, ord("h"), struct.pack(f">{len(values)}h", *values), 2)


def _text(key: str, value: str) -> bytes:
    """Build a text/FourCC leaf record."""
    return _klv(key, ord("c"), value.encode("latin-1"), 1)


def _stream(
    scen: list[str],
    *,
    duration_seconds: float = 2.0,
    gps_speeds_mm_s: list[int] | None = None,
) -> bytes:
    """Build a multi-record GPMF stream with a 0.5 s clock step."""
    scene_per_record = scen or ["WATR"]
    steps = max(1, len(scene_per_record))
    step_us = int(duration_seconds * 1_000_000 / steps)

    records: list[bytes] = []
    for index, code in enumerate(scene_per_record):
        leaves = (
            _text("SCEN", code)
            + _i32("YAVG", [120, 110])
            + _i16("GYRO", [3, 4, 0])
            + _i16("CORI", [10 * (index + 1), -5, 0])
            + _u32("FACE", [1])
        )
        if gps_speeds_mm_s:
            speed = gps_speeds_mm_s[min(index, len(gps_speeds_mm_s) - 1)]
            leaves += _i32("GPS5", [1000 + index, 2000 + index, 500, speed, 0])
        records.append(_stmp(1_000_000 + index * step_us))
        records.append(_container("DEVC", _container("STRM", leaves)))
    return b"".join(records)


@pytest.fixture
def gpmf_track():
    """A two-second synthetic telemetry track inside the ``underwater`` domain."""
    return parse_gpmf(_stream(["WATR"] * 4, gps_speeds_mm_s=[500, 1500, 3000, 2500]), duration=2.0)


# --- stream classification --------------------------------------------------


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (["gpmd"], GOPRO),
        (["djmd"], DJI),
        (["insta"], INSTA360),
        (["avc1", "mp4a"], ""),
        ([], ""),
    ],
)
def test_classify_telemetry_type(tags: list[str], expected: str):
    assert classify_telemetry_type(tags) == expected


def test_classify_telemetry_type_is_case_insensitive():
    assert classify_telemetry_type(["GPMD"]) == GOPRO


# --- GPMF decoding ----------------------------------------------------------


def test_parse_gpmf_returns_track(gpmf_track):
    assert gpmf_track is not None
    assert gpmf_track.kind == GOPRO
    assert gpmf_track.decoder == "gpmf"
    assert gpmf_track.provenance["supported"] is True


def test_parse_gpmf_normalizes_scen_codes(gpmf_track):
    assert gpmf_track.series("SCEN").values == ["WATR"] * 4


def test_parse_gpmf_attributes_samples_to_stmp_clock(gpmf_track):
    timestamps = gpmf_track.series("YAVG").timestamps
    assert timestamps[0] == pytest.approx(0.0)
    assert timestamps[-1] == pytest.approx(1.5)


def test_parse_gpmf_groups_gps_and_vector_samples(gpmf_track):
    gps_sample = gpmf_track.series("GPS5").values[0]
    assert isinstance(gps_sample, tuple)
    assert len(gps_sample) == 5
    cori_sample = gpmf_track.series("CORI").values[0]
    assert isinstance(cori_sample, tuple)
    assert len(cori_sample) == 3


def test_parse_gpmf_returns_none_without_payload():
    assert parse_gpmf(b"") is None


def test_gps_speed_samples_converts_mm_s_to_m_s(gpmf_track):
    samples = gps_speed_samples(gpmf_track)
    assert samples == [
        (0.0, 0.5),
        (0.5, 1.5),
        (1.0, 3.0),
        (1.5, 2.5),
    ]


# --- window aggregation -----------------------------------------------------


def test_aggregate_window_reports_sensor_signals(gpmf_track):
    record = aggregate_window(gpmf_track, 0.0, 1.0)

    assert record["interval"] == [0.0, 1.0]
    assert record["scen_dominant"] == "WATR"
    assert record["scen_confidence"] == 1.0
    assert record["yavg"] == pytest.approx(115.0)
    assert record["roll_deg"] == pytest.approx(15.0)
    assert record["pitch_deg"] == pytest.approx(-5.0)
    assert record["gps_speed_ms"] == pytest.approx(1.5)
    assert record["face_detected"] is True
    assert record["sample_count"] > 0


def test_aggregate_window_keeps_gyro_as_axis_rms(gpmf_track):
    record = aggregate_window(gpmf_track, 0.0, 0.5)
    # One record inside the half-open window: RMS over the 3 axes, not the mean.
    assert record["gyro_rms"] == pytest.approx(round(((3**2 + 4**2 + 0**2) / 3) ** 0.5, 3))


def test_aggregate_window_without_track_returns_nulls():
    record = aggregate_window(None, 1.0, 2.0)

    assert record["interval"] == [1.0, 2.0]
    assert record["scen_dominant"] is None
    assert record["yavg"] is None
    assert record["roll_deg"] is None
    assert record["gyro_rms"] is None
    assert record["face_detected"] is None
    assert record["sample_count"] == 0


def test_aggregate_window_omits_missing_signals_as_null():
    track = parse_gpmf(_stream(["WATR"]), duration=1.0)
    record = aggregate_window(track, 0.0, 1.0)

    assert record["gyro_rms"] is not None
    assert record["gps_speed_ms"] is None
    assert record["unif"] is None


# --- telemetry score --------------------------------------------------------


def test_compute_telemetry_score_renormalizes_weights():
    result = compute_telemetry_score({"gyro_rms": 0.0})

    assert result["score"] == 1.0
    assert result["weights"] == {"gyro_stability": 1.0}
    assert result["available"] == ["gyro_stability"]


def test_compute_telemetry_score_without_signals_is_none():
    result = compute_telemetry_score({})

    assert result["score"] is None
    assert result["weights"] == {}
    assert result["components"] == {}


def test_compute_telemetry_score_does_not_penalize_missing_sensors():
    assert compute_telemetry_score({"face_detected": True})["score"] == 1.0
    assert compute_telemetry_score({"gps_speed_ms": 3.0})["score"] == 1.0


def test_compute_telemetry_score_mixes_available_components():
    result = compute_telemetry_score({"gyro_rms": 3.5, "gps_speed_ms": 3.0, "face_detected": False})

    # lean_speed 1.0 (0.35), gyro_stability 0.0 (0.25), face 0.0 (0.20)
    total = 0.35 + 0.25 + 0.20
    assert result["score"] == pytest.approx(round(1.0 * 0.35 / total, 4))
    assert result["weights"] == {
        "lean_speed": pytest.approx(round(0.35 / total, 4)),
        "gyro_stability": pytest.approx(round(0.25 / total, 4)),
        "face": pytest.approx(round(0.20 / total, 4)),
    }


# --- advisory gates ---------------------------------------------------------


def test_gates_are_advisory_by_default():
    gates = evaluate_gates({"yavg": 5.0})

    dark = next(gate for gate in gates if gate["gate"] == "dark")
    assert dark["outcome"] == "fail"
    assert dark["mode"] == "advisory"
    assert dark["reason"]
    assert gates_reject(gates) is False


def test_gates_reject_when_explicitly_enabled():
    gates = evaluate_gates({"yavg": 5.0}, mode="enabled")

    assert gates_reject(gates) is True


def test_gates_skip_missing_signals():
    gates = evaluate_gates({})

    assert {gate["outcome"] for gate in gates} == {"skip"}
    assert gates_reject(gates) is False


def test_gates_accept_threshold_overrides():
    gates = evaluate_gates({"unif": 0.5}, thresholds={"unif_max": 0.4})

    featureless = next(gate for gate in gates if gate["gate"] == "featureless")
    assert featureless["outcome"] == "fail"
    assert featureless["threshold"] == 0.4


# --- Pass 0 domain prior ----------------------------------------------------


def test_domain_prior_fast_path_for_confident_scen(gpmf_track):
    prior = telemetry_domain_prior(gpmf_track, duration=2.0)

    assert prior is not None
    assert prior["domain"] == "underwater"
    assert prior["scen_dominant"] == "WATR"
    assert prior["confidence"] == 1.0
    assert prior["coverage"] == 1.0
    assert prior["fast_path"] is True
    assert prior["fallback_reason"] == ""


def test_domain_prior_falls_back_when_coverage_is_low(gpmf_track):
    prior = telemetry_domain_prior(gpmf_track, duration=60.0)

    assert prior["coverage"] == pytest.approx(round(4 / 60, 3))
    assert prior["fast_path"] is False
    assert "coverage" in prior["fallback_reason"]


def test_domain_prior_falls_back_for_unmapped_codes():
    track = parse_gpmf(_stream(["ZZZZ"]), duration=1.0)
    prior = telemetry_domain_prior(track, duration=1.0)

    assert prior["domain"] == "generic"
    assert prior["fast_path"] is False
    assert "unmapped" in prior["fallback_reason"]


def test_domain_prior_returns_none_without_scen():
    track = parse_gpmf(_stream(["WATR"]), duration=1.0)
    del track.signals["SCEN"]

    assert telemetry_domain_prior(track, duration=1.0) is None
    assert telemetry_domain_prior(None, duration=1.0) is None


# --- unsupported camera families -------------------------------------------


@pytest.mark.parametrize("kind", [DJI, INSTA360])
def test_unsupported_families_fail_soft(tmp_path: Path, kind: str):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")

    assert load_telemetry(video, telemetry_type=kind, duration=1.0) is None
    assert telemetry_unsupported_reason(kind)

    provenance = telemetry_provenance(None, kind)
    assert provenance["supported"] is False
    assert provenance["reason"] == telemetry_unsupported_reason(kind)
    assert provenance["sample_count"] == 0


def test_load_telemetry_without_type_returns_none(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")

    assert load_telemetry(video) is None


def test_load_telemetry_uses_and_writes_cache(tmp_path: Path, monkeypatch, gpmf_track):
    calls: list[Path] = []

    def fake_extract(path: Path, *_args, **_kwargs) -> bytes:
        calls.append(path)
        return _stream(["WATR"] * 2)

    monkeypatch.setattr("utils.telemetry.extract_telemetry_bytes", fake_extract)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    cache = tmp_path / "cache" / "clip.gpmd"

    first = load_telemetry(video, telemetry_type=GOPRO, duration=1.0, cache_path=cache)
    second = load_telemetry(video, telemetry_type=GOPRO, duration=1.0, cache_path=cache)

    assert cache.exists()
    assert len(calls) == 1
    assert first is not None
    assert second is not None
    assert second.series("SCEN").values == ["WATR"] * 2
    assert gpmf_track is not None
