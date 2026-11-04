"""Tests for utils/prescreen.py: the pre-VLM screening funnel."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from utils.models import PrescreenTraceOutput
from utils.prescreen import (
    DEFAULT_AUDIO_SPIKE_DB,
    DEFAULT_AUDIO_SPIKE_FLOOR_DB,
    PrescreenConfig,
    audio_window_metrics,
    build_gates,
    build_prescreen_record,
    candidate_windows,
    detect_audio_spikes,
    hsv_entropy,
    hsv_histogram,
    hsv_similarity,
    hue_change_rate,
    laplacian_variance,
    luma_uniformity,
    mean_luma,
    resolve_action,
    rgb_to_luma,
    screen_clip,
    visual_metrics,
)

SIZE = 8


def make_frame(red: int, green: int, blue: int, size: int = SIZE) -> bytes:
    """Build one packed rgb24 frame filled with a single colour."""
    return bytes([red, green, blue] * (size * size))


def make_checker(size: int = SIZE) -> bytes:
    """Build a high-frequency black/white checker so luma spread and edges exist."""
    pixels: list[int] = []
    for row in range(size):
        for column in range(size):
            value = 255 if (row + column) % 2 else 0
            pixels.extend([value, value, value])
    return bytes(pixels)


def make_split(
    red: int, green: int, blue: int, other: tuple[int, int, int], size: int = SIZE
) -> bytes:
    """Build a frame whose left half carries one hue and whose right half another."""
    pixels: list[int] = []
    for _row in range(size):
        for column in range(size):
            color = (red, green, blue) if column < size // 2 else other
            pixels.extend(color)
    return bytes(pixels)


# --- candidate windows -------------------------------------------------------


@pytest.mark.parametrize(
    ("tags", "duration", "expected"),
    [
        ([10.0], 60.0, [(6.0, 14.0)]),
        ([1.0], 60.0, [(0.0, 5.0)]),
        ([58.0], 60.0, [(54.0, 60.0)]),
        ([59.0, 61.0], 60.0, [(55.0, 60.0), (57.0, 60.0)]),
        ([], 60.0, []),
    ],
)
def test_candidate_windows_clamps_to_duration(
    tags: list[float], duration: float, expected: list[tuple[float, float]]
) -> None:
    """HiLight windows pad by 4 s each side and stay inside the clip."""
    assert candidate_windows(tags, duration) == expected


def test_candidate_windows_respects_custom_pad() -> None:
    """The pad is configurable and windows collapse when clamped to nothing."""
    assert candidate_windows([30.0], 60.0, pad=2.0) == [(28.0, 32.0)]
    # A marker past the clip end collapses away once clamped.
    assert candidate_windows([10.0], 0.5, pad=4.0) == []


def test_candidate_windows_accepts_tuple_sequence() -> None:
    """Tags may arrive as any sequence of numbers, not just floats."""
    assert candidate_windows((2,), 30.0) == [(0.0, 6.0)]


# --- pixel primitives --------------------------------------------------------


def test_rgb_to_luma_uses_rec601_weights() -> None:
    """Luma conversion matches the documented integer Rec. 601 weights."""
    assert rgb_to_luma(make_frame(255, 255, 255), SIZE) == [255] * (SIZE * SIZE)
    assert rgb_to_luma(make_frame(0, 0, 0), SIZE) == [0] * (SIZE * SIZE)


def test_rgb_to_luma_ignores_trailing_partial_pixels() -> None:
    """A truncated payload yields only the complete pixels it contains."""
    assert len(rgb_to_luma(b"\x10\x20\x30\x40", SIZE)) == 1


def test_mean_luma_and_uniformity_on_flat_frame() -> None:
    """A flat frame is uniform; a checker is not."""
    assert mean_luma(rgb_to_luma(make_frame(40, 40, 40), SIZE)) == 40.0
    assert luma_uniformity(rgb_to_luma(make_frame(40, 40, 40), SIZE)) == 1.0
    assert luma_uniformity(rgb_to_luma(make_checker(), SIZE)) < 0.2


def test_uniformity_needs_two_pixels() -> None:
    """Uniformity is undefined for a single sample."""
    assert luma_uniformity([7]) is None


def test_laplacian_variance_separates_flat_from_textured() -> None:
    """Out-of-focus footage has no edge response; a checker has plenty."""
    assert laplacian_variance(rgb_to_luma(make_frame(40, 40, 40), SIZE), SIZE) == 0.0
    assert laplacian_variance(rgb_to_luma(make_checker(), SIZE), SIZE) > 1000.0


def test_laplacian_variance_needs_interior_pixels() -> None:
    """A 1x1 or 2x2 frame has no interior and reports nothing."""
    assert laplacian_variance(rgb_to_luma(make_frame(9, 9, 9), 1), 1) is None


# --- hue histograms ----------------------------------------------------------


def test_hsv_histogram_is_empty_for_achromatic_frames() -> None:
    """Grey and black carry no hue, so callers get no similarity signal."""
    assert hsv_histogram(make_frame(40, 40, 40), SIZE) == []
    assert hsv_histogram(make_frame(0, 0, 0), SIZE) == []


def test_hsv_histogram_is_normalized() -> None:
    """The hue histogram sums to 1 for a chromatic frame."""
    histogram = hsv_histogram(make_split(255, 0, 0, (0, 0, 255)), SIZE)
    assert len(histogram) == 8
    assert sum(histogram) == pytest.approx(1.0, abs=1e-3)


def test_hsv_entropy_bounds() -> None:
    """Entropy is 0 for a single hue bin, higher for a spread, None with no data."""
    assert hsv_entropy([1.0] + [0.0] * 7) == 0.0
    assert hsv_entropy([]) is None
    spread = hsv_entropy([0.25] * 4 + [0.0] * 4)
    assert spread is not None
    assert spread > 0.5


def test_hsv_similarity_returns_none_for_missing_histograms() -> None:
    """An achromatic frame yields ``None`` instead of a fake similarity score."""
    assert hsv_similarity([], [0.5, 0.5]) is None
    assert hsv_similarity([0.5, 0.5], []) is None
    assert hsv_similarity([0.5, 0.5], [0.5, 0.5, 0.0]) is None


def test_hsv_similarity_is_intersection() -> None:
    """Identical histograms score 1 and disjoint histograms score 0."""
    assert hsv_similarity([0.5, 0.5], [0.5, 0.5]) == 1.0
    assert hsv_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_hue_change_rate_counts_transitions() -> None:
    """Alternating hues are changes; a missing histogram is not compared."""
    red = make_frame(255, 0, 0)
    blue = make_frame(0, 0, 255)
    metrics = visual_metrics([red, blue, red, blue], size=SIZE)
    assert metrics["hue_changes"] == 3
    assert metrics["hue_changes_per_s"] == 1.0

    grey = make_frame(40, 40, 40)
    mixed = hue_change_rate(
        [hsv_histogram(red, SIZE), hsv_histogram(red, SIZE), hsv_histogram(grey, SIZE)],
        [0.0, 1.0, 2.0],
    )
    # The achromatic frame yields no histogram, so only red->red is comparable.
    assert mixed["compared_pairs"] == 1
    assert mixed["hue_changes"] == 0


# --- visual metrics ----------------------------------------------------------


def test_visual_metrics_averages_over_frames() -> None:
    """Window metrics are means, so one odd frame cannot reject the window alone."""
    dark = make_frame(10, 10, 10)
    bright = make_frame(200, 200, 200)
    metrics = visual_metrics([dark, bright], timestamps=[0.0, 2.0], size=SIZE)
    assert metrics["frame_count"] == 2
    assert metrics["mean_luma"] == 105.0
    assert metrics["timestamps"] == [0.0, 2.0]
    assert len(metrics["histograms"]) == 2


def test_visual_metrics_handles_no_frames() -> None:
    """An empty sample keeps the metric keys present and null."""
    metrics = visual_metrics([], size=SIZE)
    assert metrics["frame_count"] == 0
    assert metrics["mean_luma"] is None
    assert metrics["hue_changes_per_s"] is None


def test_visual_metrics_tolerates_truncated_frames() -> None:
    """Short payloads contribute the pixels they have instead of raising."""
    metrics = visual_metrics([b"\x10\x20\x30"], size=SIZE)
    assert metrics["frame_count"] == 1
    assert metrics["laplacian_sharpness"] is None


# --- audio -------------------------------------------------------------------


def test_detect_audio_spikes_finds_transient_over_steady_bed() -> None:
    """A loud transient above the rolling median is reported with its baseline."""
    peaks = [(index * 0.5, -45.0) for index in range(40)]
    peaks[20] = (10.0, -12.0)
    spikes = detect_audio_spikes(peaks, DEFAULT_AUDIO_SPIKE_DB, DEFAULT_AUDIO_SPIKE_FLOOR_DB, 5)
    assert len(spikes) == 1
    assert spikes[0]["timestamp"] == 10.0
    assert spikes[0]["baseline_db"] == -45.0
    assert spikes[0]["delta_db"] == 33.0


def test_detect_audio_spikes_ignores_steady_quiet_audio() -> None:
    """Underwater flow noise stays quiet and flat, so it never spikes."""
    peaks = [(index * 0.5, -60.0) for index in range(40)]
    assert detect_audio_spikes(peaks) == []


def test_detect_audio_spikes_ignores_quiet_jumps_below_floor() -> None:
    """A relative jump that stays near silence is not an audible event."""
    peaks = [(0.0, -80.0), (0.5, -78.0), (1.0, -76.0)]
    assert detect_audio_spikes(peaks, spike_db=1.0, floor_db=-35.0) == []


def test_audio_window_metrics_reports_peak_and_spike() -> None:
    """Peak and spike signals are restricted to the candidate window."""
    peaks = [(index * 0.5, -45.0) for index in range(40)]
    peaks[20] = (10.0, -12.0)
    metrics = audio_window_metrics(peaks, 9.0, 11.0)
    assert metrics["audio_peak_db"] == -12.0
    assert metrics["audio_spike_triggered"] is True
    assert metrics["audio_spike_time"] == 10.0
    assert metrics["audio_spike_count"] == 1

    outside = audio_window_metrics(peaks, 0.0, 4.0)
    assert outside["audio_spike_triggered"] is False
    assert outside["audio_peak_db"] == -45.0


def test_audio_window_metrics_without_peaks() -> None:
    """A clip with no audio track degrades to null signals, not an error."""
    metrics = audio_window_metrics([], 0.0, 5.0)
    assert metrics["audio_peak_db"] is None
    assert metrics["audio_spike_triggered"] is False


# --- gates -------------------------------------------------------------------

DARK_SIGNALS = {"yavg": 5.0, "unif": 0.5, "gyro_rms": 0.2, "laplacian_sharpness": 40.0}


def test_build_gates_records_thresholds_before_filtering() -> None:
    """Every gate keeps its input, threshold, and reason for the audit trail."""
    gates = build_gates(DARK_SIGNALS, PrescreenConfig())
    by_name = {gate.name: gate for gate in gates}
    assert set(by_name) == {"dark", "featureless", "shake", "orientation", "blur"}
    assert by_name["dark"].value == 5.0
    assert by_name["dark"].threshold == 15.0
    assert by_name["dark"].triggered is True
    assert "visibility floor" in by_name["dark"].reason


def test_build_gates_advisory_preserves_candidates() -> None:
    """In advisory mode a triggered gate is recorded but cannot reject."""
    config = PrescreenConfig(gate_mode="advisory")
    gates = build_gates(DARK_SIGNALS, config)
    assert any(gate.triggered for gate in gates)
    assert {gate.outcome for gate in gates} <= {"pass", "advisory"}
    action, reasons = resolve_action(gates, config.gate_mode)
    assert action == "defer"
    assert reasons and reasons[0].startswith("dark:")


def test_build_gates_enabled_mode_rejects() -> None:
    """Enabled mode turns the identical evaluation into a reject."""
    config = PrescreenConfig(gate_mode="enabled")
    gates = build_gates(DARK_SIGNALS, config)
    assert any(gate.outcome == "reject" for gate in gates)
    action, _ = resolve_action(gates, config.gate_mode)
    assert action == "reject"


def test_build_gates_skips_missing_signals() -> None:
    """Absent signals are skipped, not failed, so clips without telemetry survive."""
    gates = build_gates({"unif": 0.1}, PrescreenConfig(gate_mode="enabled"))
    triggered = [gate for gate in gates if gate.triggered]
    assert triggered == []
    assert resolve_action(gates, "enabled")[0] == "keep"


def test_blur_gate_flags_out_of_focus_frames() -> None:
    """A Laplacian variance under ``blur_min`` triggers the focus gate."""
    gates = build_gates(
        {"laplacian_sharpness": 3.0}, PrescreenConfig(gate_mode="enabled", blur_min=12.0)
    )
    blur = next(gate for gate in gates if gate.name == "blur")
    assert blur.triggered is True
    assert blur.outcome == "reject"
    assert "Laplacian" in blur.reason


def test_blur_gate_uses_configured_threshold() -> None:
    """The focus floor is configurable per run."""
    gates = build_gates(
        {"laplacian_sharpness": 40.0}, PrescreenConfig(gate_mode="enabled", blur_min=50.0)
    )
    blur = next(gate for gate in gates if gate.name == "blur")
    assert blur.triggered is True
    assert blur.threshold == 50.0


def test_resolve_action_without_triggers_keeps() -> None:
    """A clean window keeps and carries no reasons."""
    gates = build_gates(
        {"yavg": 120.0, "unif": 0.2, "gyro_rms": 0.1, "laplacian_sharpness": 900.0},
        PrescreenConfig(gate_mode="enabled"),
    )
    assert resolve_action(gates, "enabled") == ("keep", [])


# --- config ------------------------------------------------------------------


def test_prescreen_config_defaults_are_disabled_and_advisory() -> None:
    """The funnel is opt-in and never rejects until explicitly enabled."""
    config = PrescreenConfig.from_config(None)
    assert config.enabled is False
    assert config.gate_mode == "advisory"


def test_prescreen_config_coerces_yaml_scalars() -> None:
    """String and numeric YAML values coerce instead of raising."""
    config = PrescreenConfig.from_config({
        "prescreen": {
            "enabled": "yes",
            "sample_fps": "1.0",
            "sample_size": "32",
            "keyframes_only": 0,
            "blur_min": "20",
            "weights": {"face": "0.5", "bogus": "not-a-number"},
        }
    })
    assert config.enabled is True
    assert config.sample_fps == 1.0
    assert config.sample_size == 32
    assert config.keyframes_only is False
    assert config.blur_min == 20.0
    assert config.weights == {"face": 0.5}


@pytest.mark.parametrize("raw_mode", ["ENABLED", " Enabled ", "bogus", None, 7])
def test_prescreen_config_gate_mode_falls_back_to_advisory(raw_mode: object) -> None:
    """Unknown gate modes never silently enable rejection."""
    config = PrescreenConfig.from_config({"prescreen": {"gate_mode": raw_mode}})
    assert config.gate_mode in ("advisory", "enabled")
    if raw_mode in ("ENABLED", " Enabled "):
        assert config.gate_mode == "enabled"
    else:
        assert config.gate_mode == "advisory"


def test_prescreen_config_thresholds_merge_over_defaults() -> None:
    """Run-specific overrides extend, and never erase, the documented defaults."""
    config = PrescreenConfig.from_config({
        "prescreen": {"gate_thresholds": {"yavg_min": 20.0}, "weights": {"face": 0.9}}
    })
    thresholds = config.telemetry_gate_thresholds()
    assert thresholds["yavg_min"] == 20.0
    assert thresholds["unif_max"] == 0.92
    assert config.telemetry_weights()["face"] == 0.9
    assert config.telemetry_weights()["lean_speed"] == 0.35
    assert config.flat_thresholds()["blur_min"] == config.blur_min


def test_prescreen_config_ignores_non_mapping_section() -> None:
    """A malformed ``prescreen`` section degrades to defaults."""
    assert PrescreenConfig.from_config({"prescreen": "nonsense"}) == PrescreenConfig()


# --- record composition ------------------------------------------------------


def test_build_prescreen_record_captures_audit_fields() -> None:
    """The record keeps thresholds, gates, score breakdown, and the action."""
    record = build_prescreen_record(
        clip_id="clip-1",
        source_file="clip-1.mp4",
        candidate_index=2,
        start=10.0,
        end=14.0,
        config=PrescreenConfig(gate_mode="advisory"),
        window={
            "sample_count": 40,
            "yavg": 5.0,
            "unif": 0.5,
            "gyro_rms": 0.2,
            "gps_speed_ms": 2.0,
            "roll_deg": 12.0,
        },
        visual={"laplacian_sharpness": 40.0, "hsv_entropy": 0.4, "hue_changes_per_s": 0.1},
        hilight_tags=[12.0],
        hilight_candidates=[(10.0, 14.0)],
        telemetry_type="gpmf",
        bitrate_density=0.11,
        is_high_fps=True,
    )
    assert record.interval.start == 10.0
    assert record.interval.sample_count == 40
    assert record.action == "defer"
    assert record.gate_mode == "advisory"
    assert record.thresholds["yavg_min"] == 15.0
    assert record.hilight_candidates == [[10.0, 14.0]]
    assert record.telemetry_type == "gpmf"
    assert record.is_high_fps is True
    assert record.telemetry_score is not None
    assert {component.name for component in record.score_components} == {
        "lean_speed",
        "gyro_stability",
        "hue_change",
    }
    assert all(component.weight > 0 for component in record.score_components)
    assert record.gates and all(gate.reason for gate in record.gates)


def test_build_prescreen_record_falls_back_to_visual_exposure() -> None:
    """Without a sensor track the sampled frames fill ``yavg``/``unif`` and say so."""
    record = build_prescreen_record(
        clip_id="clip-2",
        source_file="clip-2.mp4",
        candidate_index=0,
        start=0.0,
        end=5.0,
        config=PrescreenConfig(),
        window={},
        visual={"mean_luma": 4.0, "luma_uniformity": 0.95, "laplacian_sharpness": 2.0},
    )
    assert record.yavg == 4.0
    assert record.unif == 0.95
    assert "sampled frames" in record.reason
    assert record.audio_peak_db is None
    assert record.vlm_call is False


def test_build_prescreen_record_prefers_telemetry_over_visual() -> None:
    """Sensor signals win when both sources are present."""
    record = build_prescreen_record(
        clip_id="clip-3",
        source_file="clip-3.mp4",
        candidate_index=0,
        start=0.0,
        end=5.0,
        window={"yavg": 90.0, "unif": 0.1},
        visual={"mean_luma": 4.0, "luma_uniformity": 0.95},
    )
    assert record.yavg == 90.0
    assert record.unif == 0.1
    assert "sampled frames" not in record.reason


def test_build_prescreen_record_validates_against_trace_schema() -> None:
    """Records serialize into the published ``prescreen_trace.json`` schema."""
    record = build_prescreen_record(
        clip_id="clip-4",
        source_file="clip-4.mp4",
        candidate_index=0,
        start=0.0,
        end=5.0,
        window={"yavg": 120.0, "unif": 0.2, "gyro_rms": 0.1},
        visual={"laplacian_sharpness": 900.0, "hsv_entropy": 0.7},
    )
    trace = PrescreenTraceOutput(total_clips=1, total_records=1, records=[record])
    payload = trace.model_dump()
    assert payload["version"] == 1
    assert payload["records"][0]["action"] == "keep"
    assert payload["records"][0]["interval"]["end"] == 5.0


# --- screen_clip -------------------------------------------------------------


def _patch_sampling(frames: list[bytes], peaks: list[tuple[float, float]]) -> tuple[MagicMock, ...]:
    """Patch the FFmpeg sampling helpers used by :func:`screen_clip`."""
    frames_mock = MagicMock(return_value=frames)
    audio_mock = MagicMock(return_value=peaks)
    return frames_mock, audio_mock


def test_screen_clip_is_disabled_by_default() -> None:
    """With the funnel off, no sampling happens and no window is filtered."""
    with (
        patch("utils.prescreen.extract_raw_frames") as frames_mock,
        patch("utils.prescreen.extract_audio_peak_levels") as audio_mock,
    ):
        result = screen_clip("clip.mp4", clip_id="c", duration=12.0)
    frames_mock.assert_not_called()
    audio_mock.assert_not_called()
    assert result.records == []
    assert result.action_at(3.0) == ("keep", None)


def test_screen_clip_samples_once_and_records_each_window() -> None:
    """One frame pass and one audio pass produce one record per candidate window."""
    frames = [make_checker() for _ in range(20)]
    frames_mock, audio_mock = _patch_sampling(frames, [(0.0, -45.0)])
    config = PrescreenConfig(enabled=True, gate_mode="advisory", sample_size=SIZE, sample_fps=1.0)
    with (
        patch("utils.prescreen.extract_raw_frames", frames_mock),
        patch("utils.prescreen.extract_audio_peak_levels", audio_mock),
    ):
        result = screen_clip(
            "clip.mp4",
            clip_id="c1",
            duration=20.0,
            config=config,
            hilight_tags=[10.0],
        )
    assert frames_mock.call_count == 1
    assert audio_mock.call_count == 1
    assert len(result.records) == 1
    record = result.records[0]
    assert (record.interval.start, record.interval.end) == (6.0, 14.0)
    assert result.duration == 20.0
    assert result.action_at(10.0)[1] is record


def test_screen_clip_rejects_when_enabled() -> None:
    """A lens cap (flat dark frames) rejects only in enabled mode."""
    dark = [make_frame(2, 2, 2) for _ in range(20)]
    frames_mock, audio_mock = _patch_sampling(dark, [])
    config = PrescreenConfig(enabled=True, gate_mode="enabled", sample_size=SIZE, sample_fps=1.0)
    with (
        patch("utils.prescreen.extract_raw_frames", frames_mock),
        patch("utils.prescreen.extract_audio_peak_levels", audio_mock),
    ):
        result = screen_clip("clip.mp4", clip_id="c2", duration=20.0, config=config)
    assert result.rejected_intervals == [(0.0, 20.0)]
    assert result.intervals_for("keep") == []
    assert result.action_at(5.0)[0] == "reject"


def test_screen_clip_advisory_keeps_coverage() -> None:
    """Advisory mode records the same decision but preserves the candidate."""
    dark = [make_frame(2, 2, 2) for _ in range(10)]
    frames_mock, audio_mock = _patch_sampling(dark, [])
    config = PrescreenConfig(enabled=True, gate_mode="advisory", sample_size=SIZE, sample_fps=1.0)
    with (
        patch("utils.prescreen.extract_raw_frames", frames_mock),
        patch("utils.prescreen.extract_audio_peak_levels", audio_mock),
    ):
        result = screen_clip("clip.mp4", clip_id="c3", duration=10.0, config=config)
    assert result.rejected_intervals == []
    assert result.deferred_intervals == [(0.0, 10.0)]
    assert result.action_at(5.0)[0] == "defer"


def test_screen_clip_honours_explicit_intervals() -> None:
    """Callers can screen specific windows instead of HiLight candidates."""
    frames_mock, audio_mock = _patch_sampling([make_checker() for _ in range(10)], [])
    config = PrescreenConfig(enabled=True, sample_size=SIZE, sample_fps=1.0)
    with (
        patch("utils.prescreen.extract_raw_frames", frames_mock),
        patch("utils.prescreen.extract_audio_peak_levels", audio_mock),
    ):
        result = screen_clip(
            "clip.mp4",
            clip_id="c4",
            duration=10.0,
            config=config,
            intervals=[(1.0, 3.0), (5.0, 8.0), (9.0, 9.0)],
        )
    assert [(record.interval.start, record.interval.end) for record in result.records] == [
        (1.0, 3.0),
        (5.0, 8.0),
    ]
    assert [record.candidate_index for record in result.records] == [0, 1]


def test_screen_clip_skips_audio_when_disabled() -> None:
    """``audio_enabled: false`` avoids the astats pass entirely."""
    frames_mock, audio_mock = _patch_sampling([make_checker()], [])
    config = PrescreenConfig(enabled=True, audio_enabled=False, sample_size=SIZE, sample_fps=1.0)
    with (
        patch("utils.prescreen.extract_raw_frames", frames_mock),
        patch("utils.prescreen.extract_audio_peak_levels", audio_mock),
    ):
        result = screen_clip("clip.mp4", clip_id="c5", duration=5.0, config=config)
    audio_mock.assert_not_called()
    assert result.audio_peak_count == 0


def test_screen_clip_skips_audio_when_clip_has_no_audio_track() -> None:
    """A clip known to carry no audio track avoids the astats pass entirely."""
    frames_mock, audio_mock = _patch_sampling([make_checker()], [])
    config = PrescreenConfig(enabled=True, sample_size=SIZE, sample_fps=1.0)
    with (
        patch("utils.prescreen.extract_raw_frames", frames_mock),
        patch("utils.prescreen.extract_audio_peak_levels", audio_mock),
    ):
        result = screen_clip(
            "clip.mp4", clip_id="c5b", duration=5.0, config=config, has_audio=False
        )
    audio_mock.assert_not_called()
    assert result.audio_peak_count == 0


def test_screen_clip_runs_audio_pass_when_clip_has_audio() -> None:
    """``has_audio`` defaults to true so existing callers keep probing audio."""
    frames_mock, audio_mock = _patch_sampling([make_checker()], [(0.0, -12.0)])
    config = PrescreenConfig(enabled=True, sample_size=SIZE, sample_fps=1.0)
    with (
        patch("utils.prescreen.extract_raw_frames", frames_mock),
        patch("utils.prescreen.extract_audio_peak_levels", audio_mock),
    ):
        result = screen_clip("clip.mp4", clip_id="c5c", duration=5.0, config=config, has_audio=True)
    audio_mock.assert_called_once()
    assert result.audio_peak_count == 1


def test_screen_clip_survives_unsampleable_video() -> None:
    """A clip FFmpeg cannot decode still yields a record with null metrics."""
    frames_mock, audio_mock = _patch_sampling([], [])
    config = PrescreenConfig(enabled=True, sample_size=SIZE, sample_fps=1.0)
    with (
        patch("utils.prescreen.extract_raw_frames", frames_mock),
        patch("utils.prescreen.extract_audio_peak_levels", audio_mock),
    ):
        result = screen_clip("clip.mp4", clip_id="c6", duration=6.0, config=config)
    assert len(result.records) == 1
    record = result.records[0]
    assert record.yavg is None
    assert record.laplacian_sharpness is None
    assert result.action_at(1.0)[0] == "keep"
