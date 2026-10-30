"""Tests for utils/timecode_math.py Timecode class."""

from __future__ import annotations

from fractions import Fraction

from utils.timecode_math import Timecode


def test_from_seconds_25fps() -> None:
    """Seconds to timecode at 25 fps."""
    tc = Timecode.from_seconds(37.5, 25.0)
    assert tc.hours == 0
    assert tc.minutes == 0
    assert tc.seconds == 37
    assert tc.frames == 13  # 0.5 * 25 = 12.5 frames, rounded


def test_from_seconds_30fps() -> None:
    """Seconds to timecode at 30 fps."""
    tc = Timecode.from_seconds(61.5, 30.0)
    assert tc.minutes == 1
    assert tc.seconds == 1
    assert tc.frames == 15


def test_from_seconds_24fps() -> None:
    """Seconds to timecode at 24 fps."""
    tc = Timecode.from_seconds(0.5, 24.0)
    assert tc.seconds == 0
    assert tc.frames == 12


def test_from_seconds_with_hours() -> None:
    """Large second values produce hours."""
    tc = Timecode.from_seconds(3661.0, 25.0)
    assert tc.hours == 1
    assert tc.minutes == 1
    assert tc.seconds == 1


def test_from_frames_25fps() -> None:
    """Frame count to timecode at 25 fps."""
    tc = Timecode.from_frames(100, 25.0)
    assert tc.minutes == 0
    assert tc.seconds == 4
    assert tc.frames == 0  # 100 / 25 = 4 seconds exactly


def test_from_frames_30fps() -> None:
    """Frame count to timecode at 30 fps."""
    tc = Timecode.from_frames(91, 30.0)
    assert tc.seconds == 3
    assert tc.frames == 1


def test_to_seconds_roundtrip() -> None:
    """to_seconds should invert from_seconds."""
    original = 123.456
    tc = Timecode.from_seconds(original, 30.0)
    back = tc.to_seconds()
    assert abs(back - original) < 0.05  # within rounding tolerance


def test_str_format() -> None:
    """String representation should be HH:MM:SS:FF."""
    tc = Timecode(hours=1, minutes=2, seconds=3, frames=4, fps=25.0)
    assert str(tc) == "01:02:03:04"


def test_str_format_single_digits() -> None:
    """String should zero-pad all components."""
    tc = Timecode(hours=0, minutes=0, seconds=5, frames=7, fps=30.0)
    assert str(tc) == "00:00:05:07"


def test_drop_frame_2997() -> None:
    """Drop-frame mode should be available for 29.97 fps."""
    tc = Timecode.from_seconds(10.0, 29.97, drop_frame=True)
    assert abs(float(tc.fps) - 29.97) < 0.001
    assert tc.drop_frame is True


def test_non_drop_frame_2997() -> None:
    """Non-drop-frame mode should be available for 29.97 fps."""
    tc = Timecode.from_seconds(10.0, 29.97, drop_frame=False)
    assert tc.drop_frame is False


def test_from_seconds_zero() -> None:
    """Zero seconds should produce all zeros."""
    tc = Timecode.from_seconds(0.0, 25.0)
    assert tc.hours == 0
    assert tc.minutes == 0
    assert tc.seconds == 0
    assert tc.frames == 0


def test_from_frames_zero() -> None:
    """Zero frames should produce all zeros."""
    tc = Timecode.from_frames(0, 25.0)
    assert tc.hours == 0
    assert tc.minutes == 0
    assert tc.seconds == 0
    assert tc.frames == 0


def test_fps_as_fraction() -> None:
    """FPS can be passed as a Fraction."""
    tc = Timecode.from_seconds(1.5, Fraction(25, 1))
    # 1.5 seconds * 25 fps = 37.5 frames -> 1 second + 12-13 frames
    assert tc.seconds == 1
    assert tc.frames in (12, 13)
