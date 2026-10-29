"""Timecode math utilities for precise frame/FPS to SMPTE conversion.

Handles exact timecode calculation for various frame rates including
29.97 drop-frame, 30.0 non-drop-frame, 25.0 PAL, 24.0 film, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass
class Timecode:
    """SMPTE timecode representation."""

    hours: int
    minutes: int
    seconds: int
    frames: int
    fps: Fraction
    drop_frame: bool = False

    def __str__(self) -> str:
        return f"{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d}:{self.frames:02d}"

    @classmethod
    def from_seconds(
        cls, seconds: float, fps: float | Fraction, drop_frame: bool = False
    ) -> Timecode:
        """Convert seconds (float) to SMPTE timecode."""
        fps_frac = Fraction(fps) if isinstance(fps, float) else fps
        total_frames = round(seconds * float(fps_frac))
        return cls.from_frames(total_frames, fps_frac, drop_frame)

    @classmethod
    def from_frames(
        cls, total_frames: int, fps: float | Fraction, drop_frame: bool = False
    ) -> Timecode:
        """Convert absolute frame count to SMPTE timecode."""
        fps_frac = Fraction(fps) if isinstance(fps, float) else fps
        fps_float = float(fps_frac)

        if drop_frame and fps_float in (29.97, 59.94):
            return cls._from_frames_dropframe(total_frames, fps_float)

        total_seconds = total_frames / fps_float
        hours = int(total_seconds // 3600)
        remaining = total_seconds - hours * 3600
        minutes = int(remaining // 60)
        remaining = remaining - minutes * 60
        seconds = int(remaining)
        frames = round((remaining - seconds) * fps_float)

        return cls(hours, minutes, seconds, frames, fps_frac, False)

    @classmethod
    def _from_frames_dropframe(
        cls, total_frames: int, fps_float: float
    ) -> Timecode:
        """Convert actual frame count to 29.97/59.94 drop-frame timecode.

        In drop frame, frame labels :00 and :01 are skipped at the start
        of each minute except minutes 0, 10, 20, 30, 40, 50.
        This means 108 labels are dropped per hour of actual time.
        """
        frames_per_hour_actual = fps_float * 3600
        hours = int(total_frames // frames_per_hour_actual)
        remainder = total_frames - int(hours * frames_per_hour_actual)

        # Add dropped labels for completed hours (108 per hour)
        display_frames = remainder + hours * 108

        # Account for drops within the current hour
        frames_per_minute_actual = fps_float * 60
        minutes_actual = int(display_frames // frames_per_minute_actual)
        remainder_in_minute = display_frames - int(minutes_actual * frames_per_minute_actual)

        decade_minutes = {0, 10, 20, 30, 40, 50}
        dropped_in_hour = sum(
            2 for m in range(minutes_actual) if m % 10 not in decade_minutes
        )
        display_frames += dropped_in_hour

        minutes = minutes_actual % 60
        remaining_seconds = remainder_in_minute / fps_float
        seconds = int(remaining_seconds)
        frames = round((remaining_seconds - seconds) * fps_float)

        return cls(hours, minutes, seconds, frames, Fraction(fps_float), True)

    def to_seconds(self) -> float:
        """Convert timecode back to seconds (float)."""
        fps_float = float(self.fps)
        total_frames = (
            self.hours * 3600 * fps_float
            + self.minutes * 60 * fps_float
            + self.seconds * fps_float
            + self.frames
        )
        if self.drop_frame:
            total_frames -= self._count_dropped_frames()
        return total_frames / fps_float

    def _count_dropped_frames(self) -> int:
        """Count dropped frame labels up to this timecode (drop-frame only)."""
        total_minutes = self.hours * 60 + self.minutes
        decade_minutes = {0, 10, 20, 30, 40, 50}
        dropped = 0
        for m in range(total_minutes):
            if m % 10 not in decade_minutes:
                dropped += 2
        return dropped
