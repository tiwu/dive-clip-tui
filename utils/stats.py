"""Pipeline run statistics models, formatting, and persistence.

Collects and summarizes execution metrics across ingest, coarse scan (phase 1),
fine scan (phase 2), and highlight rendering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rich import box
from rich.panel import Panel
from rich.table import Table


def parse_duration(value: Any, fps: float = 25.0) -> float:
    """Parse a duration value (float, int, or SMPTE/colon string) into seconds.

    Examples:
        12.5 -> 12.5
        "12.5" -> 12.5
        "01:30" -> 90.0
        "00:01:30:10" -> 90.4
    """
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return max(0.0, float(value))

    s = str(value).strip()
    if not s:
        return 0.0

    # Try plain float string
    try:
        return max(0.0, float(s))
    except ValueError:
        pass

    # Try colon-separated timecode (HH:MM:SS or HH:MM:SS:FF)
    parts = s.split(":")
    try:
        if len(parts) == 2:
            minutes, secs = float(parts[0]), float(parts[1])
            return max(0.0, minutes * 60 + secs)
        if len(parts) == 3:
            hours, minutes, secs = float(parts[0]), float(parts[1]), float(parts[2])
            return max(0.0, hours * 3600 + minutes * 60 + secs)
        if len(parts) == 4:
            hours, minutes, secs, frames = (
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            )
            fps_divisor = max(1.0, fps)
            return max(0.0, hours * 3600 + minutes * 60 + secs + frames / fps_divisor)
    except ValueError:
        pass

    return 0.0


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string.

    Examples:
        0.0 -> "0.0s"
        12.4 -> "12.4s"
        65.0 -> "1m 05s"
        3665.0 -> "1h 01m 05s"
    """
    if seconds < 0:
        return "0.0s"
    if seconds < 60:
        return f"{seconds:.1f}s"

    total_secs = int(seconds)
    hours, remainder = divmod(total_secs, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


class Phase0Stats(BaseModel):
    """Statistics for Phase 0 clip domain classification."""

    clips_classified: int = 0
    frames_analyzed: int = 0
    ollama_calls: int = 0
    ollama_failures: int = 0
    elapsed_seconds: float = 0.0
    domain_counts: dict[str, int] = Field(default_factory=dict)


class Phase1Stats(BaseModel):
    """Statistics for Phase 1 coarse scanning."""

    candidate_windows: int = 0
    candidate_duration: float = 0.0
    frames_extracted: int = 0
    frames_analyzed: int = 0
    detections_detected: int = 0
    ollama_calls: int = 0
    ollama_failures: int = 0
    elapsed_seconds: float = 0.0


class Phase2Stats(BaseModel):
    """Statistics for Phase 2 fine scanning."""

    windows_scanned: int = 0
    frames_extracted: int = 0
    frames_analyzed: int = 0
    detections_found: int = 0
    detections_duration: float = 0.0
    detections_dedup: int = 0
    detections_dedup_duration: float = 0.0
    ollama_calls: int = 0
    ollama_failures: int = 0
    elapsed_seconds: float = 0.0


class RenderStats(BaseModel):
    """Statistics for clip rendering."""

    clips_total: int = 0
    clips_rendered: int = 0
    clips_skipped: int = 0
    clips_failed: int = 0
    total_duration: float = 0.0
    elapsed_seconds: float = 0.0


class PipelineStats(BaseModel):
    """Cumulative run statistics across the entire pipeline."""

    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    total_elapsed_seconds: float = 0.0
    video_count: int = 0
    total_video_duration: float = 0.0
    phase0: Phase0Stats = Field(default_factory=Phase0Stats)
    phase1: Phase1Stats = Field(default_factory=Phase1Stats)
    phase2: Phase2Stats = Field(default_factory=Phase2Stats)
    render: RenderStats = Field(default_factory=RenderStats)
    step_durations: dict[str, float] = Field(default_factory=dict)

    @property
    def speed_ratio(self) -> float:
        """Ratio of source footage duration to execution elapsed time."""
        if self.total_elapsed_seconds <= 0 or self.total_video_duration <= 0:
            return 0.0
        return self.total_video_duration / self.total_elapsed_seconds

    @property
    def fps_throughput(self) -> float:
        """Total frames analyzed per second of elapsed execution."""
        total_frames = (
            self.phase0.frames_analyzed + self.phase1.frames_analyzed + self.phase2.frames_analyzed
        )
        if self.total_elapsed_seconds <= 0 or total_frames <= 0:
            return 0.0
        return total_frames / self.total_elapsed_seconds

    def to_dict(self) -> dict[str, Any]:
        """Export dictionary including computed metrics."""
        data = self.model_dump()
        data["speed_ratio"] = round(self.speed_ratio, 2)
        data["fps_throughput"] = round(self.fps_throughput, 2)
        return data


def save_stats_json(stats: PipelineStats, path: Path) -> None:
    """Save pipeline statistics to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2, ensure_ascii=False)


def load_stats_json(path: Path) -> PipelineStats:
    """Load pipeline statistics from a JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return PipelineStats.model_validate(data)


def format_stats_table(stats: PipelineStats, title: str = "Pipeline Run Summary") -> Panel:
    """Build a Rich Panel containing structured statistics tables."""
    table = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0, 1))
    table.add_column("Category", style="bold cyan", width=22)
    table.add_column("Metrics", style="white")

    # Source footage
    table.add_row(
        "📹 Source Footage",
        f"[bold]{stats.video_count}[/] videos  •  "
        f"Total duration: [bold]{format_duration(stats.total_video_duration)}[/]",
    )
    table.add_row("", "")

    # Phase 0 (if any clips classified)
    if stats.phase0.clips_classified > 0 or stats.phase0.frames_analyzed > 0:
        p0 = stats.phase0
        dom_parts = [f"{d}: {c}" for d, c in p0.domain_counts.items()]
        dom_str = f" ({', '.join(dom_parts)})" if dom_parts else ""
        p0_details = (
            f"[bold]{p0.clips_classified}[/] clips classified{dom_str}\n"
            f"[dim]{p0.frames_analyzed} frames analyzed  •  "
            f"{p0.ollama_calls} Ollama calls ({p0.ollama_failures} failed)  •  "
            f"Elapsed: {format_duration(p0.elapsed_seconds)}[/dim]"
        )
        table.add_row("🏷️ Phase 0 (Classify)", p0_details)
        table.add_row("", "")

    # Phase 1
    p1 = stats.phase1
    p1_details = (
        f"[bold]{p1.candidate_windows}[/] candidate windows "
        f"([bold]{format_duration(p1.candidate_duration)}[/])\n"
        f"[dim]{p1.frames_analyzed} frames analyzed  •  "
        f"{p1.ollama_calls} Ollama calls ({p1.ollama_failures} failed)  •  "
        f"Elapsed: {format_duration(p1.elapsed_seconds)}[/dim]"
    )
    table.add_row("🔍 Phase 1 (Coarse)", p1_details)
    table.add_row("", "")

    # Phase 2
    p2 = stats.phase2
    p2_details = (
        f"[bold]{p2.detections_dedup}[/] refined detections "
        f"([bold]{format_duration(p2.detections_dedup_duration)}[/]) "
        f"[dim]from {p2.detections_found} raw hits across {p2.windows_scanned} windows[/dim]\n"
        f"[dim]{p2.frames_analyzed} frames analyzed  •  "
        f"{p2.ollama_calls} Ollama calls ({p2.ollama_failures} failed)  •  "
        f"Elapsed: {format_duration(p2.elapsed_seconds)}[/dim]"
    )
    table.add_row("🎯 Phase 2 (Fine)", p2_details)
    table.add_row("", "")

    # Render
    r = stats.render
    r_details = (
        f"[bold]{r.clips_rendered}[/] clips rendered  •  "
        f"[bold]{format_duration(r.total_duration)}[/] highlight duration\n"
        f"[dim]{r.clips_skipped} skipped (already rendered)  •  "
        f"{r.clips_failed} failed  •  "
        f"Elapsed: {format_duration(r.elapsed_seconds)}[/dim]"
    )
    table.add_row("🎬 Highlights", r_details)
    table.add_row("", "")

    # Performance & Timing
    speed_text = f"{stats.speed_ratio:.2f}x realtime" if stats.speed_ratio > 0 else "n/a"
    fps_text = f"{stats.fps_throughput:.1f} fps" if stats.fps_throughput > 0 else "n/a"

    step_parts = [
        f"{step}: {format_duration(dur)}" for step, dur in stats.step_durations.items() if dur > 0
    ]
    step_str = f" ({', '.join(step_parts)})" if step_parts else ""

    perf_details = (
        f"Total time: [bold green]{format_duration(stats.total_elapsed_seconds)}[/]{step_str}\n"
        f"Throughput: [bold yellow]{speed_text}[/]  •  [bold yellow]{fps_text}[/]"
    )
    table.add_row("⏱️ Performance", perf_details)

    header_title = f"[bold green]{title}[/bold green]"
    if stats.run_id:
        header_title += f" [dim]({stats.run_id})[/dim]"

    return Panel(table, title=header_title, border_style="cyan")


def format_stats_log(stats: PipelineStats) -> str:
    """Format pipeline statistics as a plain-text multi-line string for logging."""
    speed_text = f"{stats.speed_ratio:.2f}x realtime" if stats.speed_ratio > 0 else "n/a"
    fps_text = f"{stats.fps_throughput:.1f} fps" if stats.fps_throughput > 0 else "n/a"

    lines = [
        "=== Run Statistics Summary ===",
        f"Run ID: {stats.run_id or 'n/a'}",
        (
            f"Source footage: {stats.video_count} videos, "
            f"duration {format_duration(stats.total_video_duration)}"
        ),
    ]

    if stats.phase0.clips_classified > 0 or stats.phase0.frames_analyzed > 0:
        p0 = stats.phase0
        dom_parts = [f"{d}: {c}" for d, c in p0.domain_counts.items()]
        dom_str = f" ({', '.join(dom_parts)})" if dom_parts else ""
        lines.append(
            f"Phase 0 (Classify): {p0.clips_classified} clips classified{dom_str}, "
            f"{p0.frames_analyzed} frames, "
            f"{p0.ollama_calls} calls ({p0.ollama_failures} failed), "
            f"elapsed {format_duration(p0.elapsed_seconds)}"
        )

    lines.extend([
        (
            f"Phase 1 (Coarse): {stats.phase1.candidate_windows} windows, "
            f"duration {format_duration(stats.phase1.candidate_duration)}, "
            f"{stats.phase1.frames_analyzed} frames, "
            f"{stats.phase1.ollama_calls} calls ({stats.phase1.ollama_failures} failed), "
            f"elapsed {format_duration(stats.phase1.elapsed_seconds)}"
        ),
        (
            f"Phase 2 (Fine): {stats.phase2.detections_dedup} detections, "
            f"duration {format_duration(stats.phase2.detections_dedup_duration)} "
            f"(from {stats.phase2.detections_found} raw hits), "
            f"{stats.phase2.frames_analyzed} frames, "
            f"{stats.phase2.ollama_calls} calls ({stats.phase2.ollama_failures} failed), "
            f"elapsed {format_duration(stats.phase2.elapsed_seconds)}"
        ),
        (
            f"Highlights Rendered: {stats.render.clips_rendered} clips rendered, "
            f"{stats.render.clips_skipped} skipped, {stats.render.clips_failed} failed, "
            f"duration {format_duration(stats.render.total_duration)}, "
            f"elapsed {format_duration(stats.render.elapsed_seconds)}"
        ),
        (
            f"Total time: {format_duration(stats.total_elapsed_seconds)}, "
            f"Speed: {speed_text}, Throughput: {fps_text}"
        ),
    ])

    if stats.step_durations:
        steps_str = ", ".join(f"{k}: {format_duration(v)}" for k, v in stats.step_durations.items())
        lines.append(f"Step durations: {steps_str}")

    return "\n".join(lines)
