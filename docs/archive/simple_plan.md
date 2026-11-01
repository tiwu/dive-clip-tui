# Simplified MVP Plan: Local GoPro Quick-like Highlight Pipeline

> **Archived — historical reference only.**
>
> This plan records the earlier CLI-first MVP and its completed milestones. Use
> the current [product roadmap](../roadmap/roadmap.md) for active priorities.
>

**Owner:** Tech-savvy diver with dive + motorcycle footage  
**Goal:** Local, subscription-free replacement for the discontinued GoPro Quick desktop app. No cloud upload, no DaVinci Resolve required, CLI-first.

---

## User goal (simplified)

```
videos > find interesting bits > list out > storyline > video
```

Point at a folder of dive/motorcycle footage, let AI find the interesting bits, review them, and get back a watchable highlight video — all on your own machine.

---

## Guiding principles

- **Local-first:** Ollama + FFmpeg on your machine. No GoPro cloud, no subscription, no API keys required for the core flow.
- **CLI first:** Tech-savvy user is comfortable with terminal commands. A web UI can come later.
- **Video only, to start:** Dive and motorcycle footage is all video. Photos/EXIF can wait.
- **Resolve optional:** Your existing marker/EDL flow still works if you want to fine-tune in DaVinci Resolve, but it is no longer required.
- **Small steps:** Each phase ships something usable before the next begins.

---

## Phase 1: Liberate from DaVinci Resolve (Completed)

**Goal:** One command: point at a folder of videos → get a folder of MP4 highlight clips. No Resolve, no manual console steps.

| Step | What | Files involved | Status |
|------|------|----------------|--------|
| 1.1 | Folder ingestion: walk a directory, enumerate videos, extract duration/FPS/resolution/codec, emit a plain JSON manifest | `scripts/00_ingest_media.py` | Completed |
| 1.2 | Point the existing Ollama scanner at that manifest directly (bypass Resolve keyword export) | `scripts/02_scan_footage.py` | Completed |
| 1.3 | Render each detection window as a standalone MP4 clip via FFmpeg | `scripts/05_render_clips.py` | Completed |
| 1.4 | Single CLI entry point tying it together | `scripts/dive_clip.py` (`dive-clip scan`) | Completed |

**Definition of done:** `dive-clip scan --input <folder>` produces playable MP4 clips without opening DaVinci Resolve. (Shipped)

---

## Phase 2: Storyline → Video (Completed)

**Goal:** Automatically assemble the best clips into a watchable video with music and transitions.

| Step | What | Files involved | Status |
|------|------|----------------|--------|
| 2.1 | Simple concatenation of top-N clips ordered by quality score | `scripts/06_assemble_video.py` | Completed |
| 2.2 | Add a background music track with basic ducking | `scripts/07_create_aftermovie.py` | Completed |
| 2.3 | Cross-dissolve transitions between clips | `scripts/07_create_aftermovie.py` | Completed |
| 2.4 | Output one `aftermovie.mp4` + optional per-subject reels | `scripts/07_create_aftermovie.py` | Completed |

**Definition of done:** One command produces a highlight video from a folder of raw footage. (Shipped)

---

## Phase 2.5: TUI Review Interface (Completed)

**Goal:** An interactive terminal interface for browsing footage, reviewing AI
detections before rendering, and triggering pipeline steps — without remembering
CLI flags. Built with [Textual](https://textual.textualize.io/), a Rich-based
Python TUI framework.

| Step | What | Files involved | Status |
|------|------|----------------|--------|
| 2.5.1 | Add `textual` dependency and `dive-clip-tui` entry point | `pyproject.toml`, `requirements.txt` | Completed |
| 2.5.2 | Browse screen: point at a footage folder, load and view the manifest | `tui/screens/browse.py` | Completed |
| 2.5.3 | Review screen: list AI detections, filter by quality/subject, accept/reject | `tui/screens/review.py` | Completed |
| 2.5.4 | Pipeline screen: run ingest/scan/EDL/storyline steps with live log output | `tui/screens/pipeline.py` | Completed |
| 2.5.5 | Folder ingestion script that walks a directory and emits a JSON manifest | `scripts/00_ingest_media.py` | Completed |

**Definition of done:** `dive-clip-tui` launches an interactive TUI that can
browse a footage folder, view the manifest, review AI detections, and trigger
pipeline steps with live log output. (Shipped)

---

## Phase 3: Polish, Reliability & Advanced AI

- [x] **Run archives & self-contained snapshots:** Per-run directories with `<output>/latest` symlink (`scripts/dive_clip.py`, `utils/run_archive.py`)
- [x] **Pipeline resume:** Automatic or targeted `--resume` restoring settings and state (`utils/checkpoint.py`)
- [x] **Intra-clip progress checkpointing:** Periodic frame flush every N frames to resume long clip scans without restarting
- [x] **End-of-run processing statistics & Rich summary table:** Throughput (FPS), phase breakdown, and `run_stats.json` artifact
- [x] **3-stage domain-aware scanning:** Pass 0 video domain classification, Pass 1 domain coarse screen, Pass 2 subject-routed fine scan (underwater, motorcycle, generic, custom)
- [x] **Config presets:** Aftermovie, social, and subject config presets
- [x] **End-to-end CLI highlight assembly:** `dive-clip scan --assemble` flag plus first-class `dive-clip assemble` and `dive-clip aftermovie` subcommands with automatic latest-run resolution (`scripts/dive_clip.py`)
- [ ] **Social aspect ratios:** 9:16 vertical render for Reels/TikTok / Shorts
- [ ] **Face/subject-aware reframing:** Smart cropping for portrait output
- [ ] **TUI review-to-render integration:** Wire accepted detection review directly into clip rendering from the UI
- [ ] **Web UI:** Browser-based review interface (optional)

---

## Explicitly out of scope for now

- **Photos / EXIF support** — video is the priority
- **Platform publishing / social upload APIs** — keep rendering local
- **Docker containerization / cloud hosting** — local-first workstation focus
