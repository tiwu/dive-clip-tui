# Simplified MVP Plan: Local GoPro Quick-like Highlight Pipeline

**Status:** Active (supersedes `docs/roadmap/roadmap.md`, which is kept for reference)  
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

## Phase 1: Liberate from DaVinci Resolve (the blocker)

**Goal:** One command: point at a folder of videos → get a folder of MP4 highlight clips. No Resolve, no manual console steps.

| Step | What | Files involved |
|------|------|----------------|
| 1.1 | Folder ingestion: walk a directory, enumerate videos, extract duration/FPS/resolution/codec, emit a plain JSON manifest | New `scripts/00_ingest_media.py` |
| 1.2 | Point the existing Ollama scanner at that manifest directly (bypass Resolve keyword export) | Modify `scripts/02_scan_footage.py` |
| 1.3 | Render each detection window as a standalone MP4 clip via FFmpeg | New `scripts/05_render_clips.py` |
| 1.4 | Single CLI entry point tying it together | `dive-clip scan --input ./footage --output ./highlights` |

**Definition of done:** `dive-clip scan --input <folder>` produces playable MP4 clips without opening DaVinci Resolve.

---

## Phase 2: Storyline → Video (the "Quick" magic)

**Goal:** Automatically assemble the best clips into a watchable video with music and transitions.

| Step | What |
|------|------|
| 2.1 | Simple concatenation of top-N clips ordered by quality score |
| 2.2 | Add a background music track with basic ducking |
| 2.3 | Cross-dissolve transitions between clips |
| 2.4 | Output one `aftermovie.mp4` + optional per-subject reels (turtle, shark, motorcycle, etc.) |

**Definition of done:** One command produces a 3–5 minute highlight video from a folder of raw footage.

---

## Phase 2.5: TUI Review Interface (interactive terminal UI)

**Goal:** An interactive terminal interface for browsing footage, reviewing AI
detections before rendering, and triggering pipeline steps — without remembering
CLI flags. Built with [Textual](https://textual.textualize.io/), a Rich-based
Python TUI framework.

| Step | What | Files involved |
|------|------|----------------|
| 2.5.1 | Add `textual` dependency and `dive-clip-tui` entry point | `pyproject.toml`, `requirements.txt` |
| 2.5.2 | Browse screen: point at a footage folder, load and view the manifest | `tui/screens/browse.py` |
| 2.5.3 | Review screen: list AI detections, filter by quality/subject, accept/reject | `tui/screens/review.py` |
| 2.5.4 | Pipeline screen: run ingest/scan/EDL/storyline steps with live log output | `tui/screens/pipeline.py` |
| 2.5.5 | Folder ingestion script that walks a directory and emits a JSON manifest | New `scripts/00_ingest_media.py` |

**Definition of done:** `dive-clip-tui` launches an interactive TUI that can
browse a footage folder, view the manifest, review AI detections, and trigger
pipeline steps with live log output.

---

## Phase 3: Polish & UX (later, optional)

- Social aspect ratios (9:16 vertical for Reels/TikTok)
- Face/subject-aware reframing
- Web UI to review/trim AI suggestions before render
- Checkpointing/resume for long batches
- Config presets (aftermovie / social / subject)

---

## Explicitly out of scope for now

- **Photos / EXIF support** — video is the priority
- **Cloud LLM storyline** — keep it local; optional Claude/Gemini later
- **Web UI, Docker, platform publishing, batch project management** — Phase 3+

---

## Recommended first action

Start **Phase 1, Step 1.1**: add `scripts/00_ingest_media.py` so you can point at a folder and get a JSON manifest of videos. Then wire it into the existing `02_scan_footage.py` scanner.

Ship Phase 1, measure "time to first clip," then decide whether Phase 2 (assembly) is worth building.
