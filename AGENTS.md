# AGENTS.md

This file helps AI agents (and humans) navigate, understand, and contribute to the
**dive-clip-ai** repository. It also defines the team's feature-workflow convention.

---

## 🌊 Project overview

**Dive Clip TUI** — local-first AI curation for dive and motorcycle footage, with a
Textual terminal UI and optional DaVinci Resolve marker export.

A simplified MVP (`docs/roadmap/simple_plan.md`) is in progress to make the pipeline
CLI-first and local-first (no Resolve required for the core flow): point at a folder of
videos → AI finds the interesting bits → get back a watchable highlight video.

**Key docs**
- `README.md` — full setup, configuration, and execution guide
- `CHANGELOG.md` — unreleased changes and release history
- `docs/roadmap/simple_plan.md` — active simplified MVP plan (supersedes `roadmap.md`)
- `docs/roadmap/roadmap.md` — kept for reference
- `docs/plans/` - folder for architecture plans
- `docs/plans/README.md` — required issue, plan, PR, changelog, and release workflow
- `config.yaml` — paths, thresholds, Ollama endpoints, API keys
- `projectplan_technische_specificatie.md` — technical specification

---

## 🗂️ Repository structure

```
├── README.md                 # Full setup, configuration, and execution guide
├── CHANGELOG.md              # Keep a Changelog release history
├── config.yaml               # Paths, thresholds, Ollama endpoints, API keys
├── config.aftermovie.yaml    # Aftermovie config preset (untracked)
├── config.social.yaml        # Social config preset (untracked)
├── config.subject.yaml       # Subject config preset (untracked)
├── pyproject.toml            # Project metadata, uv config, console scripts
├── requirements.txt          # Python dependencies
├── uv.lock                   # Locked dependencies
├── ruff.toml                 # Ruff linting configuration
├── AGENTS.md                 # This file
├── .github/workflows/        # Pull-request CI and main-branch release automation
├── scripts/                  # Pipeline scripts (numbered in run order)
│   ├── 00_ingest_media.py    # Folder ingestion → JSON manifest
│   ├── 01_extract_manifest.py# DaVinci Resolve metadata export
│   ├── 02_scan_footage.py    # Worker pipeline (FFmpeg + Ollama coarse-to-fine)
│   ├── 03_generate_edl.py    # AI detections → CMX 3600 EDL & CSV
│   ├── 04_plan_storyline.py  # Cloud API story architect (Claude / Gemini)
│   ├── 05_render_clips.py    # Render each detection window as MP4
│   ├── 06_assemble_video.py  # Assemble clips into a highlight video
│   ├── 07_create_aftermovie.py# Create aftermovie with music/transitions
│   ├── release.py            # Changelog, SemVer, and release helpers
│   └── dive_clip.py          # Single CLI entry point (ingest → scan → render)
├── utils/                    # Shared helpers
│   ├── checkpoint.py         # Pipeline checkpoint / resume state
│   ├── ffmpeg_streamer.py    # Frame sampling helpers (/dev/shm & buffers)
│   ├── models.py             # Pydantic data models (manifest, detections)
│   ├── ollama_client.py      # Shared Ollama requests and metrics
│   ├── prompt_templates.py   # Dynamic prompts per keyword category
│   └── timecode_math.py      # Frame/FPS → SMPTE timecode conversion
├── tui/                      # Textual-based terminal UI
│   ├── app.py                # Main TUI application
│   └── screens/              # browse.py, pipeline.py, review.py
├── tests/                    # Pytest tests (+ conftest.py fixtures)
├── docs/
|   └── roadmap/              # roadmap.md, simple_plan.md
|   └── plans/                # architecture plans
└── outputs/                  # Generated outputs (manifest, detections, EDL, etc.)
```

**Entry points** (defined in `pyproject.toml` `[project.scripts]`):
- `dive-clip scan --input <folder> --output <folder>` — full pipeline (ingest → scan → render)
- `scan-footage`, `generate-edl`, `plan-storyline`, `ingest-media`, `render-clips`, `assemble-clips`, `create-aftermovie`, `dive-clip-tui`, `dive-clip`

**Python version:** `>=3.10`. Dependency manager: **uv** (`uv sync`).

---

## 🚦 How to run

```bash
# Install dependencies
uv sync

# Run the full pipeline
uv run scripts/dive_clip.py scan --input ./footage --output ./highlights --config config.yaml

# Or use individual scripts
uv run scripts/00_ingest_media.py --input ./footage --output outputs/manifest_raw.json
uv run scripts/02_scan_footage.py --config config.yaml
uv run scripts/03_generate_edl.py --config config.yaml

# Run the TUI
uv run tui/app.py   # or: dive-clip-tui

# Run tests
uv run python -m pytest

# Lint
uv run ruff check .
uv run ruff format --check .
```

Note: `scripts/` and `utils/` are known first-party packages for ruff/isort
(`known-first-party = ["scripts", "utils"]` in `pyproject.toml`/`ruff.toml`).
PR CI applies Ruff lint and formatting checks to changed Python files; full-repository
checks currently include existing unrelated violations.

---

## 🔁 Feature workflow (required convention)

When we discuss a feature and make a plan, follow the two-stage issue and PR process
in `docs/plans/README.md`. Include the proposed SemVer impact in the issue plan, then
add a `CHANGELOG.md` entry under `[Unreleased]` to each releasable PR. Set one SemVer
label to choose the bump; an unlabeled PR defaults to minor. CI enforces the tests,
lint, formatting, and changelog requirements.

---

## 🧪 Conventions

- **Tests:** `pytest`, with shared fixtures in `tests/conftest.py`
  (`tmp_output_dir`, `sample_detections`, `sample_detections_file`, `sample_manifest`).
- **Changelog and releases:** use Keep a Changelog categories and follow the PR
  label/bump policy in `docs/plans/README.md`; merged PRs are released automatically.
- **Linting/format:** `ruff` (config in `pyproject.toml` and `ruff.toml`). Line length
  `100`, target `py310`. `E501` is ignored (line length handled by formatter).
- **Data models:** Pydantic v2 in `utils/models.py` — keep manifest/detection JSON
  strictly validated.
- **Console scripts:** registered in `pyproject.toml` `[project.scripts]`.
- **Checkpointing:** `utils/checkpoint.py` (`PipelineState`) for resume support in
  long-running batches.
- **Ollama JSON mode:** all Ollama API calls use `"format": "json"` so Qwen2.5-VL
  returns pure JSON.
- **Ollama requests:** use `utils/ollama_client.py` for shared request handling
  and metrics rather than calling the API directly from scripts.

---

## 💬 Ask before acting

If anything in this file or the repo is unclear, or if you want to propose a change to
the workflow, ask the user before making changes. Do not assume issue numbers, branch
names, or PR details — confirm them.
