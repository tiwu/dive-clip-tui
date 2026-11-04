# AGENTS.md

This file helps AI agents (and humans) navigate, understand, and contribute to the
**dive-clip-ai** repository. It also defines the team's feature-workflow convention.

---

## 🌊 Project overview

**Dive Clip** — local-first AI curation for dive and motorcycle footage, with a
CLI-first pipeline and optional DaVinci Resolve marker export. The existing
Textual TUI is an unmaintained prototype; future UI work starts from scratch.

The active direction is in [`docs/roadmap/roadmap.md`](docs/roadmap/roadmap.md).
The core flow needs no Resolve: point at a folder of videos → AI finds the
interesting bits → get back a watchable highlight video.

**Key docs**
- `README.md` — full setup, configuration, and execution guide
- `CHANGELOG.md` — unreleased changes and release history
- `docs/roadmap/roadmap.md` — active product roadmap
- `docs/archive/simple_plan.md` — archived CLI-first MVP plan
- `docs/archive/original_roadmap.md` — archived original product roadmap
- `docs/plans/` - folder for architecture plans
- `docs/plans/README.md` — required issue, plan, PR, changelog, and release workflow
- `config.yaml` — paths, thresholds, Ollama endpoints, API keys
- `docs/archive/projectplan_technische_specificatie.md` — technical specification (archived)

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
│   ├── config.py             # Config validation and loading
│   ├── ffmpeg_streamer.py    # Frame sampling helpers (/dev/shm & buffers)
│   ├── logging_setup.py      # Shared stderr + per-run logfile configuration
│   ├── models.py             # Pydantic data models (manifest, detections)
│   ├── ollama_client.py      # Shared Ollama requests and metrics
│   ├── prescreen.py          # Pre-VLM screening funnel (gates, metrics, trace)
│   ├── prompt_templates.py   # Domain prompts and subject-routed templates
│   ├── run_archive.py        # Run directory snapshots and latest symlink
│   ├── selection.py          # Maximal Marginal Relevance (MMR) highlight selection
│   ├── stats.py              # Scan throughput and pipeline metrics
│   ├── telemetry.py          # GoPro GPMF / DJI / Insta360 sensor decoding + scoring
│   ├── timecode_math.py      # Frame/FPS → SMPTE timecode conversion
│   └── version.py            # Dynamic package version helper
├── tui/                      # Legacy Textual UI prototype; future UI is a rebuild
│   ├── app.py                # Main TUI application
│   └── screens/              # browse.py, pipeline.py, review.py
├── tests/                    # Pytest tests (+ conftest.py fixtures)
├── docs/
|   └── roadmap/              # active product roadmap
|   └── archive/              # superseded roadmaps and historical documents
|   └── plans/                # architecture plans
└── outputs/                  # Generated outputs (manifest, detections, EDL, etc.)
```

**Entry points** (defined in `pyproject.toml` `[project.scripts]`):
- `dive-clip scan --input <folder> --output <folder>` — full pipeline (ingest → scan → render)
- `scan-footage`, `generate-edl`, `plan-storyline`, `ingest-media`, `render-clips`, `assemble-clips`, `create-aftermovie`, `dive-clip-tui`, `dive-clip`

**Python version:** `>=3.14`. Dependency manager: **uv** (`uv sync`).

---

## 🚦 How to run

```bash
# Install dependencies
uv sync

# Run the full pipeline
uv run scripts/dive_clip.py scan --input ./footage --output ./highlights --config config.yaml

# Or use individual scripts
uv run scripts/00_ingest_media.py --input ./footage --output outputs/manifest.json
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
in `docs/plans/README.md`:
1. **Stage 1 (Plan)**: Once requirements are aligned, if no GitHub issue exists yet, create one with `gh issue create --label "agent:plan-review"`, write `docs/plans/PLAN-<id>-<slug>.md` (including the proposed SemVer impact and draft changelog entry), embed the plan in the issue body, and stop to wait for user approval before writing implementation code.
2. **Stage 2 (Implement & PR)**: After approval, implement on `feature/issue-<id>-<slug>`, add a `CHANGELOG.md` entry under `[Unreleased]`, and open a linked PR (`Closes #<id>`) with one SemVer label (unlabeled defaults to minor). CI enforces the tests, lint, formatting, and changelog requirements.

When using the GitHub CLI:
- Pass `--json title,body,labels,comments` to `gh issue view` (or `gh pr view`) to avoid the deprecated GraphQL `projectCards` failure.
- When creating or editing PR descriptions with Markdown, write the body to a file and pass `--body-file` to prevent shell command substitution and redirection syntax errors.
- If `gh pr edit` fails due to the GraphQL `projectCards` deprecation, update the PR via the GitHub REST API: `gh api repos/:owner/:repo/pulls/:number -X PATCH -F body=@<file>`.
- Do not use LaTeX math (`$...$` or `$$...$$`) for multi-character identifiers or variables containing underscores (e.g. `\text{var_name}`) in plans, issue bodies, or PR descriptions. GitHub's KaTeX renderer fails on underscores with `'_' allowed only in math mode`. Use fenced code blocks (`python`) or backtick code spans (`` `var_name` ``) instead.

---

## 🧪 Conventions

- **Tests:** `pytest`, with shared fixtures in `tests/conftest.py`
  (`tmp_output_dir`, `sample_detections`, `sample_detections_file`, `sample_manifest`).
- **Changelog and releases:** use Keep a Changelog categories and follow the PR
  label/bump policy in `docs/plans/README.md`; merged PRs are released automatically.
- **Linting/format:** `ruff` (config in `pyproject.toml` and `ruff.toml`). Line length
  `100`, target `py314`. `E501` is ignored (line length handled by formatter).
- **GitHub Actions:** use recent action versions to avoid deprecated Node.js runner
  warnings. When updating workflows, check external sources (GitHub API / releases)
  to validate action versions, as internal knowledge may be outdated. Do not use
  major-version-only tags for actions that do not publish them (e.g. use
  `astral-sh/setup-uv@v10.3.3`, not `v10`).
- **Data models:** Pydantic v2 in `utils/models.py` — keep manifest/detection JSON
  strictly validated.
- **Console scripts:** registered in `pyproject.toml` `[project.scripts]`.
- **Checkpointing:** `utils/checkpoint.py` (`PipelineState`) for resume support in
  long-running batches.
- **Ollama JSON mode:** all Ollama API calls use `"format": "json"` so Qwen2.5-VL
  returns pure JSON.
- **Ollama requests:** use `utils/ollama_client.py` for shared request handling
  and metrics rather than calling the API directly from scripts.
- **Telemetry:** `utils/telemetry.py` decodes GoPro GPMF (`gpmd`) tracks; DJI
  (`djmd`) and Insta360 (`insta`) tracks are detected but not decoded. Telemetry
  streams must be dynamically detected across camera models by codec tag / handler
  name rather than assuming fixed track indices (e.g. never hardcode `0:d:0`, which
  is often `tmcd` timecode). Payloads in GPMF must be stepped by 32-bit aligned length
  `((payload_length + 3) & ~3)`. A missing sensor must never act as a penalty — drop
  the signal and renormalize weights. Telemetry specifications, coordinate axes, and
  conversion formulas must be permanently cited in `README.md` and module docstrings.
- **Pre-VLM screening:** `utils/prescreen.py` runs before Pass 1 and is off by
  default (`prescreen.enabled`). Keep image math dependency-free (pure Python over
  ffmpeg `rgb24` bytes; no NumPy/Pillow). Screening decisions are recorded in
  `prescreen_trace.json`, one record per candidate window, never per raw sample.
  `gate_mode: "advisory"` must only annotate; only `"enabled"` may drop windows.

---

## 💬 Ask before acting

If anything in this file or the repo is unclear, or if you want to propose a change to
the workflow, ask the user before making changes. Do not assume issue numbers, branch
names, or PR details — confirm them.
