# 🌊 Dive Clip TUI

Local-first video curation for dive and motorcycle footage. Point the pipeline
at a folder of videos, use Ollama and FFmpeg to find interesting moments, and
render the detections as standalone MP4 clips. DaVinci Resolve is optional:
the existing EDL/CSV marker workflow is still available for editors who want it.

The project is CLI-first, with an experimental Textual terminal UI (TUI) for
browsing manifests and reviewing detections. The core scan flow does not require
cloud uploads, a subscription, API keys, or DaVinci Resolve.

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) and `ffprobe`
- [Ollama](https://ollama.com/) with `qwen2.5-vl:7b` available locally
- Enough memory and GPU resources for the selected vision model; a CUDA GPU
  with 12 GB VRAM is recommended for the configured model
- DaVinci Resolve only if you want to use the optional Resolve export/import flow

## Installation

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
ollama pull qwen2.5-vl:7b
```

Alternatively, install with pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Review `config.yaml` before scanning. In particular, set `general.output_dir`,
the Ollama model/host, and any `general.path_mapping` needed to locate media
across machines.

## Quick start: scan a footage folder

The single-command CLI chains folder ingestion, AI scanning, and clip rendering:

```bash
uv run dive-clip scan --input ./footage --output ./highlights
```

The scanner reads and writes its manifest and detections under
`general.output_dir` in the selected config. Set that directory to match the
command's `--output` directory (for example, `./highlights`) so the pipeline
steps use the same files. A custom config can be selected with `--config`.

The command supports checkpointing and can resume completed steps:

```bash
uv run dive-clip scan --input ./footage --output ./highlights --resume
```

To start over and clear the saved pipeline state:

```bash
uv run dive-clip scan --input ./footage --output ./highlights --reset
```

The pipeline writes a JSON manifest, refined detections, and rendered MP4 clips
to the configured output location.

## Terminal UI (experimental)

Launch the Textual interface with:

```bash
uv run dive-clip-tui
```

The current TUI opens on the browse screen. Enter a footage folder containing
`manifest_raw.json`, or use the manifest under `outputs/`, to view clip metadata.
From there, open the review screen to load `detections_refined.json` and filter
detections by quality score.

The TUI is still a prototype and does not yet provide the full review-to-render
workflow: accepting detections and rendering clips are placeholders, and the
pipeline screen is not yet reachable from the main interface. Use the CLI
workflow above to run the pipeline.

## Optional: DaVinci Resolve markers

The original Resolve workflow remains available for projects that use Resolve
metadata or want timeline markers:

1. In Resolve, open **Workspace → Console**, choose **Py3**, and run:
   ```python
   exec(open("scripts/01_extract_manifest.py").read())
   ```
2. Scan the exported manifest:
   ```bash
   uv run scripts/02_scan_footage.py --config config.yaml
   ```
3. Generate EDL and CSV marker files:
   ```bash
   uv run scripts/03_generate_edl.py --config config.yaml
   ```
4. Import `outputs/markers_import.edl` into the desired Resolve timeline using
   **Timelines → Import → Timeline Markers from EDL...**.

You can also create the EDL/CSV files from a folder-ingested manifest; Resolve is
not required to run the local scan and render flow.

## Repository structure

```text
scripts/
  00_ingest_media.py       # Find videos and write a metadata manifest
  01_extract_manifest.py   # Export clip metadata from DaVinci Resolve
  02_scan_footage.py       # Scan footage with Ollama
  03_generate_edl.py       # Create EDL and CSV timeline markers
  04_plan_storyline.py     # Optional story plan generation
  05_render_clips.py       # Render detection windows as MP4 clips
  06_assemble_video.py     # Assemble rendered clips
  07_create_aftermovie.py  # Create an aftermovie
  dive_clip.py             # Single-command ingest → scan → render CLI
tui/
  app.py                   # Textual TUI entry point
  screens/                 # Browse, review, and pipeline screens
utils/
  checkpoint.py            # Pipeline resume state
  ffmpeg_streamer.py       # Frame sampling and temporary storage
  models.py                # Pydantic manifest and detection models
  prompt_templates.py      # Prompts by subject/category
  timecode_math.py         # Frame-rate and SMPTE timecode helpers
outputs/                   # Generated manifests, detections, and exports
```

## Pipeline notes

- Folder ingestion uses `ffprobe` to record video duration, frame rate,
  resolution, and codec in `manifest_raw.json`.
- The scanner uses Ollama's JSON response mode and writes
  `detections_refined.json`. It prints request counts, failures, elapsed time,
  and available Ollama token/timing metrics after the scan.
- `dive-clip scan` stores progress in `<output>/.pipeline_state.json`; use
  `--resume` to skip completed steps or `--reset` to clear that state.
- Frame sampling uses `/dev/shm` when available and falls back to a temporary
  directory on systems such as macOS.
- The optional Resolve marker export supports frame-rate-aware timecodes,
  including 29.97 drop-frame and non-drop-frame formats.

For the current implementation plan and project scope, see
[`docs/roadmap/simple_plan.md`](docs/roadmap/simple_plan.md).

## Contributing changes and releases

Changes follow the two-stage GitHub issue and plan workflow in
[`docs/plans/README.md`](docs/plans/README.md): first review and approve the issue
plan, then implement it in a linked pull request. Each releasable PR must add a
user-facing entry under `[Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md). Select
one `semver:major`, `semver:minor`, or `semver:patch` label to set the bump; an
unlabeled PR defaults to minor. Use `semver:none` for changes that should not
trigger a release.

Pull requests run the test suite, Ruff lint and formatting checks on changed Python
files, and changelog validation. When PRs merge to `main`, the release workflow
groups unreleased changes, updates the package and lockfile versions, archives the
changelog entries, and creates a `vX.Y.Z` tag and GitHub Release. Repository
settings must allow Actions to write contents and let `github-actions[bot]` update
`main`.
