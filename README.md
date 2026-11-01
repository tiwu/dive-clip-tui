# 🌊 Dive Clip TUI

Local-first video curation for dive and motorcycle footage. Point the pipeline
at a folder of videos, use Ollama and FFmpeg to find interesting moments, and
render the detections as standalone MP4 clips. DaVinci Resolve is optional:
the existing EDL/CSV marker workflow is still available for editors who want it.

The project is CLI-first, with an experimental Textual terminal UI (TUI) for
browsing manifests and reviewing detections. The core scan flow does not require
cloud uploads, a subscription, API keys, or DaVinci Resolve.

## Requirements

- Python 3.14+
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

Review `config.yaml` before scanning. In particular, set `general.output_dir`
and the Ollama model/host. Optionally set `general.input_dir` to the footage
folder so `dive-clip scan` can run without `--input`.

### Using a remote Ollama

`OLLAMA_HOST` overrides `ollama.host` from the config file, so a remote server can
be selected per run without editing the tracked config:

```bash
export OLLAMA_HOST=plotter.local:11434
uv run dive-clip scan --input ./footage --output ./highlights
```

Precedence is `OLLAMA_HOST`, then `ollama.host` in the config file, then
`http://localhost:11434`. A missing `http://` scheme is added automatically, so the
scheme-less form the Ollama CLI accepts works here too.

## Quick start: scan a footage folder

The single-command CLI chains folder ingestion, AI scanning, and clip rendering:

```bash
uv run dive-clip scan --input ./footage --output ./highlights
```

The scanner reads and writes its manifest and detections under the resolved
output folder. `--output` takes precedence over `general.output_dir`, and the
resolved folder is passed to every pipeline step, so the two can no longer drift
apart. Likewise `--input` takes precedence over `general.input_dir`:

```bash
# Footage and output folders both come from config.yaml
uv run dive-clip scan

# Flags still override the config values
uv run dive-clip scan --input ./footage --output ./highlights
```

A custom config can be selected with `--config`.

> [!CAUTION]
> Keep the input and output folders separate and non-nested. Ingestion walks the
> input folder recursively, so an output folder placed inside it (such as
> `./footage/highlights`) gets picked up as more footage on the next run and
> silently inflates the manifest. Use sibling folders, for example `./footage`
> and `./highlights`, not `./footage` and `./footage/highlights`.

The command supports checkpointing and can resume completed steps:

```bash
uv run dive-clip scan --input ./footage --output ./highlights --resume
```

Interrupting a scan with `Ctrl-C` (or `SIGINT`/`SIGTERM`) cleanly terminates processing: completed clip detections are preserved in `detections_refined.json`, checkpoint progress is flushed, and temporary working frames are deleted.

To start over and clear the saved pipeline state:

```bash
uv run dive-clip scan --input ./footage --output ./highlights --reset
```

For verbose logging during long-running AI scanning and rendering, use `--debug`:

```bash
uv run dive-clip scan --input ./footage --output ./highlights --debug
```

This enables timestamped debug logs showing per-frame Ollama request timing,
token counts, and full FFmpeg commands.

### 3-stage scanning & domain awareness

The scanner uses a 3-stage coarse-to-fine architecture:
1. **Pass 0 (Video Classification)**: Samples representative frames to determine the video domain (`underwater`, `motorcycle`, or `generic`), unless specified via `scan_parameters.domain` or clip tags.
2. **Pass 1 (Domain Coarse Scan)**: Screens frames at low FPS (default: 0.5 FPS) using a domain-specific prompt that filters out unusable camera angles (e.g. dangling on a BCD underwater vs. pointed at a fuel tank/ground on a motorcycle) while tagging detected subjects and categories.
3. **Pass 2 (Subject-Routed Fine Scan)**: Dynamically routes each highlight candidate window to a specialized rubric (such as `pelagics`, `macro`, `diver`, `moto_riding`, `moto_scenic`, `moto_group`, or `generic_highlight`) for precise quality scoring and deduplication.

Custom domains and prompt templates can also be configured or extended directly under `prompts:` in `config.yaml`.

In debug mode the scanner also logs a verdict for every frame it analyzes, which
is the main tool for tuning the confidence thresholds:

```
[Pass 1] Analyzing frame 12 of 50
[coarse] Ollama response in 8.4s: 4210 prompt tokens, 57 generated tokens
[Pass 1] Frame 12 @ 24.0s: detected=True confidence=0.82 passes_threshold=True subject=shark (8.4s)
[Pass 1] Summary clip001.mp4: 50 frames, 47 detected, 44 over threshold 0.65,
  confidence min/avg/max = 0.05/0.78/0.95, 1 failed calls, 421.3s total
```

Each pass ends with a summary line at info level, so a normal run (without
`--debug`) still reports how selective the coarse pass was and how many
detections survived deduplication. Use those numbers to adjust
`coarse_confidence_threshold`, `coarse_fps`, and `fine_fps`.

Every run also writes a transcript of its terminal output to
`<output>/logs/run-<YYYYmmdd-HHMMSS>.log`. The file is timestamped per run and
older runs are never deleted, so long unattended batches stay inspectable.
Ingest, scan, and render all append to the same file. To pick the location
yourself:

```bash
uv run dive-clip scan --input ./footage --output ./highlights --log-file ./my-run.log
```

The individual step scripts accept `--log-file` too, so a hand-run
`scripts/02_scan_footage.py` can log somewhere other than the default output
folder.

The pipeline writes a JSON manifest, refined detections, and rendered MP4 clips
to the configured output location.

### Pipeline state file & run archives

`dive-clip scan` organizes each execution into a self-contained snapshot under
`<output>/runs/<run_id>/` (with an `<output>/latest` symlink pointing to the most
recent run). Each run directory contains `pipeline_state.json`, `manifest.json`,
`detections_refined.json`, `run.log`, and the extracted `highlights/` clips.

Progress is recorded in `pipeline_state.json` and saved after every clip and scan window,
so an interrupted run leaves an up-to-date checkpoint. The file also preserves full run
context (`run_id`, `input_dir`, `output_dir`, `config_path`) and timing:

| Field | Meaning |
| --- | --- |
| `updated_at` | Timestamp of the most recent save |
| `activity`, `activity_at` | What the pipeline was doing at that save, e.g. `scan: clip001.mp4 window 2/5 @30s` |
| `<step>_done` | Whether `ingest`, `scan`, or `render` finished |
| `<step>_started_at` / `<step>_finished_at` | Step start and end timestamps (UTC, ISO 8601) |
| `<step>_duration_seconds` | Wall-clock duration of the completed step |
| `<step>_progress` | Per-step list of completed clip names |

To resume an interrupted or completed run:
- `dive-clip scan --output ./outputs --resume`: Resumes the most recent run automatically, restoring footage path and settings.
- `dive-clip scan --output ./outputs --resume run-20261031-120000`: Resumes or inspects a specific run folder.

Saves merge with the existing file on disk, so the orchestrator and the scan and
render workers can share one state path without overwriting each other's work.

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
  resolution, and codec in `manifest.json` (with backward compatibility for `manifest_raw.json`).
- The scanner uses Ollama's JSON response mode and writes
  `detections_refined.json`. It prints request counts, failures, elapsed time,
  and available Ollama token/timing metrics after the scan. Use `--debug` for
  per-frame verdicts, per-request timing, and token counts during the run.
- `dive-clip scan` stores progress and full run context in `pipeline_state.json` inside each run directory, updated
  after every clip and scan window with `updated_at`, `activity`, and per-step
  timing; use `--resume [run_id_or_path]` to resume previous runs or `--reset` to clear that state.
- Frame sampling creates isolated private temporary directories under `/dev/shm`
  when available and enabled via `scan_parameters.use_ramdisk: true` (default).
  Writing temporary JPEG frames to shared memory maximizes extraction and reading
  throughput while eliminating heavy SSD write cycles when processing large batches.
  The pipeline automatically falls back to standard temporary storage on systems without
  `/dev/shm` (e.g. macOS) or when `use_ramdisk` is set to `false`.
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
