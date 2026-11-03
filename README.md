# 🌊 Dive Clip TUI

Local-first video curation for dive and motorcycle footage. Point the pipeline
at a folder of videos, use Ollama and FFmpeg to find interesting moments, and
render the detections as standalone MP4 clips. DaVinci Resolve is optional:
the existing EDL/CSV marker workflow is still available for editors who want it.

The project is CLI-first. The existing Textual terminal UI (TUI) is an
unmaintained prototype; future UI work will start from scratch. The core scan
flow does not require cloud uploads, a subscription, API keys, or DaVinci Resolve.

## Requirements

- Python 3.14+
- [FFmpeg](https://ffmpeg.org/) and `ffprobe`
- [Ollama](https://ollama.com/) with `qwen2.5-vl:7b` available locally
- Enough memory and GPU resources for the selected vision model; a CUDA GPU
  with 12 GB VRAM is recommended for the configured model. NVIDIA GPUs with NVENC
  support are automatically leveraged for hardware-accelerated video rendering.
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
1. **Pass 0 (Video Classification & Usability)**: Uses fast targeted input seeking (<0.5s) to sample isolated frames across video duration (scaled dynamically: min 3, max 8, 1 frame per 5 min) without decoding full files. Classifies the clip domain (`underwater`, `motorcycle`, `generic`, or custom domains) and screens for unusable footage (camera inside a bag/pocket, covered lens, dangling on gear). Unusable or non-target clips exit immediately with terminal alerts, avoiding wasted coarse frame decoding and VLM calls.
2. **Pass 1 (Domain Coarse Scan)**: Screens frames at low FPS (default: 0.25–0.5 FPS) using a domain-specific screening prompt that filters out false positives while tagging detected subjects and categories. Coarse frame extraction for upcoming clips is prefetched in the background in `/dev/shm` to keep the GPU saturated.
3. **Pass 2 (Subject-Routed Fine Scan)**: Dynamically routes each highlight candidate window to a specialized rubric (such as `pelagics`, `macro`, `diver`, `moto_riding`, `moto_scenic`, `moto_group`, or `generic_highlight`) for precise quality scoring and deduplication.

Even in the `generic` domain, the model identifies and preserves the concrete subject and action (e.g. `"downhill mountain bike jump"`, `"scenic mountain pass"`, `"dog catching frisbee"`) alongside detailed descriptions, making it easy to discover what subjects exist in your footage and add custom templates for them.

Custom domains and prompt templates can be configured or extended directly under `prompts:` in `config.yaml`.

### Running Pass 0 on its own (classify)

To quickly classify an unorganized folder of unknown videos without running highlight detection:

```bash
uv run dive-clip classify --input ./footage --output ./outputs
```

Pass 0 samples targeted representative frames dynamically based on clip duration (configured via `scan_parameters.classify_frames_count`, default 3, plus 1 per 5 min up to `classify_frames_max`, default 8) to identify the domain and verify usability without scanning the entire video. Results display each clip's domain, mean confidence, usability status, and uncertainty flags (flagged when frames disagree, when confidence is below 0.6, or when VLM responses are unusable).

Outputs are organized into standard timestamped runs (`<output>/runs/run-<timestamp>/classifications.json`) with an updated `<output>/latest` symlink and mirrored to `<output>/classifications.json`. Results are persisted incrementally and atomically after each clip completes, and interrupted runs can be resumed at any time with `--resume`.

To synthesize a reviewable prompt candidate for `generic` clips, add `--specialize-generic` (optionally `--candidate-output <path.json>`; defaults to `prompt_candidates.json` next to the output). Candidates are never used by `dive-clip scan` until you copy them into your config yourself.

#### Reusing classifications in later scans

To reuse confident domain labels and avoid reclassifying clips in a full scan:

```bash
uv run dive-clip scan --input ./footage --output ./highlights --classifications ./outputs/classifications.json
```

Confident labels are reused by matching `clip_id`. Any uncertain labels are skipped and re-evaluated by Pass 0 during the scan. Without `--classifications`, `dive-clip scan` runs normal Pass 0 domain classification as usual.

In debug mode the scanner also logs a verdict for every frame it analyzes, which
is the main tool for tuning the confidence thresholds:

```text
[Pass 0] Extracting frames @ 0.5 FPS: clip001.mp4
[Pass 0] Extracted 3 frames from clip001.mp4
[Pass 0] Classifying domain for clip001.mp4 across 3 sample frames...
[Pass 0] Classified clip001.mp4 as domain 'underwater'
[Pass 1] Coarse scan @ 0.5 FPS (underwater): clip001.mp4
[Pass 1] Analyzing frame 12 of 150
[coarse] Ollama response in 4.2s: 4210 prompt tokens, 57 generated tokens
[Pass 1] Frame 12 @ 24.0s: detected=True confidence=0.82 quality=8 passes_threshold=True subject=shark category=pelagics (4.2s)
[Pass 1] Summary clip001.mp4: 150 frames, 47 detected, 44 over threshold 0.80,
  confidence min/avg/max = 0.05/0.78/0.95, 0 failed calls, 120.3s total
[Pass 2] Fine scan @ 1.0 FPS: clip001.mp4 (1 windows)
[Pass 2] Window 1 of 1: 20.0s-32.0s (12.0s) @ 1.0 FPS [template: pelagics, domain: underwater]
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

The pipeline writes a JSON manifest, refined detections, rendered MP4 clips, and
end-of-run statistics to the configured output location.

### Pipeline state file & run archives

`dive-clip scan` organizes each execution into a self-contained snapshot under
`<output>/runs/<run_id>/` (with an `<output>/latest` symlink pointing to the most
recent run). Each run directory contains `pipeline_state.json`, `manifest.json`,
`detections_refined.json`, `run_stats.json`, `run.log`, and the extracted `highlights/` clips.

Progress is recorded in `pipeline_state.json` and saved after every clip and scan window.
Additionally, intra-clip progress is checkpointed periodically (every `checkpoint_interval_frames`,
default: 20) in `scan_checkpoint.json` inside each clip's temporary work directory, so an interrupted
run on long video files resumes without re-evaluating previously scanned frames.

The state file preserves full run context (`run_id`, `input_dir`, `output_dir`, `config_path`) and timing:

| Field | Meaning |
| --- | --- |
| `updated_at` | Timestamp of the most recent save |
| `activity`, `activity_at` | What the pipeline was doing at that save, e.g. `scan: clip001.mp4 window 2/5 @30s` |
| `<step>_done` | Whether `ingest`, `scan`, or `render` finished |
| `<step>_started_at` / `<step>_finished_at` | Step start and end timestamps (UTC, ISO 8601) |
| `<step>_duration_seconds` | Wall-clock duration of the completed step |
| `<step>_progress` | Per-step list of completed clip names |

At the end of scanning, a Rich summary table is displayed in the terminal and saved to `run_stats.json`, detailing total run duration, throughput (FPS), phase breakdown (Pass 0 domain counts, Pass 1 frames scanned, Pass 2 windows), and highlight counts.

To resume an interrupted or completed run:
- `dive-clip scan --output ./outputs --resume`: Resumes the most recent run automatically, restoring footage path and settings.
- `dive-clip scan --output ./outputs --resume run-20261031-120000`: Resumes or inspects a specific run folder.

Saves merge with the existing file on disk, so the orchestrator and the scan and
render workers can share one state path without overwriting each other's work.

### Reviewing and recording clip feedback

To review rendered highlight clips from a scan, open each in your default media player, and record judgments in the terminal:

```bash
uv run dive-clip review
```

Options:
- `--run <RUN_ID_OR_PATH>`: Review a specific run ID or folder (defaults to the latest run)
- `--all`: Re-review all clips in the run, including those with existing saved feedback

For each clip, your platform's video player launches automatically while the terminal prompts for:
- `[k]eep`: Highlight is accurate and worth keeping
- `[r]eject`: False detection or unwanted highlight
- `[w]rong subject`: Accurate highlight window but detected subject was wrong (prompts for corrected subject)
- `[s]kip`: Skip this clip without recording feedback
- `[b]ack`: Step back to the previous clip
- `[q]uit`: Exit review and display feedback summary

Feedback is saved atomically after each response to `<run_dir>/feedback.json`. When review ends, a summary table displays decision counts alongside all subject corrections and notes, providing concrete evidence for manual prompt refinement.

## Legacy terminal UI

The old Textual prototype remains available but is not maintained. It cannot
complete the review-to-render workflow. Use the CLI for pipeline work; see the
[product roadmap](docs/roadmap/roadmap.md) for future UI direction.

To launch the legacy interface:

```bash
uv run dive-clip-tui
```

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
  02_scan_footage.py       # Scan footage with Ollama (Pass 0, 1, and 2)
  03_generate_edl.py       # Create EDL and CSV timeline markers
  04_plan_storyline.py     # Optional story plan generation
  05_render_clips.py       # Render detection windows as MP4 clips
  06_assemble_video.py     # Assemble rendered clips
  07_create_aftermovie.py  # Create an aftermovie with music/transitions
  release.py               # Changelog, SemVer, and release automation helpers
  dive_clip.py             # Single-command ingest → scan → render CLI
tui/
  app.py                   # Legacy Textual TUI entry point
  screens/                 # Browse, review, and pipeline screens
utils/
  checkpoint.py            # Pipeline checkpoint and intra-clip resume state
  config.py                # Configuration loading and validation
  encoder.py               # Hardware encoder detection (NVENC) and CPU fallback
  ffmpeg_streamer.py       # Frame sampling and temporary storage (/dev/shm)
  logging_setup.py         # Shared console and file logging setup
  models.py                # Pydantic manifest and detection models
  ollama_client.py         # Ollama API client and metrics
  prompt_templates.py      # Domain prompts and subject-routed templates
  run_archive.py           # Self-contained run snapshots and latest symlink
  selection.py             # Maximal Marginal Relevance (MMR) highlight selection and diversity
  stats.py                 # Scan throughput, phase metrics, and summary tables
  timecode_math.py         # Frame-rate and SMPTE timecode helpers
  version.py               # Dynamic package version helper
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
- **Hardware-accelerated encoding**: Highlight clip rendering (`05_render_clips.py`)
  and aftermovie creation (`07_create_aftermovie.py`) probe and use `h264_nvenc` GPU
  encoding by default (`video_encoder: "auto"`), reducing CPU load and rendering
  times by over 90%. If hardware encoding is unavailable or encounters errors, the
  pipeline automatically falls back to thread-capped CPU `libx264`.
- **Fast input seeking**: Clip extraction places `-ss` and `-to` before `-i`, enabling
  instant keyframe seeking and decoding only the target highlight segment rather than
  decoding the full container from 0.0s.
- **Pipelined frame prefetching**: During multi-clip scans, coarse frame extraction
  for Video $N+1$ runs in the background in `/dev/shm` while Video $N$ executes VLM
  inference on GPU, overlapping decoding and inference with a strict 1-clip memory
  bound (`buffer_size=1`).
- **HTTP connection pooling**: `OllamaClient` maintains a persistent `requests.Session()`
  connection pool across all frame evaluations to eliminate TCP handshake latency and
  socket churn.
- The optional Resolve marker export supports frame-rate-aware timecodes,
  including 29.97 drop-frame and non-drop-frame formats.

For current project priorities, see
[`docs/roadmap/roadmap.md`](docs/roadmap/roadmap.md).

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
