# 🌊 Dive Clip TUI

Local-first video curation for dive and motorcycle footage. Point it at a folder
of videos, let Ollama and FFmpeg find the interesting moments, and get back
standalone MP4 clips or a watchable highlight reel.

The core flow needs no cloud, no subscription, no API keys, and no DaVinci
Resolve. The bundled Textual TUI is an unmaintained prototype; future UI work
starts from scratch.

## Contents

- [Requirements](#requirements) - [Installation](#installation) - [Quick start](#quick-start)
- [Commands](#commands): [scan](#dive-clip-scan) - [classify](#dive-clip-classify) - [assemble / aftermovie](#dive-clip-assemble-and-dive-clip-aftermovie) - [review](#dive-clip-review) - [shared options](#options-shared-by-every-command)
- [Configuration](#configuration): [precedence](#what-wins) - [reference](#configuration-reference) - [VLM providers](#choosing-a-vlm-provider) - [domains](#domains-and-prompt-templates)
- [How the pipeline works](#how-the-pipeline-works)
- [Outputs, run archives, and resuming](#outputs-run-archives-and-resuming)
- [Tuning and troubleshooting](#tuning-and-troubleshooting)
- [Optional: DaVinci Resolve markers](#optional-davinci-resolve-markers)
- [Legacy terminal UI](#legacy-terminal-ui) - [Repository structure](#repository-structure) - [Contributing](#contributing-changes-and-releases)

## Requirements

- Python 3.14+
- [FFmpeg](https://ffmpeg.org/) and `ffprobe` on `PATH`
- [Ollama](https://ollama.com/) with a vision model pulled locally —
  `ollama pull qwen2.5vl:3b` matches the default config. Swap the model via
  `ollama.model`, or use a cloud VLM instead (see
  [Choosing a VLM provider](#choosing-a-vlm-provider)).
- Enough memory and GPU for the chosen model; a CUDA GPU with 12 GB VRAM is
  recommended. NVIDIA NVENC is used automatically for rendering.
- DaVinci Resolve only for the optional [marker export](#optional-davinci-resolve-markers).

NumPy and Pillow are not required — the screening funnel's image math is pure Python.

## Installation

```bash
uv sync
ollama pull qwen2.5vl:3b
```

Or with pip (requires Python 3.14+):

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then review [`config.yaml`](config.yaml). At minimum set `general.output_dir` and
your Ollama model or host; every option is documented under
[Configuration](#configuration).

Copy [`.env.example`](.env.example) to `.env` only if you plan to use a cloud VLM.
`.env` is git-ignored and read from the config file's folder or the working
directory; variables already set in your shell win over it.

## Quick start

```bash
# Scan a folder and render the detected highlights as clips
uv run dive-clip scan --input ./footage --output ./highlights

# Same, plus a single highlight video with music under it
uv run dive-clip scan --input ./footage --output ./highlights \
  --assemble --music ./track.mp3

# Play the clips back and record which ones were good
uv run dive-clip review
```

> [!CAUTION]
> Keep the input and output folders separate and non-nested. Ingestion walks the
> input folder recursively, so an output folder placed inside it (such as
> `./footage/highlights`) gets picked up as more footage on the next run and
> silently inflates the manifest. Use siblings, for example `./footage` and
> `./highlights`, not `./footage` and `./footage/highlights`.

## Commands

| Command | What it does | Uses the VLM |
| --- | --- | --- |
| [`scan`](#dive-clip-scan) | Ingest → screen → coarse/fine scan → render clips (→ optional assemble) | yes |
| [`classify`](#dive-clip-classify) | Pass 0 only: label domains and usability, no highlight detection | yes |
| [`assemble`](#dive-clip-assemble-and-dive-clip-aftermovie) / [`aftermovie`](#dive-clip-assemble-and-dive-clip-aftermovie) | Cut existing clips into one video, plain or with music and fades | no |
| [`review`](#dive-clip-review) | Play clips back and record keep/reject feedback | no |

`assemble` and `aftermovie` only read existing run artifacts, so re-cutting with a
different `--top-n` costs nothing in scanning.

### `dive-clip scan`

```bash
uv run dive-clip scan --input ./footage --output ./highlights
```

Runs the full pipeline. Each step records its progress in the run directory, so an
interrupted run resumes rather than restarting.

| Option | Default | Meaning |
| --- | --- | --- |
| `--input PATH` | `general.input_dir` | Footage folder, walked recursively |
| `--output PATH` | `general.output_dir` | Output folder for runs, clips, and logs |
| `--resume [RUN_ID_OR_PATH]` | latest run | Resume a run; bare flag resumes the newest |
| `--reset` | off | Clear saved pipeline state and start fresh |
| `--assemble` | off | Add an assembly step after rendering |
| `--assemble-mode {aftermovie,simple}` | `aftermovie` | `aftermovie` = music/fades (`07`), `simple` = plain concat (`06`) |
| `--music PATH` | none | Music bed; ignored by `--assemble-mode simple` |
| `--top-n N` | `aftermovie.top_n_clips`, else `10` | Clips to include when assembling |
| `--classifications PATH` | none | Reuse confident domain labels from a previous `classify` run |
| `--config`, `--debug`, `--log-file` | — | [Shared options](#options-shared-by-every-command) |

`Ctrl-C` (or `SIGINT`/`SIGTERM`) is clean: completed detections are kept,
checkpoint progress is flushed, and temporary frames are removed.

### `dive-clip classify`

Pass 0 only — label a disorganized folder without looking for highlights:

```bash
uv run dive-clip classify --input ./footage --output ./outputs
```

Each clip gets a domain (`underwater`, `motorcycle`, `generic`, or a custom one), a
mean confidence, a usability verdict, and uncertainty flags (frames disagree,
confidence below 0.6, or unusable VLM replies). Sampling is targeted seeking, not
full decode: `classify_frames_count` frames minimum plus one per
`classify_interval_seconds`, capped at `classify_frames_max`.

Results land in `<output>/runs/run-<timestamp>/classifications.json`, are mirrored
to `<output>/classifications.json`, and are written atomically after each clip, so
`--resume` works mid-run. Feed them into a later scan with
`dive-clip scan --classifications <path>`: confident labels are reused by
`clip_id`, uncertain ones are re-evaluated.

| Option | Default | Meaning |
| --- | --- | --- |
| `--input PATH` | `general.input_dir` | Footage folder |
| `--output PATH` | `<general.output_dir>/classifications.json` | Classification output path |
| `--resume` | off | Continue an interrupted classification run |
| `--specialize-generic` | off | Synthesize a reviewable prompt candidate for `generic` clips |
| `--candidate-output PATH` | `<output_dir>/prompt_candidates.json` | Candidate path; requires `--specialize-generic` |
| `--config`, `--debug`, `--log-file` | — | [Shared options](#options-shared-by-every-command) |

Candidates are never picked up automatically — copy them into `config.yaml`
yourself once you are happy with them.

### `dive-clip assemble` and `dive-clip aftermovie`

Both cut existing clips into one video; `aftermovie` adds music, fades, and
transitions.

```bash
uv run dive-clip assemble   --output ./outputs --top-n 8
uv run dive-clip aftermovie --output ./outputs --music ./track.mp3
```

| Option | `assemble` | `aftermovie` |
| --- | --- | --- |
| `--input PATH` | `<output>/latest/detections_refined.json` | same |
| `--clips-dir PATH` | `<output>/latest/highlights` (or `<output>/highlights`) | same |
| `--output PATH` | `<output>/highlight_reel.mp4` | `<output>/aftermovie.mp4` |
| `--top-n N` | `10` or config | same |
| `--music PATH` | — | music bed |
| `--music-volume F` | — | `0.15`, or `aftermovie.music_volume` |
| `--fade-duration S` | — | `1.0`, or `aftermovie.fade_duration` |
| `--config`, `--debug`, `--log-file` | [shared](#options-shared-by-every-command) | [shared](#options-shared-by-every-command) |

### `dive-clip review`

```bash
uv run dive-clip review                                  # latest run
uv run dive-clip review --run run-20261031-120000        # a specific run
uv run dive-clip review --all                            # re-review everything
```

Each clip opens in the platform player while the terminal prompts: `[k]eep`,
`[r]eject`, `[w]rong subject` (prompts for the correction), `[s]kip`, `[b]ack`,
`[q]uit`. Feedback is saved atomically to `<run_dir>/feedback.json` after every
answer, and the closing summary lists decision counts plus every subject
correction — useful evidence for prompt refinement.

### Options shared by every command

- `--config PATH` — config file to load (default `config.yaml`).
- `--debug` — per-frame verdicts, Ollama request timing and token counts, full
  FFmpeg command lines.
- `--log-file PATH` — transcript location; defaults to a timestamped file in the
  run directory. Older logs are never deleted.
- `-h` / `--help` — the authoritative, always-current option list.
  `dive-clip --version` prints the package version.

## Configuration

### What wins

Commands load a single configuration file. When a setting is specified across
multiple layers, the following precedence applies:

| Setting | Highest | Then | Then |
| --- | --- | --- | --- |
| Footage folder | `--input` | `general.input_dir` | — |
| Output folder | `--output` | `general.output_dir` | — |
| Ollama server | `OLLAMA_HOST` | `ollama.host` | `http://localhost:11434` |
| API keys | shell environment | `.env` file | — |
| Clip count | `--top-n` | `aftermovie.top_n_clips` | `10` |
| Music level | `--music-volume` | `aftermovie.music_volume` | `0.15` |
| Fade length | `--fade-duration` | `aftermovie.fade_duration` | `1.0` |
| Everything else | the config file | built-in default | — |

A missing `http://` scheme on the Ollama host is added for you. The resolved output
folder is handed to every pipeline step, so flags and config cannot drift apart
mid-run.

> [!NOTE]
> `load_config()` only checks that the file parses as YAML with a mapping at the top
> level. Unknown keys are silently ignored, missing keys fall back to per-consumer
> defaults, and the tables below are maintained by hand rather than generated — a
> typo fails quietly. Keep `config.yaml` in version control and review diffs.

### Configuration reference

[`config.yaml`](config.yaml) is the canonical, fully commented example. Sections
marked *optional* are absent from the tracked file and only take effect once you
add them.

**`general`**

| Key | Default | Meaning |
| --- | --- | --- |
| `project_name` | `"Madagascar_Dives_2026"` | Label recorded in the Resolve manifest; the folder CLI uses the folder name |
| `input_dir` | none | Footage folder used when `--input` is omitted |
| `output_dir` | `./outputs` | Output folder used when `--output` is omitted |

**`ollama`**

| Key | Default | Meaning |
| --- | --- | --- |
| `host` | `http://localhost:11434` | Ollama server; `OLLAMA_HOST` overrides it |
| `model` | `qwen2.5vl:3b` | Vision model for Pass 0/1/2 |
| `timeout_seconds` | `120` | Per-request timeout |
| `num_ctx` | unset | Optional context window in tokens, sent as `options.num_ctx` (lower it to fit a model in memory; unset keeps the model's default) |

**`vlm`**

| Key | Default | Meaning |
| --- | --- | --- |
| `provider` | `ollama` | `ollama` or `openrouter` |
| `openrouter.model` | `nvidia/nemotron-3.5-super-vl:free` | Cloud vision model |
| `openrouter.timeout_seconds` | `120` | Per-request timeout |
| `openrouter.requests_per_minute` | `20` | Client-side rate limit |
| `openrouter.max_retries` | `3` | Retries on HTTP 429/5xx with exponential backoff |

Any other `provider` value fails at startup with the list of supported backends.

**`scan_parameters`**

| Key | Default | Meaning |
| --- | --- | --- |
| `domain` | `"auto"` | `auto` lets Pass 0 classify each clip; or force `underwater`, `motorcycle`, `generic` |
| `classify_clips` | `"auto"` | `auto` = run Pass 0 unless the domain is pinned or tags exist; `always`; `off` |
| `classify_frames_count` | `3` | Minimum Pass 0 sample frames per clip |
| `classify_interval_seconds` | `300` | One extra Pass 0 frame per this many seconds |
| `classify_frames_max` | `8` | Maximum Pass 0 sample frames per clip |
| `prefetch_coarse` | `true` | Extract clip N+1's coarse frames while clip N runs inference |
| `video_encoder` | `"auto"` | `auto` probes `h264_nvenc` then falls back to `libx264`; or force either |
| `video_quality.cq` | `19` | NVENC constant-quality level applied to every encode (lower = better/bigger) |
| `video_quality.crf` | `19` | `libx264` CRF level applied to every encode (lower = better/bigger) |
| `coarse_fps` | `0.25` | Pass 1 sample rate (one frame every 4 s) |
| `coarse_scale` | `"426:-2"` | Downscale for cheap Pass 1 screening |
| `fine_fps` | `1.0` | Pass 2 sample rate inside candidate windows |
| `fine_scale` | `"1280:-2"` | Pass 2 resolution (`"iw/2:-2"` for half, `"null"` for none) |
| `coarse_confidence_threshold` | `0.80` | Minimum confidence to open a fine-scan window |
| `coarse_quality_threshold` | `4` | Minimum 1–10 score to keep a coarse frame (rejects dangling/upside-down) |
| `fine_padding_seconds` | `2` | Extra context around a detected action |
| `fine_pre_roll_seconds` | `1.5` | Lead-in before a fine highlight |
| `fine_post_roll_seconds` | `2.0` | Tail after a fine highlight |
| `merge_gap_seconds` | `1.5` | Merge same-subject detections across gaps this small |
| `max_clip_duration` | `12.0` | Cap on a merged window, in seconds |
| `quality_threshold` | `8` | Minimum 1–10 score for a detection to be kept |
| `checkpoint_interval_frames` | `20` | Intra-clip resume checkpoint cadence |
| `use_ramdisk` | `true` | Use `/dev/shm` for frames when available |

`use_ramdisk` falls back to normal temporary storage on systems without `/dev/shm`
(such as macOS) or when set to `false`.

**`selection`** — which detections become clips.

| Key | Default | Meaning |
| --- | --- | --- |
| `lambda_quality` | `0.65` | MMR trade-off: `0.0` pure diversity, `1.0` pure quality |
| `per_clip_soft_limit` | `1` | Soft quota per source clip before an exponential penalty applies |

Lower `lambda_quality` for more variety across a long trip; raise it when you only
want the best moments even if they cluster in one file.

**`prescreen`** — the pre-VLM screening funnel, described in
[How the pipeline works](#pre-vlm-screening-funnel). It ships **disabled**: with
`enabled: false` no extra FFmpeg pass runs and behaviour is unchanged.

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Turn the funnel on at all |
| `gate_mode` | `"advisory"` | `advisory` records gate failures but drops nothing; `enabled` rejects failing windows |
| `sample_fps` | `0.5` | Frames per second sampled for visual metrics |
| `sample_size` | `64` | Downscale samples to 64×64 for cheap luma/Laplacian math |
| `keyframes_only` | `true` | Prefer keyframes, fall back to a full seek pass |
| `audio_enabled` | `true` | Run the FFmpeg `astats` pass for peak levels |
| `audio_spike_db` | `10.0` | dB above the rolling median that counts as a spike |
| `audio_spike_floor_db` | `-35.0` | Absolute floor so steady quiet audio never reads as a spike |
| `audio_window` | `5` | Rolling median window, in peak samples |
| `blur_min` | `12.0` | Median Laplacian variance below this reads as soft or blurred |
| `contact_sheet` | `false` | Opt-in 2×2 contact sheets: up to 4× fewer Pass 1 calls |
| `weights.lean_speed` | `0.35` | Lean angle + speed magnitude |
| `weights.gyro_stability` | `0.25` | Inverse of gyro RMS — steadier footage scores higher |
| `weights.face` | `0.20` | Detected subject/face presence |
| `weights.hue_change` | `0.20` | Colour churn across the window |
| `gate_thresholds.yavg_min` | `15.0` | Mean luma below this is too dark to use |
| `gate_thresholds.unif_max` | `0.92` | Luma uniformity above this is flat or fogged (1.0 is featureless) |
| `gate_thresholds.gyro_max` | `3.5` | Gyro RMS above this is too shaky |
| `gate_thresholds.tilt_max` | `100.0` | Max tilt angle (degrees) relative to upright horizontal before orientation gate rejects |
| `domain_prior_thresholds.min_confidence` | `0.80` | Minimum Pass 0 confidence to trust a domain prior |
| `domain_prior_thresholds.min_coverage` | `0.80` | Minimum Pass 0 coverage to trust a domain prior |

Start at `gate_mode: "advisory"`, read `prescreen_trace.json`, then switch to
`"enabled"` once you are satisfied the gates reject what you expect.

**`cloud_storyline`** *(optional)* — used only by `scripts/04_plan_storyline.py`.

| Key | Default | Meaning |
| --- | --- | --- |
| `provider` | `"claude"` | `claude` or `gemini` |
| `model` | `claude-3-5-sonnet-latest` | Model for story planning |
| `target_edit_duration_minutes` | `4` | Target length of the planned edit |

**`aftermovie`** *(optional)* — consulted only when the matching flag is absent.

| Key | Default | Meaning |
| --- | --- | --- |
| `top_n_clips` | `10` | Default clip count |
| `music_volume` | `0.15` | Default music mix level |
| `fade_duration` | `1.0` | Default fade/transition length |
| `min_transition_gap` | `3.0` | Minimum gap in seconds between same-source clips before fading out/in (dip-to-black) |
| `bridge_gap_seconds` | `1.5` | Max gap in seconds between same-source clips to bridge into a single take |
| `max_clip_duration` | `12.0` | Max merged duration in seconds for bridged clip takes |

**`prompts`** *(optional)* — custom domains and prompt templates, with a commented
example at the bottom of [`config.yaml`](config.yaml).

### Choosing a VLM provider

**Local Ollama (default).** Uses the local instance specified in `ollama.model`
without sending frames off-machine.

**Remote Ollama.** `OLLAMA_HOST` overrides `ollama.host` without editing the
tracked config:

```bash
export OLLAMA_HOST=plotter.local:11434
uv run dive-clip scan --input ./footage --output ./highlights
```

**Cloud VLM (OpenRouter).** Set the key, then switch the provider:

```bash
cp .env.example .env   # then set OPENROUTER_API_KEY in .env (git-ignored)
# or: export OPENROUTER_API_KEY=sk-or-...
```

```yaml
vlm:
  provider: "openrouter"
  openrouter:
    model: "nvidia/nemotron-3.5-super-vl:free"
```

Replies that are not pure JSON are recovered when they contain a JSON object. Free-tier
models have their own quotas; managing them is up to you.

> [!WARNING]
> **Privacy:** with `provider: "openrouter"`, sampled video frames are uploaded to
> OpenRouter and the model provider behind it. Nothing leaves your machine with the
> default `ollama` provider. The scan logs a warning at startup when a cloud provider
> is active.

### Domains and prompt templates

Four domains ship by default: `underwater`, `motorcycle`, `generic`, and `auto`
(meaning "let Pass 0 decide"). Each has its own screening prompt in
`utils/prompt_templates.py`, and each candidate window is routed to a specialized
rubric — `pelagics`, `macro`, `diver`, `moto_riding`, `moto_scenic`, `moto_group`,
or `generic_highlight`.

Even in the `generic` domain the model keeps the concrete subject and action
(`"downhill mountain bike jump"`, `"scenic mountain pass"`, `"dog catching
frisbee"`), which makes it easy to discover what your footage contains and then add
a template for it. Declare new domains and templates under `prompts:` in
`config.yaml`.

## How the pipeline works

`dive-clip scan` runs three steps, plus a fourth with `--assemble`:

| Step | Script | What happens |
| --- | --- | --- |
| 1. Ingest | `00_ingest_media.py` | `ffprobe` every video into `manifest.json` |
| 2. Scan | `02_scan_footage.py` | Optional pre-VLM funnel, then Pass 0, 1, and 2 |
| 3. Render | `05_render_clips.py` | Cut each detection window into an MP4 |
| 4. Assemble | `06_`/`07_` | Only with `--assemble` |

### The three scan passes

1. **Pass 0 — classification and usability.** Targeted seeking (<0.5 s per clip)
   instead of full decode. Assigns a domain and rejects unusable footage — camera in
   a bag or pocket, covered lens, dangling on gear — so the rest of the scan skips it.
2. **Pass 1 — coarse scan.** Samples at `coarse_fps` with a domain-specific prompt
   that filters false positives while tagging subject and category. Coarse frames for
   the next clip are prefetched in `/dev/shm` so the GPU stays busy.
3. **Pass 2 — fine scan.** Each candidate window is routed to a specialized rubric
   for precise quality scoring and deduplication.

Both passes end with a summary line at info level, so a normal run still reports how
selective the coarse pass was and how many detections survived deduplication.
Detections that clear `quality_threshold` become candidates; Maximal Marginal
Relevance then picks the final set using `selection.lambda_quality`, so a long trip
yields variety instead of ten near-identical shots.

### Pre-VLM screening funnel

An optional cheap pass ahead of every VLM call that decides per candidate window
whether to `keep`, `defer`, or `reject` it. Configured under `prescreen:`; off by
default.

1. **Ingestion** — `00_ingest_media.py` probes every stream and records sensor tracks
   and container hints in the manifest (`telemetry_type`, `telemetry_stream_index`,
   `has_telemetry`, `is_high_fps`, `hilight_tags`, `bitrate_density`). Telemetry
   streams are detected dynamically across container tracks by matching codec tag
   (`gpmd`, `djmd`, `insta`) or handler name (`GoPro MET`, `DJI.Meta`) rather than
   assuming fixed track numbers. GoPro `gpmf` telemetry is decoded into timestamped
   series; the funnel consumes `SCEN`, `GYRO`, `GPS5`/`GPS9`/`GPS2` speed, `GRAV`
   gravity vector, `CORI`/`IORI` orientation proxies, `YAVG`/`UNIF` luma, and `FACE`
   presence. DJI and Insta360 tracks are detected but not yet decoded, so they yield
   no score rather than a penalty.
2. **HiLight tags** — GoPro `moov.udta.hmmt` markers are parsed out of the atom tree
   and used to align screening windows, so per-window metrics land on moments a human
   already marked. They do not create windows of their own; they are echoed into
   every prescreen record.
3. **Metrics** — one FFmpeg pass samples frames at `prescreen.sample_fps` for mean
   luma, luma uniformity, Laplacian sharpness, and HSV histograms; a second `astats`
   pass reads audio peaks and detects spikes above a rolling median.
4. **Gates** — `yavg_min`, `unif_max`, `gyro_max`, `tilt_max`, and `blur_min` plus a weighted
   telemetry score decide each window. Missing sensors never reject footage: a gate
   with no signal reports `signal unavailable` and is skipped, and weights renormalize
   over whichever signals exist.

The funnel runs first for each clip, before Pass 0 domain classification. With
`gate_mode: "enabled"`, a clip whose windows are all rejected is skipped outright
(recorded as `skipped_prescreen` in `classifications.json`) without any VLM call
or coarse frame extraction. Advisory mode never skips clips.

#### Telemetry Specifications & Coordinates Reference
* **GoPro GPMF Format**: Based on the [GoPro GPMF parser specification](https://github.com/gopro/gpmf-parser).
  Every KLV payload is 32-bit aligned (`((payload_length + 3) & ~3)`).
* **GoPro Coordinate System**: Camera body axes are defined as:
  * `+X`: Camera right
  * `+Y`: Camera bottom
  * `+Z`: Optical axis (lens forward)
* **Orientation and Gravity Decoding**:
  When upright and horizontal, the gravity vector `GRAV` points down `+Y` (`gy = 1.0`, `tilt = 0°`).
  Tilt angle relative to upright horizontal is calculated as `arccos(clamp(gy, -1.0, 1.0)) * (180.0 / pi)`.
  When a camera dangles inverted from diving gear or a lanyard, `gy < 0` (`tilt > 90°`), triggering the `orientation` gate (`tilt_max: 100.0`).
* **Research & Implementation Notes**: See `docs/research/highlight-detection-and-telemetry.md`.

All image math is pure Python over FFmpeg `rgb24` bytes, so the funnel needs no NumPy
or Pillow and stays optional.

Opt into 2×2 contact sheets with `contact_sheet: true`: frames are composed into one
2×2 image and a single VLM call returns up to four per-tile verdicts, cutting Pass 1
calls up to 4×. Partial final batches, failed compositions, and unparseable sheet
responses fall back to the per-frame path automatically, so the feature can only
reduce calls, never detection coverage.

### Performance features

- **Hardware encoding** — rendering probes `h264_nvenc` (`video_encoder: "auto"`) and
  falls back to thread-capped `libx264`. Both paths use constant-quality rate control
  (`video_quality.cq`/`.crf`, default 19) so 4K sources are not starved of bitrate.
- **Fast input seeking** — `-ss`/`-to` precede `-i`, decoding only the target segment.
- **Pipelined prefetch** — clip N+1's coarse frames extract while clip N runs
  inference, bounded to one clip in memory.
- **RAM disk** — frames go to `/dev/shm` when available, avoiding SSD write wear.
- **Connection pooling** — one persistent `requests.Session()` per Ollama client.

## Outputs, run archives, and resuming

### Run directory

Pipeline runs produce a self-contained snapshot under `<output>/runs/<run_id>/`, with
`<output>/latest` symlinked to the newest:

| Artifact | Contents |
| --- | --- |
| `manifest.json` | Ingested clip metadata (container, duration, frame rate, telemetry hints) |
| `pipeline_state.json` | Which steps finished, timings, current activity |
| `detections_refined.json` | Final detections |
| `classifications.json` | Pass 0 labels, when classification ran |
| `prescreen_trace.json` | Per-window funnel decisions |
| `pass1_coarse_trace.json` / `pass2_fine_trace.json` | Per-frame and per-window verdicts |
| `selection_trace.json` | Which detections became clips, and why |
| `run_stats.json` | Throughput, phase breakdown, highlight counts |
| `run.log` | Transcript of the run |
| `highlights/` | Rendered MP4 clips |
| `feedback.json` | Review judgments, once you have run `review` |

Traces are mirrored next to the other outputs too. A scan ends by printing a Rich
summary table and saving the same numbers to `run_stats.json`: total duration,
throughput in FPS, Pass 0 domain counts, Pass 1 frames scanned, Pass 2 windows, and
highlight counts.

### Pipeline state file

Progress is saved after every clip and scan window. Intra-clip progress is
checkpointed every `checkpoint_interval_frames` into `scan_checkpoint.json` inside
each clip's temporary work directory, so an interrupted run over long files does not
re-evaluate frames it already scanned.

The state file keeps full run context (`run_id`, `input_dir`, `output_dir`,
`config_path`) and timing:

| Field | Meaning |
| --- | --- |
| `updated_at` | Timestamp of the most recent save |
| `activity`, `activity_at` | What the pipeline was doing, e.g. `scan: clip001.mp4 window 2/5 @30s` |
| `<step>_done` | Whether `ingest`, `scan`, or `render` finished |
| `<step>_started_at` / `<step>_finished_at` | Step start and end timestamps (UTC, ISO 8601) |
| `<step>_duration_seconds` | Wall-clock duration of the completed step |
| `<step>_progress` | Per-step list of completed clip names |

Saves merge with what is on disk, so the orchestrator and the scan and render workers
can share one state path without clobbering each other.

### Resuming, resetting, logs

```bash
uv run dive-clip scan --output ./highlights --resume                  # newest run
uv run dive-clip scan --output ./highlights --resume run-20261031-120000
uv run dive-clip scan --output ./highlights --reset                   # start over
```

Transcripts default to `<output>/runs/run-<timestamp>/run.log`, timestamped per run
and never deleted, so long unattended batches stay inspectable; ingest, scan, and
render all append to the same file. Override per run with `--log-file`, which the
individual step scripts also accept.

## Tuning and troubleshooting

Most tuning comes down to four knobs:

| Symptom | Knob |
| --- | --- |
| Coarse pass too eager or too shy | `coarse_confidence_threshold`, `coarse_quality_threshold`, `coarse_fps` |
| Windows too short or too long | `fine_pre_roll_seconds`, `fine_post_roll_seconds`, `merge_gap_seconds`, `max_clip_duration` |
| Output repetitive or all from one file | `selection.lambda_quality`, `selection.per_clip_soft_limit` |
| Domain misdetected on many clips | pin `scan_parameters.domain`, or add a domain under `prompts:` |

`--debug` prints every frame verdict, which is the main tool for setting thresholds:

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

With the funnel enabled, `prescreen_trace.json` answers "why was this window
skipped?" without re-running anything. It holds one record per candidate window —
never per raw sensor sample — carrying container signals, telemetry signals,
audio/visual metrics, the configured thresholds, every gate outcome and reason,
score components with renormalized weights, the chosen action, and whether a VLM call
followed.

To judge whether a classification run is trustworthy, read the uncertainty flags:
disagreeing frames, mean confidence below 0.6, or unusable VLM replies. Uncertain
labels are always re-evaluated rather than reused.

## Optional: DaVinci Resolve markers

For projects that use Resolve metadata or want timeline markers:

1. In Resolve, open **Workspace → Console**, choose **Py3**, and run
   `exec(open("scripts/01_extract_manifest.py").read())`.
2. Scan the exported manifest: `uv run scripts/02_scan_footage.py --config config.yaml`.
3. Generate EDL and CSV markers: `uv run scripts/03_generate_edl.py --config config.yaml`.
4. Import `outputs/markers_import.edl` via **Timelines → Import → Timeline Markers
   from EDL...**.

You can also generate the EDL/CSV from a folder-ingested manifest; Resolve is never
required for the local scan and render flow. The export supports frame-rate-aware
timecodes, including 29.97 drop-frame and non-drop-frame.

## Legacy terminal UI

The old Textual prototype is unmaintained and cannot complete the
review-to-render workflow. Use the CLI for pipeline work; see the
[product roadmap](docs/roadmap/roadmap.md) for where the UI is heading.

```bash
uv run dive-clip-tui
```

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
  config.py                # Configuration loading
  encoder.py               # Hardware encoder detection (NVENC) and CPU fallback
  env.py                   # Minimal .env loader for secrets like OPENROUTER_API_KEY
  ffmpeg_streamer.py       # Frame sampling and temporary storage (/dev/shm)
  logging_setup.py         # Shared console and file logging setup
  models.py                # Pydantic manifest and detection models
  ollama_client.py         # Ollama API client and host resolution
  vlm_common.py            # Shared VLM client base, JSON extraction and metrics
  prescreen.py             # Pre-VLM screening funnel: gates, metrics, trace records
  prompt_templates.py      # Domain prompts and subject-routed templates
  run_archive.py           # Self-contained run snapshots and latest symlink
  selection.py             # Maximal Marginal Relevance (MMR) highlight selection
  signals.py               # Graceful Ctrl-C / SIGTERM shutdown handling
  stats.py                 # Scan throughput, phase metrics, and summary tables
  telemetry.py             # GoPro GPMF / DJI / Insta360 telemetry decoding and scoring
  timecode_math.py         # Frame-rate and SMPTE timecode helpers
  version.py               # Dynamic package version helper
  vlm_client.py            # VLM client protocol, OpenRouter backend, provider factory
tests/                     # Pytest suite
outputs/                   # Generated manifests, detections, and exports
```

## Contributing changes and releases

Changes follow the two-stage GitHub issue and plan workflow in
[`docs/plans/README.md`](docs/plans/README.md): first review and approve the issue
plan, then implement it in a linked pull request. Each releasable PR must add a
user-facing entry under `[Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md). Select one
`semver:major`, `semver:minor`, or `semver:patch` label to set the bump; an unlabeled
PR defaults to minor. Use `semver:none` for changes that should not trigger a release.

Pull requests run the test suite, Ruff lint and formatting checks on changed Python
files, and changelog validation. When PRs merge to `main`, the release workflow groups
unreleased changes, updates the package and lockfile versions, archives the changelog
entries, and creates a `vX.Y.Z` tag and GitHub Release. Repository settings must allow
Actions to write contents and let `github-actions[bot]` update `main`.

For current project priorities, see [`docs/roadmap/roadmap.md`](docs/roadmap/roadmap.md).
