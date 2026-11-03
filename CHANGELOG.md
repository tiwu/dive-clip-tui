# Changelog

All notable changes to this project are documented here. This changelog follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.25.0] - 2026-11-03

### Added
- Add `suggested_domain` cross-frame aggregation for generic clip classification in `classify_clip`.
- Refine generic coarse screening and highlight prompt criteria with structured rejection rules.

## [0.24.0] - 2026-11-02

### Added
- Add timestamped run directory archiving (`runs/run-YYYYMMDD-HHMMSS/`) and `latest` symlink creation to `dive-clip classify`.
- Add atomic incremental result saving after each clip during domain classification.
- Add `--resume` support to `dive-clip classify` to resume interrupted classification runs.

## [0.23.0] - 2026-11-02

### Added
- Add first-class `wildlife` domain with coarse screening prompt, specialist fine templates (`wildlife_animal`, `wildlife_bird`, `wildlife_action`), and fallback (`broad_wildlife`).
- Standardize universal coarse orientation rejection ("accidental, inverted, dangling, or unusable") and 1–10 fine quality rubrics across all domains.
- Add generic footage context inference (`infer_generic_context`, `create_prompt_candidate_for_generic`) and `--specialize-generic` CLI option in `dive-clip classify` with scan isolation.
- Expand multi-domain evaluation seed cases in `docs/evaluation/seed-cases.yaml` for underwater, motorcycle, wildlife, and generic footage.

## [0.22.0] - 2026-11-02

### Changed
- Enable hardware-accelerated NVENC video encoding with automatic fallback, fast input seeking for highlight renders, dynamic duration-based Pass 0 frame sampling with in-bag/dangling footage rejection, connection pooling in OllamaClient, and memory-bounded pipelined frame extraction prefetching.

## [0.21.0] - 2026-11-01

### Added
- Add standalone Pass 0 classification with reviewable confidence and uncertainty results that can be reused in later scans.

## [0.20.0] - 2026-11-01

### Added
- Add a CLI flow to review rendered clips and save keep, reject, and subject-correction feedback per run.

## [0.19.2] - 2026-11-01

### Fixed
- Prevent scan-time and final clip renders from racing, and retry incomplete clips safely after interruption.

## [0.19.1] - 2026-11-01

### Added
- Add root `.jscpd.json` configuration for automated copy-paste and duplication detection across pipeline scripts, utilities, and TUI.

### Changed
- Consolidate duplicated signal handling, FFmpeg concatenation, and artifact resolution into shared `utils/` helpers.

## [0.19.0] - 2026-11-01

### Added
- Add end-to-end highlight assembly to `dive-clip scan` via `--assemble` and introduce first-class `assemble` and `aftermovie` CLI subcommands.

## [0.18.2] - 2026-11-01

### Added
- Add repository-scoped agent definitions for Copilot (`.github/agents/`) and Antigravity skills (`_agents/skills/`) for Product Owner, Architecture Expert, Code Quality Review, and Project Review Team.

## [0.18.1] - 2026-11-01

### Changed
- Align coarse frame extraction logging with Pass 0 domain classification so logs and pipeline activities remain strictly chronological.

## [0.18.0] - 2026-11-01

### Added
- Add 3-stage domain-aware VLM scanning pipeline (Pass 0 clip domain classification, Pass 1 domain coarse screening, and Pass 2 subject-routed fine evaluation) supporting underwater, motorcycle, and generic highlight footage with custom prompt configuration in `config.yaml`.

## [0.17.0] - 2026-11-01

### Added
- Add intra-clip progress checkpointing for long video scans, periodically saving Pass 1 and Pass 2 frame evaluations to disk so interrupted scans can resume without re-running completed frames.

## [0.16.0] - 2026-11-01

### Added
- Add GitHub Actions step summaries to CI pipeline reporting test results, code coverage, Ruff lint status, and changelog/SemVer validation.
- Add `pytest-cov` dev dependency and extend unit tests covering entrypoints, Ollama connection checks, storyline planning, and TUI screens.

## [0.15.0] - 2026-11-01

### Fixed
- Fixed highlight clip collisions by including timestamp ranges in output filenames.
- Fixed FFmpeg extraction timeouts on long 4K video files by dynamically scaling extraction timeouts to video duration.
- Added guaranteed single-frame extraction fallback for clips shorter than the sampling interval.
- Merged consecutive detection windows of the same subject into continuous highlight clips.
- Added graceful fallback for `latest` symlink creation on exFAT and non-symlink filesystems.

## [0.14.1] - 2026-10-31

### Fixed
- Cleanly handle configuration errors across the CLI and pipeline scripts, reporting syntax errors with file and line details instead of printing raw Python tracebacks.

## [0.14.0] - 2026-10-31

### Added
- Add end-of-run processing statistics, Rich summary table, and `run_stats.json` artifact for full pipeline and standalone scanning.

## [0.13.0] - 2026-10-31

### Added
- Add self-contained run snapshots in `<output>/runs/<run_id>/` storing `pipeline_state.json`, `manifest.json`, `detections_refined.json`, `run.log`, and `highlights/` clips with an `<output>/latest` symlink.
- Support resuming previous runs via `dive-clip scan --resume [run_id_or_path]`, automatically continuing the latest run or targeting a specific run folder while restoring run parameters.
- Rename manifest artifact from `manifest_raw.json` to `manifest.json` with backward-compatible fallback for legacy manifests.

## [0.12.0] - 2026-10-31

### Changed
- Wire up `scan_parameters.use_ramdisk` configuration setting in `scripts/02_scan_footage.py` and `utils/ffmpeg_streamer.py`, allowing users to explicitly control RAM-disk usage (`/dev/shm`) and documenting its I/O performance and SSD wear reduction benefits.

## [0.11.0] - 2026-10-31

### Added
- Add per-run artifact archiving under `<output>/runs/run-<timestamp>/`, storing snapshots of the run log, `manifest_raw.json`, and `detections_refined.json` while maintaining latest files in the output directory.

## [0.10.1] - 2026-10-31

### Security
- Secure temporary frame directory in `/dev/shm` using restricted private directories (`0700` permissions) to prevent local symlink attacks and race conditions on multi-user systems.
- Sanitize `clip_id` against directory traversal in video scan, clip render, and aftermovie generation.
- Enforce strict escaping of backslashes and single quotes for FFmpeg concat demuxer lists and reject path names with newline characters.
- Sanitize exported CSV marker cells to neutralize spreadsheet formula injection vulnerabilities (`=`, `+`, `-`, `@`).

### Fixed
- Add signal and interrupt handlers (`SIGINT`, `SIGTERM`, `KeyboardInterrupt`) across all pipeline scripts (`dive-clip`, `scan_footage`, `render_clips`, `create_aftermovie`, `ingest_media`) to ensure clean shutdown, progress persistence, and temporary file cleanup without terminal tracebacks.

## [0.10.0] - 2026-10-31

### Added
- Split VLM screening into coarse gatekeeping and fine quality evaluation:
  Coarse pass instructs Qwen2.5-VL to reject upside-down, tumbling, and dangling
  camera footage (`detected: false`), enforced by a configurable
  `coarse_quality_threshold` (default `4`). Fine pass scores candidate highlights
  using a calibrated 1-10 rubric covering lighting, focus sharpness, camera
  stability, and composition.

### Fixed
- Mute background render logging on console to eliminate cursor displacement and duplicate progress bar lines during video scanning, while preserving complete logs in run logfiles.
- Add version reporting to `dive-clip` startup banner panel, log transcripts, and new `--version` CLI flag.

## [0.9.0] - 2026-10-30

### Added
- Add frame downscaling for fast coarse scanning, atomic per-clip detection
  persistence with resume support, and concurrent highlight clip rendering.
  Coarse scanning now downscales frames (configurable `coarse_scale`, default
  `"640:-2"`) to drastically reduce VLM vision tokens and HTTP transport latency.
  Detections are persisted atomically to `detections_refined.json` after each
  video completes rather than waiting for the entire batch. `dive-clip scan`
  now streams newly detected clips to a background render worker that cuts MP4
  highlights concurrently while subsequent videos are being scanned.

### Fixed
- Fix terminal line wrapping and progress bar interleaving during video scanning
  by sharing the process-wide Rich `Console` instance between `RichHandler` and
  `Progress`, and constraining task columns with bounded widths and text truncation.

## [0.8.0] - 2026-10-30

### Added
- Add Rich console logging and inline updating progress bars for video scanning.
  Terminal output displays color-coded log levels, styled pipeline step banners,
  and real-time inline progress during coarse and fine scanning passes with frame
  counters, detection hits, and ETA indicators. File transcripts keep plain text
  formatting.

## [0.7.0] - 2026-10-30

### Added
- `dive-clip scan` now writes a timestamped transcript of each run to
  `<output>/logs/run-<YYYYmmdd-HHMMSS>.log`, alongside the usual stderr output.
  Every run gets its own file and older logs are kept. The chained ingest, scan,
  and render steps receive the path so their log records land in the same
  transcript. Pass `--log-file PATH` to choose a location, or to write to a
  specific file when running the step scripts directly. Logging configuration now
  lives in `utils/logging_setup.py` instead of being repeated per script.

## [0.6.1] - 2026-10-30

### Added
- Add optional `general.input_dir` to the config. When set, `dive-clip scan` can
  run without `--input`, mirroring how `general.output_dir` already supplied the
  output folder. Explicit flags still win.

### Fixed
- `dive-clip scan` now passes the resolved output folder to
  `scripts/02_scan_footage.py` via a new `--output-dir` flag. The scan step
  previously re-derived its input and output paths from `general.output_dir`, so
  a `--output` that differed from the config silently read a manifest from a
  different folder and wrote detections where the render step did not look.

### Removed
- Remove `general.path_mapping`, which only rewrote media paths exported by the
  DaVinci Resolve manifest extractor (`scripts/01_extract_manifest.py`). The
  CLI-first ingest flow records the paths it is given on the command line, so
  the setting was dead config; the shipped value also mapped a prefix onto
  itself.

## [0.6.0] - 2026-10-30

### Added
- Honour the `OLLAMA_HOST` environment variable, which now takes precedence over
  `ollama.host` in the config file. A remote Ollama can be selected at runtime
  with `export OLLAMA_HOST=plotter.local:11434` instead of maintaining an
  untracked config file. Scheme-less values are normalised to `http://`, since
  the Ollama CLI accepts them but `requests` rejects them.

## [0.5.0] - 2026-10-30

### Added
- Add `--debug` flag to `dive-clip scan`, `scripts/00_ingest_media.py`,
  `scripts/02_scan_footage.py`, and `scripts/05_render_clips.py` to enable
  verbose logging with timestamps. In debug mode, every Ollama request is logged
  with its phase, elapsed time, and token counts (prompt and generated), and
  FFmpeg commands are shown in full. This makes the long waits during AI frame
  analysis and clip rendering observable in real time.
- Log every frame analyzed in debug mode with the full VLM verdict
  (`detected`, `confidence`, `passes_threshold`, `subject`/`quality`) and the
  per-frame elapsed time, so confidence thresholds can be tuned from real data
  instead of guesswork.
- Log per-pass summaries at info level: coarse pass frame counts, detections,
  over-threshold count, confidence min/avg/max, failed calls, and total seconds;
  fine pass per-window ranges, frame counts, pass/fail totals, and detection
  counts before and after deduplication.
- Log per-clip progress at info level for both scanning and rendering, so long
  runs stay observable without `--debug`.
- Record timestamps and progress in `.pipeline_state.json`: `updated_at` on every
  save, `activity`/`activity_at` heartbeats, and per-step
  `{step}_started_at`/`{step}_finished_at`/`{step}_duration_seconds`.
- Share one `--state-path` between the pipeline orchestrator and the scan/render
  workers so a single state file reflects all steps.
- Add `scripts/check_ollama_connection.py` to verify Ollama reachability and model
  loading before a long run.

### Changed
- Save `.pipeline_state.json` after every clip and window rather than only at step
  boundaries, so an interrupted run leaves an up-to-date checkpoint.

### Fixed
- Fix `dive-clip scan` crashing with `unrecognized arguments: --debug` when the
  ingest step actually ran; `scripts/00_ingest_media.py` now accepts `--debug`.
- Fix concurrent writes to the pipeline state file losing data: saves now merge
  the on-disk state with in-memory changes instead of overwriting the whole file,
  so the orchestrator no longer clobbers progress written by its workers.
- Fix `logging.basicConfig` being a no-op on repeated in-process runs (for example
  from the TUI or tests) by passing `force=True`.
- Remove a leftover debug-only setting from `config.yaml` and restore
  `ollama.host` to `http://localhost:11434`; use an untracked config file for
  machine-specific hosts.

## [0.4.0] - 2026-10-30

### Added
- Add broad marine-life fallback prompt for untagged clips, covering corals, worms, nudibranchs, fish, octopus, moray eels, sharks, turtles, whales, frogfish, crocodilefish, seahorses, and stonefish

## [0.3.0] - 2026-10-30

### Changed
- Raise the minimum supported Python version from `>=3.10` to `>=3.14`, aligning
  the Ruff and mypy targets, CI matrix, and release workflow runtime to the new
  floor. Python 3.14 and 3.15 are the currently validated bug-fix releases.
- Update GitHub Actions to recent versions (`checkout@v7`, `setup-python@v7`,
  `setup-uv@v10.3.3`) to resolve Node.js 20 deprecation runner warnings.

## [0.2.0] - 2026-10-30

### Added
- Add Keep a Changelog, SemVer, pull-request CI, and automated GitHub releases.
