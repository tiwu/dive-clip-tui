# Changelog

All notable changes to this project are documented here. This changelog follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
