# Changelog

All notable changes to this project are documented here. This changelog follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add `--debug` flag to `dive-clip scan`, `scripts/02_scan_footage.py`, and
  `scripts/05_render_clips.py` to enable verbose logging with timestamps. In
  debug mode, every Ollama request is logged with its phase, elapsed time,
  and token counts (prompt and generated), and FFmpeg commands are shown in
  full. This makes the long waits during AI frame analysis and clip rendering
  observable in real time.

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
