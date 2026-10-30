# Changelog

All notable changes to this project are documented here. This changelog follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
