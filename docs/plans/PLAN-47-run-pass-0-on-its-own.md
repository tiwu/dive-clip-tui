# Architecture Plan: Issue #47 — Run Pass 0 on its own

## Problem and approach

Pass 0 classifies video domains only inside the full scan. Users cannot classify
unknown footage without also running coarse and fine highlight detection.

Add a standalone `dive-clip classify` command that ingests a folder, samples only
the configured number of representative frames per video, and runs existing
Pass 0 domain classification without starting highlight scans. Print each
domain, confidence, and uncertainty status. Save results as a separate JSON file
so users can inspect it and reuse reliable classifications in a later scan.

Extend `dive-clip scan` with `--classifications <path>`. Match saved records to
ingested clips by stable clip identity and reuse confident labels. Ignore saved
uncertain labels and let scan-time Pass 0 classify those clips again. Preserve
existing scan behavior when this option is absent.

Aggregate frame-level classification confidence and domain agreement. Mark a
result uncertain when there are no usable responses, mean confidence for the
winning domain is below 0.6, or sampled frames disagree.

## Requirements and acceptance criteria

- `dive-clip classify` classifies videos from an input folder without starting
  coarse or fine highlight detection.
- Classification samples no more than `classify_frames_count` representative
  frames per video, using the existing config setting and a bounded sampling
  strategy.
- The command supports `--input`, `--output`, and `--config`. It saves a
  separate JSON classification file, defaulting to
  `<general.output_dir>/classifications.json`; `--output` overrides that path.
- The command prints each clip's domain, aggregate confidence, and uncertainty
  status, plus a useful summary. It does not prompt users to edit labels.
- A classification is uncertain when no sampled frame yields a usable response,
  mean confidence for the winning domain is below 0.6, or usable sampled frames
  disagree on domain.
- Saved results retain stable clip identity and enough manifest context to match
  and inspect them later. Validate JSON instead of silently accepting malformed
  records.
- `dive-clip scan --classifications <path>` reuses confident saved labels.
  Without this option, existing scan behavior remains unchanged.
- Scan ignores uncertain saved labels and performs its normal Pass 0
  classification for those clips. Unmatched saved entries are reported.
- Classification data stays separate from detections and pipeline checkpoints;
  no detection or checkpoint schema migration is required.
- Focused tests cover bounded sampling, no highlight inference in classify,
  confidence and disagreement, saved output, confident-label reuse, uncertain
  reclassification, malformed or unmatched records, and default scan behavior.
- README documents classify and reuse commands. The implementation PR adds an
  `[Unreleased]` changelog entry.

## Architecture and data flow

Keep CLI orchestration in `scripts/dive_clip.py` and Pass 0 extraction and
classification logic in `scripts/02_scan_footage.py`. Share classification
logic instead of implementing another prompt or Ollama path. Reuse manifest
ingestion, frame extraction, domain registry, Ollama client, and
`classify_frames_count`.

```text
dive-clip classify
  -> ingest input folder into clip manifest
  -> sample representative frames (bounded by classify_frames_count)
  -> classify frames with existing Pass 0 logic
  -> save and print per-clip domain, confidence, and uncertainty

dive-clip scan --classifications <path>
  -> ingest input folder
  -> load and validate classification records
  -> reuse confident labels by clip identity
  -> run existing Pass 0 for clips without a confident saved label
  -> continue existing coarse/fine scan
```

Keep classification aggregation explicit: retain usable frame-level domain and
confidence results, derive the winning domain and its mean confidence, and mark
disagreement when usable frames select different domains. Missing or
unparseable responses must not become confident `generic` labels. Represent
unavailable confidence clearly and mark that result uncertain.

Keep output records in a focused classification model alongside shared models.
Use manifest `clip_id` as primary identity and preserve file name/path for
review. Reject malformed files with a clear error; report valid records that do
not match an input clip.

For scan reuse, merge each validated confident classification into the
in-memory manifest clip before invoking the existing scanner. Do not rewrite the
source manifest or detections. If a clip has no saved record or only an
uncertain record, existing Pass 0 behavior handles it.

## Change log and release impact

- **Change category**: Added
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**: Add standalone Pass 0 classification with
  reviewable confidence and uncertainty results that can be reused in later
  scans.

## Test-driven development

1. **Red**: Test CLI arguments, input validation, configured defaults, and
   output JSON. Test bounded representative sampling and that classify never
   calls coarse or fine inference. Test confident domain aggregation, low mean
   confidence, disagreement, and unusable responses. Test confident reuse,
   uncertain reclassification, unmatched records, malformed files, and
   unchanged behavior without `--classifications`.
2. **Green**: Add classify CLI flow and documentation. Extract/share Pass 0
   classification and aggregation logic with bounded sampling and validated
   records. Add optional classification loading to scan and reuse only confident
   matched labels.
3. **Refactor and verification**: Run focused CLI, scanner, and model tests;
   run `uv run python -m pytest`; run Ruff lint and format checks on changed
   Python files.

## Verification and rollout checklist

- [x] Classify processes an input folder without highlight detection and emits
      reviewable, validated results.
- [x] Later scans reuse confident records and reclassify uncertain ones.
- [x] Scans without `--classifications` retain current behavior.
- [x] Focused and full test suites pass.
- [x] Changed Python files pass Ruff lint and formatting checks.
- [x] README documents standalone classification and scan reuse.
- [x] Implementation PR adds a changelog entry, uses `semver:minor`, and
      includes `Closes #47` with this plan linked.
