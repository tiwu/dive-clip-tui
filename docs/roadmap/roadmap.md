# Product Roadmap

**Status:** Active
**Updated:** 2026-11-04

## Product direction

Build a local-first video understanding pipeline: classify what kind of footage a
video contains, then run a specialized scan to find useful moments. Start with
diving and motorcycle footage, but keep the classifier open to other video
domains. Highlight generation is a workflow built on classification, not the
product's only identity.

Use **domain** for a broad kind of footage (underwater, motorcycle, wildlife,
generic). Use **subject** for what appears within a video (pufferfish, coral,
rider). “Source” can also mean camera or file origin, so use it only for that
meaning.

## Current foundation

- Pass 0 already classifies unknown clips by domain during a full scan.
- The CLI already renders detected clips in the background while scanning.
- The aftermovie renderer supports 9:16 output by padding footage to fit; it
  does not yet fill the portrait frame with a useful crop.
- Runs are archived and resumable. CLI commands can assemble clips into an
aftermovie.
- Highlight selection applies Maximal Marginal Relevance (MMR) for diversity with
  chronological assembly sort order, contiguous shot merging, and per-stage decision
  traceability artifacts.
- Scan-time rendering is race-safe: one writer owns each output, and interrupted
  renders are retried instead of appearing as completed clips.
- Pass 0 runs on its own, and a human review loop records clip feedback per run.
  Both reuse the same run-artifact contracts as a full scan.
- Ingest records optional action-camera telemetry (GoPro GPMF `gpmd`, DJI `djmd`,
  Insta360) and manual MP4 `hmmt` HiLight tags. A pre-VLM screening funnel can
  score candidate windows from that telemetry plus cheap FFmpeg signals, write its
  decisions to `prescreen_trace.json`, and opt into 2×2 temporal contact sheets for
  Pass 1. Screening is shipped disabled by default; scans behave as before until it
  is turned on.
- The existing Textual TUI is an unmaintained prototype. A future TUI starts
  from a clean design after its CLI and data contracts are stable.

## Prioritized roadmap

### 1. Turn on pre-VLM screening and widen telemetry coverage

The screening funnel (telemetry gates, FFmpeg blur/brightness/audio signals, 2×2
contact sheets, `prescreen_trace.json`) ships disabled. Run it in advisory mode on
representative dive and motorcycle footage, compare its decisions against the
review-loop judgments, and only then consider enabling rejection. Extend decoding
to the telemetry containers ingest already detects, and decide whether HiLight
tags should seed coarse candidate windows instead of only aligning screening
windows.

**Done when:** Advisory `prescreen_trace.json` records show measured VLM call
savings on representative footage with no loss of reviewed highlights, at least
one non-GoPro telemetry format is decoded, and the documented default is backed by
that evidence.

### 2. Grow the human-judged evaluation set

The review loop records per-clip feedback, but the
[evaluation seed set](../evaluation/seed-cases.yaml) still needs cases covering
false positives, wrong subject labels, false negatives, and reported confidence.
Keep reviewed examples with the model identity, prompt/template revision, and a
secret-safe configuration fingerprint that produced them. Use this evidence to
compare prompt and classifier changes before adoption; do not silently train or
rewrite prompts.

**Done when:** A repeatable local evaluation set covers false positives, wrong
subject labels, false negatives, and reported confidence. Runs record enough
secret-safe provenance to reproduce comparisons, and focused regression tests
protect existing behavior.

### 3. Specialize generic footage with evaluated prompts

Add domains based on real footage and evaluation examples, not an open-ended
category list. For footage that remains generic, infer likely context and create
subject-specific coarse and fine prompt candidates. Treat generated prompts as
candidates to evaluate, not trusted prompts that change behavior automatically.

**Done when:** Each supported domain has a clear label, representative
validation clips, and prompts that improve detection without regressing existing
domains.

### 4. Optimize measured bottlenecks

Profile representative workloads before changing concurrency, frame sampling,
or memory use. Measure the now race-safe scan/render overlap; pursue more
parallelism only when benchmarks show a safe end-to-end gain. Treat screening
savings from item 1 as part of the same measurement so VLM calls, wall-clock
time, and quality are compared together.

**Done when:** Benchmarks show a repeatable improvement on representative
hardware and footage, with output and resume behavior unchanged.

### 5. Improve portrait and social export

Replace padded 9:16 output with useful portrait framing while keeping 16:9 as
the default. Add subject-aware reframing only after basic vertical export works
on real dive and motorcycle clips.

**Done when:** The CLI produces playable 1080×1920 output, preserves landscape
behavior, and keeps important action visible on representative clips.

### 6. Revisit product name and rebuild the TUI

Once classifier-first workflows and supported domains prove useful, choose a
name that describes the broader product. Then design a new TUI around the stable
CLI and run-artifact contracts. Do not extend the current prototype as the new
architecture.

**Done when:** The name reflects the validated product direction, and a new TUI
can classify footage, run specialized scans, show progress, review results, and
render selected clips through supported workflows.

## Parked

- Photo and EXIF processing.
- Direct publishing or uploads to social platforms.
- Web UI, mobile apps, Docker, and cloud hosting.

Revisit parked items only when user needs or project direction change.

## Planning and status

This roadmap is a prioritized direction, not approval to implement every item.
Plan each feature through the issue and review workflow in
[`../plans/README.md`](../plans/README.md). Update this file when priorities or
completion status change.
