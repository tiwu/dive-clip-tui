# Product Roadmap

**Status:** Active
**Updated:** 2026-11-01

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
- The existing Textual TUI is an unmaintained prototype. A future TUI starts
  from a clean design after its CLI and data contracts are stable.

## Prioritized roadmap

### 1. Make scan-time rendering safe

Background rendering already overlaps with scanning, but the worker can outlive
its two-second shutdown wait while the final render starts. Ensure a single writer
owns each output and that incomplete renders cannot appear as completed clips.
Keep the existing overlap only if it remains correct and resume-safe.

**Done when:** Tests prove scan-time and final rendering cannot race on the same
output, interrupted renders are safely retried, and each detection produces one
valid clip after a normal run or resume.

### 2. Build a human-judged evaluation loop

Let users judge highlight relevance and subject-label correctness separately,
and identify interesting moments the scan missed. Keep reviewed examples with
the model identity, prompt/template revision, and a secret-safe configuration
fingerprint that produced them. Use this evidence to compare prompt and
classifier changes before adoption; do not silently train or rewrite prompts.
Add focused behavioral regression tests for each change, including resume and
output behavior where affected.

Initial user-reviewed cases live in the local-safe
[evaluation seed set](../evaluation/seed-cases.yaml). Keep run-specific examples,
expected judgments, and unresolved questions there; keep raw videos and
machine-specific paths out of the repository. Track implementation work in a
GitHub issue linked to this roadmap.

**Done when:** A repeatable local evaluation set covers false positives, wrong
subject labels, false negatives, and reported confidence. Runs record enough
secret-safe provenance to reproduce comparisons, and focused regression tests
protect existing behavior.

### 3. Run Pass 0 on its own

Add a quick classify-only workflow for folders with unknown or unclassified
videos. It labels footage by broad domain without running coarse/fine highlight
scans. Save results so users can inspect and reuse classifications in a later
scan. Sample only enough frames to keep this pass fast, and report uncertain
classifications rather than presenting guesses as facts.

**Done when:** A user can classify a folder without starting highlight detection,
review the labels and uncertainty, and use the saved results in a later scan.

### 4. Expand domains and specialize generic footage

Add domains based on real footage and evaluation examples, not an open-ended
category list. For footage that remains generic, infer likely context and create
subject-specific coarse and fine prompt candidates. Treat generated prompts as
candidates to evaluate, not trusted prompts that change behavior automatically.

**Done when:** Each supported domain has a clear label, representative
validation clips, and prompts that improve detection without regressing existing
domains.

### 5. Optimize measured bottlenecks

Profile representative workloads before changing concurrency, frame sampling,
or memory use. Measure the existing scan/render overlap after its correctness is
secured; pursue more parallelism only when benchmarks show a safe end-to-end gain.

**Done when:** Benchmarks show a repeatable improvement on representative
hardware and footage, with output and resume behavior unchanged.

### 6. Improve portrait and social export

Replace padded 9:16 output with useful portrait framing while keeping 16:9 as
the default. Add subject-aware reframing only after basic vertical export works
on real dive and motorcycle clips.

**Done when:** The CLI produces playable 1080×1920 output, preserves landscape
behavior, and keeps important action visible on representative clips.

### 7. Revisit product name and rebuild the TUI

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
