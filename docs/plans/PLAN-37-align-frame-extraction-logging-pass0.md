# Architecture Plan: [Issue #37] Align coarse frame extraction logging with Pass 0 domain classification

- **Issue**: [#37](https://github.com/tiwu/dive-clip-tui/issues/37)
- **Status**: Completed
- **Author**: Autonomous Agent / Contributor
- **Date**: 2026-11-01
- **Target Branch**: `feature/issue-37-align-frame-extraction-logging-pass0`
- **Proposed SemVer**: `semver:patch`

---

## 1. Problem Statement & Motivation

During video scanning when clip domain classification is enabled (`domain: "auto"`), the pipeline reuses the coarse frame extraction to avoid running FFmpeg twice. However, the logger currently logs `[Pass 1] Coarse scan @ 0.5 FPS` and `[Pass 1] Extracted 150 frames` before Pass 0 samples the frames for domain classification.

This produces confusing interleaved log output:
```text
[Pass 1] Coarse scan @ 0.5 FPS: x001.mp4
[Pass 1] Extracted 150 frames from x001.mp4
[Pass 0] Classifying domain for x001.mp4 across 3 sample frames...
[Pass 0] Classified x001.mp4 as domain 'generic'
[Pass 1] Summary x001.mp4: ...
```

This makes it appear as though Pass 1 has started before Pass 0, or that Pass 0 and Pass 1 are running out of order / concurrently.

---

## 2. Requirements & Acceptance Criteria

- [x] When Pass 0 classification is active (`detected_domain is None` and `classify_clips != "off"`):
  - Frame extraction is logged under `[Pass 0]`:
    - `[Pass 0] Extracting frames @ <fps> FPS: <filename>`
    - `[Pass 0] Extracted <count> frames from <filename>`
  - Pass 0 classifies the domain:
    - `[Pass 0] Classifying domain for <filename> across <n> sample frames...`
    - `[Pass 0] Classified <filename> as domain '<domain>'`
  - Pass 1 coarse inference start is logged explicitly:
    - `[Pass 1] Coarse scan @ <fps> FPS (<domain>): <filename>`
  - `PipelineState.current_activity` reflects `(pass 0 extract)` during extraction and `(pass 1 coarse)` during Pass 1 evaluation.
- [x] When the domain is already known (override via config, inferred from tags, or `classify_clips: "off"`):
  - Pass 0 logs its resolution (e.g. `[Pass 0] Using configured domain...`).
  - Frame extraction is logged under `[Pass 1]`:
    - `[Pass 1] Coarse scan @ <fps> FPS (<domain>): <filename>`
    - `[Pass 1] Extracted <count> frames from <filename>`
- [x] Preserve existing stats tracking (`phase1_stats.frames_extracted`) and checkpointing behavior.
- [x] All tests pass without regressions.

---

## 3. Implementation Details

In `scripts/02_scan_footage.py` (`process_clip()`):
1. Detect whether Pass 0 classification will run:
   ```python
   need_pass0_classification = detected_domain is None and classify_clips != "off"
   ```
2. Set the log prefix and activity string accordingly:
   ```python
   if need_pass0_classification:
       logger.info("[Pass 0] Extracting frames @ %.1f FPS: %s", coarse_fps, clip["file_name"])
       if state is not None:
           state.set_activity(f"scan: {clip['file_name']} (pass 0 extract, {coarse_fps} FPS)")
   else:
       logger.info(
           "[Pass 1] Coarse scan @ %.1f FPS (%s): %s",
           coarse_fps,
           detected_domain or "underwater",
           clip["file_name"],
       )
       if state is not None:
           state.set_activity(f"scan: {clip['file_name']} (pass 1 coarse, {coarse_fps} FPS)")
   ```
3. After `extract_frames`:
   ```python
   extract_prefix = "[Pass 0]" if need_pass0_classification else "[Pass 1]"
   logger.info("%s Extracted %d frames from %s", extract_prefix, len(coarse_frames), clip["file_name"])
   ```
4. Right before the coarse inference loop in Pass 1:
   ```python
   if not skip_coarse_inference and need_pass0_classification:
       logger.info(
           "[Pass 1] Coarse scan @ %.1f FPS (%s): %s",
           coarse_fps,
           detected_domain,
           clip["file_name"],
       )
       if state is not None:
           state.set_activity(f"scan: {clip['file_name']} (pass 1 coarse, {coarse_fps} FPS)")
   ```

---

## 4. Proposed SemVer & Changelog

- **SemVer Impact**: `semver:patch`
- **Draft Changelog Entry**:
  ```markdown
  ### Changed
  - Align coarse frame extraction logging with Pass 0 domain classification so logs and pipeline activities remain strictly chronological.
  ```
