# Architecture Plan: [Issue #79] Scan Efficiency: Ollama Thinking Cap, Segmented Pre-screen Windows, Smarter Pass 1→2 Promotion, and Trace Resume Fix

- **Issue**: [#79](https://github.com/tiwu/dive-clip-tui/issues/79)
- **Status**: Plan Review
- **Author**: Autonomous Agent / Contributor
- **Date**: 2026-11-07
- **Target Branch**: `feature/issue-79-scan-efficiency-and-prescreen-refinements`

---

## 1. Problem Statement & Motivation

Analysis of run `run-20261106-123452` on the `Nosy Be` dataset (111 clips, ~2.83 hours of footage) against earlier baseline runs (`run-20261031-230736` at 7.1h and `run-20261103-022348` at 8.5h) revealed four compounding bottlenecks and one data-loss bug on resume:

1. **VLM Thinking Token Explosion & Undocumented Ollama Service Defaults (6.5x per-frame slowdown)**:
   - Under Ollama `0.40.15+`, `qwen3-vl:8b` (`RENDERER qwen3-vl-thinking`, `PARSER qwen3-vl-thinking`) ignores `"think": false` and defers `"format": "json"` grammar constraints until after `</think>` closes. While prompt evaluation remained fast (`1.71s` for ~1,478 tokens, which includes `--image-min-tokens 1024` injected automatically by Ollama's `qwen3vl` runner), token generation jumped from `~110` tokens/call (`1.9s`) to `1,100–2,100` tokens/call (`19s–36s`), increasing Pass 1 from `3.57s/frame` to `23.80s/frame` and Pass 2 from `4.85s/frame` to `28.01s/frame`.
   - Additionally, Ollama upgrades overwrite `/etc/systemd/system/ollama.service`, resetting `OLLAMA_KV_CACHE_TYPE=q8_0` and `OLLAMA_KEEP_ALIVE` unless persisted in a systemd drop-in override (`/etc/systemd/system/ollama.service.d/override.conf`).

2. **Pre-screen Overgeneralization on Multi-Minute Clips (`0%` intra-clip rejection)**:
   - After pre-screening was moved ahead of Pass 0 (`5857d47`), `screen_clip()` in `utils/prescreen.py` falls back to a single full-clip interval `[(0.0, clip_duration)]` whenever `intervals` is `None` and the clip has no GoPro `HiLight` tags (`0/111` clips in `Nosy Be` had `HiLight` tags).
   - Averaging luma, uniformity, Laplacian sharpness, gyro RMS, and gravity tilt across an entire 60s–396s clip dilutes localized garbage segments (e.g., dangling camera on boat deck, in-bag start/end, mask clearing, fin kicks), resulting in `prescreen_rej=0` across all intra-clip frames.

3. **Over-Permissive Pass 1 Coarse Scan & Touching-Window Snowballing (`89.4%` pass rate into Pass 2)**:
   - `BROAD_MARINE_TEMPLATE.coarse_prompt` marks `detected: true` whenever any coral, fish, diver action, or reef setting is visible in usable footage. Across `pass1_coarse_trace.json`, `429/497` (`89.4%`) coarse frames passed, with `272/429` (`63.4%`) classified as generic `story_role: "establishing"` shots (`"corals"`, `"diver action"`).
   - Because `coarse_fps = 0.25` samples every `4.0s` and `fine_padding_seconds = 2.0` expands each hit to `[t - 2.0, t + 2.0]`, consecutive hits at `t` and `t + 4.0` produce `[t - 2, t + 2]` and `[t + 2, t + 6]`. The merge check `start <= merged[-1][1]` (`t + 2 <= t + 2`) merges every consecutive hit into a single full-clip window, causing Pass 2 (`1.0 FPS`) to scan almost 100% of every reef clip (`4x` more VLM calls than Pass 1).

4. **Trace Collector Data Loss on `--resume`**:
   - In `scripts/02_scan_footage.py`, `prescreen_collector`, `pass0_collector`, `pass1_collector`, and `pass2_collector` are initialized as empty lists `[]` even when `--resume` is active. When `_flush_traces()` runs after the first post-resume clip, it overwrites `prescreen_trace.json`, `classifications.json`, `pass1_coarse_trace.json`, and `pass2_fine_trace.json` with only post-resume clips (losing trace records for clips completed before the resume).

---

## 2. Requirements & Acceptance Criteria

- [ ] **Documentation (`README.md`)**:
  - Document recommended systemd drop-in override settings (`/etc/systemd/system/ollama.service.d/override.conf`) including `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_KEEP_ALIVE=2h0m0s`, and `OLLAMA_FLASH_ATTENTION=1`.
  - Document why Ollama's `qwen3vl` runner passes `--image-min-tokens 1024` (~1,450–1,480 prompt tokens per 640×360 frame) and how to avoid `qwen3-vl-thinking` token explosion under Ollama `0.40.15+` (via non-thinking model variants, custom Modelfile `TEMPLATE`/`RENDERER` override, and `ollama.num_predict`).
- [ ] **VLM Generation Cap & Runaway Thinking Diagnostic (`utils/config.py`, `utils/ollama_client.py`, `config.yaml`)**:
  - Add optional `num_predict: int | None = None` to `OllamaConfig` (`utils/config.py`) and forward it in `OllamaClient.generate()` request `options` when configured.
  - Log a clear warning in `OllamaClient.generate()` when `think=False` is requested but `eval_count` exceeds a runaway threshold (e.g., `>= 512` generated tokens for a compact JSON call), alerting the user to `qwen3-vl-thinking` renderer overhead.
- [ ] **Segmented Intra-Clip Pre-screen Windows (`utils/prescreen.py`, `utils/config.py`, `scripts/02_scan_footage.py`, `config.yaml`)**:
  - Add `window_duration: float = 8.0` (must be `> 0`) to `PrescreenConfig` (`utils/config.py`) and `config.yaml`.
  - In `screen_clip()` (`utils/prescreen.py`), when `intervals is None` and `not hilight_tags`, partition `(0.0, clip_duration)` into contiguous windows of `window_duration` seconds (merging a trailing remainder shorter than `0.5 * window_duration` into the last window) instead of a single whole-clip `(0.0, clip_duration)` window.
  - Preserve the `prescreen_trace.json` contract: one `PrescreenRecord` per candidate window, never per raw sample.
  - When `prescreen.gate_mode == "enabled"`:
    - If **all** windows in a clip are rejected, `_prescreen_rejects_whole_clip()` still skips the whole clip before Pass 0.
    - If **some** windows in a clip are rejected, Pass 1 skips any coarse frame whose timestamp `ts` falls inside a rejected pre-screen window (`start_s <= ts < end_s`), incrementing `prescreen_rejected` metrics accordingly.
- [ ] **Smarter Pass 1 → Pass 2 Promotion & Window Merging (`scripts/02_scan_footage.py`, `utils/prompt_templates.py`, `utils/config.py`, `config.yaml`)**:
  - Refine `BROAD_MARINE_TEMPLATE.coarse_prompt` (`utils/prompt_templates.py`) to explicitly reject in-bag/covered-lens frames, dangling camera on boat/surface, featureless water, heavy fin-wash, and monotonous distant reef swim-overs with no focal point.
  - Add `coarse_establishing_quality_threshold: int = 7` (`ge=1, le=10`) to `ScanningConfig` (`utils/config.py`) and `config.yaml`: in Pass 1, frames with `story_role == "establishing"` require `quality_score >= coarse_establishing_quality_threshold` to promote a window to Pass 2, whereas `"subject"`, `"interaction"`, and `"action"` frames continue to promote at `coarse_quality_threshold` (default `4`).
  - Fix Pass 1 → Pass 2 window merging in `scripts/02_scan_footage.py`: group consecutive passing coarse timestamps using `coarse_merge_gap_seconds` (defaulting to `1.0 / coarse_fps` so adjacent passing coarse samples merge into a cluster, while non-adjacent samples separated by a rejected sample do not bridge via padding), and apply `fine_padding_seconds` around each merged cluster.
- [ ] **Preserve Trace Files on `--resume` (`scripts/02_scan_footage.py`)**:
  - When `resume=True`, pre-populate `prescreen_collector`, `pass0_collector`, `pass1_collector`, and `pass2_collector` from existing `prescreen_trace.json`, `classifications.json`, `pass1_coarse_trace.json`, and `pass2_fine_trace.json` in `output_dir`, retaining records for clips already in `state.completed_clips` so `_flush_traces()` preserves the complete run history.
- [ ] Automated tests cover all acceptance criteria.
- [ ] All relevant documentation (`README.md`, `CHANGELOG.md`, `config.yaml`) has been updated.

---

## 3. Technical Design

### 3.1 Segmented Pre-screen Window Partitioning (`utils/prescreen.py`)

Currently in `screen_clip()`:
```python
if intervals is not None:
    windows = [(max(0.0, float(s)), min(clip_duration, float(e))) for s, e in intervals if e > s]
elif hilight_tags:
    windows = [
        (max(0.0, float(t) - half), min(clip_duration, float(t) + half))
        for t in sorted(hilight_tags)
    ]
else:
    windows = [(0.0, clip_duration)] if clip_duration > 0 else []
```

We will introduce a helper `_build_default_windows(clip_duration: float, window_duration: float) -> list[tuple[float, float]]`:
- If `clip_duration <= 0`: return `[]`.
- If `window_duration <= 0` or `clip_duration <= window_duration`: return `[(0.0, clip_duration)]`.
- Otherwise, slice `0.0 .. clip_duration` in steps of `window_duration`. If the final remainder `clip_duration - last_start < 0.5 * window_duration` and there is at least one prior window, extend the last window to `(prev_start, clip_duration)` so short tail fragments still have enough duration for 2 visual samples and telemetry RMS.

In `scripts/02_scan_footage.py`:
- When `prescreen.enabled` is true and `prescreen.gate_mode == "enabled"`, build a list of rejected intervals `[(r.start_s, r.end_s) for r in clip_prescreen_records if not r.passed]`.
- During Pass 1 coarse frame iteration, before sending frame `(frame_idx, ts, img_bytes)` to Ollama, check if `ts` falls in any rejected interval (`start_s <= ts < end_s` or `ts == end_s == clip_duration`). If so, skip the VLM call for that timestamp and increment `clip_metric.prescreen_rejected`.

### 3.2 Pass 1 → Pass 2 Promotion & Cluster-Then-Pad Window Merging (`scripts/02_scan_footage.py`)

1. **Role-Aware Promotion Gate**:
   - A Pass 1 coarse response promotes timestamp `ts` to Pass 2 when:
     - `res.get("detected", False)` is true, AND
     - `confidence >= coarse_confidence_threshold`, AND
     - If `res.get("story_role") == "establishing"`: `quality_score >= coarse_establishing_quality_threshold` (default `7`),
     - Else (`"subject"`, `"interaction"`, `"action"`, or unspecified): `quality_score >= coarse_quality_threshold` (default `4`).
   - All coarse responses (promoted or not) are still recorded in `pass1_coarse_trace.json` with a boolean `"promoted_to_pass2"` field for observability.

2. **Cluster-Then-Pad Window Construction**:
   - Instead of padding every point `[ts - pad, ts + pad]` first and merging on `start <= merged[-1][1]` (which turns adjacent 4s samples with 2s padding into an unbroken chain and nearly bridges 8s gaps), we:
     1. Sort promoted timestamps `t_0, t_1, ..., t_k`.
     2. Group consecutive promoted timestamps into clusters where `t_i - t_{i-1} <= step_gap` (with `step_gap = (1.0 / coarse_fps) * 1.1` so adjacent passing coarse samples merge into `[cluster_start, cluster_end]`, while a rejected coarse frame in between breaks the cluster).
     3. Expand each cluster by `fine_padding_seconds`: `[max(0.0, cluster_start - pad), min(duration_sec, cluster_end + pad)]`.
     4. Merge any resulting windows that strictly overlap (`start < merged[-1][1]`).

### 3.3 Trace Collector Hydration on `--resume` (`scripts/02_scan_footage.py`)

Add a helper `_load_existing_traces_for_resume(output_dir: Path, completed_clips: set[str])`:
- For each of `prescreen_trace.json`, `classifications.json`, `pass1_coarse_trace.json`, and `pass2_fine_trace.json`:
  - If the file exists and parses as a JSON list of dicts, retain all entries whose `clip_id` (or `filename` fallback) is in `completed_clips`.
  - If a file is missing or corrupt, log a warning and start that collector with `[]`.

---

## 4. Change Log and Release Impact

- **Change category**: Changed / Fixed
- **Proposed SemVer impact**: `semver:minor`
- **Draft changelog entry**:
  - **Changed**: Partition clips without HiLight tags into fixed-duration pre-screen windows (`prescreen.window_duration`, default `8.0s`), add role-aware Pass 1→Pass 2 promotion (`scanning.coarse_establishing_quality_threshold`, default `7`) with cluster-then-pad window merging, support `ollama.num_predict` token caps, and document systemd Ollama service overrides.
  - **Fixed**: Preserve existing `prescreen_trace.json`, `classifications.json`, `pass1_coarse_trace.json`, and `pass2_fine_trace.json` records when resuming a scan with `--resume`.

---

## 5. Test-Driven Development (TDD) Plan

1. **Red (Tests First)**:
   - `tests/test_prescreen.py`:
     - Verify `screen_clip()` partitions a clip without `hilight_tags` into `window_duration` segments (e.g., a `25.0s` clip with `window_duration=8.0` yields `[(0.0, 8.0), (8.0, 16.0), (16.0, 25.0)]`).
     - Verify partial-clip rejection: when window 1 fails pre-screen and window 2 passes, `_prescreen_rejects_whole_clip()` is `False`, and Pass 1 skips coarse frames in `[0.0, 8.0)` while scanning `[8.0, 16.0)`.
   - `tests/test_scan_footage.py`:
     - Verify Pass 1 → Pass 2 role-aware promotion: `story_role="establishing"` with `quality_score=5` is not promoted when `coarse_establishing_quality_threshold=7`, whereas `story_role="subject"` with `quality_score=5` is promoted.
     - Verify cluster-then-pad window merging: promoted hits separated by a rejected coarse sample do not merge unless their padded windows strictly overlap.
     - Verify `--resume` trace preservation: existing `prescreen_trace.json`, `classifications.json`, `pass1_coarse_trace.json`, and `pass2_fine_trace.json` records for completed clips are retained after resuming on remaining clips.
   - `tests/test_ollama_client.py` & `tests/test_config.py`:
     - Verify `OllamaConfig.num_predict` validation and forwarding in `OllamaClient.generate()` payload `options`.
2. **Green (Implementation)**:
   - Update `utils/config.py`, `utils/ollama_client.py`, `utils/prescreen.py`, `utils/prompt_templates.py`, `scripts/02_scan_footage.py`, `config.yaml`, and `README.md`.
3. **Refactor & Quality**:
   - Run focused tests, then full suite (`uv run python -m pytest`).
   - Run `uv run ruff check` and `uv run ruff format --check` on changed files.

---

## 6. Verification & Rollout Checklist

- [ ] Tests pass: `uv run python -m pytest`
- [ ] Changed Python files pass Ruff lint and formatting checks
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR has no conflicting SemVer labels (unlabeled defaults to minor)
- [ ] PR submitted with `Closes #79` and plan linked
