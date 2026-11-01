> **Archived — historical reference only.**
>
> This document described the original roadmap to a full GoPro Quick-like experience.
>
> Use the current [product roadmap](../roadmap/roadmap.md) for project direction.
>

# Roadmap: Automated AI Highlight Reel Creator

**Project:** dive-clip-ai
**Goal:** Transform raw holiday/event media (videos + photos) into curated highlight reels — long-form aftermovies, short-form social clips, and subject-specific videos — with a GoPro Quick-like "one-click" experience.

---

## 1. Current State Assessment

### What exists today (v0.1.0)

| Component | File | Purpose |
|-----------|------|---------|
| Manifest extractor | `scripts/01_extract_manifest.py` | Runs inside DaVinci Resolve, exports clips with keywords to `manifest_raw.json` |
| Coarse-to-fine VLM scanner | `scripts/02_scan_footage.py` | Qwen2.5-VL via Ollama, two-pass scanning (0.5 FPS coarse → 2.0 FPS fine) |
| EDL/CSV generator | `scripts/03_generate_edl.py` | Converts detections to CMX 3600 EDL + CSV markers for Resolve import |
| Storyline planner | `scripts/04_plan_storyline.py` | Cloud LLM (Claude/Gemini) builds narrative edit plan from detections |
| Pydantic models | `utils/models.py` | `ClipManifest`, `ManifestOutput`, `Detection`, `RefinedDetection`, `DetectionsRefined` |
| FFmpeg streaming | `utils/ffmpeg_streamer.py` | Frame extraction with RAM-disk support and corrupt-frame recovery |
| Timecode math | `utils/timecode_math.py` | SMPTE timecode conversion (drop-frame, non-drop-frame, multiple FPS) |
| Prompt templates | `utils/prompt_templates.py` | Qwen2.5-VL prompts for pelagics/macro/diver categories |
| Config | `config.yaml` | Paths, Ollama endpoint, scan thresholds, cloud storyline settings |

### Current pipeline
```
Resolve Media Pool → 01_extract_manifest → manifest_raw.json
                                              ↓
                                      02_scan_footage (Ollama Qwen2.5-VL)
                                              ↓
                                      detections_refined.json
                                              ↓
                                  03_generate_edl → EDL + CSV markers
                                              ↓
                                  04_plan_storyline → storyline_edit_guide.md
```

### Key limitations vs. the target vision

1. **Requires DaVinci Resolve** — cannot ingest a raw folder of videos/photos directly
2. **Video-only** — no photo support
3. **No actual rendering** — produces markers/EDL, not finished video files
4. **No subject-specific clip grouping** — no "turtle video" or "shark video" outputs
5. **No social media optimization** — no 9:16 vertical, no duration limits, no platform-specific exports
6. **No long-form aftermovie assembly** — no automated timeline construction
7. **No audio/music integration** — no automatic music sync, beat detection, or audio ducking
8. **No transitions/effects/color grading** — purely marker-based
9. **No one-click experience** — requires manual steps: Resolve console → Ubuntu worker → Resolve import → cloud API
10. **No progress tracking/resume** — no checkpointing for long batch jobs
11. **No thumbnail/preview generation** — no visual preview of suggested clips
12. **No face/person-aware framing** — no auto-reframing for vertical social clips
13. **No multi-language/localization** — hardcoded English/Dutch keywords
14. **No web UI or simple CLI** — requires technical knowledge to run scripts
15. **No batch project management** — single project at a time

---

## 2. Target Vision (GoPro Quick-like experience)

**User flow:**
1. Point the tool at a folder of videos + photos
2. Click "Create Highlights" (or run one command)
3. AI automatically: scans media, identifies subjects (turtle, shark, diver, etc.), groups clips by theme, selects best moments
4. Outputs: long-form aftermovie (5–10 min), short-form social clips (15–60s, 9:16), subject-specific videos
5. User reviews, tweaks, exports

**Core principles:**
- **Zero-config default:** sensible defaults for frame rates, durations, music styles
- **Automated everything:** scan → select → edit → render with minimal user input
- **Subject-aware:** tag-based grouping (turtle, shark, macro, diver, landscape)
- **Platform-native:** vertical for TikTok/Reels/Shorts, horizontal for YouTube/Vimeo
- **Local-first:** run on user's machine with optional cloud LLM for storyline

---

## 3. Gap Analysis & Required Capabilities

### P0 — Must have (for MVP highlight reel creator)

| Gap | Description | Priority |
|-----|-------------|----------|
| **Folder ingestion** | Scan a directory tree for videos + photos directly, no Resolve required | P0 |
| **Photo support** | Extract EXIF, detect subjects in stills, include in slideshow/montage sections | P0 |
| **Actual clip rendering** | FFmpeg-based clip extraction from detection windows into standalone MP4 files | P0 |
| **Long-form aftermovie assembly** | Concatenate selected clips with transitions, music, titles into one video | P0 |
| **Short-form social clips** | 9:16 vertical crops, 15–60s duration, face-aware reframing, platform presets | P0 |
| **Subject-specific videos** | Group detections by subject (turtle, shark, macro, diver) and render per-subject reels | P0 |
| **Simple CLI entry point** | Single command: `dive-clip --input ./media --output ./highlights` | P0 |
| **Config presets** | "Aftermovie", "Social Reels", "Subject Reels" preset configurations | P0 |

### P1 — Should have (for polished experience)

| Gap | Description | Priority |
|-----|-------------|----------|
| **Web UI** | Browser-based review/trim of AI-suggested clips before render | P1 |
| **Thumbnail/preview generation** | Contact sheets and hover previews for suggested clips | P1 |
| **Music/audio integration** | Auto music sync, beat detection, audio ducking for voiceover | P1 |
| **Transitions & effects** | Cross-dissolves, fades, color grade presets applied automatically | P1 |
| **Face/person-aware reframing** | Auto-crop vertical clips to keep people centered | P1 |
| **Progress tracking & resume** | Checkpointing so long scans can be interrupted and resumed | P1 |
| **Multiple project support** | Manage several holiday/event projects simultaneously | P1 |
| **Export presets** | YouTube, TikTok, Instagram, Vimeo presets with correct resolution/fps/bitrate | P1 |

### P2 — Nice to have (v1.5+)

| Gap | Description | Priority |
|-----|-------------|----------|
| **Cloud LLM storyline** | Optional cloud-based narrative planning (already partially exists) | P2 |
| **Voiceover generation** | TTS narration based on storyline | P2 |
| **Auto color grading** | Underwater color correction (red filter simulation, white balance) | P2 |
| **Duplicate/similar shot detection** | Remove near-identical clips from selection | P2 |
| **Batch processing** | Process multiple folders/projects overnight | P2 |
| **Mobile companion app** | iOS/Android app to trigger and review renders | P2 |
| **Platform publishing** | Direct upload to YouTube/TikTok/Instagram | P2 |

---

## 4. Phased Roadmap

### Phase 1: Foundation — Direct media ingestion + clip rendering
**Goal:** Replace Resolve dependency with direct folder scanning, render actual video clips from AI detections.

**Key changes:**
- New `scripts/00_ingest_media.py` — walk input folder, enumerate videos + photos, extract basic metadata (duration, FPS, resolution, EXIF)
- New `ingest.py` — unified manifest builder (replaces Resolve dependency), supports video + photo entries
- Extend `02_scan_footage.py` — accept manifest directly (no Resolve), add photo scanning (EXIF + VLM on stills)
- New `clip_renderer.py` — FFmpeg-based extraction of detection windows into MP4 clips
- Update `utils/models.py` — add `PhotoManifest`, `RenderedClip`, `RenderJob` models
- Add `config/presets/` — aftermovie, social, subject preset YAML files

**Deliverable:** Point at a folder → get rendered highlight clips as MP4 files

### Phase 2: Automated assembly — aftermovie + social reels
**Goal:** Auto-assemble long-form aftermovie and short-form social clips from rendered highlights.

**Key changes:**
- New `aftermovie_builder.py` — timeline assembly with transitions, music sync, titles
- New `social_reel_builder.py` — 9:16 vertical crops, face-aware reframing, duration limits
- New `subject_grouper.py` — cluster detections by subject (turtle/shark/macro/diver), render per-subject reels
- New `audio_mixer.py` — background music, audio ducking, beat-matched cuts
- Update `utils/ffmpeg_streamer.py` — add concatenation, vertical crop, transition filter support
- New `utils/transition_effects.py` — cross-dissolve, fade-in/out, dip-to-white

**Deliverable:** One-click aftermovie + social reels output

### Phase 3: Polish — UI, previews, resume
**Goal:** Make it usable by non-technical users with review and iteration.

**Key changes:**
- New `web_ui/` — FastAPI/Streamlit app for reviewing AI suggestions, trimming clips, reordering
- New `preview_generator.py` — contact sheets, animated thumbnails, scrubbable previews
- New `checkpoint.py` — save/resume state for long scans and renders
- New `project_manager.py` — multiple project support, project metadata
- CLI improvements — `dive-clip init`, `dive-clip scan`, `dive-clip render`, `dive-clip review`

**Deliverable:** Web UI for reviewing and tweaking before final render

### Phase 4: Scale — batch, export, cloud
**Goal:** Production-ready for multiple projects and platform publishing.

**Key changes:**
- Batch processing queue with concurrency limits
- Platform-specific export presets (YouTube 4K, TikTok 1080×1920, Instagram 1080×1350)
- Optional cloud LLM storyline integration (improve existing `04_plan_storyline.py`)
- Direct platform publishing (YouTube API, TikTok API)
- Docker container for one-command setup

**Deliverable:** Full GoPro Quick-like experience

---

## 5. Detailed Technical Recommendations

### 5.1 Folder ingestion (replaces Resolve manifest)

Create `scripts/00_ingest_media.py`:

```python
# Walk input directory, collect videos + photos
# Extract: duration, fps, resolution, codec, EXIF (date, GPS, camera)
# Output: manifest.json (unified format for video + photo)
# No DaVinci Resolve required
```

### 5.2 Subject-specific video grouping

Extend `utils/models.py`:

```python
class SubjectCluster(BaseModel):
    subject: str  # "turtle", "shark", "macro", "diver", "landscape"
    clips: list[RefinedDetection]
    total_duration: float
    representative_clip: str  # best clip for thumbnail
```

New `scripts/05_group_by_subject.py` — clusters detections by subject keyword, renders per-subject reels.

### 5.3 Social media optimization

New `utils/social_formats.py`:

```python
SOCIAL_PRESETS = {
    "tiktok": {"width": 1080, "height": 1920, "max_duration": 60},
    "reels": {"width": 1080, "height": 1920, "max_duration": 90},
    "shorts": {"width": 1080, "height": 1920, "max_duration": 60},
    "youtube": {"width": 1920, "height": 1080, "max_duration": 600},
}
```

Face-aware reframing using OpenCV DNN or MediaPipe face detection to keep subjects centered in vertical crops.

### 5.4 Aftermovie assembly

New `scripts/06_build_aftermovie.py`:

- Takes top-N detections across all subjects
- Orders by narrative arc (establishing → action → macro → close → outro)
- Applies transitions between clips
- Syncs cuts to music beats
- Adds title cards and credits

### 5.5 Config presets

`config/presets/aftermovie.yaml`:
```yaml
output:
  format: "mp4"
  resolution: [1920, 1080]
  fps: 30
  duration_min: 5
  duration_max: 10
  music: "upbeat_tropical"
  transitions: "crossfade"
  title: "Holiday Aftermovie"

social:
  platforms: ["tiktok", "reels", "shorts"]
  duration_max: 60
  aspect_ratio: "9:16"

subjects:
  - turtle
  - shark
  - macro
  - diver
```

---

## 6. Implementation Sequence (Suggested PRs)

### PR 1: Foundation (Phase 1)
- [ ] Add `scripts/00_ingest_media.py` — folder scanning for videos + photos
- [ ] Extend `utils/models.py` — `PhotoManifest`, `RenderedClip` models
- [ ] Update `02_scan_footage.py` — accept manifest without Resolve, add photo support
- [ ] Add `scripts/05_render_clips.py` — FFmpeg clip extraction from detections
- [ ] Add `config/presets/` — aftermovie, social, subject presets
- [ ] Tests for ingest + render pipeline

### PR 2: Assembly (Phase 2)
- [ ] Add `scripts/06_build_aftermovie.py` — timeline assembly with music + transitions
- [ ] Add `scripts/07_build_social_reels.py` — vertical crops, face-aware reframing
- [ ] Add `scripts/08_group_subjects.py` — per-subject reel generation
- [ ] Add `utils/audio_mixer.py` — music sync, ducking
- [ ] Tests for aftermovie + social builders

### PR 3: Polish (Phase 3)
- [ ] Add `web_ui/` — Streamlit/FastAPI review interface
- [ ] Add `utils/preview_generator.py` — thumbnails, contact sheets
- [ ] Add `utils/checkpoint.py` — save/resume state
- [ ] CLI unification — single `dive-clip` entry point
- [ ] Tests for UI + checkpoint

### PR 4: Scale (Phase 4)
- [ ] Batch processing queue
- [ ] Platform export presets
- [ ] Cloud LLM storyline improvements
- [ ] Docker container
- [ ] Publishing integrations

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| FFmpeg complexity — many filter combinations can fail | High | Extensive integration tests, fallback to simple concatenation |
| VLM scanning is slow (Ollama Qwen2.5-VL) | Medium | Coarse-to-fine already helps; add frame skipping, batch processing |
| Face-aware reframing needs additional models | Medium | Use OpenCV Haar cascades as fallback, MediaPipe as upgrade |
| Music licensing for aftermovies | Low | Provide royalty-free music bundle, let user supply own |
| Large video files = high disk I/O | Medium | RAM-disk support already exists; add SSD recommendation |
| Cloud LLM costs for storyline | Low | Keep optional, local-first default |

---

## 8. Success Metrics

| Metric | Target |
|--------|--------|
| Time from "point at folder" to "first highlight clip" | < 10 min |
| One-command setup (Docker/uv) | Single command |
| Subject-specific reels generated automatically | ≥ 3 subjects (turtle, shark, macro) |
| Social clips meet platform specs (9:16, duration) | 100% |
| Aftermovie has transitions + music + titles | Yes |
| No DaVinci Resolve required for core flow | Yes |

---

## 9. Open Questions for the Team

1. **Primary target platform:** macOS desktop app, CLI tool, or web service?
2. **Preferred AI stack:** Keep Ollama (local) or add cloud VLM APIs (OpenAI, Gemini) as option?
3. **Music:** Bundle royalty-free tracks or require user to provide?
4. **Photo handling:** Slideshow sections with Ken Burns effect, or photos only as B-roll thumbnails?
5. **Output format:** MP4 files only, or also project files for Resolve/Premiere round-trip?

---

## 10. Appendix: File Map (Current → Target)

```
Current structure:
├── scripts/01_extract_manifest.py    (Resolve-dependent)
├── scripts/02_scan_footage.py        (Ollama VLM scan)
├── scripts/03_generate_edl.py        (markers only)
├── scripts/04_plan_storyline.py      (cloud LLM)
├── utils/ffmpeg_streamer.py          (frame extraction)
├── utils/timecode_math.py            (SMPTE conversion)
├── utils/prompt_templates.py         (VLM prompts)
├── utils/models.py                   (Pydantic models)
└── config.yaml                       (single config)

Target structure (after roadmap):
├── scripts/00_ingest_media.py        (NEW: folder scanning)
├── scripts/01_extract_manifest.py    (keep, optional Resolve path)
├── scripts/02_scan_footage.py        (extended: photo + direct manifest)
├── scripts/03_generate_edl.py        (keep)
├── scripts/04_plan_storyline.py      (keep, optional)
├── scripts/05_render_clips.py        (NEW: clip extraction)
├── scripts/06_build_aftermovie.py    (NEW: long-form assembly)
├── scripts/07_build_social_reels.py  (NEW: vertical social clips)
├── scripts/08_group_subjects.py      (NEW: per-subject reels)
├── utils/ffmpeg_streamer.py          (extended: concat, crop, transitions)
├── utils/timecode_math.py            (keep)
├── utils/prompt_templates.py         (extended: photo prompts)
├── utils/models.py                   (extended: Photo, RenderJob, SubjectCluster)
├── utils/audio_mixer.py              (NEW: music + ducking)
├── utils/social_formats.py           (NEW: platform presets)
├── utils/preview_generator.py        (NEW: thumbnails, contact sheets)
├── utils/checkpoint.py               (NEW: save/resume)
├── web_ui/                           (NEW: Streamlit/FastAPI review app)
├── config/presets/                   (NEW: aftermovie, social, subject YAMLs)
├── config.yaml                       (kept, extended)
└── pyproject.toml                    (new console entry: dive-clip)
```

---

*This roadmap is a living document. Update it as phases are completed and new requirements emerge.*
