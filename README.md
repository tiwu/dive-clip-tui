# 🌊 Underwater Resolve AI Curation Pipeline

Automated AI curation for underwater dive footage. Uses local GPU power (`Qwen2.5-VL` via Ollama) to scan footage based on existing DaVinci Resolve keywords, and imports the best moments directly back as timeline markers.

---

## 📋 Requirements

### Hardware
- **Ubuntu Machine:** Nvidia GPU with at least 12GB VRAM and 32GB RAM.
- **Mac (optional):** M-series Mac for Resolve editing.
- **External SSD:** Contains the raw video files.

### Software
- **DaVinci Resolve Studio or Free** (with Python scripting enabled).
- **FFmpeg:** `sudo apt install ffmpeg`
- **Ollama:** With `qwen2.5-vl:7b` model.
- **Python 3.10+**

---

## 🗂️ Repository Structure

```text
├── README.md                 # Full setup, configuration, and execution guide
├── config.yaml               # Paths, thresholds, Ollama endpoints, API keys
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project metadata and uv-compatible config
├── ruff.toml                 # Ruff linting configuration
├── uv.lock                   # Locked dependencies
├── scripts/
│   ├── 01_extract_manifest.py    # Runs in DaVinci Resolve (Mac or Ubuntu)
│   ├── 02_scan_footage.py        # Worker pipeline (FFmpeg + Ollama coarse-to-fine)
│   ├── 03_generate_edl.py        # Converts AI detections to CMX 3600 EDL & CSV
│   └── 04_plan_storyline.py      # Cloud API story architect (Claude / Gemini)
├── utils/
│   ├── ffmpeg_streamer.py        # Frame sampling helpers (/dev/shm & buffers)
│   ├── timecode_math.py          # Frame/FPS to SMPTE timecodes
│   ├── prompt_templates.py       # Dynamic prompts per keyword category
│   └── models.py                 # Pydantic data models for manifest and detections
└── outputs/
    ├── manifest_raw.json
    ├── detections_refined.json
    ├── markers_import.edl
    ├── markers_import.csv
    └── storyline_edit_guide.md
```

---

## 🛠️ Installation & Preparation

1. **Clone repository and install dependencies with uv:**
   ```bash
   cd dive-clip-ai
   uv sync
   ```

   *Alternative (if you don't have uv):*
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start Ollama and pull the model on Ubuntu:**
   ```bash
   ollama pull qwen2.5-vl:7b
   ollama serve
   ```

3. **Configure:**
   Check `config.yaml` and adjust any SSD mount paths (path_mapping).

---

## 🎬 Step-by-Step Execution

### Step 1: Export Metadata from DaVinci Resolve

1. Open DaVinci Resolve with the Madagascar project.
2. Go to **Workspace -> Console** and choose **Py3**.
3. Run the extract script:
   ```python
   exec(open("scripts/01_extract_manifest.py").read())
   ```
   *Result:* `outputs/manifest_raw.json` is created.

### Step 2: Run Local Scan on Ubuntu

Run the batch worker on the Ubuntu machine:
```bash
uv run scripts/02_scan_footage.py --config config.yaml
```
* The script processes all videos pass-by-pass via the RAM disk (`/dev/shm`).
* On systems without `/dev/shm` (e.g. macOS), it falls back to a temporary directory.
* *Result:* `outputs/detections_refined.json`.

### Step 3: Generate Markers and Import into Resolve

1. Generate the EDL and CSV files:
   ```bash
   uv run scripts/03_generate_edl.py --config config.yaml
   ```
2. In DaVinci Resolve, go to your desired timeline.
3. Right-click the timeline in the Media Pool:
   **Timelines -> Import -> Timeline Markers from EDL...**
4. Select `outputs/markers_import.edl`.
5. All AI-identified actions and macro moments appear directly as colored markers on your timeline.

   **Color codes:**
   * **Cyan:** Pelagics / large sea life (turtles, sharks, rays)
   * **Yellow:** Macro (nudibranchs, shrimp, corals)
   * **Green:** Diver action / Good composition

### Step 4 (Optional): Create Storyline Edit Plan

Let Claude or Gemini determine a narrative order:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
uv run scripts/04_plan_storyline.py --config config.yaml
```
Check `outputs/storyline_edit_guide.md` for the recommended shot order and music suggestions.

---

## 🧩 Technical Details

### Data Model (Pydantic)
All JSON structures are strictly validated with Pydantic models (`utils/models.py`):
- `ClipManifest` / `ManifestOutput` — for the Resolve metadata export
- `RefinedDetection` / `DetectionsRefined` — for the AI scan results

### Timecode Precision
`utils/timecode_math.py` calculates timecodes exactly based on clip FPS, with support for:
- 29.97 drop-frame and non-drop-frame
- 30.0, 25.0, 24.0 and other frame rates
- Conversion between seconds, frame counts, and SMPTE timecodes (HH:MM:SS:FF)

### I/O Safety
`utils/ffmpeg_streamer.py` checks if `/dev/shm` exists and falls back to `tempfile.mkdtemp()` if not available. Corrupt frames are skipped with a warning without stopping the entire batch.

### Ollama JSON Mode
All Ollama API calls use `"format": "json"` in the payload, so Qwen2.5-VL returns pure JSON without markdown filler.

---

## ✅ Checklist

- [x] **Data Model:** Use `pydantic` for strict validation of manifest and detection JSON structures.
- [x] **I/O Safety:** Check if `/dev/shm` exists; fall back to `tempfile.mkdtemp()` if not available.
- [x] **Error Recovery:** If a corrupt video frame crashes FFmpeg, skip the frame range and log a warning without stopping the entire batch.
- [x] **Timecode Precision:** Ensure timecodes are calculated exactly based on clip FPS (e.g. 29.97 vs 30.0 vs 25.0).
- [x] **Ollama JSON Mode:** Force `format: "json"` in the Ollama API payload to prevent Qwen2.5-VL from returning markdown filler instead of pure JSON.
