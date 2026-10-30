# Architecture & Technical Implementation Plan: Underwater Dive Footage AI Curation Pipeline

## 1. Executive Summary & Objective

Het doel van dit project is het automatiseren van de selectie en curatie van onderwater videobeelden (Madagascar reis) in DaVinci Resolve Studio/Free. 

De grove selectie is reeds gedaan in DaVinci Resolve en georganiseerd met behulp van **Smart Bins & Keywords**. Dit plan specificeert een geautomatiseerde end-to-end pipeline:
1. **Metadata-extractie:** Resolve-project uitlezen op bestanden en tags.
2. **Two-Pass AI Computer Vision:** Een lokale Ubuntu GPU worker met **Qwen2.5-VL-7B** (via Ollama) scant de beelden coarse-to-fine op actie, macro en onderwaterleven, gestuurd door de Resolve-tags.
3. **Resolve Marker Sync:** Resultaten exporteren naar een CMX 3600 EDL en CSV voor naadloze import als gekleurde markers in DaVinci Resolve.
4. **Storyline Architect:** Geaggregeerde highlight-metadata aanbieden aan een frontier LLM (Claude 3.5 Sonnet / Gemini API) voor een redactioneel montageplan.

---

## 2. Architectuur & Workflow

```
┌────────────────────────────────────────────────────────┐
│ Stage 1: Resolve Metadata Ingestion                    │
│   DaVinci Python API -> export manifest_raw.json       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: Coarse-to-Fine Computer Vision Scanner        │
│   • Pass 1 (Coarse): 0.5 FPS Sampling naar RAM-disk    │
│   • Pass 2 (Fine): 2-4 FPS Window rond candidates      │
│   • Inference: Qwen2.5-VL-7B (Ollama on Ubuntu)        │
│   • Prompt injection gebaseerd op clip keywords        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Resolve Marker Synchronization                │
│   Generate CMX 3600 EDL with Marker Comments / CSV     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: Storyline Sequence Builder (Cloud API)        │
│   Aggregate markers -> Claude 3.5 Sonnet / Gemini API  │
│   Produces edit narrative & sequence order             │
└───────────────────────────┬────────────────────────────┘
```

---

## 3. Repository- & Mappenstructuur

De coding agent moet het project opbouwen volgens deze structuur:

```text
dive-clip-tui/
├── README.md                 # Volledige setup-, configuratie- en uitvoeringsgids
├── config.yaml               # Paden, thresholds, Ollama endpoints, API keys
├── requirements.txt          # Python dependencies
├── scripts/
│   ├── 01_extract_manifest.py    # Draait in DaVinci Resolve (Mac of Ubuntu)
│   ├── 02_scan_footage.py        # Worker pipeline (FFmpeg + Ollama coarse-to-fine)
│   ├── 03_generate_edl.py        # Converteert AI detecties naar CMX 3600 EDL & CSV
│   └── 04_plan_storyline.py      # Cloud API story-architect (Claude / Gemini)
├── utils/
│   ├── ffmpeg_streamer.py        # Frame sampling helpers (/dev/shm & buffers)
│   ├── timecode_math.py          # Frame/FPS naar SMPTE timecodes
│   └── prompt_templates.py       # Dynamische prompts per keyword categorie
└── outputs/
    ├── manifest_raw.json
    ├── detections_refined.json
    ├── markers_import.edl
    └── storyline_edit_guide.md
```

---

## 4. Configuratiebestand: `config.yaml`

```yaml
# Configuration for Dive Clip TUI

general:
  project_name: "Madagascar_Dives_2026"
  # Pad mapping: maakt het script draagbaar tussen macOS en Linux
  path_mapping:
    from_prefix: "/Volumes/ExtremeSSD/Madagascar"
    to_prefix: "/media/ubuntu/ExtremeSSD/Madagascar"
  output_dir: "./outputs"

ollama:
  host: "http://localhost:11434"
  model: "qwen2.5-vl:7b"
  timeout_seconds: 90

scan_parameters:
  coarse_fps: 0.5                   # 1 frame per 2 seconden
  fine_fps: 2.0                     # 2 frames per seconde tijdens highlight-windows
  coarse_confidence_threshold: 0.65 # Minimum confidence om fine scan te triggeren
  fine_padding_seconds: 4           # Extra context voor/na gedetecteerde actie
  quality_threshold: 7              # Min. score (1-10) om in EDL/Storyline te komen
  use_ramdisk: true                 # Gebruik /dev/shm op Linux voor snelle I/O

cloud_storyline:
  provider: "claude"                # "claude" of "gemini"
  model: "claude-3-5-sonnet-latest"
  target_edit_duration_minutes: 4
```

---

## 5. Technische Specificaties per Component

### 5.1 Script 1: DaVinci Manifest Extractor (`scripts/01_extract_manifest.py`)
* **Taak:** Traverseert de Media Pool van het actieve Resolve project en leest alle clips uit die door de gebruiker zijn voorzien van minstens 1 keyword.
* **Resolve API calls:**
  * `resolve.GetProjectManager().GetCurrentProject()`
  * `project.GetMediaPool().GetRootFolder()`
  * Recursieve inspectie via `GetSubFolderList()` en `GetClipList()`.
  * Metadata extractie: `clip.GetMetadata("Keywords")`, `clip.GetClipProperty()`.
* **Output:** `outputs/manifest_raw.json` bevattende clip-id, bestandsnaam, absoluut pad, framerate, resolutie en tags.

### 5.2 Script 2: Coarse-to-Fine VLM Scanner (`scripts/02_scan_footage.py`)
* **Taak:** Voert de video-analyse uit op de Ubuntu machine via Ollama.
* **I/O Strategie:** Schrijf frames naar `/dev/shm/dive_worker_tmp/` (Linux RAM-disk). Dit voorkomt slijtage en latency op de externe SSD.
* **Pass 1 (Coarse Scan @ 0.5 FPS):**
  * FFmpeg extracteert `fps=0.5,scale=640:-1`.
  * Evalueert frames met Qwen2.5-VL: detecteer of er iets interessants gebeurt dat matcht met de tags (vis, schildpad, macro, actie).
* **Pass 2 (Fine Scan @ 2.0 FPS):**
  * Voor elke timestamp met `confidence >= 0.65`: neem een venster van `-4s` tot `+4s`.
  * Extract frames op hogere resolutie (`scale=1280:-1`) aan 2 FPS.
  * Gestructureerde JSON extractie: start/eind frame, subject, scherpte-/kwaliteitsscore (1-10) en actiebeschrijving.
* **Output:** `outputs/detections_refined.json`.

### 5.3 Script 3: EDL & Marker Sync (`scripts/03_generate_edl.py`)
* **Taak:** Converteert de JSON detecties naar een direct in DaVinci te importeren **CMX 3600 EDL** en **CSV Marker file**.
* **Kleurcodes:**
  * Cyaan: Pelagics / Groot zeeleven (schildpadden, haaien, roggen).
  * Geel: Macro (naaktslakken, garnalen, koralen).
  * Groen: Duikeractie / Goede compositie.
* **CMX 3600 Syntax:**
  ```text
  TITLE: MADAGASCAR_AI_MARKERS
  FCM: NON-DROP FRAME

  001  AX       V     C        00:00:12:00 00:00:15:00 01:00:12:00 01:00:15:00
  * FROM CLIP NAME: GOPR0042.MP4
  * LOC: 01:00:12:00 Cyan Turtle - feeding sharp macro
  ```

### 5.4 Script 4: Cloud Storyline Assembler (`scripts/04_plan_storyline.py`)
* **Taak:** Neemt de `detections_refined.json` (alleen tekst, kost slechts enkele duizenden tokens) en stuurt dit naar de Anthropic of Google GenAI SDK.
* **Output:** `outputs/storyline_edit_guide.md` met een professionele tijdlijnvolgorde (Establishing -> Reef Scape -> Encounters -> Macro Treasures -> Sunset/Safety Stop).

---

## 6. Volledige `README.md` Specificatie (Voor de Gebruiker / Agent)

```markdown
# 🌊 Dive Clip TUI

Geautomatiseerde AI-curatie voor onderwater duikvideo's. Gebruikt lokale GPU-kracht (`Qwen2.5-VL` via Ollama) voor het scannen van beelden op basis van bestaande DaVinci Resolve keywords, en importeert de beste momenten direct terug als tijdlijnmarkers.

---

## 📋 Vereisten

### Hardware
- **Ubuntu Machine:** Nvidia GPU met minimaal 12GB VRAM en 32GB RAM.
- **Mac (optioneel):** M-series Mac voor Resolve editing.
- **Externe SSD:** Bevat de ruwe videobestanden.

### Software
- **DaVinci Resolve Studio of Free** (met Python scripting enabled).
- **FFmpeg:** `sudo apt install ffmpeg`
- **Ollama:** Met `qwen2.5-vl:7b` model.

---

## 🛠️ Installatie & Voorbereiding

1. **Clone repository en virtuele omgeving aanmaken:**
   ```bash
   cd dive-clip-tui
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Ollama starten en model ophalen op Ubuntu:**
   ```bash
   ollama pull qwen2.5-vl:7b
   ollama serve
   ```

3. **Configuratie instellen:**
   Controleer `config.yaml` en pas eventuele SSD mount-paden aan.

---

## 🎬 Stap-voor-Stap Uitvoering

### Stap 1: Metadata exporteren uit DaVinci Resolve
1. Open DaVinci Resolve met het Madagascar project.
2. Ga naar **Workspace -> Console** en kies **Py3**.
3. Draai het extractiescript:
   ```python
   exec(open("scripts/01_extract_manifest.py").read())
   ```
   *Resultaat:* `outputs/manifest_raw.json` is aangemaakt.

### Stap 2: Lokale Scan draaien op Ubuntu
Draai de batchworker op de Ubuntu machine:
```bash
python scripts/02_scan_footage.py --config config.yaml
```
* Het script verwerkt alle video's pass-by-pass via de RAM-disk (`/dev/shm`).
* *Resultaat:* `outputs/detections_refined.json`.

### Stap 3: Markers genereren en importeren in Resolve
1. Genereer de EDL en CSV bestanden:
   ```bash
   python scripts/03_generate_edl.py --config config.yaml
   ```
2. Ga in DaVinci Resolve naar je gewenste tijdlijn.
3. Klik met de rechtermuisknop op de tijdlijn in de Media Pool:
   **Timelines -> Import -> Timeline Markers from EDL...**
4. Selecteer `outputs/markers_import.edl`.
5. Alle AI-geïdentificeerde acties en macro-momenten staan direct als gekleurde markers op je tijdlijn.

### Stap 4 (Optioneel): Storyline Montageplan opstellen
Laat Claude of Gemini een narrative volgorde bepalen:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/04_plan_storyline.py --config config.yaml
```
Bekijk `outputs/storyline_edit_guide.md` voor de geadviseerde shotvolgorde en muzieksuggesties.
```

---

## 7. Checklist voor de Coding Agent

- [ ] **Data Model:** Gebruik `pydantic` voor strikte validatie van manifest- en detectie-JSON-structuren.
- [ ] **I/O Veiligheid:** Controleer altijd of `/dev/shm` bestaat; val terug op `tempfile.mkdtemp()` indien niet aanwezig.
- [ ] **Error Recovery:** Als een corrupt videoframe FFmpeg laat crashen, skip dan de frame-range en log een waarschuwing zonder de hele batch te stoppen.
- [ ] **Timecode Precision:** Zorg dat timecodes exact berekend worden op basis van de clip FPS (bijv. 29.97 vs 30.0 vs 25.0).
- [ ] **Ollama JSON Mode:** Forceer `format: "json"` in de Ollama API payload om te voorkomen dat Qwen2.5-VL markdown-babbels retourneert in plaats van pure JSON.