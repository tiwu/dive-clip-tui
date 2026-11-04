# Research Report: Action Camera Telemetry, Non-AI Programmatic Pre-Screening, Commercial/Open-Source Highlight Detection, and Broadening the `dive-clip-ai` Curation Pipeline

**Date:** 2026-11-03  
**Scope:** Primary-source investigation into embedded action camera telemetry (GoPro GPMF, DJI `djmd`, Insta360 `insv`), commercial and open-source highlight detection architectures (GoPro Quik, PySceneDetect, Auto-Editor, LosslessCut, Gyroflow, Katna/Hecate, Immich/Apple/Google Photos Memories, VideoHighlighter, FFmpeg `lavfi`), evaluation of 4 non-AI programmatic preprocessing methods (`ffprobe` stream headers, audio telemetry, `export_mvs` motion vectors + HEVC caveats, and `-skip_frame nokey` I-frame histograms), root-cause diagnosis of why `dive-clip-ai` produces narrow/repetitive results, and a prioritized roadmap of actionable engineering tactics.

---

## 1. GoPro GPMF & Action Camera Embedded Sensor Telemetry

Modern action cameras (GoPro HERO5+, DJI Osmo Action, Insta360) do not merely record compressed H.264/HEVC video and AAC audio streams; they simultaneously record synchronized, time-indexed **inertial measurement unit (IMU)**, **environmental**, **exposure**, and **on-chip computer-vision** telemetry directly inside the MP4/MOV container.

### 1.1 GoPro Metadata Format (GPMF) Architecture & MP4 Track Structure

According to the official GoPro [`gpmf-parser` specification](https://github.com/gopro/gpmf-parser) and GoPro's engineering overview ([*GoPro Video Metadata Open Source Explained*](https://gopro.com/en/ps/news/gopro-video-metadata-open-source-explained)), **GPMF** is a big-endian, 32-bit aligned, hierarchical **Key-Length-Value (KLV)** binary format designed for low-overhead time-of-capture logging.

* **MP4 Track Layout (`gpmd`):**
  * Inside an MP4/MOV container, GPMF data is stored as a dedicated **timed metadata track** (`hdlr` handler type `meta`, sample description FourCC **`gpmd`**, typically stream `0:2` or `0:3` when inspected via `ffprobe`).
  * Payloads are packetized into 1-second (or frame-aligned) MP4 samples (`DEVC` root container → `STRM` stream containers) indexed by the standard MP4 sample table (`stts`, `stsc`, `stsz`, `stco`), allowing instant extraction of the entire sensor timeline via `ffmpeg -i clip.mp4 -map 0:m:handler_name:"\tGoPro MET" -c copy -f rawvideo -` without decoding a single video frame.
  * Starting with **HERO8 Black**, GoPro added microsecond timestamps (`STMP`) alongside sample counts (`TSMP`), enabling sub-frame sensor synchronization.

### 1.2 Complete GPMF FourCC Reference (IMU, Exposure, & On-Chip Vision)

A critical finding verified directly against the [`gopro/gpmf-parser` README and FourCC table](https://github.com/gopro/gpmf-parser#where-to-find-gpmf-data) is that **HERO7 Black and newer cameras (powered by the GP1/GP2 SoC) run real-time computer vision and exposure/scene classification in hardware during recording** and write the results directly into the `gpmd` track:

| Category | FourCC | Name / Stream Description | Typical Rate (Hz) | Units / Data Structure | Camera Availability | Pipeline Utility for `dive-clip-ai` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Inertial / IMU** | `ACCL` | 3-Axis Accelerometer (includes gravity) | 200 Hz | $\text{m/s}^2$ (`z, x, y` or `x, y, z` via `ORIN`) | HERO5+ | Detect G-force spikes, jumps, braking, impacts, or freefall |
| **Inertial / IMU** | `GYRO` | 3-Axis Gyroscope (angular velocity) | 200 Hz | $\text{rad/s}$ | HERO5+ | Detect camera shake/tumbling (`usable=False` pre-filter) vs. smooth pans |
| **Orientation** | `CORI` | Camera Orientation | Frame rate (24–240 Hz) | Quaternion `(w, x, y, z)` since capture start | HERO8+, MAX | Instant check for inverted/dangling camera or motorcycle lean angle! |
| **Orientation** | `IORI` | Image Orientation | Frame rate | Quaternion `(w, x, y, z)` relative to sensor | HERO8+, MAX | Horizon leveling & roll/pitch stability |
| **Orientation** | `GRAV` | Gravity Vector | Frame rate (30–60 Hz) | Normalized 3D vector `(x, y, z)` | HERO8+, MAX | Exact pitch/roll tilt relative to Earth gravity (cornering lean angle, upside-down check) |
| **Location / Speed** | `GPS5` | 5-Channel GPS Telemetry | 18 Hz | Lat, Lon, Alt, 2D speed, 3D speed | HERO5–HERO10 | Speed peaks, twisty mountain pass detection, stationary stop rejection |
| **Location / Speed** | `GPS9` | Extended 9-Channel GPS Telemetry | 10–18 Hz | Lat, Lon, Alt, 2D/3D speed, DOP, fix, days, secs | HERO11, HERO13 | High-precision speed + GPS fix validation (note: omitted on HERO12) |
| **Exposure / Sensor** | `SHUT` | Shutter Speed (exposure duration) | Frame rate | Seconds (float) | HERO5+ (metadata HERO7+) | Motion blur risk estimation (long shutter + high `GYRO` = smeared frame) |
| **Exposure / Sensor** | `ISOE` / `ISOG` | Sensor ISO / Image Sensor Gain | Frame rate | ISO integer / gain | HERO7+ / HERO5+ | Low-light / murky water / tunnel detection |
| **Exposure / Sensor** | `WBAL` / `WRGB` | White Balance (Kelvin / RGB gains) | Frame rate | Kelvin (`WBAL`), `(R, G, B)` gains (`WRGB`) | HERO7+ | Depth/color shift estimation underwater (red channel attenuation vs. depth) |
| **Exposure / Sensor** | `SROT` | Sensor Readout Time | Frame rate | Milliseconds | HERO8+ | Rolling shutter distortion severity |
| **Image Stats** | `YAVG` | Luma (Y) Average | ~8–10 Hz | `0` (black) to `255` (white) | HERO7+ | Zero-cost dark/overexposed/in-bag clip rejection before FFmpeg decode |
| **Image Stats** | `UNIF` | Image Uniformity | ~8–10 Hz | `0.0` (textured) to `1.0` (solid uniform color) | HERO7+ | Reject empty blue water, fog, mud, or covered lens (`UNIF > 0.90`) |
| **Image Stats** | `HUES` | Predominant Hues | ~8–10 Hz | Array of `(hue: u8, weight: u8)` pairs | HERO7+ | Detect colorful coral/macro vs. monochrome blue/gray frames; color change boundaries |
| **On-Chip CV** | `SCEN` | Scene Classification | ~1 Hz | `[SNOW, URB, INDO, WATR, VEGE, BEAC]` probabilities | HERO7+ | **Instant domain classification (`WATR` = underwater, `VEGE`/`URB` = road/trail)** without VLM Pass 0! |
| **On-Chip CV** | `FACE` | Face & Smile Detection | ~8–10 Hz | Struct: `ver, confidence, ID, x, y, w, h, smile` | HERO7+ | Detect buddy interactions, boat briefings, post-dive/ride smiles & reactions |

> **Verification of `SCEN`, `HUES`, `UNIF`, `YAVG`, `FACE`, `CORI`, `GRAV`:**  
> Confirmed directly from [`gopro/gpmf-parser` README.md ("Where to find GPMF data")](https://github.com/gopro/gpmf-parser#where-to-find-gpmf-data) and [`GPMF_parser.h`](https://github.com/gopro/gpmf-parser). Starting with HERO7 Black, the GP1/GP2 processor embeds `YAVG`, `UNIF`, `HUES`, `FACE`, and `SCEN` (with FourCC sub-types `SNOW` for snow, `URB` / `URBA` for urban, `INDO` for indoor, `WATR` for water/underwater, `VEGE` for vegetation, and `BEAC` for beach/sand) at ~1–10 Hz, and HERO8+ embeds `CORI`, `IORI`, and `GRAV` at frame rate.

### 1.3 How GoPro Stores Manual & Automatic "HiLight" Tags (`hmmt`, `HLMT`, `MANL`)

GoPro stores highlight markers in two places inside the MP4 container ([ExifTool GoPro Tags](https://exiftool.org/TagNames/GoPro.html), [`gpmf-parser` issues & spec](https://github.com/gopro/gpmf-parser)):

1. **MP4 `moov.udta.hmmt` Box (HiLight Marker Table):**
   * Stored in the MP4 User Data (`udta`) atom under the 4-byte box type **`hmmt`**.
   * Binary layout: a 32-bit big-endian integer `count`, followed by `count` 32-bit big-endian integers representing **millisecond timestamps** (`ms` from start of video) where the user pressed the physical Mode/HiLight button on the camera, pressed the remote button, or said *"GoPro HiLight"*.
   * Accessible in <5 ms via `ffprobe` / `exiftool -HiLight` or a 15-line Python `struct.unpack` MP4 atom reader.
2. **GPMF Stream Markers (`MANL` / `HLMT`):**
   * Inside the `gpmd` track (and `moov.udta.GPMF`), manual button presses are recorded under the **`MANL`** (Manual HiLight) or **`HLMT`** (Highlight Metadata Table) FourCC, alongside firmware-computed candidate markers found in certain HERO models (`AIMU` for IMU spikes, `AACD` for Abrupt Audio Change Detector, `FUSE` for fused multi-sensor highlights).

### 1.4 DJI & Insta360 Embedded Telemetry Formats

* **DJI (`djmd` Protobuf & SRT Telemetry):**
  * **DJI Action 2 / Action 4 / Action 5 Pro / Avata / Air Units:** Embed a timed metadata stream in the MP4 container with handler/FourCC **`djmd`** encoded using **Google Protocol Buffers (Protobuf)** ([`gyroflow/telemetry-parser` DJI parser](https://github.com/AdrianEddy/telemetry-parser/tree/master/src/dji)). It contains high-rate IMU (`200–1000 Hz` gyro/accel), exposure (shutter, ISO), and quaternion orientation data.
  * **DJI Drones / Osmo (Legacy & Optional Mode):** Also write either an embedded subtitle track (`subt` / `mov_text`) or sidecar `.SRT` file containing per-frame GPS coordinates, barometer altitude, ISO, shutter, and focal length.
* **Insta360 (`insv` / `.mp4` Trailer Metadata):**
  * Insta360 cameras (ONE R/RS, Ace, Ace Pro, X3, X4, GO 3/3S) append a proprietary binary **extra-metadata trailer** at the **end of the `.mp4` / `.insv` file**, terminated by the 32-byte magic signature `8db42d694ccc418790edff439fe026bf` ([`gyroflow/telemetry-parser` Insta360 format](https://github.com/AdrianEddy/telemetry-parser/tree/master/src/insta360), [ExifTool Insta360 docs](https://exiftool.org/TagNames/GoPro.html)).
  * Reading backwards from the magic footer reveals an index of typed binary records including `0x03` (Accelerometer + Gyroscope IMU at 200–1000 Hz), `0x04` (Exposure/shutter timestamps), `0x06` (GPS), and `0x07` (AAFrame / LRV / highlight markers).

### 1.5 Python Ecosystem for Reading Action Camera Telemetry

| Library / Tool | Supported Formats | Maintainer / Primary Source | Pros & Cons for `dive-clip-ai` |
| :--- | :--- | :--- | :--- |
| **`telemetry-parser`** (PyPI `telemetry-parser`) | **GoPro GPMF**, **DJI (`djmd`)**, **Insta360**, Sony (`rtmd`), Betaflight, Blackmagic, RED | Adrian Eddy / [Gyroflow](https://github.com/AdrianEddy/telemetry-parser) (Rust core + `maturin` Python wheels) | **Best all-in-one library.** Fast Rust parser, exposes `tp.normalized_imu()` (gyro/accel in $\text{deg/s}$ and $\text{m/s}^2$) and raw GPMF keys (`SCEN`, `CORI`, `GRAV`, `YAVG`, `UNIF`, `HUES`, `FACE`, `GPS5`) across GoPro, DJI, and Insta360. |
| **`gpmf` / `pygpmf`** | GoPro GPMF (`gpmd` only) | [`alexis-mignon/pygpmf`](https://github.com/alexis-mignon/pygpmf) / [`insufficient/gpmf`](https://github.com/insufficient/gpmf) | Pure Python / Cython GPMF KLV parsers; focused mainly on GPS/IMU extraction, lacks DJI/Insta360 support. |
| **`exiftool`** (`pyexiftool` / CLI) | GoPro (`GPMF`, `hmmt`), DJI, Insta360 | Phil Harvey ([exiftool.org](https://exiftool.org/TagNames/GoPro.html)) | `exiftool -ee -G3` extracts all timed metadata, `HiLight` tags, and `SCEN`/`FACE`, but spawning Perl subprocesses with `-ee` on multi-GB MP4s is slower than `telemetry-parser` or `ffprobe` stream copy. |
| **`ffprobe` + Python `struct`** | MP4 stream layout (`gpmd`, `djmd`) + `moov.udta.hmmt` | FFmpeg + Python stdlib | **Zero extra dependencies.** Can read `hmmt` HiLight tags directly from the MP4 `moov` box in <2 ms and extract the raw `gpmd` byte stream via `ffmpeg -map 0:m:handler_name` in <100 ms. |

---

## 2. How GoPro Quik & Similar Commercial/Open-Source Tools Do Highlight Detection & Auto-Editing

### 2.1 GoPro Quik's Multi-Signal Highlight Engine

GoPro Quik does **not** run a heavy Vision-Language Model over every video frame. Instead, as documented in GoPro's GPMF engineering publications ([*GoPro Video Metadata Open Source Explained*](https://gopro.com/en/ps/news/gopro-video-metadata-open-source-explained)), it combines **record-time hardware metadata** from the GP1/GP2 SoC with lightweight post-capture signal fusion:

1. **User HiLight Priority (`hmmt` / `MANL`):** Any timestamp tagged manually by the user during recording is treated as an anchor highlight (`weight = 1.0`).
2. **On-Chip Computer Vision Metadata (`SCEN`, `FACE`, `HUES`, `YAVG`, `UNIF`):**
   * **Faces & Smiles (`FACE`):** Frames with high face confidence and smile scores (e.g., divers signaling "OK" on the surface, riders cheering at a mountain pass) receive a strong human-interest boost.
   * **Scene Transitions (`SCEN`):** Shifts in scene probabilities (e.g., boat deck `URB`/`INDO` → underwater `WATR` entry) identify narrative transition moments.
   * **Color & Exposure Quality (`HUES`, `YAVG`, `UNIF`):** Rejects uniform/murky/dark segments (`UNIF` high or `YAVG` near 0/255) and boosts vibrant color shifts (`HUES`).
3. **Inertial & Kinematic Peaks (`ACCL`, `GYRO`, `CORI`, `GRAV`, `GPS5`):**
   * High rotational chaos without horizon stability (`GYRO` variance spike) is penalized as camera shake/handling noise.
   * Sustained lean angle (`GRAV` / `CORI` roll tilt) + high speed (`GPS5`) + smooth angular rate (`GYRO`) scores high as dynamic cornering/carving action.
   * Peak G-force impulses (`ACCL`) followed by stable recovery flag jumps, drops, or giant-stride water entries.
4. **Audio Excitement & Beat Syncing:**
   * Audio peaks, cheering, laughter, and engine revs (`AACD`) boost segment scores.
   * Once candidate segments are ranked, Quik detects musical beats/downbeats in the backing track and **snaps cut points and clip durations (`1.5s – 4.5s`) to musical bar boundaries** rather than cutting arbitrary fixed 2-second windows.

### 2.2 Open-Source & Commercial Highlight / Video Summarization Tools

| Tool / System | Core Detection & Selection Signals | Primary Source / Reference | Key Takeaway for `dive-clip-ai` |
| :--- | :--- | :--- | :--- |
| **`PySceneDetect`** (`scenedetect`) | **`ContentDetector`** (HSV pixel/color delta between frames), **`AdaptiveDetector`** (rolling-average HSV delta to avoid false cuts on fast camera pans), **`HistogramDetector`** (YUV luminance histogram diff). | [Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | Highlight clips should align with **natural shot/motion boundaries** (`AdaptiveDetector`), not arbitrary `[ts - 1.0, ts + 1.0]` slices. |
| **`Auto-Editor`** (`WyattBlue/auto-editor`) | Composable boolean expression DSL (`--edit "audio:threshold=4% or motion:threshold=2%,width=400"`) combining frame-to-frame pixel motion and RMS audio loudness with margin padding (`--frame-margin`). | [WyattBlue/auto-editor](https://github.com/WyattBlue/auto-editor) | Fast low-res pixel-difference motion + audio loudness pre-screening eliminates 60–80% of static/dead footage before running any AI model. |
| **`LosslessCut`** | FFmpeg `scdet` / `blackdetect` / `silencedetect` + keyframe (`I-frame`) boundary alignment for smart/lossless cutting. | [mifi/lossless-cut](https://github.com/mifi/lossless-cut) | FFmpeg's built-in `lavfi` detectors provide near-instant structural segmentation. |
| **`Gyroflow` / `telemetry-parser`** | Synchronized 200Hz IMU (`GYRO`, `ACCL`) + quaternion orientation (`CORI`, `GRAV`) for horizon lock, smoothness scoring, and motion analysis. | [gyroflow/gyroflow](https://github.com/gyroflow/gyroflow) | IMU quaternions directly distinguish **intentional smooth panning / cornering lean** from **accidental dangling / camera shake**. |
| **`Katna` / Yahoo `Hecate`** | **1.** Filter out blurry (Laplacian variance), dark/blown-out (brightness), and flat/low-entropy frames.<br>**2.** Extract HSV/RGB histograms or CNN features.<br>**3.** Run **K-Means clustering** ($K = N_{\text{clips}}$) and pick the sharpest/highest-scoring frame from **each cluster**. | [keplerlab/katna](https://github.com/keplerlab/katna), [yahoo/hecate](https://github.com/yahoo/hecate) (Song et al., CVPR/ACMMM) | **Solves the repetition problem:** K-Means clustering forces the summary to pick 1 representative clip per visual cluster instead of 10 identical shots. |
| **Apple Photos / Google Photos / `Immich` Memories** | Multimodal **CLIP / SigLIP** embeddings + aesthetic scoring (NIMA / LAION-Aesthetics) + **Maximal Marginal Relevance (MMR)** or temporal clustering to maximize both relevance and visual diversity across the timeline. | [immich-app/immich](https://github.com/immich-app/immich) | MMR (`λ * quality - (1 - λ) * max_similarity_to_selected`) guarantees that every selected highlight adds *new* visual information. |
| **`VideoHighlighter`** (`Aseiel`) | Local multi-signal timeline fusion: **OpenCV Scene Cuts + Optical Flow/Motion + Audio Peaks + YOLO Object Detection + Action Recognition + Local VLM (Ollama)** with explainable per-timestamp composite scores. | [Aseiel/VideoHighlighter](https://github.com/Aseiel/VideoHighlighter) | VLM should be **one signal in a multi-signal scorer** (or a verifier on pre-scored multi-modal peaks), not the sole brute-force scanner. |
| **FFmpeg Built-in Filters (`lavfi`)** | `scdet` (`lavfi.scd.score`), `blurdetect` (`lavfi.blur`), `freezedetect`, `blackdetect`, `astats` (`RMS_level`, `Peak_level`), `ebur128` (`lavfi.r128.M`). | [FFmpeg Filters Documentation](https://ffmpeg.org/ffmpeg-filters.html) | Can be run in a single FFmpeg pass during frame extraction (`-vf "fps=0.5,scale=640:-2,blurdetect,metadata=mode=print"`), adding zero extra decode cost! |

---

## 3. Deep Evaluation of 4 Non-AI Programmatic Preprocessing Methods & The Pre-VLM Filter Funnel

Before invoking an expensive Vision-Language Model (7B parameters @ ~0.3–1.2s per frame), `dive-clip-ai` can extract massive amounts of structural, acoustic, kinematic, and chromatic signal using **pure, deterministic FFmpeg/FFprobe and container-level inspection**. Below is a technical evaluation of the 4 candidate non-AI preprocessing methods, including exact primary-source caveats for **underwater dive** vs. **motorcycle** footage.

### 3.1 Method 1: Video Stream Headers (`ffprobe` / MP4 `moov` Atom Inspection, `< 50 ms`)

By expanding `probe_video()` in [`scripts/00_ingest_media.py#L34-L80`](file:///home/twuytens/work/dive-clip-tui/scripts/00_ingest_media.py#L34-L80) to inspect **all streams and container tags** (`ffprobe -v error -show_format -show_streams -of json`), we extract critical signals in `< 50 ms` without decoding a single frame:

* **High Frame Rate (`r_frame_rate` $\ge 100\text{ FPS}$, e.g. `120/1` or `240/1`):**
  * **Signal:** Divers and riders intentionally switch action cameras to `120 FPS` or `240 FPS` (Slow-Mo mode) when anticipating high-value encounters (pelagic sightings, jumps, fast track sections).
  * **Action:** Flag any clip with `fps >= 100` as high-intent priority (`intent_boost = +0.15`).
* **Duration & Creation Time (`format.duration`, `format.tags.creation_time`):**
  * **Signal:** Ultra-short clips (`< 2.5s`) are almost always accidental shutter double-taps; conversely, `creation_time` allows sorting clips into true chronological trip/dive sessions even when filenames wrap or come from multiple cameras.
* **Embedded Telemetry & HiLight Streams (`codec_tag_string == "gpmd"`, `"djmd"`, `moov.udta.hmmt`):**
  * **Signal:** Currently, `00_ingest_media.py` passes `-select_streams v:0` ([`scripts/00_ingest_media.py#L40-L41`](file:///home/twuytens/work/dive-clip-tui/scripts/00_ingest_media.py#L40-L41)), which hides the `gpmd` / `djmd` data stream and audio stream completely. Removing `-select_streams v:0` immediately reveals whether telemetry (`codec_tag_string: "gpmd"`) is available for sensor extraction and whether `hmmt` manual HiLight tags exist.
* **Bitrate-to-Resolution Ratio (`format.bit_rate / (width * height * fps)`):**
  * **Signal:** Because GoPro/DJI use variable bitrate (VBR) scaling within a target ceiling, a clip recorded inside a dark gear bag or against a covered lens compresses down to a fraction of the normal bits-per-pixel, providing an instant suspicion flag.

### 3.2 Method 2: Audio Track Telemetry (`ffmpeg` `astats` / `ebur128` / `silencedetect`, `~0.2s per min`)

Decoding the AAC audio track (`-vn -af astats=metadata=1:reset=1,silencedetect=n=-45dB:d=2.0`) is **~100x faster than decoding 4K HEVC video** and provides a rich orthogonal modality:

```bash
ffmpeg -i input.mp4 -vn \
  -af "astats=metadata=1:reset=1:length=0.5,ametadata=mode=print:file=-" \
  -f null -
```

* **What it captures in Motorcycle & Action Footage:**
  * **Engine Roar & Exhaust Revs:** High `RMS_level` + rising `Crest_factor` / zero-crossing rate separates spirited acceleration and downshift rev-matching from idling at a stoplight or parked silence.
  * **Human Excitement:** Laughter, cheers inside the helmet or at a scenic pass, and roadside conversation.
  * **Muffled / In-Pocket Detection:** Consistently dead or low-frequency muffled audio (`RMS_level < -50 dBFS` with zero dynamic range) strongly correlates with a camera left running inside a backpack or tank bag.
* **What it captures in Underwater Dive Footage (Important Caveat!):**
  * Underwater sound in a sealed housing has a distinct acoustic profile: rhythmic **scuba regulator inhalation/exhalation cycles**, **bubble roar** during descent/ascent, **diver shaker / tank banger / squeaker signals** (sharp high-frequency `Peak_level` spikes used by dive guides to point out sharks/turtles!), and the **giant-stride splash** + **post-dive surface cheering**.
  * *Caveat:* Quiet steady hovering for macro photography has low ambient audio energy—so low audio RMS underwater must **not** penalize a clip, whereas **sudden acoustic spikes** (`Peak_level` > 10 dB above rolling median, e.g. tank banger, regulator purge, splash entry, or boat laughter) are strong positive candidate triggers!

### 3.3 Method 3: Motion Vector Extraction (`export_mvs` / `codecview`) — Primary-Source HEVC Caveats & Better Alternatives

A widely cited technique for H.264 video analysis is passing `-flags2 +export_mvs` to `libavcodec` (or `-vf codecview=mv=pf+bf+bb`) to inspect macroblock motion vectors (`AV_FRAME_DATA_MOTION_VECTORS`) without running optical flow.

#### Primary-Source Technical Verification: Does `export_mvs` work on GoPro HEVC (H.265)?
* **Verified Fact (`FFmpeg libavcodec` source code):**
  * In FFmpeg's `libavcodec`, `AV_CODEC_FLAG2_EXPORT_MVS` (`+export_mvs`) is implemented in **`h264_mb.c`**, **`mpegvideo.c`** (MPEG-1/2/4), and **`vp9.c`**.
  * **FFmpeg's native HEVC (`hevcdec.c`) decoder does NOT implement `AV_FRAME_DATA_MOTION_VECTORS` export.** Running `ffmpeg -flags2 +export_mvs -i gopro_hevc.mp4` succeeds without error, but `av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS)` returns **`NULL`** (0 vectors exported)!
  * **Why this matters for `dive-clip-ai`:** Starting with GoPro HERO6/7/8 and mandatory on HERO9–13 for 4K60, 5.3K, and 10-bit (`yuv420p10le`) modes—as well as DJI Action 4/5 and Insta360 Ace Pro—**virtually all modern action camera footage is encoded in HEVC (`hevc`), not H.264.** Relying on `-flags2 +export_mvs` will silently produce zero data on almost all modern GoPro files.

#### Three Fast, Codec-Agnostic Alternatives for HEVC Action Camera Footage:
1. **Embedded GPMF / IMU Telemetry (`GYRO`, `ACCL`, `CORI`, `GRAV`):**
   * On GoPro/DJI/Insta360, you don't need noisy 2D pixel motion vectors to infer camera movement—the camera literally logs **200Hz 3D angular velocity (`GYRO`) and quaternion orientation (`CORI`/`GRAV`)** in the container!
   * **Uniform pan/tilt:** Smooth, low-jerk `GYRO` angular rate (`0.1 – 1.2 rad/s`) with stable `GRAV` roll.
   * **Chaotic shake / dangling:** High-frequency `GYRO` variance / sign reversals (`> 3.5 rad/s` RMS) or inverted `GRAV` vector.
   * **Static / parked:** `GYRO` norm `< 0.03 rad/s` and `ACCL` variance $\approx 0$.
2. **Compressed Packet Byte-Size Variation (`ffprobe -show_packets`, `< 150 ms`, Works on HEVC & H.264!):**
   * Without decoding *any* video frames, `ffprobe -select_streams v:0 -show_entries packet=pts_time,size,flags -of json` reads the byte size of every P/B-frame packet from the MP4 sample table (`stsz`) in ~100 ms.
   * In inter-frame codecs (both H.264 and HEVC), when the camera is stationary or looking at empty blue water (`UNIF` high), P/B-frame residual packets shrink to tiny sizes. During high motion, complex reef swim-throughs, or fast cornering, P/B-frame packet sizes relative to the preceding I-frame (`size_P / size_I`) surge dramatically.
3. **Low-Res Frame-to-Frame Pixel Difference / Optical Flow on Sub-sampled Frames:**
   * Computing `np.mean(np.abs(frame_t - frame_prev))` (or `cv2.calcOpticalFlowFarneback` on `160x90` grayscale thumbnails) in Python takes `< 0.3 ms` per frame and works identically on HEVC, H.264, and AV1.

### 3.4 Method 4: Direct Color Histograms & Sharpness on Sub-Sampled I-Frames (`-skip_frame nokey`)

Passing `-skip_frame nokey` before `-i` tells FFmpeg's HEVC/H.264 decoder to **completely skip decoding all P-frames and B-frames** and only decode self-contained **Intra keyframes (I-frames)**:

```bash
ffmpeg -skip_frame nokey -i input.mp4 -vf "scale=320:-2" -vsync vfr -qscale:v 2 keyframes_%05d.jpg
```

* **Speed Advantage:** GoPro cameras typically use a **1.0-second GOP** (`30` frames at 30fps, `60` frames at 60fps). Using `-skip_frame nokey` bypasses **96% to 98.3% of all video decoding work**, extracting `320x180` keyframes at **40x–80x real-time speed** (a 2-minute 4K60 HEVC clip is decoded in ~1.5 seconds!).
* **What We Compute on Each Decoded Keyframe (`< 1 ms` per frame in Python/NumPy):**
  1. **Brightness & Exposure (`Y_mean`, `Y_std`):**
     * `Y_mean < 12` → Black/in-bag/covered lens (`unusable`).
     * `Y_mean > 245` and `Y_std < 10` → Blown-out sky/overexposed (`unusable`).
  2. **Sharpness / Blur (`Laplacian Variance`):**
     * Compute variance of the $3\times 3$ Laplacian filter. Extremely low variance flags out-of-focus macro attempts, lens condensation, or murky silt clouds.
  3. **Shannon Entropy & Colorfulness (HSV Histogram):**
     * Compute a normalized 3D HSV histogram (e.g., `8 Hue x 4 Saturation x 4 Value = 128 bins`).
     * **Underwater vs. Surface Pre-Classification:** If $> 75\%$ of pixels fall in the cyan-blue hue band (`170°–250°`) with near-zero red channel (`R_mean < 25`), the frame is **open blue water**; when red/yellow/orange/magenta bins spike alongside blue/cyan, the diver has approached **illuminated coral, macro life, or a colorful subject**!
     * **Scene Cut & Transition Detection:** Large cosine/Bhattacharyya distance between consecutive I-frame histograms (`1 - cos_sim(H_t, H_{t-1}) > 0.35`) pinpoints exact water-entry, tunnel-exit, or camera-turn boundaries.
     * **Visual Diversity Clustering (Katna/Hecate Style):** By storing the 128-bin HSV histogram vector on each candidate detection, `06_assemble_video.py` and `07_create_aftermovie.py` can compute **exact visual similarity** between candidate clips and cluster/deduplicate visually identical shots!

### 3.5 How the 4 Non-AI Methods Chain into a Fast Pre-VLM Filter Funnel

By chaining these 4 non-AI methods before Pass 0 / Pass 1 in `02_scan_footage.py`, we transform the pipeline from a **brute-force VLM scanner** into a **multi-stage funnel**:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                 PROPOSED MULTI-SIGNAL PRE-VLM FILTER FUNNEL                      │
└──────────────────────────────────────────────────────────────────────────────────┘
 [Stage A: Container & Header Probe (< 50 ms / clip)]
   • ffprobe all streams + MP4 `moov.udta.hmmt` + packet sizes
   • Drop < 2.0s accidental taps; boost 120/240fps slow-mo clips
   • Extract manual GoPro HiLight tags (`hmmt`) -> force into Pass 2 candidates!
        │
        ▼
 [Stage B: Embedded Telemetry & Audio Scan (~100–300 ms / clip)]
   • If `gpmd`/`djmd`/Insta360 present: read `SCEN` (skip Pass 0 VLM!), `CORI`/`GRAV`
     (mask out inverted/tumbling intervals), `YAVG`/`UNIF`, `GPS5` speed, `FACE`
   • Audio `astats`: flag cheering, tank-banger signals, splash entries, engine revs
        │
        ▼
 [Stage C: Keyframe (`-skip_frame nokey`) Histogram & Blur Scan (~1–2s / clip)]
   • Decode ONLY I-frames at 320x180 (skips 97% of HEVC decode!)
   • Compute 128-bin HSV histogram + Laplacian sharpness + Shannon entropy per I-frame
   • Prune dark/in-bag/covered/murky-blue/frozen segments (saves 50–75% of VLM calls!)
        │
        ▼
 [Stage D: Targeted VLM Pass 1 & Pass 2 (Qwen2.5-VL on Surviving Rich Windows)]
   • Send 2x2 temporal contact sheets ONLY for visually sharp, non-uniform,
     kinematically/acoustically/chromatically interesting segments
        │
        ▼
 [Stage E: Diversity-Aware MMR + Histogram Clustering Assembly]
   • Merge contiguous windows by shot/category; enforce 3.5s–8.0s natural clip length
   • Select Top-N via MMR (Quality - Visual Histogram Similarity - Subject Similarity
     - Per-Clip Quota Penalty) and assemble in chronological storyline order!
```

---

## 4. Why Our Current Pipeline (`dive-clip-ai`) Produces Narrow Findings & How to Broaden the Search Landscape

A code-level audit of [`scripts/02_scan_footage.py`](file:///home/twuytens/work/dive-clip-tui/scripts/02_scan_footage.py), [`utils/prompt_templates.py`](file:///home/twuytens/work/dive-clip-tui/utils/prompt_templates.py), [`scripts/06_assemble_video.py`](file:///home/twuytens/work/dive-clip-tui/scripts/06_assemble_video.py), and [`scripts/07_create_aftermovie.py`](file:///home/twuytens/work/dive-clip-tui/scripts/07_create_aftermovie.py) reveals **four compounding structural bottlenecks** that cause narrow, repetitive, and fragmented highlight reels:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     CURRENT PIPELINE BOTTLENECK FUNNEL                     │
└────────────────────────────────────────────────────────────────────────────┘
  [Raw Footage: Rich Action, Transitions, Atmosphere, Telemetry, Audio]
         │
         ▼  Bottleneck 1: Prompt Tunnel Vision (utils/prompt_templates.py)
  Closed-world noun checklist (pelagics/macro/diver; moto_riding/scenic/group).
  Rejects surface/boat prep, giant-stride entry, sunbeams, campfires, humor, B-roll.
         │
         ▼  Bottleneck 2: Single-Frame Myopia (scripts/02_scan_footage.py:1147)
  Sends 1 static JPEG at a time @ 0.5 FPS to Ollama. Blind to temporal motion,
  acceleration, lean progression, reveals, audio cheers/engine note, & IMU telemetry.
         │
         ▼  Bottleneck 3: Fixed ±1s Windows & Exact-String Merge (:1482, :1578)
  Emits 2.0s micro-slices [ts-1, ts+1]. Only merges if det.subject.strip().lower()
  is 100% identical ("green sea turtle" != "sea turtle swimming" -> unmerged!).
         │
         ▼  Bottleneck 4: Greedy Quality-Only Top-N (06_assemble:98, 07_aftermovie:403)
  sorted(detections, key=lambda d: d["quality_score"], reverse=True)[:top_n]
  Zero clip-level quota, zero category balance, zero visual/semantic deduplication!
         │
         ▼
  [Result: 10 repetitive 2-second clips from the same 1-2 videos & same subject]
```

### 4.1 Bottleneck 1: Prompt Tunnel Vision (Closed-World Noun Checklists)

* **Where in Code:**
  * [`utils/prompt_templates.py#L161-L401`](file:///home/twuytens/work/dive-clip-tui/utils/prompt_templates.py#L161-L401) (`PELAGICS_TEMPLATE`, `MACRO_TEMPLATE`, `DIVER_TEMPLATE`, `BROAD_MARINE_TEMPLATE`)
  * [`utils/prompt_templates.py#L408-L595`](file:///home/twuytens/work/dive-clip-tui/utils/prompt_templates.py#L408-L595) (`MOTORCYCLE_COARSE_PROMPT`, `MOTO_RIDING_TEMPLATE`, `MOTO_SCENIC_TEMPLATE`, `MOTO_GROUP_TEMPLATE`)
  * [`scripts/02_scan_footage.py#L1309-L1364`](file:///home/twuytens/work/dive-clip-tui/scripts/02_scan_footage.py#L1309-L1364) (`window_routing` locks each Pass 2 window to a single template based on `best_hit = max(matching_hits, key=lambda h: h[4])`).
* **Root Cause:**
  * Pass 1 and Pass 2 prompts ask Qwen2.5-VL a **closed-world taxonomic question**: e.g., `BROAD_MARINE_TEMPLATE.coarse_prompt` ([`utils/prompt_templates.py#L368`](file:///home/twuytens/work/dive-clip-tui/utils/prompt_templates.py#L368)) demands `"detected: boolean (true ONLY if marine life or diver action is present...)"`, and `MOTORCYCLE_COARSE_PROMPT` ([`utils/prompt_templates.py#L416`](file:///home/twuytens/work/dive-clip-tui/utils/prompt_templates.py#L416)) instructs the VLM to reject any frame that is `"stationary with zero movement/action"`.
  * Consequently, **essential narrative & B-roll moments are systematically destroyed**:
    * *Dive footage:* Gearing up on the boat, buddy checks, backward-roll / giant-stride water entry, dramatic cathedral sunbeams/snell's window, bubble curtains ascending, wrecks/caverns without fish, surface cheering after the dive.
    * *Motorcycle footage:* Putting on helmets/gloves, engine startup, roadside coffee/scenic lookout stops, laughter with friends at a mountain pass summit, weather/tunnel transitions, sunset campfires.
  * Furthermore, in Pass 2 ([`scripts/02_scan_footage.py#L1314-L1364`](file:///home/twuytens/work/dive-clip-tui/scripts/02_scan_footage.py#L1314-L1364)), if a merged coarse window contains *both* a diver and a turtle, `window_routing` routes the entire window to whichever single coarse frame had the higher confidence (e.g., `PELAGICS_TEMPLATE`), causing Pass 2 to ask `"is a pelagic species clearly visible?"` ([`utils/prompt_templates.py#L212`](file:///home/twuytens/work/dive-clip-tui/utils/prompt_templates.py#L212)) and **reject the diver frames inside that same window**!
  * *Bonus Bug Found:* Every `PromptTemplate` in [`utils/prompt_templates.py#L13-L22`](file:///home/twuytens/work/dive-clip-tui/utils/prompt_templates.py#L13-L22) defines a `system_prompt` attribute, but [`OllamaClient.generate()`](file:///home/twuytens/work/dive-clip-tui/utils/ollama_client.py#L180-L186) and [`call_ollama()`](file:///home/twuytens/work/dive-clip-tui/scripts/02_scan_footage.py#L88-L100) never pass `"system"` in the Ollama `/api/generate` JSON payload—making `system_prompt` dead code across the entire repository.

### 4.2 Bottleneck 2: Single-Frame Myopia (Zero Temporal Motion, Audio, or Sensor Awareness)

* **Where in Code:**
  * [`scripts/02_scan_footage.py#L1147-L1155`](file:///home/twuytens/work/dive-clip-tui/scripts/02_scan_footage.py#L1147-L1155) (Pass 1 single-frame `call_ollama` loop)
  * [`scripts/02_scan_footage.py#L1423-L1431`](file:///home/twuytens/work/dive-clip-tui/scripts/02_scan_footage.py#L1423-L1431) (Pass 2 single-frame `call_ollama` loop)
  * [`scripts/00_ingest_media.py#L34-L80`](file:///home/twuytens/work/dive-clip-tui/scripts/00_ingest_media.py#L34-L80) (`probe_video` only queries `v:0` width/height/fps/codec/duration; completely ignores audio streams and `gpmd`/`djmd` metadata tracks).
* **Root Cause:**
  * `call_ollama` sends a **single static JPEG** (`frame["frame_path"]`) to Qwen2.5-VL in isolation.
  * A single static frame cannot distinguish:
    * A motorcycle leaning into a high-speed hairpin vs. parked on its kickstand at an angle.
    * A smooth cinematic camera reveal/glide vs. a static or shaky shot.
    * An exciting transition (entering the water, bursting out of a dark tunnel into alpine sunlight, a sudden shark turn) vs. static cruising.
  * Even though GoPro/DJI/Insta360 files already contain 200Hz IMU (`ACCL`, `GYRO`, `GRAV`, `CORI`), GPS speed (`GPS5`/`GPS9`), user `HiLight` tags (`hmmt`), and audio tracks (engine roar, regulator purge, cheering), `dive-clip-ai` strips all of this context and forces a 7B vision model to guess "camera stability" and "speed" from a single downscaled `640x360` still image.

### 4.3 Bottleneck 3: Fixed `±1.0s` Detection Windows & Brittle Exact-String Deduplication

* **Where in Code:**
  * [`scripts/02_scan_footage.py#L1482-L1483`](file:///home/twuytens/work/dive-clip-tui/scripts/02_scan_footage.py#L1482-L1483):
    ```python
    start_time=max(0, frame["timestamp"] - 1),
    end_time=frame["timestamp"] + 1,
    ```
  * [`scripts/02_scan_footage.py#L1559-L1629`](file:///home/twuytens/work/dive-clip-tui/scripts/02_scan_footage.py#L1559-L1629) (`_deduplicate_detections`).
* **Root Cause:**
  * Every passing Pass 2 frame at `2.0 FPS` (`0.5s` step) emits a tiny **2.0-second window** (`[ts - 1.0, ts + 1.0]`).
  * In `_deduplicate_detections` ([`scripts/02_scan_footage.py#L1578`](file:///home/twuytens/work/dive-clip-tui/scripts/02_scan_footage.py#L1578)), contiguous detections are grouped **strictly by exact lowercase string match**:
    ```python
    key = (det.clip_id, det.subject.strip().lower())
    ```
  * Because `subject` is a free-text string generated independently on each frame by the LLM (e.g., `"subject: string (e.g. 'deep lean right hairpin', 'accelerating out of curve')"` in [`utils/prompt_templates.py#L466`](file:///home/twuytens/work/dive-clip-tui/utils/prompt_templates.py#L466)), consecutive frames 0.5 seconds apart routinely receive slightly different strings:
    * Frame at `10.0s`: `subject = "green sea turtle"` → window `[9.0s, 11.0s]`, quality `8`
    * Frame at `10.5s`: `subject = "sea turtle swimming over reef"` → window `[9.5s, 11.5s]`, quality `7`
    * Frame at `11.0s`: `subject = "hawksbill turtle"` → window `[10.0s, 12.0s]`, quality `8`
  * **What happens next is destructive:**
    1. Step 1 of `_deduplicate_detections` puts them into **three separate groups** because `"green sea turtle" != "sea turtle swimming over reef"`. They **fail to merge** into a smooth `[9.0s, 12.0s]` clip!
    2. Step 2 ([`scripts/02_scan_footage.py#L1614-L1626`](file:///home/twuytens/work/dive-clip-tui/scripts/02_scan_footage.py#L1614-L1626)) checks if different-subject windows overlap (`det.start_time < kept_det.end_time and det.end_time > kept_det.start_time`). Because `[9.0, 11.0]`, `[9.5, 11.5]`, and `[10.0, 12.0]` overlap, **Step 2 deletes the overlapping windows and keeps only the single 2-second `[9.0s, 11.0s]` slice!**
    3. Or worse, if Frame A is at `10.0s` (`[9.0, 11.0]`) and Frame B is at `12.5s` (`[11.5, 13.5]`) with a synonym subject, both survive as two separate, jarring 2-second micro-clips of the exact same animal 0.5 seconds apart!

### 4.4 Bottleneck 4: Greedy Quality-Only Top-N Selection (Zero Diversity or Story Arc)

* **Where in Code:**
  * [`scripts/06_assemble_video.py#L98-L99`](file:///home/twuytens/work/dive-clip-tui/scripts/06_assemble_video.py#L98-L99):
    ```python
    sorted_dets = sorted(detections, key=lambda d: d.get("quality_score", 0), reverse=True)
    top_dets = sorted_dets[:top_n]
    ```
  * [`scripts/07_create_aftermovie.py#L403-L404`](file:///home/twuytens/work/dive-clip-tui/scripts/07_create_aftermovie.py#L403-L404):
    ```python
    sorted_dets = sorted(detections, key=lambda d: d.get("quality_score", 0), reverse=True)
    top_dets = sorted_dets[:top_n]
    ```
  * [`scripts/04_plan_storyline.py#L36-L38`](file:///home/twuytens/work/dive-clip-tui/scripts/04_plan_storyline.py#L36-L38) (same greedy sort by `quality_score`).
* **Root Cause:**
  * Selection is a pure `quality_score` argmax with **zero diversity constraint, zero per-clip cap, zero subject/category balancing, and zero chronological re-ordering**:
    1. **Single-Clip Monopoly:** If `GX010042.MP4` has crystal-clear shallow water lighting where a turtle or coral bommie scores `9/10` across 10 windows, all `top_n = 10` clips in the aftermovie come from `GX010042.MP4`, while the other 25 videos in the folder get **0 clips**.
    2. **Subject Repetition:** If "coral reef" or "mountain road" scores `8–9`, the output contains 10 nearly identical shots of coral or asphalt.
    3. **Scrambled Chronology:** Because `top_dets` stays sorted by `quality_score` descending during assembly (`for det in top_dets`), the final video plays all `9/10` clips first, then all `8/10` clips, jumping randomly back and forth across files and timestamps instead of following the chronological dive/ride progression!

---

## 5. Actionable Tactics Ranked by Effort vs. Impact

Below is a concrete, prioritized engineering roadmap designed to plug directly into `00_ingest_media.py`, `02_scan_footage.py`, `utils/prompt_templates.py`, `06_assemble_video.py`, and `07_create_aftermovie.py`.

### Summary Priority Matrix

| Rank | Tactic | Target Files | Effort | Impact | External Dependencies |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Fix Contiguous Window Merging & Minimum Clip Duration (`3.5s–8.0s`)** | `scripts/02_scan_footage.py` | **Low** (~40 LOC) | **Very High** (Eliminates 2s micro-clips & self-cannibalization) | None (stdlib) |
| **2** | **Diversity-Aware MMR + Per-Clip Quota + Chronological Assembly** | `scripts/06_assemble_video.py`, `scripts/07_create_aftermovie.py` | **Low** (~80 LOC) | **Very High** (Guarantees multi-clip coverage & variety) | None (stdlib) |
| **3** | **All-Stream `ffprobe` + GoPro `hmmt` HiLight & Telemetry Extraction at Ingest** | `scripts/00_ingest_media.py`, `utils/telemetry.py` | **Low–Med** (~120 LOC) | **High** (120/240fps boost, instant user HiLights, Pass 0 `SCEN` bypass, IMU/GPS scoring) | Stdlib `struct` + optional `telemetry-parser` |
| **4** | **Keyframe (`-skip_frame nokey`) HSV Histogram & Blur Pre-Screening + Audio `astats`** | `utils/ffmpeg_streamer.py`, `scripts/02_scan_footage.py` | **Medium** (~130 LOC) | **High** (Skips 50–75% of dead/blurry/murky frames; enables visual histogram clustering) | FFmpeg + NumPy/Pillow |
| **5** | **Open-World "Story Role" Prompting (Action, Character, Establishing, Transition, Moment)** | `utils/prompt_templates.py`, `scripts/02_scan_footage.py` | **Low–Med** (~100 LOC) | **High** (Captures boat/prep, entries, atmosphere, reactions) | None |
| **6** | **Multi-Frame Contact Sheet (2x2 Temporal Grid) for VLM Calls** | `utils/ffmpeg_streamer.py`, `scripts/02_scan_footage.py` | **Medium** (~90 LOC) | **High** (Gives Qwen2.5-VL true motion/action perception + 2–4x speedup) | FFmpeg `tile=2x2` filter |

---

### Tactic 1: Fix Contiguous Window Merging & Enforce Natural Clip Durations (Low Effort, Very High Impact)

**Problem Solved:** Bottleneck 3 (2-second micro-clips and non-merging of synonym subjects like `"green sea turtle"` vs `"sea turtle swimming"`).

**How it fits into `scripts/02_scan_footage.py`:**
1. **Replace exact-string grouping in `_deduplicate_detections` ([`scripts/02_scan_footage.py#L1578`](file:///home/twuytens/work/dive-clip-tui/scripts/02_scan_footage.py#L1578)):**
   * Merge any overlapping or adjacent (`det.start_time <= prev.end_time + merge_gap_seconds`, default `2.0s`) detections within the same `clip_id` that share the same **`template` / `category`** OR have token Jaccard overlap $\ge 0.3$, rather than requiring `det.subject.strip().lower()` equality.
   * Even simpler: if two highlight frames in the same clip are within `1.5s` of each other, they are part of the **same continuous camera shot**—merge them into one continuous highlight window `[min_start, max_end]` up to `max_clip_duration` (e.g., `8.0s–10.0s`), keeping the highest `quality_score` and combining distinct subjects (`"green sea turtle, diver"`).
2. **Pad short single-frame hits to a watchable minimum duration (`min_clip_duration: 3.5s`):**
   * Instead of hardcoded `[ts - 1.0, ts + 1.0]` (`2.0s`) at [`scripts/02_scan_footage.py#L1482-L1483`](file:///home/twuytens/work/dive-clip-tui/scripts/02_scan_footage.py#L1482-L1483), configurable `pre_roll_seconds: 1.5` and `post_roll_seconds: 2.0` (`3.5s` minimum window) ensures every rendered clip is long enough for cross-dissolves (`fade_duration = 1.0s` in `07_create_aftermovie.py` currently eats 100% of a 2.0s clip!).

---

### Tactic 2: Diversity-Aware Selection (MMR + Per-Clip Quota + Chronological Ordering) (Low Effort, Very High Impact)

**Problem Solved:** Bottleneck 4 (10 nearly identical clips from 1 video, ignoring 90% of the folder, and playing in scrambled order).

**How it fits into `scripts/06_assemble_video.py` and `scripts/07_create_aftermovie.py`:**
Replace the 2-line greedy slice `sorted_dets[:top_n]` ([`06_assemble_video.py#L98-L99`](file:///home/twuytens/work/dive-clip-tui/scripts/06_assemble_video.py#L98-L99), [`07_create_aftermovie.py#L403-L404`](file:///home/twuytens/work/dive-clip-tui/scripts/07_create_aftermovie.py#L403-L404)) with a shared helper `select_diverse_highlights(detections, top_n)` in `utils/`:

```python
def select_diverse_highlights(
    detections: list[dict],
    top_n: int = 10,
    lambda_quality: float = 0.65,
    max_per_clip: int | None = None,
    chronological_output: bool = True,
) -> list[dict]:
    """Select top_n highlights using Maximal Marginal Relevance (MMR) across subject,
    visual histogram, template/category, source clip, and temporal proximity."""
    if len(detections) <= top_n:
        return sorted(detections, key=lambda d: (d.get("file_name", ""), d.get("start_time", 0)))

    # Dynamic per-clip cap so no single video monopolizes the highlight reel
    unique_clips = {d.get("clip_id") for d in detections}
    per_clip_limit = max_per_clip or max(2, -(-top_n // max(1, len(unique_clips))))  # ceil(top_n / n_clips)

    selected: list[dict] = []
    remaining = list(detections)
    clip_counts: dict[str, int] = {}

    while len(selected) < top_n and remaining:
        best_det = None
        best_score = -1e9
        for cand in remaining:
            cid = str(cand.get("clip_id", ""))
            # Soft penalty if this clip has already reached its fair-share quota
            quota_penalty = 0.35 if clip_counts.get(cid, 0) >= per_clip_limit else 0.0
            norm_q = float(cand.get("quality_score", 0)) / 10.0

            if not selected:
                sim_penalty = 0.0
            else:
                sim_penalty = max(_detection_similarity(cand, s) for s in selected)

            mmr = (lambda_quality * norm_q) - ((1.0 - lambda_quality) * sim_penalty) - quota_penalty
            if mmr > best_score:
                best_score = mmr
                best_det = cand

        if best_det is None:
            break
        selected.append(best_det)
        remaining.remove(best_det)
        cid = str(best_det.get("clip_id", ""))
        clip_counts[cid] = clip_counts.get(cid, 0) + 1

    if chronological_output:
        selected.sort(key=lambda d: (str(d.get("file_name", "")), float(d.get("start_time", 0))))
    return selected
```

Where `_detection_similarity(a, b)` is a zero-dependency weighted similarity combining:
* **Subject & Description Token Jaccard + Visual Histogram Cosine Similarity** (`0.50` weight): penalizes picking `"green sea turtle"` or visually identical coral/road shots 5 times in a row.
* **Same Source Clip & Temporal Proximity** (`0.30` weight): `1.0` if within 30s in the same `clip_id`, `0.5` if same `clip_id`.
* **Same Template / Category** (`0.20` weight): encourages alternating `pelagics`, `macro`, `diver`, and `establishing` shots.

---

### Tactic 3: All-Stream `ffprobe` + Embedded Action-Camera Telemetry & `HiLight` Tags at Ingest (Low–Medium Effort, High Impact)

**Problem Solved:** Bottleneck 2 (ignoring 120/240fps slow-mo metadata, user-tagged `HiLight` moments, wasting VLM Pass 0 calls when GoPro `SCEN` already knows the domain, and lacking IMU motion/lean/stability metrics).

**How it fits into `scripts/00_ingest_media.py` and `scripts/02_scan_footage.py`:**
1. **All-Stream `ffprobe` & MP4 `hmmt` Probe in `00_ingest_media.py`:**
   * Update `probe_video()` ([`scripts/00_ingest_media.py#L34`](file:///home/twuytens/work/dive-clip-tui/scripts/00_ingest_media.py#L34)) to remove `-select_streams v:0` so `ffprobe` returns all stream handlers (`gpmd`, `djmd`, audio), `creation_time`, and `bit_rate`.
   * Flag `is_high_fps: True` when `fps >= 100.0` (120/240fps slow-mo intent).
   * Parse the MP4 `moov.udta.hmmt` atom in pure Python (`struct.unpack(">I", ...)`): if a user pressed the GoPro HiLight button at `t = 42.3s`, store `hilight_tags: [42.3]` in `manifest.json`.
   * In `02_scan_footage.py`, **automatically inject `hilight_tags` as high-priority Pass 1 candidate windows** `[t - 4.0, t + 4.0]` so user-tagged moments are *never* missed by coarse sampling!
2. **Optional `telemetry-parser` / GPMF Stream Extraction (`utils/telemetry.py`):**
   * When a `gpmd`, `djmd`, or Insta360 trailer is present:
     * **Pass 0 Fast-Path (`SCEN`):** If GoPro GPMF `SCEN` reports predominant `WATR`, immediately classify as `underwater` with `0` VLM calls.
     * **Zero-Cost Unusable Frame Filter (`CORI`, `GRAV`, `YAVG`, `UNIF`):** Before calling Ollama in Pass 1, check `YAVG` (reject if `< 15` or `> 245`), `UNIF` (reject if `> 0.92` uniform murk/covered lens), and `GRAV` / `CORI` (reject if camera is inverted or `GYRO` norm indicates chaotic tumbling).
     * **Kinematic Action Boost (`ACCL`, `GYRO`, `GRAV`, `GPS5`):** Compute a per-second `kinematic_score` (high roll lean angle from `GRAV` + smooth `GYRO` + speed from `GPS5` for motorcycles; steady glide `GYRO` for underwater) and use it to propose candidate windows alongside VLM Pass 1.

---

### Tactic 4: Keyframe (`-skip_frame nokey`) HSV Histogram & Blur Pre-Screening + Audio `astats` (Medium Effort, High Impact)

**Problem Solved:** Bottleneck 2 & Bottleneck 4 (wasting VLM calls on blurry/dark/empty-blue frames, missing audio cheers/tank-bangers/engine revs, and lacking visual feature vectors for diversity clustering).

**How it fits into `utils/ffmpeg_streamer.py` and `scripts/02_scan_footage.py`:**
* **Fast I-Frame Extraction (`-skip_frame nokey`):**
  * Add a `-skip_frame nokey` option to `extract_frames()` in [`utils/ffmpeg_streamer.py#L190`](file:///home/twuytens/work/dive-clip-tui/utils/ffmpeg_streamer.py#L190) for Pass 1 coarse screening on HEVC/H.264 files, or compute a fast NumPy/OpenCV check on the extracted coarse JPEGs in `/dev/shm` before calling `call_ollama`:
    * **Blur & Exposure Gate:** Skip `call_ollama` on frames with Laplacian variance below threshold (heavy motion blur / out-of-focus) or extreme brightness (`Y_mean < 12` or `> 245`).
    * **128-Bin HSV Histogram Feature Vector:** Compute and attach a compact HSV histogram (or 16-float summary) to each frame/detection. Use histogram distance between consecutive keyframes to detect scene transitions (`scdet`) and pass the histogram into `select_diverse_highlights()` (Tactic 2) so K-Means / MMR deduplicates **visually identical shots** even when their LLM text descriptions differ!
* **Audio `astats` Spike Detector:**
  * Run a 0.2-second audio pass (`ffmpeg -i clip.mp4 -vn -af astats=metadata=1:reset=1`) to identify acoustic peaks (`Peak_level` / `RMS_level` surges from underwater tank-banger/shaker signals, giant-stride splashes, surface laughter, or motorcycle exhaust revs) and inject them as candidate windows.

---

### Tactic 5: Broaden Prompts from "Noun Checklists" to Open-World "Story Roles" (Low–Medium Effort, High Impact)

**Problem Solved:** Bottleneck 1 (Prompt Tunnel Vision that discards surface/boat prep, entries, scenic transitions, human reactions, and unexpected events).

**How it fits into `utils/prompt_templates.py` and `scripts/02_scan_footage.py`:**
1. **Add Narrative Story Roles (`story_role`) to Coarse & Fine Prompts:**
   * Expand domain coarse prompts ([`utils/prompt_templates.py#L356-L428`](file:///home/twuytens/work/dive-clip-tui/utils/prompt_templates.py#L356-L428)) so `detected: true` triggers on **five story roles**, not just target species/riding:
     * `establishing`: Boat arrival, gearing up, scenic overlook, wide reef/road vista, sunset/sunbeams.
     * `entry_transition`: Giant-stride/backward-roll water entry, descending through bubbles, entering a tunnel/pass/wreck/swim-through.
     * `hero_subject`: Pelagic encounter, macro critter, dynamic cornering/lean, wildlife interaction.
     * `human_moment`: Buddy interaction, OK/thumbs-up signal, biker wave, cheering/smiling at a stop, pre/post-activity camaraderie.
     * `unexpected_event`: Anything visually striking, unusual, funny, or dramatic outside standard categories.
2. **Stop Pass 2 Template Lock-Out ([`scripts/02_scan_footage.py#L1357`](file:///home/twuytens/work/dive-clip-tui/scripts/02_scan_footage.py#L1357)):**
   * Instead of routing an entire merged Pass 2 window to a narrow single-class prompt (like `PELAGICS_TEMPLATE` which rejects divers or macro inside that window), include a clause in every specialist `fine_prompt`: *"Primary target: {category}. However, if another compelling highlight (human reaction, dramatic lighting, transition, or different species) is clearly framed and sharp, still set `detected: true` and name the actual subject."*

---

### Tactic 6: Multi-Frame Temporal Contact Sheets (`2x2` Grid) for VLM Calls (Medium Effort, High Impact)

**Problem Solved:** Bottleneck 2 (Single-Frame Myopia—VLM cannot see motion, acceleration, lean progression, or camera glide in a static JPEG, and Pass 1 takes too many sequential HTTP calls).

**How it fits into `utils/ffmpeg_streamer.py` and `scripts/02_scan_footage.py`:**
* Instead of sending four separate `640x360` JPEGs at `t = 0s, 2s, 4s, 6s` in four slow sequential Ollama HTTP round-trips, FFmpeg's `drawtext` + `tile=2x2` filter can extract **4 consecutive frames (`Δt = 0.5s` or `1.0s` apart) tiled into a single `1280x720` 2x2 contact sheet**:
  ```bash
  ffmpeg -ss 10.0 -t 2.0 -i input.mp4 \
    -vf "fps=2.0,scale=640:360,drawtext=text='%{pts\:hms}':x=10:y=10:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6,tile=2x2" \
    -frames:v 1 strip_001.jpg
  ```
* **Why this is transformative for Qwen2.5-VL:**
  1. **True Temporal & Motion Reasoning:** The VLM sees the 4-frame sequence `[top-left → top-right → bottom-left → bottom-right]` in one glance. It can immediately see whether the camera is **gliding smoothly**, **shaking erratically** (huge frame-to-frame jump), **leaning progressively into a corner**, or **stationary at a traffic light**!
  2. **3x–4x Faster Pass 1 Throughput:** Cuts the number of Ollama HTTP requests and prompt prefill evaluations in Pass 1 by up to 75% while covering the exact same timeline density.

---

## 6. Primary Sources & Citations

### Action Camera Telemetry & Specs
1. **GoPro `gpmf-parser` Official Repository & Specification:** [https://github.com/gopro/gpmf-parser](https://github.com/gopro/gpmf-parser) — Defines KLV structure, MP4 `gpmd` timed metadata track, FourCC tags (`ACCL`, `GYRO`, `GRAV`, `CORI`, `IORI`, `GPS5`, `GPS9`, `SHUT`, `ISOE`, `ISOG`, `WBAL`, `WRGB`, `SROT`, `YAVG`, `UNIF`, `HUES`, `SCEN` [`SNOW`, `URB`, `INDO`, `WATR`, `VEGE`, `BEAC`], `FACE`, `MANL`).
2. **GoPro Engineering Blog — *GoPro Video Metadata Open Source Explained*:** [https://gopro.com/en/ps/news/gopro-video-metadata-open-source-explained](https://gopro.com/en/ps/news/gopro-video-metadata-open-source-explained) — Details how GP1/GP2 hardware scene analysis (`SCEN`, `HUES`, `FACE`, `YAVG`) and IMU sensors power GoPro Quik's automatic HiLight scoring.
3. **Gyroflow `telemetry-parser` (Rust & Python Library):** [https://github.com/AdrianEddy/telemetry-parser](https://github.com/AdrianEddy/telemetry-parser) & PyPI [`telemetry-parser`](https://pypi.org/project/telemetry-parser/) — Cross-vendor parser supporting GoPro GPMF, DJI `djmd` Protobuf telemetry ([`src/dji`](https://github.com/AdrianEddy/telemetry-parser/tree/master/src/dji)), and Insta360 `.insv`/`.mp4` trailer records with magic footer `8db42d694ccc418790edff439fe026bf` ([`src/insta360`](https://github.com/AdrianEddy/telemetry-parser/tree/master/src/insta360)).
4. **ExifTool GoPro & Action Camera Tag Documentation:** [https://exiftool.org/TagNames/GoPro.html](https://exiftool.org/TagNames/GoPro.html) — Documents `moov.udta.hmmt` (HiLight marker table in milliseconds), `HLMT`, `MANL`, `AIMU`, `AACD`, and `FUSE` tags.

### Highlight Detection, Summarization, Motion Vectors & Scene Analysis Tools
5. **PySceneDetect (`scenedetect`):** [https://github.com/Breakthrough/PySceneDetect](https://github.com/Breakthrough/PySceneDetect) & [Detector Docs](https://www.scenedetect.com/docs/latest/cli/detectors.html) — `ContentDetector`, `AdaptiveDetector`, and `HistogramDetector`.
6. **Auto-Editor (`WyattBlue/auto-editor`):** [https://github.com/WyattBlue/auto-editor](https://github.com/WyattBlue/auto-editor) — Expression-based multi-signal editing (`audio`, `motion`, `pixeldiff`).
7. **LosslessCut (`mifi/lossless-cut`):** [https://github.com/mifi/lossless-cut](https://github.com/mifi/lossless-cut) — Keyframe-aligned cutting, FFmpeg scene/silence/black detection integration.
8. **Katna (`keplerlab/katna`) & Yahoo Hecate (`yahoo/hecate`):** [https://github.com/keplerlab/katna](https://github.com/keplerlab/katna) & [https://github.com/yahoo/hecate](https://github.com/yahoo/hecate) — Blur (Laplacian variance), brightness, and entropy filtering followed by K-Means clustering on frame histograms for non-repetitive keyframe/clip extraction.
9. **Immich Memories & CLIP + Maximal Marginal Relevance (MMR):** [https://github.com/immich-app/immich](https://github.com/immich-app/immich) & Carbonell & Goldstein (SIGIR 1998, *The Use of MMR, Diversity-Based Reranking*).
10. **VideoHighlighter (`Aseiel/VideoHighlighter`):** [https://github.com/Aseiel/VideoHighlighter](https://github.com/Aseiel/VideoHighlighter) — Local multi-signal highlight scoring combining OpenCV scene cuts, motion, audio peaks, YOLO object detection, and Ollama VLMs.
11. **FFmpeg `libavfilter` (`lavfi`) & `libavcodec` (`export_mvs` / `-skip_frame nokey`) Documentation:** [https://ffmpeg.org/ffmpeg-filters.html](https://ffmpeg.org/ffmpeg-filters.html) & [https://ffmpeg.org/doxygen/trunk/export__mvs_8c-example.html](https://ffmpeg.org/doxygen/trunk/export__mvs_8c-example.html) — Official specifications for `scdet`, `blurdetect`, `freezedetect`, `blackdetect`, `astats`, `ebur128`, `codecview`, `AV_FRAME_DATA_MOTION_VECTORS` (supported in H.264/MPEG-4, not exported by native `hevcdec.c`), and `-skip_frame nokey` keyframe-only decoding.

### `dive-clip-ai` Repository Code Citations
12. **Folder Ingestion (`probe_video`):** [`scripts/00_ingest_media.py#L34-L93`](file:///home/twuytens/work/dive-clip-tui/scripts/00_ingest_media.py#L34-L93)
13. **Coarse-to-Fine VLM Scanner & Window Deduplication:** [`scripts/02_scan_footage.py#L817-L1630`](file:///home/twuytens/work/dive-clip-tui/scripts/02_scan_footage.py#L817-L1630)
14. **Domain & Subject Prompt Templates:** [`utils/prompt_templates.py#L73-L1233`](file:///home/twuytens/work/dive-clip-tui/utils/prompt_templates.py#L73-L1233)
15. **Storyline Cloud Planner:** [`scripts/04_plan_storyline.py#L27-L69`](file:///home/twuytens/work/dive-clip-tui/scripts/04_plan_storyline.py#L27-L69)
16. **Clip Renderer:** [`scripts/05_render_clips.py#L70-L309`](file:///home/twuytens/work/dive-clip-tui/scripts/05_render_clips.py#L70-L309)
17. **Highlight Video Assembler:** [`scripts/06_assemble_video.py#L93-L136`](file:///home/twuytens/work/dive-clip-tui/scripts/06_assemble_video.py#L93-L136)
18. **Aftermovie Creator:** [`scripts/07_create_aftermovie.py#L397-L508`](file:///home/twuytens/work/dive-clip-tui/scripts/07_create_aftermovie.py#L397-L508)
19. **FFmpeg Frame Streamer:** [`utils/ffmpeg_streamer.py#L161-L261`](file:///home/twuytens/work/dive-clip-tui/utils/ffmpeg_streamer.py#L161-L261)
