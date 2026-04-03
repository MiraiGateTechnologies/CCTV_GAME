# CCTV Casino Game - AI Vehicle Counting Engine

Real-world public traffic CCTV cameras as a betting arena. No RNG - pure real-life traffic.
AI counts vehicles crossing a line/zone in 35-second rounds, and results drive Under/Over/Range/Exact betting.

---

## Architecture

```
SequentialDownloader (1 thread, round-robin, NO GPU)
  YouTube URL pre-warm (yt-dlp Python API + cookies)
  → ffmpeg download 41s clips → quality check
         |
    download_queue (target: 3 clips)
         |
YOLOWorker (1 thread, OWNS GPU)
  GPU warmup on startup → offline_count() on each clip
  tracked_vehicles system (velocity prediction, zero flicker)
         |
    ready_queue (max 10 pre-processed clips)
         |
Scheduler (main thread)
  Cold start: waits for 5 clips → then 56-sec round cycle
  |-- BETTING  (15s)  No stream, hash published
  |-- COUNTING (35s)  Clip plays with detection overlay
  |-- WAITING  (6s)   Stream continues, result revealed
         |
FastAPI Server — REST + WebSocket + MJPEG + LiveKit WebRTC
```

---

## Quick Start

### Prerequisites

- **GPU:** NVIDIA with CUDA (tested on RTX 4060 Ti 8GB)
- **Python:** 3.10+
- **ffmpeg:** installed and in PATH

### Installation

```bash
cd D:\cctv_main\CCTV_GAME

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install ultralytics opencv-python fastapi uvicorn[standard] numpy yt-dlp livekit livekit-api
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install tensorrt-cu12

# Export TensorRT engine (one-time, 5-15 min)
yolo export model=yolo11x.pt format=engine imgsz=1248 half=True device=0 workspace=4

# (Optional) Start LiveKit server for WebRTC streaming
.\start_livekit.bat
```

### Running

```bash
# Production mode (web browser only)
python scheduler.py --config streams_config.json

# Development mode (web + debug GUI with mouse line/ROI drawing)
python scheduler.py --config streams_config.json --gui

# Single stream testing
python main.py --url "https://www.youtube.com/live/VIDEO_ID"
python main.py --video video1

# Validate config without running
python scheduler.py --test

# Download test clips
python network/download.py --url "URL" --name video1 --duration 300
```

**Web interface:** http://localhost:5000
**API docs (Swagger):** http://localhost:5000/docs
**WebSocket:** ws://localhost:5000/ws/game
**LiveKit:** ws://localhost:7880 (auto-connects, MJPEG fallback if unavailable)

---

## Project Structure

```
CCTV_GAME/
|-- core/
|   |-- counting.py            VehicleCounter + offline_count() + tracked_vehicles system
|   |-- geometry_utils.py      Line intersection, point-in-polygon
|   |-- config_manager.py      JSON config I/O
|   |-- odds_engine.py         Boundary generation from lambda_mean + constant multipliers
|   |-- provably_fair.py       SHA-256 commitment scheme for real-world event fairness
|
|-- game/
|   |-- game_api.py            Single source of truth for game state (thread-safe)
|   |-- round_manager.py       Round preparation + clip playback
|   |-- history_tracker.py     Per-stream count history for odds (max 500 rounds)
|
|-- network/
|   |-- stream_manager.py      Sequential download + YOLO worker pipeline
|   |-- download.py            yt-dlp wrapper for URL resolution
|   |-- livekit_publisher.py   Publish frames to LiveKit room via WebRTC
|
|-- ui/
|   |-- renderer.py            Shared tracking overlay (debug + playback), dashboards
|   |-- animations.py          Globe transition, IST timer
|
|-- model_training/            Custom model training pipeline (knowledge distillation)
|   |-- extract_frames.py      Extract frames from CCTV clips
|   |-- auto_label.py          Auto-label with teacher model (yolo11x)
|   |-- verify_labels.py       Visual label QA tool
|   |-- train.bat              Training command
|   |-- export_engine.bat      TensorRT export command
|   |-- deploy.py              Deployment helper (copy model + update code)
|   |-- dataset.yaml           Training dataset config
|
|-- main.py                    Single stream entry point (development/testing)
|-- scheduler.py               Multi-stream game orchestrator (production)
|-- web_server.py              FastAPI + Uvicorn + WebSocket
|
|-- streams_config.json        IST time slots + stream URLs
|-- bytetrack.yaml             ByteTrack tracker config
|-- line_configs/              Per-stream counting line/ROI configs (auto-saved)
|-- round_history.json         Persisted count history for odds
|-- cookies.txt                (Optional) YouTube cookies for anti-429
|-- templates/index.html       Web viewer (LiveKit WebRTC + MJPEG fallback)
|
|-- livekit_server/
|   |-- livekit-server.exe     LiveKit native binary (no Docker needed)
|-- start_livekit.bat          Start LiveKit server
|-- docker-compose.yml         LiveKit via Docker (alternative)
|-- livekit.yaml               LiveKit server config
```

---

## Round Cycle (56 seconds)

```
|<-- 15 sec -->|<---------- 35 sec ---------->|<-- 6 sec -->|
|              |                               |             |
|   BETTING    |        COUNTING               |   WAITING   |
|   No stream  |   Stream playing              |   Stream ON |
|   Hash shown |   AI counting ON              |   Count OFF |
|   4 options  |   Count: 0->1->3->5->7        |   Frozen: 7 |
|              |   Odds visible                |   Result    |
|              |<-------- STREAM PLAYS (41 sec) ------------>|
```

**Betting options:** Under, Range, Over, Exact

---

## AI Counting Pipeline

### Detection

- **Model:** YOLO11x TensorRT engine (default: `yolo11x.engine`, configurable via `--model`)
- **Classes:** Car (2), Motorcycle (3), Bus (5), Truck (7)
- **Confidence:** 0.10 (catches distant vehicles)
- **Input resolution:** 1248px (baked into TensorRT engine via `--imgsz`)
- **Precision:** FP16 (TensorRT half precision)
- **Tracker:** ByteTrack (persistent IDs, 180-frame buffer = 6s at 30fps)
- **NMS IoU:** 0.45

### TensorRT (GPU-Optimized Inference)

```bash
# Export (one-time, 5-15 min, GPU-specific)
yolo export model=yolo11x.pt format=engine imgsz=1248 half=True device=0 workspace=4

# Use .pt for development, .engine for production
python scheduler.py --model yolo11x.pt      # PyTorch (flexible, slower)
python scheduler.py --model yolo11x.engine  # TensorRT (fast, GPU-locked)
```

**Notes:**
- Engine file is GPU-specific -- export on the same GPU that will run it
- `imgsz` is fixed at export time -- changing requires re-export
- `.engine` files skip `.to("cuda")` automatically
- GPU warmup runs automatically on startup (10 dummy frames)

### Persistent Vehicle Tracking (tracked_vehicles system)

White tracking dots use a YOLO-independent persistence system:

```
YOLO detects vehicle    → tracked_vehicles UPDATE (position + velocity)
YOLO misses 1-5 frames → tracked_vehicles PREDICT (velocity-based motion)
Vehicle leaves frame    → tracked_vehicles REMOVE (dot disappears)
```

**Guarantees:**
- **No flicker:** Dots come from tracked_vehicles, not YOLO directly. Prediction bridges missed frames.
- **No snake trail:** Position = raw bbox center (zero EMA lag). Only velocity is smoothed.
- **No random dots:** Velocity clamped to 15px/frame max. Boundary check removes out-of-frame predictions.
- **No double dots:** Spatial dedup prevents two dots on same vehicle.
- **No miss counting:** Line crossing checked even during predicted frames.

### Counting Modes

- **Line mode:** Vehicle centroid crosses the drawn line = counted
- **ROI mode:** Vehicle enters then exits the drawn polygon = counted
- One mode at a time per stream. Config stored in `line_configs/`

### Deduplication

- **Spatial cooldown:** 20px radius + 0.5 second window (line mode)
- **ROI footprint:** IoU > 0.25 overlap check prevents double-counting

### Pre-Processing (Background)

1. Pre-warm ALL YouTube URLs at startup (yt-dlp Python API + browser cookies)
2. Download 41s clip via ffmpeg (h264_nvenc GPU encoding at 20fps)
3. Download queue maintained at exactly 3 clips (smart throttling)
4. Run YOLO on first 35 seconds only (every frame, no skip)
5. Store tracked_vehicles positions per frame (including predicted)
6. Validate: count >= 5, blur > 15, brightness 10-250
7. Push to ready queue (cold start waits for 5 clips before first round)
8. Background URL refresh thread runs every 4 hours

### Playback (During Round)

- Pre-recorded clip plays at native FPS
- **No YOLO running** during playback -- stored tracked_vehicles data is overlaid
- Green flash brackets on counted vehicles
- White tracking dots on uncounted vehicles only (disappear after counting)
- **Shared renderer** (`draw_tracking_overlay`) guarantees pixel-identical visuals between debug GUI mode and playback mode
- Stored data includes predicted positions -- playback is exact replay of debug mode
- Frame pacing ensures zero frame skip

---

## Video Streaming (LiveKit WebRTC + MJPEG Fallback)

```
Browser loads page
  -> Tries LiveKit WebRTC (low latency, adaptive quality)
    -> SUCCESS: green dot "LIVE (WebRTC)"
    -> FAIL: falls back to MJPEG (always works)
      -> orange dot "LIVE (MJPEG)"
  -> Retries LiveKit every 10 seconds
```

### LiveKit Setup

```bash
# Option 1: Native binary (recommended, no Docker)
start_livekit.bat

# Option 2: Docker
docker compose up -d
```

Config: `livekit.yaml` (API key: `devkey`, secret: `secret`)

---

## API Endpoints

### REST

| Endpoint | Description |
|----------|-------------|
| `GET /` | HTML game viewer (MJPEG stream) |
| `GET /video_feed` | Raw MJPEG video stream |
| `GET /api/round` | **Primary** -- current round data (timer, count, odds, hash) |
| `GET /api/state` | Raw game state from scheduler |
| `GET /api/verify` | Provably fair verification for last round |
| `GET /api/livekit/token` | LiveKit viewer JWT token |
| `GET /api/livekit/status` | LiveKit publisher connection status |
| `GET /docs` | Auto-generated Swagger API documentation |

### WebSocket (ws://localhost:5000/ws/game)

| Message Type | When | Data |
|-------------|------|------|
| `initial_state` | On connect | Full round state |
| `game_state` | Phase change | Phase, round_id, hash, odds |
| `count_update` | Count changes | `{vehicle_count: N}` |
| `verification` | Round ends | server_seed, result, bet_outcomes, verification_string |

---

## Configuration

### streams_config.json

```json
{
  "count_duration": 35,
  "transition_duration": 15,
  "time_slots": [
    {
      "start": "08:00",
      "end": "12:00",
      "streams": [
        {"name": "Stream 7", "url": "https://www.youtube.com/live/..."},
        {"name": "Stream 16", "url": "https://www.youtube.com/live/..."}
      ]
    }
  ]
}
```

Time slots are in **IST** (UTC+5:30). Streams rotate round-robin within the active slot.

### Per-Stream Line Config (line_configs/Stream_N.json)

```json
{
  "line": [[594, 464], [841, 576]],
  "roi_poly": null,
  "mode": "line"
}
```

Draw lines/ROI using `--gui` mode. Configs auto-save and hot-reload during processing.

### bytetrack.yaml

```yaml
tracker_type: bytetrack
track_high_thresh: 0.20    # Detections above this enter primary matching
track_low_thresh: 0.05     # Rescue even weak detections for existing tracks
new_track_thresh: 0.15     # Min confidence to create new track
track_buffer: 180          # 6 sec at 30fps -- survives pole/pillar occlusion
match_thresh: 0.60         # Relaxed for better re-association after occlusion
fuse_score: true
```

---

## YouTube URL Management

### Anti-429 Rate Limiting

- YouTube URLs pre-warmed at startup using yt-dlp Python API (no subprocess)
- Browser cookies used automatically (Chrome/Edge/Firefox)
- Optional `cookies.txt` file in project root for manual cookie export
- URLs cached for 5 hours (YouTube live URLs valid ~6 hours)
- Background refresh thread runs every 4 hours
- Exponential backoff on 429 errors: 10s -> 30s -> 60s -> 120s -> 300s max

### Manual Cookie Export (if auto-detect fails)

```bash
python -m yt_dlp --cookies-from-browser chrome --cookies cookies.txt "https://www.youtube.com"
```

---

## Development Mode (--gui)

```bash
python scheduler.py --config streams_config.json --gui
```

Opens a debug GUI window showing real-time YOLO background processing:
- Live vehicle detection with tracking
- Counting line/ROI visualization
- Flash brackets on counting events
- Pipeline stats (downloads, YOLO progress, queue)

### Mouse Controls (Debug GUI)

| Action | Effect |
|--------|--------|
| Left-drag | Draw counting line |
| Right-click x4 | Draw ROI polygon (4 corners) |
| Middle-click | Cancel in-progress polygon |

Debug window renders at 1280x720. Mouse coordinates auto-converted to original frame resolution.

---

## Custom Model Training

Train a smaller, faster YOLO model on your CCTV footage using knowledge distillation.
See `model_training/README.md` for full guide.

```bash
# Quick overview (7 steps):
python model_training/extract_frames.py          # Extract frames
python model_training/auto_label.py              # Auto-label with teacher
python model_training/verify_labels.py           # Visual QA (optional)
model_training\train.bat                         # Train student model
model_training\export_engine.bat                 # Export TensorRT
python model_training/deploy.py                  # Deploy
python scheduler.py --model cctv_yolo11n.engine  # Test
```

---

## Provably Fair System

Uses a **commitment scheme** -- the only correct approach for real-world event games where the result (vehicle count) is a physical measurement, not seed-derived.

### Commitment Contents

```
commitment_hash = SHA-256(
    server_seed:result:under:range_low:range_high:over:exact_1:exact_2:round_id
)
```

### Round Flow

```
BEFORE BETTING:
  1. YOLO processes clip -> result = 7
  2. server_seed = secrets.token_hex(32)
  3. Boundaries generated from lambda_mean
  4. commitment_hash published

BETTING (15s):
  5. Players see: hash, boundaries, multipliers
  6. Players don't see: server_seed, result

COUNTING (35s):
  7. Video plays, count increases live

WAITING (6s):
  8. REVEAL: server_seed + result
  9. Player verifies: recompute SHA-256 -> must match commitment
```

### Verification Endpoint

`GET /api/verify` returns all data needed for independent verification.

---

## Odds Generation

Odds = the **boundary numbers** for each bet option (change every round based on historical average).
Multipliers and win chances are **CONSTANT** every round.

```
Odds (change every round):
  Under < 14  |  Range 16-18  |  Over > 22  |  Exact 20 or 21

Multipliers (CONSTANT):
  Under: 3.13x  |  Range: 2.35x  |  Over: 3.76x  |  Exact: 18.8x

Win Chances (CONSTANT):
  Under: 30%  |  Range: 40%  |  Over: 25%  |  Exact: 5%
```

### Boundary Formula (lambda_mean = historical average count)

```python
under      = floor(lambda_mean * 0.75 * adjust_factor) - 1
range_low  = round(lambda_mean - 0.5)
range_high = round(lambda_mean + 0.8)
over       = ceil(lambda_mean * 1.35 * adjust_factor) + 1
exact_1    = round(lambda_mean)
exact_2    = exact_1 + 1
```

---

## Clip Validation Rules

| Check | Threshold | Action |
|-------|-----------|--------|
| Vehicle count | >= 5 | Approve / reject + retry |
| Image blur | > 15 (Laplacian variance) | Reject |
| Brightness | 10-250 (mean pixel value) | Reject |
| Clip duration | >= CLIP_DURATION - 3 sec | Reject if too short |
| Max retries per stream | 5 consecutive fails | Skip stream, reset |
| Download timeout | 90 seconds | Abort + retry |

---

## CLI Reference

### scheduler.py (Production)

```bash
python scheduler.py --config streams_config.json
python scheduler.py --config streams_config.json --gui
python scheduler.py --config streams_config.json --model yolo11x.pt --imgsz 1600
python scheduler.py --web-port 8080
python scheduler.py --test
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | streams_config.json | Path to streams config |
| `--model` | yolo11x.engine | YOLO model (.engine or .pt) |
| `--imgsz` | 1248 | YOLO input resolution |
| `--gui` | false | Enable debug GUI window |
| `--web-port` | 5000 | FastAPI server port |
| `--test` | false | Validate config and exit |

### main.py (Testing)

```bash
python main.py --url "https://youtube.com/live/..."
python main.py --video video1
python main.py --video video1 --model yolo11x.engine --conf 0.15
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--url` | - | YouTube or stream URL |
| `--video` | - | Local video file |
| `--model` | yolo11x.engine | YOLO model |
| `--conf` | 0.30 | Confidence threshold |
| `--imgsz` | 1600 | Input resolution |
| `--no-gui` | false | Headless mode |
| `--web-port` | 5000 | Web server port |

### network/download.py (Clip Download)

```bash
python network/download.py --url "YOUTUBE_URL" --name video1 --duration 300
python network/download.py --list
```

---

## Dependencies

```
ultralytics          # YOLO inference + ByteTrack tracking
opencv-python        # Video I/O, image processing, GUI
fastapi              # REST API + WebSocket server
uvicorn[standard]    # ASGI server (httptools, websockets)
numpy                # Array operations
pydantic             # API data validation (bundled with FastAPI)
yt-dlp               # YouTube URL resolution (Python API + subprocess)
livekit              # LiveKit Python SDK (WebRTC frame publishing)
livekit-api          # LiveKit server API (token generation)
torch + torchvision  # PyTorch with CUDA (cu121)
tensorrt-cu12        # TensorRT for GPU-optimized inference
```

**Hardware:** NVIDIA GPU with CUDA required (tested on RTX 4060 Ti 8GB).
**Optional:** Docker Desktop for LiveKit server (MJPEG fallback works without it).

---

## Performance (RTX 4060 Ti 8GB)

| Component | Time |
|-----------|------|
| YouTube URL resolve (cached) | 0 sec (pre-warmed) |
| ffmpeg download (41s clip, NVENC 20fps) | ~45-50 sec |
| YOLO processing (35s, 700 frames) | ~18-25 sec (TensorRT FP16) |
| Quality check | < 1 sec |
| Game round | 56 sec (15+35+6) |
| GPU warmup (one-time at startup) | ~3 sec |

### GPU Optimization Tips

```bash
# Lock GPU clocks for consistent performance
nvidia-smi -lgc 2535
nvidia-smi -pl 160

# Monitor GPU during processing
nvidia-smi dmon -s pucvmet -d 1
```
