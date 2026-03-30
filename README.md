# CCTV Casino Game - AI Vehicle Counting Engine

Real-world public traffic CCTV cameras as a betting arena. No RNG - pure real-life traffic.
AI counts vehicles crossing a line/zone in 35-second rounds, and results drive Under/Over/Range/Exact betting.

---

## Architecture

```
StreamDownloader (N parallel, no GPU)     Download 41s clips from live CCTV
         |
    download_queue
         |
YOLOWorker (1 thread, owns GPU)          offline_count() on each clip
         |
    ready_queue (pre-processed clips)
         |
Scheduler (main thread)                  56-sec round cycle
  |-- BETTING  (15s)  No stream, hash published
  |-- COUNTING (35s)  Clip plays with detection overlay
  |-- WAITING  (6s)   Stream continues, result revealed
         |
FastAPI Server                            REST + WebSocket + MJPEG
```

---

## Quick Start

```bash
# Install dependencies
pip install ultralytics opencv-python fastapi uvicorn[standard] numpy yt-dlp livekit livekit-api
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install tensorrt-cu12   # for TensorRT engine support

# Export TensorRT engine (one-time, 10-30 min)
yolo export model=yolo11x.pt format=engine imgsz=1600 half=True

# Start LiveKit server (optional, MJPEG works without it)
.\start_livekit.bat

# Production mode (web browser only)
python scheduler.py --config streams_config.json

# Development mode (web + debug GUI with mouse line/ROI drawing)
python scheduler.py --config streams_config.json --gui

# Single stream testing
python main.py --url "https://www.youtube.com/live/VIDEO_ID"
python main.py --video video1
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
|   |-- counting.py            VehicleCounter + offline_count()
|   |-- geometry_utils.py      Line intersection, point-in-polygon
|   |-- config_manager.py      JSON config I/O
|   |-- odds_engine.py         Boundary generation from lambda_mean + constant multipliers
|   |-- provably_fair.py       SHA-256 commitment scheme for real-world event fairness
|
|-- game/
|   |-- game_api.py            Single source of truth for game state
|   |-- round_manager.py       Round preparation + clip playback
|   |-- history_tracker.py     Per-stream count history for odds
|
|-- network/
|   |-- stream_manager.py      Parallel downloads + YOLO worker pipeline
|   |-- download.py            yt-dlp wrapper for URL resolution
|   |-- livekit_publisher.py   Publish frames to LiveKit room via WebRTC
|
|-- ui/
|   |-- renderer.py            Shared tracking overlay (debug + playback), dashboards
|   |-- animations.py          Globe transition, IST timer
|
|-- main.py                    Single stream entry point (development/testing)
|-- scheduler.py               Multi-stream game orchestrator (production)
|-- web_server.py              FastAPI + Uvicorn + WebSocket
|
|-- streams_config.json        IST time slots + stream URLs
|-- bytetrack.yaml             ByteTrack tracker config
|-- line_configs/              Per-stream counting line/ROI configs
|-- round_history.json         Persisted count history for odds
|-- templates/index.html       Web viewer (LiveKit WebRTC + MJPEG fallback)
|
|-- livekit_server/
|   |-- livekit-server.exe     LiveKit native binary (no Docker needed)
|-- start_livekit.bat          Start LiveKit server (double-click)
|-- docker-compose.yml         LiveKit via Docker (alternative)
|-- livekit.yaml               LiveKit server config (API key/secret)
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
- **Model:** YOLO11x TensorRT engine (default: `yolo11x.engine`, configurable via --model)
- **Classes:** Car (2), Motorcycle (3), Bus (5), Truck (7)
- **Confidence:** 0.10 (catches distant vehicles)
- **Input resolution:** 1600px (baked into TensorRT engine)
- **Precision:** FP16 (TensorRT half precision)
- **Tracker:** ByteTrack (persistent IDs, 150-frame buffer = 5s at 30fps)

### TensorRT (GPU-Optimized Inference)

The default model is a TensorRT engine — pre-compiled CUDA kernels for max speed.

```bash
# Export (one-time, 10-30 min, GPU-specific)
yolo export model=yolo11x.pt format=engine imgsz=1600 half=True
# Creates: yolo11x.engine (~200MB)

# Use .pt for development, .engine for production
python scheduler.py --model yolo11x.pt      # PyTorch (flexible, slower)
python scheduler.py --model yolo11x.engine  # TensorRT (fast, GPU-locked)
```

**Notes:**
- Engine file is GPU-specific — export on the same GPU that will run it
- `imgsz` is fixed at export time (1600) — changing requires re-export
- `.engine` files skip `.to("cuda")` automatically (already on GPU)
- `.pt` files still work (auto-moved to CUDA)

### Counting Modes
- **Line mode:** Vehicle centroid crosses the drawn line = counted
- **ROI mode:** Vehicle enters then exits the drawn polygon = counted
- One mode at a time per stream. Config stored in line_configs/

### Deduplication
- **Spatial cooldown:** 20px radius + 0.5 second window (line mode)
- **ROI footprint:** IoU > 0.25 overlap check prevents double-counting

### Pre-Processing (Background)
1. Download 41s clip via ffmpeg (-c copy, no re-encoding)
2. Download queue maintained at exactly 3 clips (smart throttling)
3. Run YOLO on first 35 seconds (every frame, no skip)
4. Validate: count >= 5, blur > 15, brightness 10-250
5. Store: clip + total count + per-frame detections + counting events
6. Push to ready queue (cold start waits for 5 clips before first round)

### Playback (During Round)
- Pre-recorded clip plays at native FPS
- **No YOLO running** during playback - stored detections are overlaid
- Green flash brackets on counted vehicles
- White tracking dots on **uncounted** vehicles only (dots disappear after counting)
- **Shared renderer** (`draw_tracking_overlay`) guarantees pixel-identical visuals
  between debug GUI mode and playback mode — single source of truth
- Frame pacing ensures zero frame skip

---

## Video Streaming (LiveKit WebRTC + MJPEG Fallback)

Two streaming modes with automatic fallback:

```
Browser loads page
  → Tries LiveKit WebRTC (low latency, adaptive quality)
    → SUCCESS: green dot "LIVE (WebRTC)"
    → FAIL: falls back to MJPEG (always works)
      → orange dot "LIVE (MJPEG)"
  → Retries LiveKit every 10 seconds
```

### LiveKit Setup

```bash
# Option 1: Native binary (recommended, no Docker needed)
start_livekit.bat
# OR:
livekit_server\livekit-server.exe --config livekit.yaml

# Option 2: Docker (alternative)
docker compose up -d
```

Config: `livekit.yaml` (API key: `devkey`, secret: `secret`)

### LiveKit API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/livekit/token?identity=player1` | Generate viewer JWT token |
| `GET /api/livekit/status` | Check publisher connection status |

### Without Docker

LiveKit is optional. Without it, MJPEG streaming works as before. The frontend auto-detects and uses the available method.

---

## API Endpoints

### REST

| Endpoint | Description |
|----------|-------------|
| `GET /` | HTML game viewer (MJPEG stream) |
| `GET /video_feed` | Raw MJPEG video stream |
| `GET /api/round` | **Primary** - current round data (timer, count, odds, hash) |
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

### /api/round Response by Phase

**BETTING (15s):**
```json
{
  "phase": "BETTING",
  "round_id": 1234,
  "stream_name": "Abbey Road London",
  "phase_timer": 12.0,
  "round_timer": 53.0,
  "commitment_hash": "e4d2f1a8b3c7d9e5f6a2b4c8...",
  "vehicle_count": 0,
  "boundaries": {
    "under": 14, "range_low": 20, "range_high": 21,
    "over": 28, "exact_1": 20, "exact_2": 21
  },
  "odds": {"under": 3.13, "range": 2.35, "over": 3.76, "exact": 18.8},
  "win_chances": {"under": 30, "range": 40, "over": 25, "exact": 5}
}
```

**COUNTING (35s):** Above + live `vehicle_count`, `elapsed`, `remaining`

**WAITING (6s):** Above + `server_seed`, `result`, `bet_outcomes`

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
        {"name": "Stream_1", "url": "https://www.youtube.com/live/..."},
        {"name": "Stream_2", "url": "https://example.com/stream.m3u8"}
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

Draw lines/ROI using --gui mode. Configs auto-save and hot-reload during processing.

### bytetrack.yaml

```yaml
tracker_type: bytetrack
track_high_thresh: 0.25    # Low-conf jittery detections excluded from primary matching
track_low_thresh: 0.10     # Extremely noisy detections ignored (matches YOLO conf floor)
new_track_thresh: 0.15     # Minimum confidence to create new track (catches side-angle vehicles)
track_buffer: 150           # 5 seconds at 30fps — survives occlusion behind poles/pillars
match_thresh: 0.65          # Relaxed for better re-association after occlusion
fuse_score: true
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

Line/ROI saves immediately to line_configs/ and hot-reloads for the next clip.
Debug window always renders at 1280x720. Mouse coordinates are auto-converted to original frame resolution when saving.

---

## Pipeline Details

### Parallel Download + Single YOLO Worker (Smart Throttling)

```
StreamDownloader threads (parallel, CPU only):
  - Each stream has its own download thread
  - ffmpeg -c copy (no re-encoding, zero CPU waste)
  - Quality check: blur, brightness validation
  - SMART THROTTLE: download_queue maintained at exactly 3 clips
    → All downloaders pause when queue has 3+ clips
    → Resume when YOLO consumes and queue drops below 3
    → Prevents YouTube 429 rate-limit spam
  - YouTube 429 exponential backoff: 10s → 30s → 60s → 120s → 300s max
  - URL cache: 4 hours (YouTube live URLs valid ~6 hours)

YOLOWorker thread (single, GPU-safe):
  - Picks clips from shared download_queue (target: 3 clips)
  - Runs offline_count() with same model/tracker as production
  - Validates count >= MIN_VEHICLE_COUNT (5)
  - Approved clips go to ready_queue (max 10)
  - Thread-safe: only one thread calls model.track()

Scheduler (main thread):
  - Cold start: waits for 5 processed clips before first round
  - Non-blocking clip polling (never freezes GUI)
  - Picks from ANY stream (fastest wins)
  - Time slot management with IST-based rotation
```

### Download Throttling Flow

```
STARTUP: download_queue empty → downloaders active → 3 clips download
         download_queue: 3 → ALL downloaders SLEEP

STEADY:  YOLO picks 1 → queue: 2 → 1 downloader wakes → downloads 1 → queue: 3
         YOLO picks 1 → queue: 2 → download 1 → queue: 3
         ...

         YouTube yt-dlp calls: max 1 every 40-86 sec (matches YOLO speed)
         429 errors: eliminated
```

### Why Pre-Processing?
Real-time YOLO during playback: 80ms/frame > 40ms budget = frame skip + stutter.
Pre-processing in background: playback uses stored detections = ~6ms/frame = smooth.

---

## Provably Fair System

Uses a **commitment scheme** — the only correct approach for real-world event games where the result (vehicle count) is a physical measurement, not seed-derived.

### Why Commitment Scheme (Not Seed Derivation)

Traditional provably fair derives results from seeds: `result = HMAC(server_seed, client_seed:nonce)`.
This game's result is a **real vehicle count** — no algorithm can derive it. The commitment proves the result was locked before betting opened, identical to how live dealer casinos prove card draws.

### Commitment Contents

```
commitment_hash = SHA-256(
    server_seed         64-char hex, cryptographically random per round
    + ":" +
    result              vehicle count (integer)
    + ":" +
    under               Under boundary number
    + ":" +
    range_low           Range lower boundary
    + ":" +
    range_high          Range upper boundary
    + ":" +
    over                Over boundary number
    + ":" +
    exact_1             First exact number
    + ":" +
    exact_2             Second exact number
    + ":" +
    round_id            global nonce (auto-incrementing)
)
```

`server_seed` prevents brute-force: vehicle counts are 0-50, trivially guessable without it. All boundary numbers are included to prove they weren't altered after seeing bets.

### Round Flow

```
BEFORE BETTING:
  1. YOLO processes clip → result = 7
  2. server_seed = secrets.token_hex(32)
  3. Boundaries generated from lambda_mean (historical average)
  4. commitment_hash = SHA-256(server_seed:result:under:range_low:range_high:over:exact_1:exact_2:round_id)

BETTING (15s):
  5. Published: commitment_hash, boundaries, multipliers, win_chances
  6. NOT published: server_seed, result (hidden)

COUNTING (35s):
  7. Video plays, count increases live

WAITING (6s):
  8. REVEAL: server_seed + result
  9. Player verification: recompute SHA-256 → must match commitment
```

### Client Seed

Client seed does NOT affect the result (real-world event). It is:
- Auto-generated on player session start (frontend)
- Sent with bet placement to betting backend
- Stored in round audit log for transparency
- Player can change it anytime; applies from next round

### Verification Endpoint

`GET /api/verify` returns all data needed for independent verification:
```json
{
  "round_id": 42,
  "server_seed": "a7f3b2c9d4e5f6...",
  "result": 7,
  "boundaries": {
    "under": 14, "range_low": 20, "range_high": 21,
    "over": 28, "exact_1": 20, "exact_2": 21
  },
  "commitment_hash": "e4d2f1a8b3c7...",
  "verification_string": "a7f3b2c9...:7:14:20:21:28:20:21:42",
  "bet_outcomes": {"under": false, "range": false, "over": false, "exact": false}
}
```

---

## Odds Generation

Odds = the **boundary numbers** for each bet option (change every round based on historical average).

Multipliers and win chances are **CONSTANT** every round.

### Terminology

```
Odds (change every round):
  Under < 14  |  Range 16-18  |  Over > 22  |  Exact 20 or 21
  ↑ these NUMBERS are the "odds"

Multipliers (CONSTANT every round):
  Under: 3.13x  |  Range: 2.35x  |  Over: 3.76x  |  Exact: 18.8x

Win Chances (CONSTANT every round):
  Under: 30%  |  Range: 40%  |  Over: 25%  |  Exact: 5%
```

### Boundary Generation Formula

Uses the stream's historical average vehicle count (lambda_mean):

```python
under      = floor(lambda_mean × 0.75 × adjust_factor) - 1
range_low  = round(lambda_mean - 0.5)
range_high = round(lambda_mean + 0.8)
over       = ceil(lambda_mean × 1.35 × adjust_factor) + 1
exact_1    = round(lambda_mean)
exact_2    = exact_1 + 1
```

### Example (lambda_mean = 20)

```
under      = floor(20 × 0.75) - 1 = 14
range_low  = round(19.5) = 20
range_high = round(20.8) = 21
over       = ceil(27) + 1 = 28
exact_1    = 20
exact_2    = 21

Player sees:
  Under < 14  |  Range 20-21  |  Over > 28  |  Exact 20 or 21

Dead zones (no bet wins): 14-19 and 22-28
  → If result falls here, house wins all main bets
```

### Winning Rules

```
Under: count < under            → Under wins
Range: range_low ≤ count ≤ range_high  → Range wins
Over:  count > over             → Over wins
Exact: count == exact_1 or exact_2     → Exact wins
Gap:   anything else            → House wins all main bets
```

### Default Odds (< 20 Rounds of History)

When insufficient data exists, boundaries use default lambda_mean = 10.0.

---

## Clip Validation Rules

| Check | Threshold | Action |
|-------|-----------|--------|
| Vehicle count | >= 5 | Approve / reject + retry |
| Image blur | > 15 (Laplacian variance) | Reject |
| Brightness | 10-250 (mean pixel value) | Reject |
| Max retries per stream | 3 consecutive fails | Skip stream |
| Download timeout | 90 seconds | Abort + retry |

---

## Dependencies

```
ultralytics          # YOLO inference + ByteTrack tracking
opencv-python        # Video I/O, image processing, GUI
fastapi              # REST API + WebSocket server
uvicorn[standard]    # ASGI server (httptools, websockets)
numpy                # Array operations
pydantic             # API data validation (bundled with FastAPI)
yt-dlp               # YouTube URL resolution
livekit              # LiveKit Python SDK (WebRTC frame publishing)
livekit-api          # LiveKit server API (token generation)
torch + torchvision  # PyTorch with CUDA (cu121)
tensorrt-cu12        # TensorRT for GPU-optimized inference
```

**Hardware:** NVIDIA GPU with CUDA required (tested on RTX 4060 Ti 8GB).
**Recommended:** Node.js (LTS) — yt-dlp uses it for better YouTube parsing, significantly reduces 429 rate-limit errors.
**Optional:** Docker Desktop for LiveKit server (native binary or MJPEG fallback works without it).

---

## CLI Reference

### scheduler.py (Production)

```bash
python scheduler.py --config streams_config.json                    # production (web only)
python scheduler.py --config streams_config.json --gui              # development (web + debug GUI)
python scheduler.py --config streams_config.json --web-port 8080    # custom port
python scheduler.py --model yolo11x.pt                              # use PyTorch instead of TensorRT
python scheduler.py --test                                          # validate config only
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | streams_config.json | Path to streams config |
| `--model` | yolo11x.engine | YOLO model (.engine or .pt) |
| `--imgsz` | 1600 | YOLO input resolution |
| `--gui` | false | Enable debug GUI window |
| `--web-port` | 5000 | FastAPI server port |
| `--test` | false | Validate config and exit |

### main.py (Single Stream Testing)

```bash
python main.py --url "https://youtube.com/live/..."                 # live stream
python main.py --video video1                                       # local video
python main.py --video video1 --model yolo11m.pt --conf 0.15       # custom model
python main.py --url URL --no-gui --web-port 5000                   # headless
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--url` | - | YouTube or stream URL |
| `--video` | - | Local video file name |
| `--model` | yolo11m.pt | YOLO model |
| `--conf` | 0.010 | Confidence threshold |
| `--imgsz` | 1600 | Input resolution |
| `--skip-frames` | 1 | Process every Nth frame |
| `--no-gui` | false | Headless mode |
| `--web-port` | 5000 | Web server port |

### download.py (Test Clips)

```bash
python network/download.py --url URL --name video1 --duration 300   # download clip
python network/download.py --list                                   # list saved videos
```
