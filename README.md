# CCTV Rush Hour

> Real-world traffic CCTV cameras as a provably fair betting arena. Zero RNG. Pure real-life traffic.

**CCTV Rush Hour** is an AI-powered betting game where players wager on how many vehicles will cross a counting zone on live traffic cameras within a 35-second window. Results are driven entirely by real-world vehicle counts detected by computer vision — no random number generation involved.

---

## Game Flow

### How It Works

A round lasts **56 seconds** and cycles through three phases:

```
BETTING (15s) --> COUNTING (35s) --> WAITING (6s) --> next round
```

**1. BETTING PHASE (15 seconds)**

- A city-themed animation plays with the upcoming stream's name and thumbnail
- The provably fair commitment hash is published
- Players choose one of four bet types:

| Bet Type | Example | Wins When | Multiplier |
|----------|---------|-----------|------------|
| **Under** | Under 6 | Vehicle count < 6 | 3.13x |
| **Range** | Range 8-10 | 8 <= count <= 10 | 2.35x |
| **Over** | Over 14 | Vehicle count > 14 | 3.76x |
| **Exact** | Exact 9 or 10 | Count matches exactly | 18.8x |

- Boundary numbers (Under/Range/Over/Exact thresholds) change every round based on the stream's historical average
- Multipliers and win chances are constant across all rounds

**2. COUNTING PHASE (35 seconds)**

- A pre-recorded CCTV traffic clip plays with real-time AI overlay
- AI detects and tracks vehicles (cars, buses, trucks, motorcycles)
- White tracking dots follow uncounted vehicles
- Green flash brackets animate when a vehicle crosses the counting line
- Vehicle count increments live: 0 -> 1 -> 3 -> 5 -> 7
- Odds bar displays at top-middle showing the round's boundaries
- Circular timer counts down at top-right (green)

**3. WAITING PHASE (6 seconds)**

- Stream continues playing (last 6 seconds of the clip)
- Vehicle count is frozen at its final value
- The winning bet option is highlighted green in the odds bar
- Server seed is revealed for provably fair verification
- Bets are settled: players can verify the result independently
- Circular timer counts down (red)

### Provably Fair Guarantee

Every round is cryptographically committed before betting opens:

```
1. AI pre-counts vehicles on a clip (result known before round starts)
2. Server generates a random 256-bit seed
3. Commitment hash = SHA-256(seed + result + boundaries + round_id)
4. Hash is published BEFORE betting opens
5. After the round: seed and result are revealed
6. Anyone can verify: recompute SHA-256 and compare with published hash
```

The result is a physical measurement (real vehicles on real cameras) — not derived from a seed. The commitment scheme proves the result was locked before any bets were placed.

### Streams & Scheduling

- Hundreds of live public CCTV feeds worldwide (YouTube Live, HLS, RTSP)
- Streams are organized by IST time slots (e.g., 08:00-12:00, 12:00-16:00)
- Each slot contains multiple streams that rotate round-robin
- Each stream has a city-specific animation video and thumbnail for the betting phase
- 24/7 non-stop operation with automatic stream rotation

---

## Architecture

```
SequentialDownloader (1 thread, round-robin, NO GPU)
  YouTube URL pre-warm (yt-dlp Python API + browser cookies)
  Batch download: 3 clips parallel (libx264 960px 18fps)
  Refill trigger: when download_queue drops to 1
         |
    download_queue (max 5 clips)
         |
YOLOWorker (1 thread, OWNS GPU exclusively)
  GPU warmup on startup (10 dummy frames)
  offline_count(): YOLO first 35s only, last 6s no processing
  tracked_vehicles system (velocity prediction, zero flicker)
         |
    ready_queue (max 10 pre-processed clips)
         |
Scheduler (main thread, 56-sec round cycle)
  Cold start: waits for 5 clips
  |-- BETTING  (15s)  City animation + thumbnail at 3s + yellow timer
  |-- COUNTING (35s)  Clip plays, AI overlay, odds bar top-middle
  |-- WAITING  (6s)   Stream continues, count frozen, result revealed
         |
  push_frame() --> web_server (MJPEG) + livekit_publisher (WebRTC)
         |
FastAPI Server -- REST /api/round + WebSocket /ws/game + LiveKit token
```

---

## Complete Workflow

### Phase 1: Download (Background)

```
1. Startup: Pre-warm ALL YouTube URLs (yt-dlp Python API + cookies)
2. SequentialDownloader picks 3 streams (round-robin)
3. Resolve URLs sequentially (1 at a time, prevents YouTube 429)
4. Download 3 clips PARALLEL via ffmpeg:
     ffmpeg -i URL -t 41 -vf "scale=960:-1,fps=18" -c:v libx264
     -preset ultrafast -crf 28 -pix_fmt yuv420p -an output.mp4
5. Quality check each: blur > 15, brightness 10-250, duration >= 38s
6. Push to download_queue (target: 3 clips)
7. When queue drops to 1 -> trigger next batch of 3
8. Background URL refresh every 4 hours
```

### Phase 2: YOLO Processing (Background, GPU)

```
1. YOLOWorker picks clip from download_queue
2. offline_count() processes clip:
   - First 35 sec: YOLO + ByteTrack on EVERY frame
   - Last 6 sec: NO YOLO (raw frames, count frozen)
3. Stores per-frame: detections, track IDs, bboxes, crossed flags
4. Stores counting_events: [{frame, timestamp, count_at}]
5. Validate: count >= 5 vehicles -> APPROVED
6. Push PreProcessedClip to ready_queue (max 10)
```

### Phase 3: Round Cycle (Main Thread, 56 sec)

```
PREPARE:
  Pick clip from ready_queue (fastest stream wins)
  prepare_round():
    -> odds_engine.generate_round_odds() -> boundaries
    -> provably_fair.generate_round_fairness() -> SHA-256 hash
    -> Package into immutable RoundData

BETTING (15 sec):
  0-3s:   City animation video plays fullscreen (slow-motion stretched to 15s)
  3s:     Thumbnail photo + stream name appears CENTER
  3-10s:  Animation continues behind thumbnail overlay
  10-15s: Frozen frame + thumbnail stays
  0-15s:  Yellow circular timer top-right (consistent with counting/waiting)
  Data:   Commitment hash published, boundaries available

COUNTING (35 sec):
  Pre-recorded clip plays with detection overlay
  White tracking dots (uncounted vehicles)
  Green flash brackets (counted vehicles)
  Counting line/ROI visible
  Count increments: 0->1->3->5->7
  Odds bar top-middle: U<6 | R 8-10 | O>14 | E 9|10
  Green circular timer top-right
  game_api.update_count() every frame -> WebSocket push

WAITING (6 sec):
  Clip continues playing (last 6s, no YOLO)
  Count frozen at final value
  Odds bar: winning option highlighted GREEN
  Red circular timer top-right
  Reveal: server_seed + result + bet_outcomes

FINALIZE:
  finalize_round() -> save to history_tracker
  push verification to web_server
  mark_clip_done() -> delete temp file
  -> Next round immediately
```

### Phase 4: Data Delivery

```
push_frame(frame) sends to BOTH:
  web_server.update_frame() -> MJPEG /video_feed
  livekit_publisher.update_frame() -> LiveKit WebRTC room

game_api.update_phase() -> WebSocket broadcast to all clients
game_api.update_count() -> WebSocket count_update on change

Browser:
  Primary: LiveKit WebRTC (green dot)
  Fallback: MJPEG stream (yellow dot)
  Auto-retry LiveKit every 10 sec
```

---

## Quick Start

### Prerequisites

- **GPU:** NVIDIA with CUDA (tested on RTX 4060 Ti 8GB)
- **Python:** 3.10+
- **ffmpeg:** installed and in PATH
- **Node.js:** v22 LTS (for yt-dlp YouTube parsing)

### Installation

```bash
cd D:\cctv_main\CCTV_GAME

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install ultralytics opencv-python fastapi uvicorn[standard] numpy yt-dlp
pip install livekit livekit-api pydantic pyjwt
pip install tensorrt-cu12

# Export TensorRT engine (one-time, 10-30 min)
yolo export model=yolo11x.pt format=engine imgsz=1280 half=True

# (Optional) Start LiveKit server for WebRTC streaming
.\start_livekit.bat
```

### Running

```bash
# Production mode (web browser only)
python scheduler.py --config streams_config.json

# Development mode (web + debug GUI with mouse line/ROI drawing)
python scheduler.py --config streams_config.json --gui

# Test with local video file
python scheduler.py --test-video path/to/video.mp4
python scheduler.py --test-video path/to/video.mp4 --gui --test-stream-name "Stream 7"

# Single stream testing
python main.py --url "https://www.youtube.com/live/VIDEO_ID"
python main.py --video video1

# Validate config without running
python scheduler.py --test
```

**Web interface:** http://localhost:5000
**API docs (Swagger):** http://localhost:5000/docs
**WebSocket:** ws://localhost:5000/ws/game
**LiveKit:** ws://localhost:7880

---

## Project Structure

```
CCTV_GAME/
|-- core/
|   |-- counting.py            VehicleCounter + offline_count() + tracked_vehicles
|   |-- geometry_utils.py      Line intersection, point-in-polygon
|   |-- config_manager.py      JSON config I/O
|   |-- odds_engine.py         Boundary generation + constant multipliers/win_chances
|   |-- provably_fair.py       SHA-256 commitment scheme
|
|-- game/
|   |-- game_api.py            Single source of truth (thread-safe, Pydantic models)
|   |-- round_manager.py       RoundData + prepare_round + play_clip_with_overlay
|   |-- history_tracker.py     Per-stream count history (max 500 rounds)
|
|-- network/
|   |-- stream_manager.py      SequentialDownloader + YOLOWorker + Pipeline
|   |-- download.py            yt-dlp wrapper for URL resolution
|   |-- livekit_publisher.py   WebRTC frame publisher to LiveKit room
|
|-- ui/
|   |-- renderer.py            Shared tracking overlay + playback overlay + odds bar
|   |-- animations.py          Globe transition, IST timer
|
|-- main.py                    Single stream entry point (development)
|-- scheduler.py               Multi-stream game orchestrator (production)
|-- web_server.py              FastAPI + Uvicorn + WebSocket + LiveKit token
|
|-- animation_videos/          City-specific animation videos for betting phase
|-- thumbnails/                City thumbnail photos for betting phase
|-- streams_config.json        IST time slots + stream URLs + animation/thumbnail
|-- bytetrack.yaml             ByteTrack tracker config
|-- line_configs/              Per-stream counting line/ROI configs (auto-saved)
|-- round_history.json         Persisted count history for odds
|-- cookies.txt                YouTube cookies (auto-read or manual export)
|-- templates/index.html       Web viewer (LiveKit WebRTC + MJPEG fallback)
|
|-- livekit_server/
|   |-- livekit-server.exe     LiveKit native binary
|-- start_livekit.bat          Start LiveKit server
|-- livekit.yaml               LiveKit server config
|-- docker-compose.yml         Docker setup (LiveKit + Redis + cookie refresher)
|-- Dockerfile                 Main app Docker image
|-- Dockerfile.cookies         Cookie refresher Docker image
|-- cookie_refresher/
|   |-- refresh.py             Playwright auto cookie refresh
|
|-- docs/                      Change requests and plans
|-- model_training/            Custom model training pipeline
```

---

## Round Cycle (56 seconds)

```
|<-- 15 sec -->|<---------- 35 sec ---------->|<-- 6 sec -->|
|              |                               |             |
|   BETTING    |        COUNTING               |   WAITING   |
|   Animation  |   Stream playing              |   Stream ON |
|   +Thumbnail |   AI counting ON              |   Count OFF |
|   +Hash      |   Odds bar visible            |   Frozen    |
|   Yellow [O] |   Green [O]                   |   Red [O]   |
|              |<-------- STREAM PLAYS (41 sec) ------------>|
```

---

## AI Counting Pipeline

### Detection & Tracking

- **Model:** YOLO11x TensorRT engine (default, configurable via `--model`)
- **Classes:** Car (2), Motorcycle (3), Bus (5), Truck (7)
- **Confidence:** 0.10
- **Input resolution:** 1280px (baked into TensorRT at export)
- **Precision:** FP16
- **Tracker:** ByteTrack (track_buffer: 180 frames = 6s at 30fps)
- **EMA smoothing:** alpha = 0.30 (small boxes) / 0.45 (large boxes)

### Counting Modes

- **Line mode:** Vehicle centroid crosses line = counted
- **ROI mode:** Vehicle enters then exits polygon = counted
- **Deduplication:** 20px spatial cooldown + 0.5s temporal window

### Pre-Processing (YOLO runs background, NOT during playback)

```
offline_count():
  Frame 0-874 (35s):  YOLO + ByteTrack every frame -> store detections
  Frame 875+ (6s):    NO YOLO -> raw frames only, count frozen
  
  Result: {
    result: 7,
    detections: {frame_no: [{track_id, bbox, crossed, ...}]},
    counting_events: [{frame, ts, count_at}],
    clip_fps: 18.0
  }
```

### Playback Overlay (during round, no YOLO)

- White tracking dots (uncounted vehicles, EMA smoothed)
- Green flash brackets (counted vehicles, sine-wave animation)
- Counting line/ROI from config
- Odds bar top-middle (boundaries only: U<6 | R 8-10 | O>14 | E 9|10)
- Circular timer top-right (yellow betting / green counting / red waiting)
- During WAITING: winning option highlighted green in odds bar

---

## Betting Phase (15 sec) — City Animation

Each stream has a city-specific animation video and thumbnail:

```json
{
  "name": "Tokyo Shinjuku",
  "url": "https://www.youtube.com/live/...",
  "animation": "tokyo_animation.mp4",
  "thumbnail": "tokyo.jpg"
}
```

**Timeline:**
- 0-3s: City animation plays fullscreen (slow-motion to fill 15s)
- 3s: Thumbnail photo + stream name appears center
- 3-15s: Animation continues behind thumbnail
- Fallback: Earth Zoom video if no city animation

**Folders:**
- `animation_videos/` — city animation MP4 files
- `thumbnails/` — city photo JPG files

---

## Video Streaming

### LiveKit WebRTC (Primary)

```bash
.\start_livekit.bat    # Start LiveKit server (ws://localhost:7880)
```

- Publisher: `livekit_publisher.py` sends frames to LiveKit room
- Viewer: Browser connects via LiveKit JS SDK
- Token: `GET /api/livekit/token?identity=viewer-123`

### MJPEG Fallback

- Always available at `/video_feed`
- Auto-fallback if LiveKit unavailable
- Browser auto-retries LiveKit every 10 sec

---

## API Endpoints

### REST

| Endpoint | Description |
|----------|-------------|
| `GET /` | HTML game viewer |
| `GET /video_feed` | MJPEG video stream |
| `GET /api/round` | **Primary** — timer, count, odds, hash |
| `GET /api/state` | Raw game state |
| `GET /api/verify` | Provably fair verification |
| `GET /api/livekit/token` | LiveKit viewer JWT |
| `GET /api/livekit/status` | LiveKit connection status |
| `GET /docs` | Swagger API docs |

### GET /api/round — Response Examples

<details>
<summary><b>BETTING phase</b> (15 sec — no stream, players placing bets)</summary>

```json
{
  "phase": "BETTING",
  "round_id": "#RId0000007",
  "stream_name": "Tokyo Shinjuku",
  "created": "2026-04-10 11:40:07 IST",
  "status": true,
  "betting_status": true,
  "betting_timer": 12,
  "round_timer": 53.0,
  "vehicle_count": 0,
  "commitment_hash": "2ca01d2233bb6e73cffb2fa71942d35c40de0bbd39a983ade976489393778a14",
  "odds": {
    "under": 19,
    "range_low": 21,
    "range_high": 23,
    "over": 25,
    "exact_1": 20,
    "exact_2": 24
  }
}
```

> **Note:** `server_seed`, `result`, and `bet_outcomes` are NOT exposed during BETTING.

</details>

<details>
<summary><b>COUNTING phase</b> (35 sec — stream playing, AI counting live)</summary>

```json
{
  "phase": "COUNTING",
  "round_id": "#RId0000007",
  "stream_name": "Tokyo Shinjuku",
  "created": "2026-04-10 11:40:07 IST",
  "status": true,
  "betting_status": false,
  "betting_timer": 0,
  "round_timer": 28.5,
  "vehicle_count": 12,
  "commitment_hash": "2ca01d2233bb6e73cffb2fa71942d35c40de0bbd39a983ade976489393778a14",
  "odds": {
    "under": 19,
    "range_low": 21,
    "range_high": 23,
    "over": 25,
    "exact_1": 20,
    "exact_2": 24
  }
}
```

> **Note:** `vehicle_count` updates every frame. `server_seed` still hidden.

</details>

<details>
<summary><b>WAITING phase</b> (6 sec — result revealed, bets settled)</summary>

```json
{
  "phase": "WAITING",
  "round_id": "#RId0000007",
  "stream_name": "Tokyo Shinjuku",
  "created": "2026-04-10 11:41:03 IST",
  "status": true,
  "betting_status": false,
  "betting_timer": 0,
  "waiting_timer": 3.4,
  "round_timer": 2.9,
  "vehicle_count": 23,
  "result": 23,
  "commitment_hash": "2ca01d2233bb6e73cffb2fa71942d35c40de0bbd39a983ade976489393778a14",
  "server_seed": "1672b2e57a8f40d8f343e76ffeaee73ea50bb567106e34760b108aa8f272d03a",
  "odds": {
    "under": 19,
    "range_low": 21,
    "range_high": 23,
    "over": 25,
    "exact_1": 20,
    "exact_2": 24
  },
  "bet_outcomes": {
    "under": false,
    "range": true,
    "over": false,
    "exact": false,
    "winning_option": "RANGE"
  }
}
```

> **Verification:** `SHA-256("1672b2e5...d03a:23:19:21:23:25:20:24:7")` must equal `commitment_hash`.
> Result 23 falls in Range 21-23, so `range: true` and `winning_option: "RANGE"`.

</details>

### WebSocket (ws://localhost:5000/ws/game)

| Message | When | Data |
|---------|------|------|
| `initial_state` | On connect | Full round state |
| `game_state` | Phase change | phase, round_id, hash, boundaries |
| `count_update` | Count changes | `{vehicle_count: N}` |
| `verification` | Round ends | server_seed, result, bet_outcomes |

---

## Configuration

### streams_config.json

```json
{
  "time_slots": [
    {
      "start": "08:00",
      "end": "12:00",
      "streams": [
        {
          "name": "Tokyo Shinjuku",
          "url": "https://www.youtube.com/live/...",
          "animation": "tokyo_animation.mp4",
          "thumbnail": "tokyo.jpg"
        }
      ]
    }
  ]
}
```

Time slots in **IST** (UTC+5:30). Streams round-robin within active slot.

### bytetrack.yaml

```yaml
tracker_type: bytetrack
track_high_thresh: 0.20
track_low_thresh: 0.05
new_track_thresh: 0.15
track_buffer: 180
match_thresh: 0.60
fuse_score: true
```

### Per-Stream Line Config (line_configs/*.json)

```json
{"line": [[594, 464], [841, 576]], "roi_poly": null, "mode": "line"}
```

Draw with `--gui` mode. Auto-saves, hot-reloads every 30 frames.

---

## Provably Fair System

Commitment scheme (NOT seed derivation — result is a real vehicle count):

```
commitment_hash = SHA-256(server_seed:result:under:range_low:range_high:over:exact_1:exact_2:round_id)
```

**Flow:** YOLO counts -> generate seed -> create hash -> publish hash -> betting -> play video -> reveal seed + result -> player verifies

`GET /api/verify` returns all verification data.

---

## Odds System

**Boundaries** (change every round): Under < 6 | Range 8-10 | Over > 14 | Exact 9|10
**Multipliers** (CONSTANT): Under 3.13x | Range 2.35x | Over 3.76x | Exact 18.8x
**Win Chances** (CONSTANT): Under 30% | Range 40% | Over 25% | Exact 5%

---

## YouTube Cookie Management

```
Priority: cookiesfrombrowser firefox -> chrome -> edge -> cookiefile cookies.txt

Manual export: Chrome -> YouTube.com -> "Get cookies.txt LOCALLY" extension -> Export
Production: Docker cookie_refresher (Playwright auto-refresh every 3 days)
```

---

## Clip Validation

| Check | Threshold | Action |
|-------|-----------|--------|
| Vehicle count | >= 5 | Reject if less |
| Blur | > 15 Laplacian | Reject if blurry |
| Brightness | 10-250 mean | Reject if dark/bright |
| Duration | >= 38 sec | Reject if short |
| Consecutive fails | 5 per stream | Skip stream, reset |
| Download timeout | 90 sec | Abort, next stream |

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `python scheduler.py --config X` | Production mode |
| `python scheduler.py --gui` | Development + debug GUI |
| `python scheduler.py --test-video X.mp4` | Test local video |
| `python scheduler.py --test` | Validate config |
| `python main.py --url URL` | Single stream test |
| `.\start_livekit.bat` | Start LiveKit server |

---

## Dependencies

```
torch + torchvision (cu124)    PyTorch with CUDA
ultralytics                    YOLO inference + ByteTrack
opencv-python                  Video I/O, image processing
fastapi + uvicorn[standard]    REST API + WebSocket
numpy                          Array operations
yt-dlp                         YouTube URL resolution
livekit + livekit-api          WebRTC streaming
pydantic                       API data validation
pyjwt                          LiveKit JWT tokens
tensorrt-cu12                  GPU-optimized inference
```

**Hardware:** NVIDIA GPU with CUDA required (tested RTX 4060 Ti 8GB).

---

## Performance (RTX 4060 Ti 8GB)

| Component | Time |
|-----------|------|
| YouTube URL resolve (cached) | 0 sec |
| ffmpeg download (41s, libx264 18fps) | ~50-65 sec |
| YOLO processing (35s, ~630 frames) | ~18-25 sec |
| Quality check | < 1 sec |
| Game round | 56 sec |
