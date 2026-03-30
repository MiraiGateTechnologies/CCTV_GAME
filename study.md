# CCTV Casino Game — Technical Study Document

## 1. System Overview

A casino betting game where real-world public CCTV traffic cameras serve as the betting arena. AI counts vehicles crossing a line/zone in 35-second rounds. Results drive Under/Over/Range/Exact betting. No RNG — outcomes are real-world vehicle counts.

**Total Codebase:** ~4,800 lines of Python across 15 source files.

---

## 2. High-Level Architecture

```
                    BACKGROUND PIPELINE (always running)
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  SequentialDownloader        YOLOWorker                  │
  │  (1 thread, CPU only)        (1 thread, GPU)             │
  │                                                          │
  │  Round-robin streams   ──>  download_queue  ──>          │
  │  ffmpeg -c copy (41s)       (target: 3)       offline_   │
  │  YouTube URL resolve                          count()    │
  │  Quality check (blur,                         YOLO11x    │
  │  brightness)                                  TensorRT   │
  │                                               ByteTrack  │
  │                                    ──>  ready_queue      │
  │                                         (max: 10)        │
  └──────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────┐
  │              SCHEDULER (main thread)                      │
  │                                                          │
  │  for each round (56 seconds):                            │
  │    1. Pick clip from ready_queue                         │
  │    2. Generate boundaries + commitment hash              │
  │    3. BETTING  (15s) — countdown, no stream              │
  │    4. COUNTING (35s) — play clip, overlay detections     │
  │    5. WAITING  (6s)  — stream continues, reveal result   │
  │    6. Finalize — save history, delete clip               │
  └──────────────────────────────────────────────────────────┘
                    │                         │
                    ▼                         ▼
  ┌───────────────────────┐   ┌─────────────────────────────┐
  │  FastAPI Web Server    │   │  LiveKit Publisher           │
  │  Port 5000             │   │  ws://localhost:7880         │
  │                        │   │                             │
  │  /video_feed  (MJPEG)  │   │  WebRTC video track         │
  │  /api/round   (REST)   │   │  1280x720 @ 25fps           │
  │  /ws/game  (WebSocket) │   │  Auto-reconnect             │
  │  /docs     (Swagger)   │   │                             │
  └───────────────────────┘   └─────────────────────────────┘
                    │                         │
                    ▼                         ▼
  ┌──────────────────────────────────────────────────────────┐
  │                  BROWSER (Player)                         │
  │                                                          │
  │  LiveKit WebRTC (primary) ──> green dot                  │
  │  MJPEG fallback (if LiveKit down) ──> yellow dot         │
  │                                                          │
  │  Plus: REST polling /api/round for game data             │
  │        WebSocket /ws/game for real-time updates          │
  └──────────────────────────────────────────────────────────┘
```

---

## 3. Threading Model

```
  ┌─────────────────────────────────────────────────────────┐
  │ Thread 1: MainThread (Scheduler)                        │
  │   - 56-second round loop                                │
  │   - Playback (frame pacing, overlay rendering)          │
  │   - Debug GUI window (--gui mode)                       │
  │   - cv2.waitKey() for Windows message pump              │
  │   - push_frame() to MJPEG + LiveKit                     │
  ├─────────────────────────────────────────────────────────┤
  │ Thread 2: DL-Sequential (SequentialDownloader)          │
  │   - Round-robin stream downloads                        │
  │   - ffmpeg subprocess calls                             │
  │   - yt-dlp URL resolution (YouTube only)                │
  │   - Quality validation                                  │
  │   - Pushes to download_queue                            │
  ├─────────────────────────────────────────────────────────┤
  │ Thread 3: YOLOWorker                                    │
  │   - Reads from download_queue                           │
  │   - offline_count() with YOLO + ByteTrack               │
  │   - ONLY thread that touches GPU                        │
  │   - Pushes approved clips to ready_queue                │
  ├─────────────────────────────────────────────────────────┤
  │ Thread 4: FastAPI-Server (uvicorn)                      │
  │   - HTTP endpoints                                      │
  │   - WebSocket connections                               │
  │   - MJPEG frame generator                               │
  ├─────────────────────────────────────────────────────────┤
  │ Thread 5: LiveKit-Publisher                             │
  │   - Async event loop                                    │
  │   - Reads frames from scheduler                         │
  │   - BGR->RGBA conversion + resize                       │
  │   - WebRTC frame publishing                             │
  └─────────────────────────────────────────────────────────┘

  Synchronization:
    download_queue: Queue(maxsize=5)   — DL-Sequential → YOLOWorker
    ready_queue:    Queue(maxsize=10)  — YOLOWorker → Scheduler
    game_api._lock: threading.Lock     — Scheduler → FastAPI/WebSocket
    web_server._frame_lock: Lock       — Scheduler → MJPEG generator
    livekit._frame_lock: Lock          — Scheduler → LiveKit publisher
```

---

## 4. Data Flow Per Round (56 Seconds)

```
  ┌─────────────────────────────────────────────────────────┐
  │ Step 1: CLIP SELECTION                                  │
  │                                                         │
  │ scheduler picks from ready_queue:                       │
  │   PreProcessedClip {                                    │
  │     clip_path: "temp/Stream_16_3.mp4"                   │
  │     stream_name: "Stream 16"                            │
  │     result: 7                                           │
  │     detections: {0: [...], 1: [...], ..., 874: [...]}   │
  │     counting_events: [{frame:115, count_at:1}, ...]     │
  │     clip_fps: 25.0                                      │
  │   }                                                     │
  └──────────────────┬──────────────────────────────────────┘
                     ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Step 2: ROUND PREPARATION                               │
  │                                                         │
  │ prepare_round(clip, round_counter, history):            │
  │   1. lambda_mean = history.get_mean("Stream 16")        │
  │   2. boundaries = odds_engine.generate_boundaries(      │
  │        lambda_mean)                                     │
  │      → {under:14, range_low:20, range_high:21,          │
  │         over:28, exact_1:20, exact_2:21}                │
  │   3. server_seed = secrets.token_hex(32)                │
  │   4. commitment_hash = SHA-256(                         │
  │        server_seed:result:under:range_low:              │
  │        range_high:over:exact_1:exact_2:round_id)        │
  │   5. bet_outcomes = settle_bets(result, boundaries)     │
  │   6. Return immutable RoundData                         │
  └──────────────────┬──────────────────────────────────────┘
                     ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Step 3: BETTING PHASE (15 seconds)                      │
  │                                                         │
  │ Published:                                              │
  │   - commitment_hash (proves result locked)              │
  │   - boundaries (Under<14, Range 20-21, Over>28, etc.)   │
  │   - multipliers (3.13x, 2.35x, 3.76x, 18.8x)          │
  │   - win_chances (30%, 40%, 25%, 5%)                     │
  │                                                         │
  │ Hidden:                                                 │
  │   - server_seed                                         │
  │   - result (vehicle count)                              │
  │                                                         │
  │ Visual: countdown timer (no video stream)               │
  │ API: game_api.update_phase("BETTING", ...)              │
  └──────────────────┬──────────────────────────────────────┘
                     ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Step 4: COUNTING PHASE (35 seconds)                     │
  │                                                         │
  │ play_clip_with_overlay():                               │
  │   for frame 0 to 874 (35s @ 25fps):                    │
  │     1. cap.read() from MP4 file                         │
  │     2. dets = clip.detections[frame_no]                 │
  │     3. count = clip.counting_events[frame_no]           │
  │     4. draw_playback_overlay(frame, dets, ...)          │
  │        → White dots (uncounted vehicles)                │
  │        → Green brackets (counted vehicles)              │
  │        → Counting line/ROI from config                  │
  │        → Timer circle (top-right)                       │
  │     5. push_frame(frame)                                │
  │        → web_server.update_frame() (MJPEG)              │
  │        → livekit_publisher.update_frame() (WebRTC)      │
  │     6. game_api.update_count(current_count)             │
  │     7. Frame pacing: sleep to maintain exact FPS        │
  │                                                         │
  │ NO YOLO RUNNING — stored detections only (~6ms/frame)   │
  └──────────────────┬──────────────────────────────────────┘
                     ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Step 5: WAITING PHASE (6 seconds)                       │
  │                                                         │
  │ Same clip continues playing (frame 875 to 1024)         │
  │ But: no detections drawn (all_detections[frame] = [])   │
  │      count frozen at final value                        │
  │      timer switches to red                              │
  │                                                         │
  │ Revealed:                                               │
  │   - server_seed                                         │
  │   - result (final vehicle count)                        │
  │   - bet_outcomes (which options won/lost)               │
  │                                                         │
  │ game_api.update_phase("WAITING", server_seed=...,       │
  │   result=..., bet_outcomes=...)                          │
  └──────────────────┬──────────────────────────────────────┘
                     ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Step 6: FINALIZE                                        │
  │                                                         │
  │ finalize_round(rd, history):                            │
  │   history.add_result("Stream 16", 7)                    │
  │                                                         │
  │ web_server.update_verification({                        │
  │   round_id, server_seed, result, boundaries,            │
  │   commitment_hash, verification_string, bet_outcomes    │
  │ })                                                      │
  │                                                         │
  │ pipeline.mark_clip_done(clip) → deletes MP4 file        │
  │                                                         │
  │ → NEXT ROUND (pick new clip from ready_queue)           │
  └─────────────────────────────────────────────────────────┘
```

---

## 5. Vehicle Counting Algorithm

```
  ┌─────────────────────────────────────────────────────────┐
  │ YOLO11x TensorRT → detections per frame                 │
  │   classes: Car(2), Motorcycle(3), Bus(5), Truck(7)      │
  │   confidence: 0.10, imgsz: 1600, FP16                  │
  │                                                         │
  │   ↓ detections: [(x1,y1,x2,y2, track_id, class, conf)] │
  └──────────────────┬──────────────────────────────────────┘
                     ▼
  ┌─────────────────────────────────────────────────────────┐
  │ ByteTrack Tracking                                      │
  │   track_buffer: 150 (5s persistence after occlusion)    │
  │   match_thresh: 0.65 (IoU for re-association)           │
  │   persist=True (IDs maintained across frames)           │
  │                                                         │
  │   ↓ tracked boxes with persistent track_ids             │
  └──────────────────┬──────────────────────────────────────┘
                     ▼
  ┌─────────────────────────────────────────────────────────┐
  │ EMA Centroid Smoothing                                  │
  │                                                         │
  │   box_area = (x2-x1) * (y2-y1)                         │
  │   alpha = 0.30 if box_area < 2000 else 0.45            │
  │                                                         │
  │   cx = alpha * cx_new + (1-alpha) * cx_old              │
  │   cy = alpha * cy_new + (1-alpha) * cy_old              │
  │                                                         │
  │   Small boxes (far vehicles): heavy smoothing (0.30)    │
  │   Large boxes (close vehicles): lighter smoothing (0.45)│
  └──────────────────┬──────────────────────────────────────┘
                     ▼
  ┌─────────────────────────────────────────────────────────┐
  │ LINE MODE: Counting Line Crossing                       │
  │                                                         │
  │   For each tracked vehicle:                             │
  │     Check last 20 trajectory points                     │
  │     Does any segment (p[i], p[i+1]) intersect           │
  │     the counting line (A, B)?                           │
  │     → Uses CCW (counter-clockwise) algorithm            │
  │       from geometry_utils.is_intersect()                │
  │                                                         │
  │   Deduplication:                                        │
  │     - track_id already in interval_counted_ids? → skip  │
  │     - Spatial cooldown: 20px radius + 0.5s window       │
  │     - recent_crossings list maintains history           │
  ├─────────────────────────────────────────────────────────┤
  │ ROI MODE: Polygon Enter-Exit                            │
  │                                                         │
  │   For each tracked vehicle:                             │
  │     is centroid inside polygon?                         │
  │     → cv2.pointPolygonTest()                            │
  │                                                         │
  │   State machine per track_id:                           │
  │     OUTSIDE → enters polygon → inside_roi_ids.add(id)   │
  │     INSIDE → exits polygon →                            │
  │       if not already counted → COUNT! increment total   │
  │                                                         │
  │   Deduplication:                                        │
  │     - IoU footprint matching (>0.25 overlap)            │
  │     - Center distance check (<60% of box size)          │
  │     - Time gap: 2.5 second window                       │
  └──────────────────┬──────────────────────────────────────┘
                     ▼
  ┌─────────────────────────────────────────────────────────┐
  │ OUTPUT                                                  │
  │                                                         │
  │   interval_total: 7  (total vehicles counted)           │
  │   counting_events: [{frame:115, ts:4.6, count_at:1},   │
  │                      {frame:198, ts:7.9, count_at:2},   │
  │                      ...]                               │
  │   detections: {frame_no: [{track_id, bbox, crossed}]}   │
  │   flash_positions: {track_id: (cx, cy, half_size)}      │
  └─────────────────────────────────────────────────────────┘
```

---

## 6. Odds & Provably Fair System

### 6.1 Boundary Generation

```
  Input: lambda_mean (historical average vehicle count per stream)

  Formula:
    under      = floor(lambda_mean * 0.75) - 1
    range_low  = round(lambda_mean - 0.5)
    range_high = round(lambda_mean + 0.8)
    over       = ceil(lambda_mean * 1.35) + 1
    exact_1    = round(lambda_mean)
    exact_2    = exact_1 + 1

  Example (lambda_mean = 20):
    Under < 14  |  Range 20-21  |  Over > 28  |  Exact 20 or 21

  Dead zones: 14-19 and 22-28 (house wins all main bets)

  CONSTANT multipliers: Under 3.13x | Range 2.35x | Over 3.76x | Exact 18.8x
  CONSTANT win chances: Under 30%   | Range 40%   | Over 25%   | Exact 5%
```

### 6.2 Commitment Scheme

```
  BEFORE betting opens:
    server_seed = secrets.token_hex(32)     ← cryptographic random
    result = 7                               ← from offline_count()
    boundaries = {under:14, range_low:20, ...}

    commitment_string = "server_seed:result:under:range_low:range_high:over:exact_1:exact_2:round_id"
    commitment_hash = SHA-256(commitment_string)

    Published: commitment_hash
    Hidden: server_seed, result

  AFTER round ends:
    Revealed: server_seed + result
    Player verifies: SHA-256(same_string) == commitment_hash

  WHY server_seed needed:
    Without it: SHA-256("7") → player pre-computes all 0-50 → knows result
    With seed: SHA-256("a7f3b2c9...:7:14:20:21:28:20:21:42") → impossible to reverse
```

---

## 7. Download Pipeline

```
  SequentialDownloader._download_loop():
  ═══════════════════════════════════════

    while running:
      ┌──────────────────────────────────┐
      │ 1. Check download_queue.qsize()  │
      │    >= 3? → sleep(5), continue    │
      │    < 3?  → proceed to download   │
      └─────────────┬────────────────────┘
                    ▼
      ┌──────────────────────────────────┐
      │ 2. Pick next stream (round-robin)│
      │    index = (index + 1) % total   │
      │    Skip if 5+ consecutive fails  │
      └─────────────┬────────────────────┘
                    ▼
      ┌──────────────────────────────────┐
      │ 3. Resolve URL                   │
      │    YouTube? → yt-dlp --get-url   │
      │      Cache: 4 hours              │
      │      429 backoff: 10s→300s max   │
      │    HLS? → use URL directly       │
      └─────────────┬────────────────────┘
                    ▼
      ┌──────────────────────────────────┐
      │ 4. Download via ffmpeg           │
      │    -c copy (no re-encoding)      │
      │    -t 41 (exact duration)        │
      │    -reconnect flags              │
      │    -movflags +faststart          │
      │    Timeout: 90 seconds           │
      └─────────────┬────────────────────┘
                    ▼
      ┌──────────────────────────────────┐
      │ 5. Quality check                 │
      │    Sample 3 frames (25%, 50%, 75%)│
      │    Blur: Laplacian variance > 15 │
      │    Brightness: mean 10-250       │
      │    Duration: >= 38 seconds       │
      └─────────────┬────────────────────┘
                    ▼
      ┌──────────────────────────────────┐
      │ 6. Push to download_queue        │
      │    DownloadedClip {              │
      │      clip_path, stream_name,     │
      │      line_config_file, imgsz,    │
      │      confidence                  │
      │    }                             │
      └──────────────────────────────────┘
```

---

## 8. API & WebSocket Protocol

### 8.1 REST Endpoints

```
  GET /                     HTML game viewer
  GET /video_feed           MJPEG stream (multipart/x-mixed-replace)
  GET /api/round            PRIMARY game data endpoint
  GET /api/state            Raw scheduler state
  GET /api/verify           Provably fair verification data
  GET /api/livekit/token    LiveKit JWT for viewer connection
  GET /api/livekit/status   LiveKit publisher connection status
  GET /docs                 Swagger auto-documentation
```

### 8.2 WebSocket Protocol

```
  Client connects: ws://localhost:5000/ws/game

  Server → Client (automatic):
    {"type": "initial_state", "data": {full round state}}
    {"type": "game_state",   "data": {phase change info}}
    {"type": "count_update", "data": {"vehicle_count": N}}
    {"type": "verification", "data": {seed, result, outcomes}}

  Client → Server:
    {"type": "ping"} → {"type": "pong"}
```

### 8.3 /api/round Response Matrix

```
  Field              │ IDLE │ BETTING │ COUNTING │ WAITING
  ───────────────────┼──────┼─────────┼──────────┼────────
  phase              │  ✅  │   ✅    │    ✅    │   ✅
  round_id           │  ✅  │   ✅    │    ✅    │   ✅
  stream_name        │  ✅  │   ✅    │    ✅    │   ✅
  phase_timer        │  ✅  │   ✅    │    ✅    │   ✅
  round_timer        │  ✅  │   ✅    │    ✅    │   ✅
  commitment_hash    │  ✅  │   ✅    │    ✅    │   ✅
  vehicle_count      │  0   │    0    │   LIVE   │ FROZEN
  boundaries         │  ❌  │   ✅    │    ✅    │   ✅
  odds (multipliers) │  ❌  │   ✅    │    ✅    │   ✅
  win_chances        │  ❌  │   ✅    │    ✅    │   ✅
  server_seed        │  ❌  │   ❌    │    ❌    │   ✅
  result             │  ❌  │   ❌    │    ❌    │   ✅
  bet_outcomes       │  ❌  │   ❌    │    ❌    │   ✅
```

---

## 9. Configuration Reference

### 9.1 streams_config.json
```
  {
    "count_duration": 35,
    "transition_duration": 15,
    "time_slots": [
      {
        "start": "08:00",     // IST (UTC+5:30)
        "end": "12:00",
        "streams": [
          {"name": "Stream 16", "url": "https://youtube.com/live/..."}
        ]
      }
    ]
  }
```

### 9.2 bytetrack.yaml
```
  tracker_type: bytetrack
  track_high_thresh: 0.25    Primary track confidence
  track_low_thresh: 0.10     Low-conf recovery attempts
  new_track_thresh: 0.15     Min confidence for new track
  track_buffer: 150          5 seconds at 30fps
  match_thresh: 0.65         IoU for re-association
  fuse_score: true
```

### 9.3 livekit.yaml
```
  keys:
    devkey: "cctv-game-secret-key-32chars-min!"
  port: 7880
  rtc:
    tcp_port: 7881
    udp_port: 7882
    port_range_start: 50000
    port_range_end: 60000
```

### 9.4 line_configs/Stream_N.json
```
  Line mode:  {"line": [[594,464],[841,576]], "roi_poly": null, "mode": "line"}
  ROI mode:   {"line": null, "roi_poly": [[x1,y1],...,[x4,y4]], "mode": "roi"}
  Hot-reload: every 30 frames during YOLO processing
```

---

## 10. Key Constants

```
  YOLO:
    Model: yolo11x.engine (TensorRT FP16)
    Input: 1600x1600
    Confidence: 0.10
    Classes: {2: Car, 3: Motorcycle, 5: Bus, 7: Truck}
    IOU: 0.35

  Counting:
    EMA alpha: 0.30 (small boxes) / 0.45 (large boxes)
    Spatial cooldown: 20px + 0.5s
    ROI dedup: IoU > 0.25, time gap 2.5s
    Track history: 30 points per vehicle
    Min vehicle count: 5 (for clip approval)

  Round Timing:
    Betting: 15 seconds
    Counting: 35 seconds
    Waiting: 6 seconds
    Total: 56 seconds
    Cold start: wait for 5 clips

  Pipeline:
    Clip duration: 41 seconds
    Download queue target: 3
    Ready queue max: 10
    Download timeout: 90s
    YouTube URL cache: 4 hours
    YouTube 429 backoff: 10s -> 300s max

  Odds:
    Multipliers (constant): Under 3.13x, Range 2.35x, Over 3.76x, Exact 18.8x
    Win chances (constant): Under 30%, Range 40%, Over 25%, Exact 5%
    Default lambda_mean: 10.0 (when < 20 rounds of data)
    Max history per stream: 500 rounds
```

---

## 11. File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `scheduler.py` | 632 | Main entry point. Round loop, cold start, time slots, debug GUI |
| `core/counting.py` | 522 | VehicleCounter (line/ROI) + offline_count() + render_debug_frame() |
| `network/stream_manager.py` | 643 | SequentialDownloader + YOLOWorker + Pipeline |
| `ui/renderer.py` | 360 | draw_tracking_overlay (shared) + draw_playback_overlay |
| `game/game_api.py` | 343 | Thread-safe game state, Pydantic models, API response builder |
| `web_server.py` | 328 | FastAPI + uvicorn + MJPEG + WebSocket + LiveKit token |
| `game/round_manager.py` | 321 | prepare_round(), play_clip_with_overlay(), RoundData |
| `network/livekit_publisher.py` | 251 | WebRTC frame publisher to LiveKit room |
| `core/odds_engine.py` | 226 | Boundary generation, bet settlement, constant multipliers |
| `network/download.py` | 226 | yt-dlp wrapper for URL resolution |
| `core/provably_fair.py` | 222 | SHA-256 commitment scheme, verification |
| `ui/animations.py` | 166 | Globe animation, IST timer |
| `game/history_tracker.py` | 113 | Per-stream count history (500 max), JSON persistence |
| `core/config_manager.py` | 60 | JSON config load/save |
| `core/geometry_utils.py` | 28 | Line intersection (CCW), point-in-polygon |
| `main.py` | 357 | Single stream testing entry point |

---

## 12. Dependencies

```
  Runtime:
    ultralytics         YOLO inference + ByteTrack
    opencv-python       Video I/O, image processing, GUI
    fastapi             REST API + WebSocket
    uvicorn[standard]   ASGI server
    numpy               Array operations
    pydantic            API data validation
    yt-dlp              YouTube URL resolution
    livekit             WebRTC frame publishing
    livekit-api         Token generation
    torch + torchvision PyTorch with CUDA 12.1
    tensorrt-cu12       TensorRT GPU inference

  System:
    NVIDIA GPU with CUDA (tested: RTX 4060 Ti 8GB)
    Node.js LTS (recommended for yt-dlp YouTube parsing)
    ffmpeg (bundled with yt-dlp or system install)
    LiveKit server binary OR Docker (optional)
```

---

*Document generated: March 30, 2026*
*Codebase version: ~4,800 lines across 15 Python source files*
