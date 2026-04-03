# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

CCTV Casino Game Engine -- AI counts vehicles on live traffic cameras, results drive Under/Over/Range/Exact betting. No RNG; results are real-world vehicle counts.

## Commands

```bash
# Install (NVIDIA GPU with CUDA required)
pip install ultralytics opencv-python fastapi uvicorn[standard] numpy yt-dlp
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install tensorrt-cu12

# Export TensorRT engine (one-time)
yolo export model=yolo11x.pt format=engine imgsz=1248 half=True device=0 workspace=4

# Production (multi-stream, IST time slots, web-only)
python scheduler.py --config streams_config.json

# Development (web + debug GUI with mouse line/ROI drawing)
python scheduler.py --config streams_config.json --gui

# Single stream testing
python main.py --url "https://youtube.com/live/VIDEO_ID"
python main.py --video video1 --web-port 5000

# Validate config without running
python scheduler.py --config streams_config.json --test

# Download test clips
python network/download.py --url "URL" --name video1 --duration 300

# Custom model training (knowledge distillation)
python model_training/extract_frames.py
python model_training/auto_label.py
model_training\train.bat
model_training\export_engine.bat
python model_training/deploy.py
```

Web UI: http://localhost:5000 | API docs: http://localhost:5000/docs | WebSocket: ws://localhost:5000/ws/game

## Architecture

```
SequentialDownloader (1 thread, round-robin, CPU)
  URL pre-warm (yt-dlp Python API + cookies)
  → download_queue (target: 3 clips)
  → YOLOWorker (1 thread, GPU)
    GPU warmup → offline_count() → tracked_vehicles → ready_queue

FastAPI (web_server.py)  <-  game_api.py  <-  scheduler.py (main thread, 56s round loop)
```

**Data flow per round:**
1. `scheduler.py` polls `ready_queue` -> gets `PreProcessedClip` (result already known)
2. `round_manager.prepare_round()` -> calls `odds_engine.generate_round_odds()` + `provably_fair.generate_round_fairness()` -> returns immutable `RoundData`
3. BETTING (15s) -> COUNTING (35s) -> WAITING (6s) -> `finalize_round()` saves history

**Single source of truth:** `game_api.py` holds all game state. REST `/api/round`, WebSocket `/ws/game`, and overlay renderer all read from here.

## Critical Terminology (MUST understand)

**"Odds" in this project means boundary NUMBERS, NOT multipliers:**
```
Odds (change every round, from lambda_mean):
  Under < 14  |  Range 20-21  |  Over > 28  |  Exact 20 or 21

Multipliers (CONSTANT, never change):
  Under: 3.13x  |  Range: 2.35x  |  Over: 3.76x  |  Exact: 18.8x

Win Chances (CONSTANT, never change):
  Under: 30%  |  Range: 40%  |  Over: 25%  |  Exact: 5%
```

## Key Design Decisions

### Persistent Vehicle Tracking (tracked_vehicles)
White dots use a YOLO-independent persistence system. When YOLO detects a vehicle, `tracked_vehicles` updates position + velocity. When YOLO misses a frame, position is predicted from velocity. Dots NEVER flicker. Velocity clamped to 15px/frame, predictions limited to 5 frames. Both debug GUI and playback use the SAME `tracked_vehicles` data -- pixel-identical.

### Provably Fair = Commitment Scheme (NOT seed derivation)
Result is a physical vehicle count. The commitment hash (`SHA-256(server_seed:result:boundaries:round_id)`) proves the result was locked before betting opened.

### Single YOLO Worker Thread
ByteTrack tracker state corrupts if shared across threads. One `YOLOWorker` owns the GPU exclusively.

### Pre-Processing, Not Real-Time
YOLO runs during background download via `offline_count()`. During playback, stored tracked_vehicles data is overlaid. GPU warmup at startup eliminates first-clip slowdown.

### YouTube Anti-429
URLs pre-warmed at startup via yt-dlp Python API (not subprocess). Browser cookies used. Cache TTL 5 hours. Background refresh every 4 hours.

### Boundary Generation Formula
```python
under      = floor(lambda_mean * 0.75) - 1
range_low  = round(lambda_mean - 0.5)
range_high = round(lambda_mean + 0.8)
over       = ceil(lambda_mean * 1.35) + 1
exact_1    = round(lambda_mean)
exact_2    = exact_1 + 1
```
`lambda_mean` = `history_tracker.get_mean(stream_name)`. Falls back to 10.0 if < 20 rounds of data.

## File Roles

| File | Role |
|------|------|
| `scheduler.py` | Production entry point. GPU warmup + URL pre-warm + main round loop |
| `core/counting.py` | `VehicleCounter` + `tracked_vehicles` system + `offline_count()` + `render_debug_frame()` (shared overlay) |
| `core/odds_engine.py` | Boundary generation from lambda_mean. Constants: MULTIPLIERS, WIN_CHANCES |
| `core/provably_fair.py` | SHA-256 commitment: generate_server_seed, create_commitment, verify |
| `game/round_manager.py` | `prepare_round()` -> `RoundData` (immutable). `play_clip_with_overlay()` |
| `game/game_api.py` | Thread-safe global state. Pydantic models. `get_api_response()`, `get_overlay_data()` |
| `game/history_tracker.py` | Per-stream count history (500 rounds max), persisted to `round_history.json` |
| `network/stream_manager.py` | `Pipeline` = `SequentialDownloader` + `YOLOWorker` + URL pre-warm + background refresh |
| `ui/renderer.py` | `draw_tracking_overlay()` (shared), `draw_playback_overlay()`, `draw_dashboard()` |
| `web_server.py` | FastAPI: MJPEG `/video_feed`, REST `/api/round`, WebSocket `/ws/game` |
| `model_training/` | Knowledge distillation pipeline: extract, label, train, export, deploy |

## Config Files

- **streams_config.json**: IST time slots + stream URLs (round-robin within slot)
- **line_configs/Stream_N.json**: Per-stream counting line/ROI (auto-saved by `--gui`, hot-reloaded every 30 frames)
- **bytetrack.yaml**: Tracker thresholds (track_buffer: 180 frames = 6s at 30fps)
- **round_history.json**: Persisted count history for lambda_mean calculation
- **cookies.txt**: (Optional) YouTube cookies for anti-429

## Platform Notes

- IST timezone (UTC+5:30) used for time slot comparisons
- `cv2.waitKey(30)` required even headless -- prevents Windows "NOT RESPONDING"
- Debug GUI renders at 1280x720; mouse coords auto-converted to original resolution
- YOLO confidence 0.10, input 1248px, classes: car/motorcycle/bus/truck
- TensorRT engine is GPU-specific -- export on same GPU that runs it
- GPU warmup (10 dummy frames) runs at startup to prevent first-clip slowdown
