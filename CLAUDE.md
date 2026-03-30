# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

CCTV Casino Game Engine — AI counts vehicles on live traffic cameras, results drive Under/Over/Range/Exact betting. No RNG; results are real-world vehicle counts.

## Commands

```bash
# Install (NVIDIA GPU with CUDA required)
pip install ultralytics opencv-python fastapi uvicorn[standard] numpy yt-dlp

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
```

Web UI: http://localhost:5000 | API docs: http://localhost:5000/docs | WebSocket: ws://localhost:5000/ws/game

## Architecture

```
StreamDownloader (N threads, CPU)  →  download_queue  →  YOLOWorker (1 thread, GPU)  →  ready_queue
                                                                                            ↓
FastAPI (web_server.py)  ←  game_api.py  ←  scheduler.py (main thread, 56s round loop)
```

**Data flow per round:**
1. `scheduler.py` polls `ready_queue` → gets `PreProcessedClip` (result already known)
2. `round_manager.prepare_round()` → calls `odds_engine.generate_round_odds()` + `provably_fair.generate_round_fairness()` → returns immutable `RoundData`
3. BETTING (15s) → COUNTING (35s) → WAITING (6s) → `finalize_round()` saves history

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

Gaps between boundaries are intentional — counts in gap zones mean house wins all main bets.

## Key Design Decisions

### Provably Fair = Commitment Scheme (NOT seed derivation)
Result is a physical vehicle count. You cannot derive a real-world measurement from `HMAC(server_seed, client_seed:nonce)`. The commitment hash (`SHA-256(server_seed:result:boundaries:round_id)`) proves the result was locked before betting opened. Same mechanism as live dealer casinos. Client seed has no functional role (stored for audit only).

### Single YOLO Worker Thread
ByteTrack tracker state corrupts if shared across threads. One `YOLOWorker` owns the GPU exclusively. All `StreamDownloader` threads are CPU-only (ffmpeg `-c copy`).

### Pre-Processing, Not Real-Time
YOLO runs during background download via `offline_count()`. During playback, stored detections are overlaid — 6ms/frame instead of 80ms. This is why there's a `ready_queue` of pre-processed clips.

### Boundary Generation Formula
```python
under      = floor(lambda_mean * 0.75) - 1
range_low  = round(lambda_mean - 0.5)
range_high = round(lambda_mean + 0.8)
over       = ceil(lambda_mean * 1.35) + 1
exact_1    = round(lambda_mean)
exact_2    = exact_1 + 1
```
`lambda_mean` = `history_tracker.get_mean(stream_name)` — historical average vehicle count. Falls back to 10.0 if < 20 rounds of data.

## File Roles

| File | Role |
|------|------|
| `scheduler.py` | Production entry point. Main loop: poll ready_queue → prepare → bet → count → wait → finalize |
| `core/counting.py` | `VehicleCounter` (line/ROI modes) + `offline_count()` batch processor. `render_debug_frame()` uses shared `draw_tracking_overlay()` |
| `core/odds_engine.py` | Boundary generation from lambda_mean. Constants: MULTIPLIERS, WIN_CHANCES |
| `core/provably_fair.py` | SHA-256 commitment: generate_server_seed, create_commitment, verify, get_verification_data |
| `game/round_manager.py` | `prepare_round()` → `RoundData` (immutable). Phase data extraction. `play_clip_with_overlay()` |
| `game/game_api.py` | Thread-safe global state. Pydantic models. `get_api_response()` for REST, `get_overlay_data()` for renderer |
| `game/history_tracker.py` | Per-stream count history (500 rounds max), persisted to `round_history.json` |
| `network/stream_manager.py` | `Pipeline` = N `StreamDownloader` threads + 1 `YOLOWorker` thread |
| `web_server.py` | FastAPI: MJPEG `/video_feed`, REST `/api/round`, WebSocket `/ws/game` |

## Config Files

- **streams_config.json**: IST time slots + stream URLs (round-robin within slot)
- **line_configs/Stream_N.json**: Per-stream counting line/ROI (auto-saved by `--gui` mouse drawing, hot-reloaded every 30 frames)
- **bytetrack.yaml**: Tracker thresholds (track_buffer: 150 frames = 5s at 30fps)
- **round_history.json**: Persisted count history for lambda_mean calculation

## Platform Notes

- IST timezone (UTC+5:30) used for time slot comparisons
- `cv2.waitKey(30)` required even headless — prevents Windows "NOT RESPONDING"
- Debug GUI renders at 1280x720; mouse coords auto-converted to original resolution
- YOLO confidence 0.10 (catches distant vehicles), input 1600px, classes: car/bus/truck
