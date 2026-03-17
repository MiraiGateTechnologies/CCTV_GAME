# CCTV Game - Vehicle Counting and Streaming System

A real-time vehicle counting and CCTV stream monitoring dashboard. Built using YOLO object detection, OpenCV, and yt-dlp.

## Features
- Real-time vehicle counting for streaming URLs and local MP4 videos
- **Line Mode**: Count vehicles that cross a designated line
- **ROI Mode**: Count vehicles that enter a selected region
- Interactive drawing controls for counting zones
- Time-gated stream scheduling supporting rotating camera feeds
- Background pre-fetching and red-light validation

## Project Structure (SOLID Architecture)

This project has been modularized according to SOLID and SoC principles:

```
CCTV_GAME/
│
├── core/
│   ├── counting.py          # Core logic: Vehicle tracking, counting, region checks.
│   ├── config_manager.py    # Managing line, ROI JSON configurations.
│   └── geometry_utils.py    # Math helpers for intersecting lines & polygons.
│
├── ui/
│   ├── renderer.py          # Drawing the HUD, dashboard, and vehicle glow brackets.
│   └── animations.py        # Visual transitions: 3D Globe loading, Results screen.
│
├── network/
│   ├── stream_manager.py    # Background worker: clip pre-fetching & validation.
│   └── download.py          # Utility to extract direct stream URLs & download MP4s.
│
├── main.py                  # Entry point for a single video stream counting session.
├── scheduler.py             # Entry point for continuous multi-camera scheduled streams.
└── web_server.py            # Simple Flask server to display the active stream in a browser.
```

## How to Run

### Installation
Ensure that the Python environment has the required dependencies:
```bash
pip install ultralytics opencv-python yt-dlp numpy supervision flask
```

### 1. Single Stream Mode (`main.py`)
Run object tracking and counting on a specific video file or live stream url.
```bash
python main.py --video my_video.mp4
python main.py --url "https://youtube.com/live/..."
```

**Controls inside the window:**
- **Left-drag**: Draw counting line (LINE mode)
- **Right-click 4 times**: Draw ROI polygon (switches to ROI mode)
- **L Key**: Reset to LINE mode and clear ROI
- **R Key**: Reset current count and timers
- **S Key**: Take a screenshot
- **+/- Keys**: Change playback speed (local videos only)
- **Q Key**: Quit immediately

### 2. Multi-camera Scheduler (`scheduler.py`)
Automatically cycle through multiple time-slotted live streams based on IST time. It includes loading globe animations and background validation of traffic streams.
```bash
python scheduler.py
```
To run the scheduler in headless mode (no OpenCV GUI window, only browser view):
```bash
python scheduler.py --no-gui
```

### 3. Background Downloader (`network/download.py`)
Download a short clip manually for testing.
```bash
python network/download.py --url "YOUTUBE_URL" --name test_video --duration 300
```

## Configuration Options

### `streams_config.json`
Define your camera feeds and time-slots here.
```json
{
  "time_slots": [
    {
      "start": "06:00",
      "end": "12:00",
      "streams": [
        {"name": "Morning Highway", "url": "https://..."}
      ]
    }
  ],
  "count_duration": 35,
  "transition_duration": 15
}
```

### Per-stream Line Configs
When you adjust the tracking line or polygon in `main.py` or `scheduler.py`, the configuration is saved locally inside the `line_configs/` folder. This ensures each camera remembers its own perspective.
