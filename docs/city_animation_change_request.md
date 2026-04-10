# Change Request: City-Specific Animation Video for Betting Phase

## Overview

Replace the single Earth Zoom animation with per-city animation videos during the 15-sec betting phase. Each stream has a real city name — play that city's animation video before the stream starts, showing the city name + stream name as thumbnail.

## Current State

```
BETTING PHASE (15 sec):
  Plays: Earth Zoom [0rWZlvK2_DY].mp4 (SAME video for ALL streams)
  Shows: Stream name overlay after video ends
  Timer: Yellow circular timer (top-right)
```

## Desired State

```
BETTING PHASE (15 sec):
  0-3 sec:   City animation video plays (fullscreen)
  3 sec:     Thumbnail photo + stream name appears in CENTER (over video)
             Video CONTINUES playing behind the thumbnail
  10-15 sec: Video ends → freeze last frame, thumbnail stays
  0-15 sec:  Yellow circular timer (top-right, unchanged)
  
  Example:
    Round plays "Tokyo Shinjuku" stream
    → animation_videos/tokyo_animation.mp4 plays fullscreen
    → At 3 sec: thumbnail photo (thumbnails/tokyo.jpg) appears center
                "Tokyo Shinjuku" text below thumbnail
    → Video continues behind thumbnail until ends
    → After video ends: frozen frame + thumbnail stays
    → Timer counts 15→0
    
    Round plays "San Marcos-Texas, US" stream
    → animation_videos/sanmarcos_animation.mp4 plays
    → At 3 sec: thumbnail (thumbnails/sanmarcos.jpg) + "San Marcos-Texas, US"
    → Timer counts 15→0
    
    NO animation video for a city?
    → Fallback: Earth Zoom video (existing behavior)
    
    NO thumbnail photo?
    → Show stream name text only (no image, just text in center)
```

## Folder Structure

```
D:\cctv_main\CCTV_GAME\
├── animation_videos/              ← NEW FOLDER (city animation videos)
│   ├── tokyo_animation.mp4
│   ├── russia_animation.mp4
│   ├── bangkok_animation.mp4
│   ├── sanmarcos_animation.mp4
│   ├── sapporo_animation.mp4
│   ├── southampton_animation.mp4
│   ├── netherlands_animation.mp4
│   ├── california_animation.mp4
│   ├── colorado_animation.mp4
│   ├── florida_animation.mp4
│   ├── newhampshire_animation.mp4
│   ├── canada_animation.mp4
│   ├── newyork_animation.mp4
│   ├── phuket_animation.mp4
│   ├── iowa_animation.mp4
│   └── ... (one per city/region)
│
├── thumbnails/                    ← NEW FOLDER (city thumbnail photos)
│   ├── tokyo.jpg
│   ├── russia.jpg
│   ├── bangkok.jpg
│   ├── sanmarcos.jpg
│   ├── sapporo.jpg
│   └── ... (one per city/region)
│
├── Earth Zoom [0rWZlvK2_DY].mp4  ← FALLBACK (existing)
└── streams_config.json            ← ADD animation + thumbnail fields
```

## streams_config.json — Add `animation` + `thumbnail` Fields

```json
{
  "time_slots": [
    {
      "start": "12:00",
      "end": "16:00",
      "streams": [
        {
          "name": "Tokyo Shinjuku",
          "url": "https://www.youtube.com/live/...",
          "animation": "tokyo_animation.mp4",
          "thumbnail": "tokyo.jpg"
        },
        {
          "name": "Tokyo Shinjuku Kabukicho",
          "url": "https://www.youtube.com/live/...",
          "animation": "tokyo_animation.mp4",
          "thumbnail": "tokyo.jpg"
        },
        {
          "name": "Russia Petersburg",
          "url": "https://www.youtube.com/live/...",
          "animation": "russia_animation.mp4",
          "thumbnail": "russia.jpg"
        },
        {
          "name": "San Marcos-Texas, US",
          "url": "https://www.youtube.com/live/...",
          "animation": "sanmarcos_animation.mp4",
          "thumbnail": "sanmarcos.jpg"
        },
        {
          "name": "El Gaucho - Bangkok, Thailand",
          "url": "https://www.youtube.com/live/...",
          "animation": "bangkok_animation.mp4",
          "thumbnail": "bangkok.jpg"
        }
      ]
    }
  ]
}
```

**Rules:**
- Same city ke multiple streams → same animation + thumbnail (e.g., 3 Tokyo streams → tokyo_animation.mp4 + tokyo.jpg)
- `animation` field optional — missing = fallback to Earth Zoom
- `thumbnail` field optional — missing = show stream name text only (no image)
- animation path relative to `animation_videos/` folder
- thumbnail path relative to `thumbnails/` folder

## City-to-Animation Mapping (from current streams_config.json)

```
Stream Name                              → Animation File
─────────────────────────────────────────────────────────────
Charles Austin, Hopkins                  → hopkins_animation.mp4
Tokyo Shinjuku                           → tokyo_animation.mp4
Tokyo Shinjuku Kabukicho                 → tokyo_animation.mp4
Shibuya Scramble Crossing - Tokyo        → tokyo_animation.mp4
Aomori City (Yanagimachi Intersection)   → aomori_animation.mp4
Russia Petersburg                        → russia_animation.mp4
Northern Ring Road - Papal Gate          → egypt_animation.mp4
Town Quay Southampton                    → southampton_animation.mp4
Capelle aan den IJssel, Netherlands      → netherlands_animation.mp4
Sapporo City, Japan                      → sapporo_animation.mp4
Sapporo City 2, Japan                    → sapporo_animation.mp4
San Marcos-Texas, US                     → sanmarcos_animation.mp4
San Marcos-Texas 2, US                   → sanmarcos_animation.mp4
San Marcos-Texas 3, US                   → sanmarcos_animation.mp4
City of Orange Plaza-California, US      → california_animation.mp4
El Gaucho - Bangkok, Thailand            → bangkok_animation.mp4
Patong-Phuket, Thailand                  → phuket_animation.mp4
Fresno-California, USA                   → california_animation.mp4
Fort Morgan-Colorado, USA                → colorado_animation.mp4
Breckenridge-Colorado, USA               → colorado_animation.mp4
The Villages-Florida, USA                → florida_animation.mp4
Tilton-New Hampshire, USA                → newhampshire_animation.mp4
Bellevue-Iowa, USA                       → iowa_animation.mp4
Derry-New Hampshire, USA                 → newhampshire_animation.mp4
North Conway-New Hampshire, USA          → newhampshire_animation.mp4
Las Vegas-New Mexico, USA                → newmexico_animation.mp4
Fort Erie-Ontario, Canada                → canada_animation.mp4
Montauk-New York, USA                    → newyork_animation.mp4
Watertown-New York, USA                  → newyork_animation.mp4
```

## Files to Change

```
MODIFY: streams_config.json
  - Add "animation" field to each stream entry

MODIFY: scheduler.py → show_betting_phase()
  - Read animation field from RoundData
  - Build video path: animation_videos/{animation}
  - If file exists → play that video
  - If not exists → fallback to Earth Zoom video

MODIFY: game/round_manager.py → RoundData
  - Add animation_video field (passed from clip's stream config)

MODIFY: network/stream_manager.py → SequentialDownloader
  - Pass animation field through to PreProcessedClip

MODIFY: network/stream_manager.py → PreProcessedClip
  - Add animation_video field

NO CHANGE: ui/renderer.py (playback overlay unchanged)
NO CHANGE: core/counting.py (counting logic unchanged)
NO CHANGE: web_server.py (API unchanged)
NO CHANGE: game/game_api.py (game state unchanged)
```

## Code Changes — Detail

### 1. streams_config.json — Add animation field

```json
{"name": "Tokyo Shinjuku", "url": "...", "animation": "tokyo_animation.mp4"}
```

### 2. PreProcessedClip — Add field

```python
class PreProcessedClip:
    def __init__(self, ..., animation_video: str = ""):
        ...
        self.animation_video = animation_video
```

### 3. SequentialDownloader — Pass animation through

```python
# In _download_loop(), when creating DownloadedClip:
dl_clip = DownloadedClip(
    ...,
    animation_video=stream.get("animation", ""),
)

# In DownloadedClip class:
class DownloadedClip:
    def __init__(self, ..., animation_video: str = ""):
        ...
        self.animation_video = animation_video
```

### 4. YOLOWorker — Pass animation through to PreProcessedClip

```python
clip = PreProcessedClip(
    ...,
    animation_video=dl_clip.animation_video,
)
```

### 5. scheduler.py → show_betting_phase()

```python
ANIMATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "animation_videos")
FALLBACK_VIDEO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "Earth Zoom [0rWZlvK2_DY].mp4")

def show_betting_phase(rd, duration, pipeline=None):
    # Determine which video to play
    anim_file = rd.clip.animation_video  # e.g., "tokyo_animation.mp4"
    video_path = os.path.join(ANIMATION_DIR, anim_file) if anim_file else ""
    
    if not video_path or not os.path.exists(video_path):
        video_path = FALLBACK_VIDEO  # Earth Zoom fallback
    
    # Rest of function same — open video, play, show stream name, timer
    cap = cv2.VideoCapture(video_path)
    ...
```

### 6. RoundData — Add animation info

```python
@dataclass
class RoundData:
    ...
    # Animation video for betting phase
    # Passed from PreProcessedClip → used by show_betting_phase()
```

## Visual Timeline (15 sec betting phase)

```
TIME    WHAT'S SHOWN
----    ----------------------------------------
0-3s    City animation video plays FULLSCREEN
        Only yellow circular timer at top-right
        NO thumbnail, NO text

3s      Thumbnail photo + stream name APPEARS in CENTER
        Animation video CONTINUES playing behind it
        Semi-transparent dark backdrop behind thumbnail

3-10s   Animation video plays + thumbnail + stream name on top
        Timer counting down (top-right)

10-15s  Video ended -> frozen last frame
        Thumbnail + stream name STAYS visible
        Timer continues counting (top-right)
```

## Thumbnail + Stream Name Layout (appears at 3 sec)

```
  +--------------------------------------------------------+
  |                                                    [O] | <- yellow timer
  |        (animation video playing behind)                |
  |                                                        |
  |               +----------------------+                 |
  |               |                      |                 |
  |               |    +----------+      |                 |
  |               |    | THUMBNAIL|      |  <- city photo  |
  |               |    |  (image) |      |    200x120px    |
  |               |    +----------+      |                 |
  |               |                      |                 |
  |               |   Tokyo Shinjuku     |  <- stream name |
  |               |   NEXT STREAM        |  <- subtitle    |
  |               |                      |                 |
  |               +----------------------+                 |
  |         (semi-transparent dark backdrop)               |
  +--------------------------------------------------------+

NO thumbnail photo available? -> text only:

  +--------------------------------------------------------+
  |                                                    [O] |
  |        (animation video playing behind)                |
  |                                                        |
  |               +----------------------+                 |
  |               |                      |                 |
  |               |   Tokyo Shinjuku     |  <- stream name |
  |               |   NEXT STREAM        |  <- subtitle    |
  |               |                      |                 |
  |               +----------------------+                 |
  |                                                        |
  +--------------------------------------------------------+
```

## Edge Cases

```
1. Animation file missing       -> fallback to Earth Zoom
2. Animation field not in config -> fallback to Earth Zoom
3. Thumbnail file missing       -> show stream name text only (no image)
4. Thumbnail field not in config -> show stream name text only (no image)
5. Video shorter than 3 sec     -> thumbnail appears when video ends (not at 3s)
6. Video shorter than 15 sec    -> freeze last frame (existing behavior)
7. Video longer than 15 sec     -> stops at 15 sec (existing behavior)
8. Same animation for multiple  -> plays same video, different stream name + thumbnail
9. New stream without animation -> Earth Zoom + text only
10. Thumbnail image very large  -> resize to max 200x120px, maintain aspect ratio
11. Thumbnail image very small  -> upscale to min 150x90px
```

## Estimated Effort

```
streams_config.json:     10 min (add animation field to all streams)
animation_videos/:       User creates folder + puts videos
PreProcessedClip:        5 min (add field)
DownloadedClip:          5 min (add field)
YOLOWorker:              5 min (pass field through)
SequentialDownloader:    5 min (pass field through)
scheduler.py:            15 min (video path logic)
Total:                   ~45 min code changes
```
