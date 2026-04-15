# Change Request: 3D Globe Night Animation for Betting Phase

## Summary

Replace per-city MP4 animation videos with a single Three.js 3D nighttime Earth globe that auto-zooms to the target city during the 15-second betting phase. Renders in browser (WebGL, 60fps). No per-city video files needed.

---

## Current State

```
BETTING (15s):
  Python renders animation video frames -> push_frame() -> LiveKit/MJPEG
  Each city needs a separate MP4 file in animation_videos/
  30+ videos = 500MB+ storage
  Adding new city = create new video manually
```

## New State

```
BETTING (15s):
  Browser renders 3D globe locally (WebGL, 60fps)
  Server sends city coordinates via API/WebSocket
  ONE earth_night texture (8MB) serves ALL cities
  Adding new city = add lat/lng in streams_config.json
```

---

## 15-Second Animation Sequence

```
TIME     WHAT HAPPENS
------   --------------------------------------------------
0-2s     Earth visible from space, slowly rotating
         Night side glowing with city lights (NASA texture)
         Stars background + blue atmosphere glow
         
2-8s     Camera flies toward target country
         Smooth ease-in-out animation
         Earth gets larger, city lights become detailed
         
8-11s    Zoom reaches city level
         Red pin drops with pulse animation
         City name label appears below pin
         
11-15s   Hold on pinned city
         Thumbnail photo + stream name appears center
         "NEXT STREAM" subtitle
         Yellow circular timer counting down (top-right)
```

---

## Architecture Change

```
BEFORE:
  scheduler.py (Python)
    -> reads MP4 video file
    -> cv2 renders frames
    -> push_frame() to LiveKit/MJPEG
    -> browser displays video stream
    
AFTER:
  scheduler.py (Python)
    -> sends {city, lat, lng, thumbnail, stream_name} via game_api
    -> push_frame() sends simple dark frame (or nothing) during BETTING
    
  index.html (Browser)
    -> detects phase=BETTING from /api/round or WebSocket
    -> hides LiveKit/MJPEG stream
    -> shows Three.js 3D globe canvas
    -> animates: rotate -> fly to city -> pin -> hold
    -> shows thumbnail at 11s
    -> detects phase=COUNTING -> hides globe, shows stream
```

---

## Files to Change

```
MODIFY:
  templates/index.html        Add Three.js globe, phase-based switching
  game/game_api.py            Add city_lat, city_lng to state
  scheduler.py                Remove video rendering, send coordinates
  streams_config.json         Replace "animation" with "lat"/"lng" fields

CREATE:
  static/earth_night.jpg      NASA night lights texture (~8MB)
  
REMOVE (after migration):
  animation_videos/*.mp4      No longer needed (saves 500MB+)

NO CHANGE:
  network/stream_manager.py   Download pipeline unchanged
  core/counting.py            YOLO/counting unchanged
  ui/renderer.py              Playback overlay unchanged
  web_server.py               Endpoints unchanged
  thumbnails/*.jpg             Still used for city photos
```

---

## streams_config.json — Replace animation with lat/lng

```json
BEFORE:
{
  "name": "Tokyo Shinjuku",
  "url": "https://...",
  "animation": "tokyo_animation.mp4",
  "thumbnail": "tokyo.jpg"
}

AFTER:
{
  "name": "Tokyo Shinjuku",
  "url": "https://...",
  "lat": 35.6762,
  "lng": 139.6503,
  "thumbnail": "tokyo.jpg"
}
```

### All Stream Coordinates

```
Stream Name                        Lat       Lng
-------------------------------------------------
Charles Austin, Hopkins            30.5083   -94.8277
Tokyo Shinjuku                     35.6938   139.7034
Tokyo Shinjuku Kabukicho           35.6938   139.7034
Shibuya Scramble - Tokyo           35.6595   139.7004
Aomori City                        40.8244   140.7400
Russia Petersburg                  59.9343   30.3351
Northern Ring Road - Papal Gate    30.0444   31.2357
Town Quay Southampton              50.8958   -1.4044
Capelle aan den IJssel             51.9292    4.5780
Sapporo City                       43.0618   141.3545
San Marcos, Texas                  29.8833   -97.9414
City of Orange Plaza, California   33.7879   -117.8531
El Gaucho, Bangkok                 13.7563   100.5018
Patong, Phuket                      7.8804    98.2920
Fresno, California                 36.7378   -119.7871
Fort Morgan, Colorado              40.2502   -103.7997
Breckenridge, Colorado             39.4817   -106.0384
The Villages, Florida              28.9348   -81.9601
Tilton, New Hampshire              43.4426   -71.5887
Bellevue, Iowa                     42.2586   -90.4218
Derry, New Hampshire               42.8806   -71.3273
North Conway, New Hampshire        44.0537   -71.1284
Las Vegas, New Mexico              35.5942   -105.2225
Fort Erie, Ontario                 42.9061   -79.0134
Montauk, New York                  41.0362   -71.9545
Watertown, New York                43.9748   -75.9108
Vicksburg, Michigan                42.1200   -85.8533
Traverse City, Michigan            44.7631   -85.6206
Park Rapids, Minnesota             46.9222   -95.0536
Apex, North Carolina               35.7327   -78.8503
```

---

## game_api.py — Add City Coordinates

```python
# Add to _state dict:
"city_lat": 0.0,
"city_lng": 0.0,
"city_name": "",

# Add to update_phase():
if city_lat is not None:
    _state["city_lat"] = city_lat
if city_lng is not None:
    _state["city_lng"] = city_lng
if city_name is not None:
    _state["city_name"] = city_name

# Add to get_api_response() during BETTING:
if phase == "BETTING":
    resp["city_lat"] = _state["city_lat"]
    resp["city_lng"] = _state["city_lng"]
    resp["city_name"] = _state["city_name"]
```

---

## scheduler.py — Send Coordinates Instead of Video

```python
# In show_betting_phase():

# REMOVE: video loading, cv2 rendering, slow-motion logic
# KEEP: circular timer on push_frame (simple dark frame)
# ADD: send city coords to game_api

game_api.update_phase(
    "BETTING",
    city_lat=clip.lat,       # from streams_config.json
    city_lng=clip.lng,
    city_name=rd.stream_name,
    ...
)

# During BETTING loop: push simple dark frame with timer only
# Browser handles 3D animation independently
```

---

## index.html — Three.js Globe

### Structure

```html
<body>
  <!-- 3D Globe (visible during BETTING only) -->
  <canvas id="globe-canvas"></canvas>
  
  <!-- Thumbnail overlay (appears at 11s during BETTING) -->
  <div id="thumbnail-overlay" style="display:none">
    <img id="thumb-img" />
    <div id="stream-name"></div>
    <div>NEXT STREAM</div>
  </div>
  
  <!-- LiveKit/MJPEG stream (visible during COUNTING + WAITING) -->
  <video id="livekit-video"></video>
  <img id="video-feed" src="/video_feed" />
  
  <!-- Status dot (always visible) -->
  <div class="status-dot"></div>
  
  <!-- Timer circle (always visible, top-right) -->
  <canvas id="timer-canvas"></canvas>
</body>
```

### Phase Switching Logic

```javascript
// Poll /api/round or listen WebSocket
function onPhaseChange(data) {
  if (data.phase === "BETTING") {
    // Hide stream, show globe
    hideStream();
    showGlobe(data.city_lat, data.city_lng, data.city_name);
    
    // Show thumbnail at 11s
    setTimeout(() => showThumbnail(data.stream_name, data.thumbnail), 11000);
  }
  else if (data.phase === "COUNTING" || data.phase === "WAITING") {
    // Hide globe, show stream
    hideGlobe();
    showStream();
  }
}
```

### Three.js Globe Setup

```javascript
// Scene: black background + stars
// Sphere: NASA night lights texture
// Atmosphere: blue glow shader
// Pin: red marker with pulse animation
// Camera: animated fly-to using TWEEN.js

// Textures needed:
//   earth_night.jpg  (NASA Black Marble, 8K or 4K)
//   earth_bump.jpg   (optional, terrain elevation)
//   stars.jpg        (starfield background)
```

### Lat/Lng to 3D Position

```javascript
function latLngToVector3(lat, lng, radius) {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lng + 180) * (Math.PI / 180);
  return new THREE.Vector3(
    -(radius * Math.sin(phi) * Math.cos(theta)),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta)
  );
}
```

---

## Required Assets

```
CREATE/DOWNLOAD:
  static/earth_night.jpg    NASA Black Marble texture
                            Source: https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144898/
                            Size: 4K version ~8MB, 8K version ~30MB
                            License: Public domain (NASA)

KEEP:
  thumbnails/*.jpg          City photos (unchanged)

REMOVE (after migration):
  animation_videos/*.mp4    No longer needed
  templates/earth_night.glb Can remove if not using model-viewer
```

---

## Timer Circle (Browser-Rendered)

```
Currently: Python draws circular timer with cv2 -> pushes as frame
After:     Browser draws timer with HTML5 Canvas (always visible)

WHY: During BETTING, Python may push minimal/dark frames
     Timer must be consistent across all phases
     Browser-rendered timer = always smooth, always visible
     
Position: top-right corner (same as current)
Colors:   BETTING=yellow, COUNTING=green, WAITING=red
```

---

## Edge Cases

```
1. Browser has no WebGL support
   -> Fallback: show dark screen + stream name text only + timer
   
2. NASA texture fails to load
   -> Fallback: dark sphere with simple wireframe outline
   
3. City lat/lng not in config (missing)
   -> Default: show earth rotating, no zoom, no pin
   
4. Thumbnail image missing
   -> Show stream name text only (no photo)
   
5. Phase switches mid-animation (betting cut short)
   -> Immediately hide globe, show stream
   
6. Slow device (low FPS WebGL)
   -> Reduce texture resolution (4K -> 2K)
   -> Disable atmosphere shader
   -> Simpler star background
```

---

## Migration Plan

```
PHASE 1 (implement):
  1. Download NASA night texture
  2. Add lat/lng to streams_config.json
  3. Add city coords to game_api.py
  4. Update scheduler.py (remove video, send coords)
  5. Build Three.js globe in index.html
  6. Add phase switching (globe <-> stream)
  7. Add thumbnail overlay at 11s
  8. Add browser-rendered timer circle

PHASE 2 (polish):
  1. Atmosphere glow shader
  2. Star particle background
  3. Smooth camera easing (TWEEN.js)
  4. Pin drop animation with bounce
  5. City name label with fade-in
  6. Mobile optimization

PHASE 3 (cleanup):
  1. Remove animation_videos/ folder
  2. Remove video playback code from scheduler.py
  3. Update README
```

---

## Estimated Effort

```
Phase 1: 1-2 days (core functionality)
Phase 2: 1 day (visual polish)
Phase 3: 30 min (cleanup)

Total: 2-3 days
```
