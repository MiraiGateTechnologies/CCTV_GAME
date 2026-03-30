"""
Core counting logic for CCTV Vehicle Counter using YOLO and OpenCV geometry.
Separated from the UI/rendering logic.

Includes:
  - VehicleCounter: real-time or frame-by-frame counting class
  - offline_count(): batch process a clip file and return result + per-frame detections
"""
import time
import math
import os
import cv2
from collections import defaultdict
from core.geometry_utils import is_intersect, point_in_polygon
from core.config_manager import load_line_config, save_line_config

# Vehicle classes for YOLO detection (used in both online and offline modes)
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

class VehicleCounter:
    def __init__(self, config_file="line_config.json", flash_frames=4, count_interval=35, wait_interval=15):
        self.config_file = config_file
        
        # External configs
        self.flash_frames = flash_frames
        self.count_interval = count_interval
        self.wait_interval = wait_interval

        # ── Geometry ──
        self.line = None          # [(x1,y1),(x2,y2)]
        self.roi_poly = None      # [(x1,y1),...,(x4,y4)]

        # ── Mode: "line" or "roi" ──
        self.mode = "line"

        # ── Frame info ──
        self.last_h = None
        self.last_w = None
        self.config_mtime = 0

        # ── Drawing state ──
        self.drawing_line = False
        self.temp_line = None
        self.poly_points = []
        self.poly_preview_pt = None

        # ── Glow brackets ──
        self.flash_timers = {}
        self.flash_positions = {}

        # ── Interval counting ──
        self.phase = "COUNT"
        self.phase_start = time.time()
        self.interval_total = 0
        self.interval_class_counts = defaultdict(int)
        self.interval_counted_ids = set()
        self.inside_roi_ids = set()

        # ── Spatial deduplication ──
        self.recent_crossings = []
        self.SPATIAL_COOLDOWN_PX   = 20
        self.SPATIAL_COOLDOWN_SECS = 0.5
        
        # ── ROI Deduplication State ──
        self.roi_track_boxes = {}  # track_id -> (x1, y1, x2, y2, timestamp)

        # ── Tracking state ──
        self.track_history = defaultdict(list)
        self.prev_centers = {}

    def set_frame_size(self, h, w):
        self.last_h = h
        self.last_w = w

        if self.line is None:
            self.line = [(int(w*0.1), int(h*0.55)), (int(w*0.4), int(h*0.55))]

        # Hot-reload JSON
        try:
            if not os.path.exists(self.config_file):
                self.save_config()
            mtime = os.path.getmtime(self.config_file)
            if mtime > self.config_mtime:
                self.config_mtime = mtime
                data = load_line_config(self.config_file)
                if data:
                    mode = data.get("mode", "line")
                    if mode == "roi" and data.get("roi_poly"):
                        self.roi_poly = [(int(p[0]), int(p[1])) for p in data["roi_poly"]]
                        self.mode = "roi"
                    elif data.get("line"):
                        self.line = [(int(p[0]), int(p[1])) for p in data["line"]]
                        self.mode = "line"
                        self.roi_poly = None
        except Exception:
            pass

    def save_config(self):
        data = {
            "line": self.line,
            "roi_poly": self.roi_poly,
            "mode": self.mode
        }
        if save_line_config(self.config_file, data):
             self.config_mtime = os.path.getmtime(self.config_file)

    def _is_duplicate_crossing(self, cx, cy) -> bool:
        """Return True if a crossing at (cx,cy) already happened nearby recently."""
        now = time.time()
        self.recent_crossings = [
            (px, py, t) for px, py, t in self.recent_crossings
            if now - t < self.SPATIAL_COOLDOWN_SECS
        ]
        for px, py, t in self.recent_crossings:
            dist = math.hypot(cx - px, cy - py)
            if dist < self.SPATIAL_COOLDOWN_PX:
                return True
        self.recent_crossings.append((cx, cy, now))
        return False

    def _is_duplicate_footprint(self, new_x1, new_y1, new_x2, new_y2, max_time_gap=2.5):
        """Return the duplicate track_id if a new detection heavily overlaps with a recently tracked object."""
        now = time.time()
        self.roi_track_boxes = {
            cid: (px1, py1, px2, py2, t) 
            for cid, (px1, py1, px2, py2, t) in self.roi_track_boxes.items()
            if now - t < max_time_gap
        }
        
        ncx, ncy = (new_x1 + new_x2) / 2, (new_y1 + new_y2) / 2
        for cid, (px1, py1, px2, py2, t) in self.roi_track_boxes.items():
            ix1, iy1 = max(new_x1, px1), max(new_y1, py1)
            ix2, iy2 = min(new_x2, px2), min(new_y2, py2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area1 = (new_x2 - new_x1) * (new_y2 - new_y1)
                area2 = (px2 - px1) * (py2 - py1)
                iou = inter / (area1 + area2 - inter)
                if iou > 0.25:
                    return cid
                    
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            dist = math.hypot(ncx - pcx, ncy - pcy)
            if dist < max(new_x2 - new_x1, new_y2 - new_y1) * 0.6:
                return cid
                
        return None

    def _clear_interval(self):
        """Called at start of each new COUNT phase."""
        self.interval_total = 0
        self.interval_class_counts.clear()
        self.interval_counted_ids.clear()
        self.inside_roi_ids.clear()
        self.recent_crossings.clear()
        self.roi_track_boxes.clear()

    def update_phase(self):
        """Update COUNT / WAIT cycle. Returns True if counting active."""
        elapsed = time.time() - self.phase_start
        if self.phase == "COUNT":
            if elapsed >= self.count_interval:
                self.phase = "WAIT"
                self.phase_start = time.time()
            return True
        else:  # WAIT
            if elapsed >= self.wait_interval:
                self.phase = "COUNT"
                self.phase_start = time.time()
                self._clear_interval()
            return False

    def process_detections(self, results, counting_active, vehicle_classes_dict, min_box_area, confidence_threshold):
        """Processes YOLO detections and updates tracking state.
        
        This handles the pure business logic of checking geometry intersections
        and region memberships without any drawing operations.
        Returns the set of track_ids that were marked to display a center dot.
        """
        seen_in_roi_this_frame = set()
        dots_to_draw = set()
        current_line = self.temp_line if self.temp_line else self.line

        if results[0].boxes is None or results[0].boxes.id is None:
            return dots_to_draw

        boxes     = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes   = results[0].boxes.cls.cpu().numpy().astype(int)
        confs     = results[0].boxes.conf.cpu().numpy()

        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
            if cls not in vehicle_classes_dict or conf < confidence_threshold:
                continue

            x1, y1, x2, y2 = box.astype(int)
            cx, cy = (x1+x2)//2, (y1+y2)//2

            box_area = (x2 - x1) * (y2 - y1)
            if box_area < min_box_area:
                continue
            
            _EMA_ALPHA = 0.30 if box_area < 2000 else 0.45
            if track_id in self.prev_centers:
                pcx, pcy = self.prev_centers[track_id]
                cx = int(_EMA_ALPHA * cx + (1 - _EMA_ALPHA) * pcx)
                cy = int(_EMA_ALPHA * cy + (1 - _EMA_ALPHA) * pcy)



            history = self.track_history[track_id]
            history.append((cx, cy))
            if len(history) > 30:
                history.pop(0)

            # Counting Line
            if self.mode == "line" and track_id not in self.interval_counted_ids:
                if current_line and len(current_line) == 2:
                    A, B = current_line
                    hist = self.track_history[track_id]
                    crossed = False
                    for i in range(max(0, len(hist) - 20), len(hist) - 1):
                        p1, p2 = hist[i], hist[i + 1]
                        if is_intersect(A, B, p1, p2):
                            crossed = True
                            break
                    if crossed and counting_active and not self._is_duplicate_crossing(cx, cy):
                        self.interval_total += 1
                        self.interval_class_counts[vehicle_classes_dict[cls]] += 1
                        self.interval_counted_ids.add(track_id)
                        self.flash_timers[track_id] = self.flash_frames
                        box_half_flash = max(x2-x1, y2-y1) // 2 + 6
                        self.flash_positions[track_id] = (cx, cy, box_half_flash)

            # ROI Mode — count on EXIT (vehicle leaves ROI → counted)
            elif self.mode == "roi" and self.roi_poly:
                inside = point_in_polygon((cx, cy), self.roi_poly)

                if track_id not in self.inside_roi_ids and track_id not in self.interval_counted_ids:
                    dup_id = self._is_duplicate_footprint(x1, y1, x2, y2)
                    if dup_id is not None:
                        if dup_id in self.interval_counted_ids:
                            self.interval_counted_ids.add(track_id)
                        if dup_id in self.inside_roi_ids:
                            self.inside_roi_ids.add(track_id)

                if inside:
                    seen_in_roi_this_frame.add(track_id)
                    self.inside_roi_ids.add(track_id)
                else:
                    if track_id in self.inside_roi_ids and track_id not in self.interval_counted_ids:
                        if counting_active and not self._is_duplicate_crossing(cx, cy):
                            self.interval_total += 1
                            self.interval_class_counts[vehicle_classes_dict[cls]] += 1
                            self.interval_counted_ids.add(track_id)
                            self.flash_timers[track_id] = self.flash_frames
                            box_half_flash = max(x2-x1, y2-y1) // 2 + 6
                            self.flash_positions[track_id] = (cx, cy, box_half_flash)

            self.prev_centers[track_id] = (cx, cy)

            if track_id in self.inside_roi_ids or track_id in self.interval_counted_ids:
                self.roi_track_boxes[track_id] = (x1, y1, x2, y2, time.time())

            if track_id not in self.interval_counted_ids:
                dots_to_draw.add((cx, cy))

        # Remove IDs that left ROI
        self.inside_roi_ids = self.inside_roi_ids.intersection(seen_in_roi_this_frame)
        return dots_to_draw


# ═══════════════════════════════════════════════════════════════════
#  OFFLINE COUNTING — Batch process a clip, return result + detections
# ═══════════════════════════════════════════════════════════════════

def offline_count(clip_path: str, line_config_file: str, model,
                  count_duration: float = 35.0, confidence: float = 0.10,
                  imgsz: int = 1600, min_box_area: int = 0,
                  debug_callback=None) -> dict | None:
    """
    Run YOLO + VehicleCounter on a pre-recorded clip file (offline/batch mode).

    Same counting algorithm as real-time, but:
      - Processes ALL frames (no skip_frames)
      - No time pressure — takes as long as needed
      - Stores per-frame detection data for playback overlay
      - Only counts within first `count_duration` seconds

    Args:
        clip_path:        Path to MP4 clip file
        line_config_file: Path to per-stream line/ROI config JSON
        model:            Loaded YOLO model instance
        count_duration:   Seconds to count (default 35, clip can be longer)
        confidence:       YOLO confidence threshold
        imgsz:            YOLO input resolution
        min_box_area:     Minimum bounding box area filter
        debug_callback:   Optional callback for --gui debug visualization.
                          Called per frame: debug_callback(annotated_frame, frame_no,
                          counter, results, counting_active).
                          Adds ~3ms overhead per frame — negligible vs YOLO.

    Returns:
        {
            "result": 7,                    # total vehicle count
            "detections": {                  # per-frame detection boxes
                0: [{"track_id": 3, "bbox": [100,200,160,250]}],
                1: [{"track_id": 3, "bbox": [102,200,162,250]}],
                ...
            },
            "counting_events": [             # moments when count incremented
                {"frame": 115, "ts": 4.6, "count_at": 1},
                {"frame": 198, "ts": 7.9, "count_at": 2},
                ...
            ],
            "total_frames_processed": 875,
            "clip_fps": 25.0
        }
        Returns None on failure.
    """
    cap = cv2.VideoCapture(clip_path, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY
    ])
    if not cap.isOpened():
        print(f"[OFFLINE_COUNT] ERROR: Cannot open clip: {clip_path}")
        return None

    clip_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    max_count_frames = int(count_duration * clip_fps)

    # Create counter with the stream's line config
    counter = VehicleCounter(config_file=line_config_file)
    counter.phase = "COUNT"
    counter.phase_start = time.time()

    # Read first frame to set frame size and load line config
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return None
    counter.set_frame_size(first_frame.shape[0], first_frame.shape[1])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset to frame 0

    all_detections = {}
    counting_events = []
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only count within count_duration (first 35 sec)
        counting_active = frame_no < max_count_frames

        # Hot-reload line config every ~1 sec (only during counting)
        if counting_active and frame_no % 30 == 0:
            counter.set_frame_size(first_frame.shape[0], first_frame.shape[1])

        # Capture count BEFORE processing to detect new crossings
        prev_count = counter.interval_total

        if counting_active:
            # First 35s: full YOLO + tracking + counting
            results = model.track(
                frame, persist=True, classes=list(VEHICLE_CLASSES.keys()),
                conf=confidence, imgsz=imgsz, half=True, verbose=False,
                device=0, tracker="bytetrack.yaml", iou=0.35,
            )
            dots = counter.process_detections(results, counting_active, VEHICLE_CLASSES, min_box_area, confidence)
        else:
            # Last 6s: NO YOLO — raw frame only, count frozen
            results = None
            dots = set()

        new_count = counter.interval_total

        # Store per-frame detection boxes for playback overlay
        frame_dets = []
        if results is not None and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = box.astype(int).tolist()
                frame_dets.append({
                    "track_id": int(tid),
                    "bbox": [x1, y1, x2, y2],
                    "counted": int(tid) in counter.interval_counted_ids,
                })

        if frame_dets:
            all_detections[frame_no] = frame_dets

        # Record counting event if count increased
        if new_count > prev_count and counting_active:
            counting_events.append({
                "frame": frame_no,
                "ts": round(frame_no / clip_fps, 2),
                "count_at": new_count,
            })

        # Store flash positions for this frame (for playback bracket animation)
        # Only mark "crossed" for vehicles that JUST crossed (flash_timers > 0)
        if counter.flash_positions:
            if frame_no not in all_detections:
                all_detections[frame_no] = []
            for tid, (fcx, fcy, fbox_half) in list(counter.flash_positions.items()):
                t = counter.flash_timers.get(tid, 0)
                if t > 0:
                    for det in all_detections.get(frame_no, []):
                        if det["track_id"] == tid:
                            det["crossed"] = True
                            det["flash_cx"] = fcx
                            det["flash_cy"] = fcy
                            det["flash_half"] = fbox_half

        # Debug visualization callback (--gui mode)
        if debug_callback is not None:
            debug_callback(frame, frame_no, counter, results, counting_active)

        # Decrement flash_timers and clean up — same as old draw_glow_bracket() did
        # Without this, flash_positions accumulates forever and blinks break
        for tid in list(counter.flash_timers.keys()):
            counter.flash_timers[tid] -= 1
            if counter.flash_timers[tid] <= 0:
                counter.flash_timers.pop(tid, None)
                counter.flash_positions.pop(tid, None)

        frame_no += 1

    cap.release()

    return {
        "result": counter.interval_total,
        "detections": all_detections,
        "counting_events": counting_events,
        "total_frames_processed": frame_no,
        "clip_fps": clip_fps,
    }


def render_debug_frame(frame, counter, results, counting_active,
                       frame_no, clip_fps, stream_name=""):
    """
    Render a debug-annotated frame for the --gui development window.
    Shows exactly what the background YOLO pipeline is doing.

    Uses draw_tracking_overlay() — the SAME shared function used by playback mode.
    This guarantees pixel-identical dots, brackets, and line/ROI rendering.
    Only the dashboard is debug-specific (shows dev info instead of game info).

    Adds ~2-3ms per frame overhead (negligible vs YOLO 50-80ms).
    """
    from ui.renderer import draw_tracking_overlay

    h, w = frame.shape[:2]

    # ── Build line_config from counter state ──
    line_config = {
        "mode": counter.mode,
        "line": list(counter.line) if counter.line else None,
        "roi_poly": list(counter.roi_poly) if counter.roi_poly else None,
    }

    # ── Build dots list (uncounted vehicles only — same as playback) ──
    dots = []
    # if results[0].boxes is not None and results[0].boxes.id is not None:
    #     boxes = results[0].boxes.xyxy.cpu().numpy()
    #     track_ids = results[0].boxes.id.cpu().numpy().astype(int)
    #     for box, tid in zip(boxes, track_ids):
    #         if tid not in counter.interval_counted_ids:
    #             x1, y1, x2, y2 = box.astype(int)
    #             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    #             dots.append((cx, cy))
    if results is not None and results[0].boxes is not None and results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        for tid in track_ids:
            if tid not in counter.interval_counted_ids:
                if tid in counter.prev_centers:
                    cx, cy = counter.prev_centers[tid]      # ← EMA-smoothed!
                    dots.append((cx, cy))

    # ── Build brackets list from counter flash state ──
    brackets = []
    for tid, (fcx, fcy, fbox_half) in list(counter.flash_positions.items()):
        t = counter.flash_timers.get(tid, 0)
        if t > 0:
            total = counter.flash_frames
            progress = (total - t) / total
            brackets.append((fcx, fcy, fbox_half, progress))

    # ── Draw shared tracking overlay (identical to playback mode) ──
    draw_tracking_overlay(frame, dots, brackets, line_config)

    # ── Debug-only dashboard (dev info — intentionally different from playback) ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    phase_label = "COUNTING" if counting_active else "BEYOND 35s (not counted)"
    phase_color = (0, 255, 100) if counting_active else (80, 80, 255)
    ts_sec = round(frame_no / (clip_fps or 25.0), 1)

    y = 30
    cv2.putText(frame, f"DEBUG: {stream_name}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    y += 25
    cv2.putText(frame, f"COUNT: {counter.interval_total}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    y += 25
    mode_label = "LINE" if counter.mode == "line" else "ROI"
    cv2.putText(frame, f"Mode: {mode_label} | Frame: {frame_no} | {ts_sec}s", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, f"Phase: {phase_label}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, phase_color, 1, cv2.LINE_AA)

    # Controls hint at bottom
    cv2.putText(frame, "LDrag=Line | RClick x4=ROI | S=Save | L=LineMode | Q=Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)

    return frame
