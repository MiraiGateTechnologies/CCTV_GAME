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

        # ── Persistent vehicle tracker (YOLO-independent dots) ──
        # Once a vehicle gets a dot, it NEVER disappears until vehicle leaves frame.
        # If YOLO misses a frame, dot continues moving at last known velocity.
        # {track_id: {"cx", "cy", "vx", "vy", "missed", "counted"}}
        self.tracked_vehicles = {}
        self.MAX_PREDICT_FRAMES = 5   # predict max 5 frames (0.25s at 20fps)
        self.MAX_VELOCITY = 15        # max pixels/frame — prevents crazy dot jumps

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
        self.tracked_vehicles.clear()

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

        Pure YOLO-based dots — no prediction, no ghost, no drift.
        Dot shown ONLY when YOLO detects the vehicle in this frame.
        tracked_vehicles used ONLY for: counted status inheritance when ByteTrack
        assigns a new track_id to the same physical vehicle.

        Returns the set of (cx, cy) for dots to draw.
        """
        seen_in_roi_this_frame = set()
        dots_to_draw = set()
        current_line = self.temp_line if self.temp_line else self.line

        if results[0].boxes is None or results[0].boxes.id is None:
            self.tracked_vehicles.clear()  # No detections = no dots
            return dots_to_draw

        # Clear tracked_vehicles each frame — only THIS frame's detections get dots.
        # Old entries caused "frozen dots" at previous positions when YOLO missed.
        old_tracked = dict(self.tracked_vehicles)  # Save for inheritance lookup
        self.tracked_vehicles.clear()

        boxes     = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes   = results[0].boxes.cls.cpu().numpy().astype(int)
        confs     = results[0].boxes.conf.cpu().numpy()

        # Minimum confidence for showing a dot (filters YOLO false positives)
        # ByteTrack tracks at 0.15+, but dots shown only at 0.20+
        # This prevents random dots on trees/poles/shadows
        MIN_DOT_CONF = 0.20

        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
            if cls not in vehicle_classes_dict:
                continue

            x1, y1, x2, y2 = box.astype(int)
            cx, cy = (x1+x2)//2, (y1+y2)//2

            box_area = (x2 - x1) * (y2 - y1)
            if box_area < min_box_area:
                continue

            # ── Inherit counted status from previous frame's track (same vehicle, new ID) ──
            if track_id not in old_tracked:
                for old_tid, old_tv in old_tracked.items():
                    dist = abs(cx - old_tv["cx"]) + abs(cy - old_tv["cy"])
                    if dist < 40:
                        if old_tv.get("counted", False):
                            self.interval_counted_ids.add(track_id)
                        if old_tid in self.track_history:
                            self.track_history[track_id] = list(self.track_history[old_tid])
                        break

            # Update tracked_vehicles (position only, no velocity/prediction)
            self.tracked_vehicles[track_id] = {
                "cx": cx, "cy": cy,
                "counted": track_id in self.interval_counted_ids,
            }

            # Update track history (for line crossing detection)
            history = self.track_history[track_id]
            history.append((cx, cy))
            if len(history) > 30:
                history.pop(0)

            # ── Counting Line ──
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
                        self.tracked_vehicles[track_id]["counted"] = True
                        self.flash_timers[track_id] = self.flash_frames
                        box_half_flash = max(x2-x1, y2-y1) // 2 + 6
                        # Store count_at for "+1" / "10!" popup display
                        self.flash_positions[track_id] = (cx, cy, box_half_flash, self.interval_total)

            # ── ROI Mode — count on EXIT ──
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
                            self.tracked_vehicles[track_id]["counted"] = True
                            self.flash_timers[track_id] = self.flash_frames
                            box_half_flash = max(x2-x1, y2-y1) // 2 + 6
                            self.flash_positions[track_id] = (cx, cy, box_half_flash, self.interval_total)

            self.prev_centers[track_id] = (cx, cy)

            if track_id in self.inside_roi_ids or track_id in self.interval_counted_ids:
                self.roi_track_boxes[track_id] = (x1, y1, x2, y2, time.time())

            # ── Dot: show ONLY for confident detected uncounted vehicles ──
            # conf >= MIN_DOT_CONF prevents random dots on false positives
            # Counting logic above still uses ALL detections (even low conf)
            if track_id not in self.interval_counted_ids and conf >= MIN_DOT_CONF:
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
            if frame_no == max_count_frames:
                counter.tracked_vehicles.clear()

        new_count = counter.interval_total

        # Store per-frame data — ONLY vehicles that had dots in debug mode.
        # This guarantees playback shows EXACTLY the same dots as debug.
        # Low-confidence false positives (conf < 0.20) are excluded.
        frame_dets = []
        if counting_active and results is not None and results[0].boxes is not None and results[0].boxes.id is not None:
            r_boxes = results[0].boxes.xyxy.cpu().numpy()
            r_tids = results[0].boxes.id.cpu().numpy().astype(int)
            r_confs = results[0].boxes.conf.cpu().numpy()
            r_classes = results[0].boxes.cls.cpu().numpy().astype(int)
            for r_box, r_tid, r_conf, r_cls in zip(r_boxes, r_tids, r_confs, r_classes):
                if r_cls not in VEHICLE_CLASSES:
                    continue
                rx1, ry1, rx2, ry2 = r_box.astype(int).tolist()
                is_counted = r_tid in counter.interval_counted_ids
                # Only store if confident enough for dot display (matches debug mode)
                show_dot = (not is_counted) and (r_conf >= 0.20)
                det = {
                    "track_id": int(r_tid),
                    "bbox": [rx1, ry1, rx2, ry2],
                    "counted": is_counted,
                    "show_dot": show_dot,
                }
                frame_dets.append(det)

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
        if counter.flash_positions:
            if frame_no not in all_detections:
                all_detections[frame_no] = []
            for tid, flash_data in list(counter.flash_positions.items()):
                fcx, fcy, fbox_half = flash_data[0], flash_data[1], flash_data[2]
                count_at = flash_data[3] if len(flash_data) > 3 else 0
                t = counter.flash_timers.get(tid, 0)
                if t > 0:
                    for det in all_detections.get(frame_no, []):
                        if det["track_id"] == tid:
                            det["crossed"] = True
                            det["flash_cx"] = fcx
                            det["flash_cy"] = fcy
                            det["flash_half"] = fbox_half
                            det["count_at"] = count_at

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

    # ── Build dots — use YOLO results directly with conf >= 0.20 filter ──
    # ── Draw YOLO bounding boxes (debug only — invisible in playback) ──
    MIN_DOT_CONF = 0.20
    dots = []
    if results is not None and results[0].boxes is not None and results[0].boxes.id is not None:
        r_boxes = results[0].boxes.xyxy.cpu().numpy()
        r_tids = results[0].boxes.id.cpu().numpy().astype(int)
        r_confs = results[0].boxes.conf.cpu().numpy()
        r_classes = results[0].boxes.cls.cpu().numpy().astype(int)
        from core.counting import VEHICLE_CLASSES as _vc
        for r_box, r_tid, r_conf, r_cls in zip(r_boxes, r_tids, r_confs, r_classes):
            if r_cls not in _vc:
                continue
            rx1, ry1, rx2, ry2 = r_box.astype(int)
 
            # Bbox color: green if counted, light blue if tracked
            if r_tid in counter.interval_counted_ids:
                box_color = (0, 180, 0)       # green = counted
            else:
                box_color = (180, 160, 50)    # light blue = tracking
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), box_color, 1, cv2.LINE_AA)
 
            # Dot for uncounted + confident vehicles only
            if r_tid not in counter.interval_counted_ids and r_conf >= MIN_DOT_CONF:
                dots.append(((rx1+rx2)//2, (ry1+ry2)//2))

    # ── Build brackets list from counter flash state ──
    brackets = []
    for tid, flash_data in list(counter.flash_positions.items()):
        fcx, fcy, fbox_half = flash_data[0], flash_data[1], flash_data[2]
        count_at = flash_data[3] if len(flash_data) > 3 else 0
        t = counter.flash_timers.get(tid, 0)
        if t > 0:
            total = counter.flash_frames
            progress = (total - t) / total
            brackets.append((fcx, fcy, fbox_half, progress, count_at))

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
