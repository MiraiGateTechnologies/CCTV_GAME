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

        Uses a persistent tracked_vehicles system:
        - YOLO detects vehicle → update position + velocity
        - YOLO misses vehicle → PREDICT position from velocity (dot stays visible)
        - Dot NEVER disappears until vehicle leaves frame or unseen for 20 frames

        Returns the set of (cx, cy) for dots to draw.
        """
        seen_in_roi_this_frame = set()
        current_line = self.temp_line if self.temp_line else self.line

        # ── Step 1: Process YOLO detections — update tracked_vehicles ──
        detected_tids = set()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes     = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes   = results[0].boxes.cls.cpu().numpy().astype(int)
            confs     = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
                # Only filter by class — NOT by confidence
                # ByteTrack already filtered via track_high/low_thresh
                if cls not in vehicle_classes_dict:
                    continue

                x1, y1, x2, y2 = box.astype(int)
                cx, cy = (x1+x2)//2, (y1+y2)//2

                box_area = (x2 - x1) * (y2 - y1)
                if box_area < min_box_area:
                    continue

                detected_tids.add(track_id)

                # Update or create tracked vehicle
                if track_id in self.tracked_vehicles:
                    tv = self.tracked_vehicles[track_id]
                    # Compute velocity from position delta
                    new_vx = cx - tv["cx"]
                    new_vy = cy - tv["cy"]
                    # Clamp velocity — CCTV vehicles never move >15px/frame
                    # Without this, YOLO bbox jumps cause dots to fly randomly
                    new_vx = max(-self.MAX_VELOCITY, min(self.MAX_VELOCITY, new_vx))
                    new_vy = max(-self.MAX_VELOCITY, min(self.MAX_VELOCITY, new_vy))
                    # Smooth velocity (prevents jitter without snake trail on position)
                    tv["vx"] = 0.6 * new_vx + 0.4 * tv["vx"]
                    tv["vy"] = 0.6 * new_vy + 0.4 * tv["vy"]
                    tv["cx"] = cx   # position = raw (no lag, no snake)
                    tv["cy"] = cy
                    tv["missed"] = 0
                else:
                    # New vehicle — check if it matches a recently-lost tracked vehicle
                    inherited = False
                    for old_tid, old_tv in list(self.tracked_vehicles.items()):
                        if old_tv["missed"] == 0:
                            continue  # Still active, not lost
                        dist = abs(cx - old_tv["cx"]) + abs(cy - old_tv["cy"])
                        if dist < 40:
                            # Inherit from lost vehicle (same physical vehicle, new track_id)
                            self.tracked_vehicles[track_id] = {
                                "cx": cx, "cy": cy,
                                "vx": old_tv["vx"], "vy": old_tv["vy"],
                                "missed": 0, "counted": old_tv["counted"],
                            }
                            # Inherit track history for line crossing detection
                            if old_tid in self.track_history:
                                self.track_history[track_id] = list(self.track_history[old_tid])
                            if old_tv["counted"]:
                                self.interval_counted_ids.add(track_id)
                            # Remove old entry
                            self.tracked_vehicles.pop(old_tid, None)
                            inherited = True
                            break

                    if not inherited:
                        self.tracked_vehicles[track_id] = {
                            "cx": cx, "cy": cy,
                            "vx": 0.0, "vy": 0.0,
                            "missed": 0, "counted": False,
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
                            self.flash_positions[track_id] = (cx, cy, box_half_flash)

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
                                self.flash_positions[track_id] = (cx, cy, box_half_flash)

                self.prev_centers[track_id] = (cx, cy)

                if track_id in self.inside_roi_ids or track_id in self.interval_counted_ids:
                    self.roi_track_boxes[track_id] = (x1, y1, x2, y2, time.time())

        # ── Step 2: Predict positions for UNDETECTED vehicles (dot stays visible) ──
        frame_h = self.last_h or 1080
        frame_w = self.last_w or 1920

        for tid in list(self.tracked_vehicles.keys()):
            if tid in detected_tids:
                continue  # Already updated above

            tv = self.tracked_vehicles[tid]
            tv["missed"] += 1

            # Only predict if velocity is meaningful (prevents stationary false dots)
            speed = abs(tv["vx"]) + abs(tv["vy"])
            if speed >= 1.0:
                # Predict position using last known velocity
                tv["cx"] += tv["vx"]
                tv["cy"] += tv["vy"]
                # Decay velocity quickly (vehicle likely stopping/occluded)
                tv["vx"] *= 0.70
                tv["vy"] *= 0.70

            # Update track history with predicted position (for line crossing)
            pcx, pcy = int(tv["cx"]), int(tv["cy"])
            history = self.track_history[tid]
            history.append((pcx, pcy))
            if len(history) > 30:
                history.pop(0)

            # Check line crossing during prediction (vehicle might cross while occluded)
            if self.mode == "line" and tid not in self.interval_counted_ids:
                if current_line and len(current_line) == 2:
                    A, B = current_line
                    hist = self.track_history[tid]
                    for i in range(max(0, len(hist) - 5), len(hist) - 1):
                        if is_intersect(A, B, hist[i], hist[i + 1]):
                            if counting_active and not self._is_duplicate_crossing(pcx, pcy):
                                self.interval_total += 1
                                self.interval_counted_ids.add(tid)
                                tv["counted"] = True
                                self.flash_timers[tid] = self.flash_frames
                                self.flash_positions[tid] = (pcx, pcy, 20)
                            break

            # Remove immediately if outside frame (zero margin — no dots outside screen)
            # OR if unseen for too long
            if (tv["missed"] > self.MAX_PREDICT_FRAMES or
                    tv["cx"] < 0 or tv["cx"] > frame_w or
                    tv["cy"] < 0 or tv["cy"] > frame_h):
                self.tracked_vehicles.pop(tid, None)
                self.track_history.pop(tid, None)
                continue

        # ── Step 3: Build dots from tracked_vehicles (NOT from YOLO directly) ──
        dots_to_draw = set()
        for tid, tv in self.tracked_vehicles.items():
            if tid not in self.interval_counted_ids and not tv.get("counted", False):
                dots_to_draw.add((int(tv["cx"]), int(tv["cy"])))

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

        # Store per-frame data from tracked_vehicles (NOT raw YOLO boxes)
        # This ensures playback shows EXACTLY what debug mode shows:
        # - Same dots (from tracked_vehicles, including predicted positions)
        # - Same persistence (no flicker — predicted dots included)
        # - Same counting status
        frame_dets = []
        for tid, tv in counter.tracked_vehicles.items():
            det = {
                "track_id": int(tid),
                "bbox": [int(tv["cx"]) - 20, int(tv["cy"]) - 15,
                         int(tv["cx"]) + 20, int(tv["cy"]) + 15],
                "counted": tv.get("counted", False) or tid in counter.interval_counted_ids,
                "predicted": tv["missed"] > 0,  # True if position was predicted (not detected)
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

    # ── Build dots from tracked_vehicles (SAME source as playback) ──
    # This guarantees pixel-identical dots between debug and playback.
    # tracked_vehicles contains both detected AND predicted positions.
    dots = []
    for tid, tv in counter.tracked_vehicles.items():
        if tid not in counter.interval_counted_ids and not tv.get("counted", False):
            dots.append((int(tv["cx"]), int(tv["cy"])))

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
