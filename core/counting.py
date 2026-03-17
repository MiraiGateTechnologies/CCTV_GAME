"""
Core counting logic for CCTV Vehicle Counter using YOLO and OpenCV geometry.
Separated from the UI/rendering logic.
"""
import time
import math
import os
import cv2
from collections import defaultdict
from core.geometry_utils import is_intersect, point_in_polygon
from core.config_manager import load_line_config, save_line_config

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
                    if "line" in data:
                        self.line = [(int(p[0]), int(p[1])) for p in data["line"]]
                    if "roi_poly" in data and data["roi_poly"]:
                        self.roi_poly = [(int(p[0]), int(p[1])) for p in data["roi_poly"]]
                        self.mode = "roi"
                    if "mode" in data:
                        self.mode = data["mode"]
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

    def _clear_interval(self):
        """Called at start of each new COUNT phase."""
        self.interval_total = 0
        self.interval_class_counts.clear()
        self.interval_counted_ids.clear()
        self.inside_roi_ids.clear()
        self.recent_crossings.clear()

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
            cx, cy = (x1+x2)//2, y2

            _EMA_ALPHA = 0.80
            if track_id in self.prev_centers:
                pcx, pcy = self.prev_centers[track_id]
                cx = int(_EMA_ALPHA * cx + (1 - _EMA_ALPHA) * pcx)
                cy = int(_EMA_ALPHA * cy + (1 - _EMA_ALPHA) * pcy)

            box_area = (x2 - x1) * (y2 - y1)
            if box_area < min_box_area:
                continue

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
                    for i in range(max(0, len(hist) - 10), len(hist) - 1):
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

            # ROI Mode
            elif self.mode == "roi" and self.roi_poly:
                inside = point_in_polygon((cx, cy), self.roi_poly)
                if inside:
                    seen_in_roi_this_frame.add(track_id)
                    if (track_id not in self.inside_roi_ids
                            and track_id not in self.interval_counted_ids
                            and counting_active
                            and not self._is_duplicate_crossing(cx, cy)):
                        self.interval_total += 1
                        self.interval_class_counts[vehicle_classes_dict[cls]] += 1
                        self.interval_counted_ids.add(track_id)
                        self.flash_timers[track_id] = self.flash_frames
                        box_half_flash = max(x2-x1, y2-y1) // 2 + 6
                        self.flash_positions[track_id] = (cx, cy, box_half_flash)
                    self.inside_roi_ids.add(track_id)

            self.prev_centers[track_id] = (cx, cy)
            
            if track_id not in self.interval_counted_ids:
                dots_to_draw.add((cx, cy))

        # Remove IDs that left ROI
        self.inside_roi_ids = self.inside_roi_ids.intersection(seen_in_roi_this_frame)
        return dots_to_draw
