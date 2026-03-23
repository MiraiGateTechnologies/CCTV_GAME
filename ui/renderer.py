"""
UI rendering functions for the CCTV Vehicle Counter.
Extracts drawing logic (dashboard, glow brackets, lines) away from business logic.

Includes:
  - Original real-time rendering (draw_dashboard, draw_glow_bracket, draw_zones)
  - Playback overlay rendering (draw_playback_overlay) for pre-processed clips
"""
import cv2
import time
import math
import numpy as np

def draw_dashboard(frame, counter, counting_active, elapsed, count_interval, wait_interval):
    """Draw the data dashboard overlay."""
    h, w = frame.shape[:2]

    # Semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (310, 230), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 35
    mode_label = "LINE MODE" if counter.mode == "line" else "ROI MODE"
    cv2.putText(frame, f"VEHICLE COUNTER  [{mode_label}]", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y += 32
    cv2.putText(frame, f"COUNT: {counter.interval_total}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    y += 30
    for cls_name, cnt in counter.interval_class_counts.items():
        cv2.putText(frame, f"  {cls_name}: {cnt}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        y += 20

    # ── Timer Bar ──────────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 10, h - 55, w - 20, 22
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (40, 40, 40), -1)

    if counting_active:
        remaining = count_interval - elapsed
        ratio = max(0, remaining / count_interval)
        fill = int(bar_w * ratio)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                      (0, 200, 60), -1)
        label = f"COUNTING: {int(remaining)+1}s left"
        label_color = (0, 255, 100)
    else:
        remaining = wait_interval - elapsed
        ratio = max(0, remaining / wait_interval)
        fill = int(bar_w * ratio)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h),
                      (0, 60, 200), -1)
        label = f"WAITING: {int(remaining)+1}s  (next cycle soon...)"
        label_color = (80, 80, 255)

    cv2.putText(frame, label, (bar_x + 8, bar_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2)

    # Controls hint
    cv2.putText(frame,
                "LDrag=Line | RClick x4=ROI | L=LineMode | R=Reset | +/-=Speed | Q=Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
    return frame


def draw_glow_bracket(frame, cx, cy, box_half, track_id, counter, flash_frames, bracket_color, bracket_size, bracket_thickness):
    """Draw animate glowing brackets around counted vehicles."""
    t = counter.flash_timers.get(track_id, 0)
    if t <= 0:
        return
    
    total = flash_frames
    progress = (total - t) / total
    alpha = abs(math.sin(math.pi * progress))
    bc = bracket_color
    color = (int(bc[2]*alpha), int(bc[1]*alpha), int(bc[0]*alpha))  # BGR
    size = bracket_size
    th = bracket_thickness
    x1, y1 = cx - box_half, cy - box_half
    x2, y2 = cx + box_half, cy + box_half

    # ── Animated green fill ──
    fill_ratio = progress
    half_w = int(box_half * fill_ratio)
    half_h = int(box_half * fill_ratio)
    fx1, fy1 = cx - half_w, cy - half_h
    fx2, fy2 = cx + half_w, cy + half_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (fx1, fy1), (fx2, fy2), (0, 180, 0), -1)
    fill_alpha = 0.25 * alpha
    cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)

    # ── Corner brackets ──
    cv2.line(frame, (x1, y1), (x1+size, y1), color, th)
    cv2.line(frame, (x1, y1), (x1, y1+size), color, th)
    cv2.line(frame, (x2, y1), (x2-size, y1), color, th)
    cv2.line(frame, (x2, y1), (x2, y1+size), color, th)
    cv2.line(frame, (x1, y2), (x1+size, y2), color, th)
    cv2.line(frame, (x1, y2), (x1, y2-size), color, th)
    cv2.line(frame, (x2, y2), (x2-size, y2), color, th)
    cv2.line(frame, (x2, y2), (x2, y2-size), color, th)
    
    counter.flash_timers[track_id] = t - 1
    if t - 1 <= 0:
        counter.flash_positions.pop(track_id, None)

def draw_zones(frame, counter, line_color, roi_color):
    """Draw counting line and ROI polygons on screen."""
    h, w = frame.shape[:2]
    current_line = counter.temp_line if counter.temp_line else counter.line

    # ── Draw ROI polygon ──
    if counter.mode == "roi" and counter.roi_poly and len(counter.roi_poly) >= 3:
        pts = np.array(counter.roi_poly, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True,
                        color=roi_color, thickness=2, lineType=cv2.LINE_AA)
        cx_l = min(p[0] for p in counter.roi_poly) + 5
        cy_l = min(p[1] for p in counter.roi_poly) + 20
        cv2.putText(frame, "ROI ZONE", (cx_l, cy_l),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, roi_color, 2)
        for i, pt in enumerate(counter.roi_poly):
            cv2.circle(frame, pt, 6, roi_color, -1)
            cv2.putText(frame, str(i+1), (pt[0]+8, pt[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, roi_color, 1)

    # ── Draw counting line ──
    if current_line and len(current_line) == 2 and (counter.mode == "line" or counter.drawing_line):
        pt1, pt2 = current_line
        cv2.line(frame, pt1, pt2, line_color, 3)
        mid = ((pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2)
        cv2.putText(frame, "COUNTING LINE", (pt1[0], pt1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

    # ── In-progress polygon preview ──
    if counter.poly_points:
        for pt in counter.poly_points:
            cv2.circle(frame, pt, 6, (0, 100, 255), -1)
        for i in range(1, len(counter.poly_points)):
            cv2.line(frame, counter.poly_points[i-1], counter.poly_points[i], (0, 100, 255), 2)
        if counter.poly_preview_pt:
            cv2.line(frame, counter.poly_points[-1], counter.poly_preview_pt, (0, 100, 255), 1)
        n = len(counter.poly_points)
        cv2.putText(frame, f"ROI: {n}/4 corners (Right-click {4-n} more | Z=cancel)",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)


# ═══════════════════════════════════════════════════════════════════
#  PLAYBACK OVERLAY — For pre-processed clips (no YOLO during play)
# ═══════════════════════════════════════════════════════════════════

# Flash animation state for playback mode
_playback_flash_timers = {}  # track_id -> remaining frames
_playback_flash_positions = {}  # track_id -> (cx, cy, half_size)
PLAYBACK_FLASH_FRAMES = 4  # match old draw_glow_bracket flash_frames=4


def draw_playback_overlay(frame, frame_no: int, current_count: int,
                          counting_active: bool, elapsed_secs: float,
                          detections: list, stream_name: str = "",
                          count_duration: float = 35.0,
                          wait_duration: float = 6.0,
                          line_config: dict = None):
    """
    Draw all overlays on a playback frame using stored detection data.
    This replaces real-time YOLO rendering during pre-processed playback.

    Args:
        frame:           OpenCV frame to draw on
        frame_no:        current frame number
        current_count:   vehicle count up to this frame
        counting_active: True during counting phase, False during waiting
        elapsed_secs:    seconds since clip started
        detections:      list of detection dicts for this frame
                         [{track_id, bbox, crossed?, flash_cx, flash_cy, flash_half}]
        stream_name:     stream display name
        count_duration:  counting phase length (35s)
        wait_duration:   waiting phase length (6s)
        line_config:     {"line": [[x1,y1],[x2,y2]], "mode": "line"} or None
    """
    global _playback_flash_timers, _playback_flash_positions

    h, w = frame.shape[:2]

    # ── Draw counting line from config ──
    if line_config:
        line = line_config.get("line")
        mode = line_config.get("mode", "line")
        roi = line_config.get("roi_poly")

        if mode == "line" and line and len(line) == 2:
            pt1 = tuple(int(x) for x in line[0])
            pt2 = tuple(int(x) for x in line[1])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 3)

        if mode == "roi" and roi and len(roi) >= 3:
            pts = np.array(roi, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 200, 255),
                          thickness=2, lineType=cv2.LINE_AA)

    # ── White tracking dots on vehicles (shows AI is tracking) ──
    for det in detections:
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

        # If vehicle just crossed counting line → start green flash bracket
        if det.get("crossed"):
            tid = det.get("track_id", 0)
            flash_cx = det.get("flash_cx", cx)
            flash_cy = det.get("flash_cy", cy)
            flash_half = det.get("flash_half", max(x2 - x1, y2 - y1) // 2 + 6)
            _playback_flash_timers[tid] = PLAYBACK_FLASH_FRAMES
            _playback_flash_positions[tid] = (flash_cx, flash_cy, flash_half)

    # ── Green blink brackets when vehicle crosses line (AI counting visual) ──
    for tid in list(_playback_flash_timers.keys()):
        t = _playback_flash_timers[tid]
        if t <= 0:
            _playback_flash_timers.pop(tid, None)
            _playback_flash_positions.pop(tid, None)
            continue

        fcx, fcy, fhalf = _playback_flash_positions[tid]
        progress = (PLAYBACK_FLASH_FRAMES - t) / PLAYBACK_FLASH_FRAMES
        alpha = abs(math.sin(math.pi * progress))
        color = (0, int(255 * alpha), 0)
        size = 28
        th = 3
        bx1, by1 = fcx - fhalf, fcy - fhalf
        bx2, by2 = fcx + fhalf, fcy + fhalf

        # Animated green fill
        hw = int(fhalf * progress)
        hh = int(fhalf * progress)
        overlay_b = frame.copy()
        cv2.rectangle(overlay_b, (fcx - hw, fcy - hh), (fcx + hw, fcy + hh), (0, 180, 0), -1)
        cv2.addWeighted(overlay_b, 0.25 * alpha, frame, 1 - 0.25 * alpha, 0, frame)

        # Corner brackets
        cv2.line(frame, (bx1, by1), (bx1 + size, by1), color, th)
        cv2.line(frame, (bx1, by1), (bx1, by1 + size), color, th)
        cv2.line(frame, (bx2, by1), (bx2 - size, by1), color, th)
        cv2.line(frame, (bx2, by1), (bx2, by1 + size), color, th)
        cv2.line(frame, (bx1, by2), (bx1 + size, by2), color, th)
        cv2.line(frame, (bx1, by2), (bx1, by2 - size), color, th)
        cv2.line(frame, (bx2, by2), (bx2 - size, by2), color, th)
        cv2.line(frame, (bx2, by2), (bx2, by2 - size), color, th)

        _playback_flash_timers[tid] = t - 1

    # ── Dashboard overlay (game data from game_api) ──
    from game.game_api import get_overlay_data
    gd = get_overlay_data()

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 30
    # Round ID
    cv2.putText(frame, f"Round #{gd['round_id']}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1)
    y += 22

    # Stream name
    if stream_name:
        cv2.putText(frame, f"STREAM: {stream_name}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        y += 28

    # Vehicle count
    cv2.putText(frame, f"COUNT: {current_count}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    y += 35

    # Phase
    phase_label = "COUNTING" if counting_active else "RESULT"
    phase_color = (0, 255, 100) if counting_active else (80, 80, 255)
    cv2.putText(frame, f"Phase: {phase_label}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 1)
    y += 22

    # Odds (shown during COUNTING and WAITING)
    odds = gd.get("odds", {})
    if odds and gd["phase"] in ("COUNTING", "WAITING"):
        ut = gd.get("under_threshold", 0)
        ot = gd.get("over_threshold", 0)
        odds_text = (f"U<{ut}:{odds.get('under', 0)}x  "
                     f"R{ut}-{ot}:{odds.get('range', 0)}x  "
                     f"O>{ot}:{odds.get('over', 0)}x")
        cv2.putText(frame, odds_text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 255), 1)

    # ── Timer bar ──
    bar_x, bar_y, bar_w, bar_h = 10, h - 55, w - 20, 22
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)

    if counting_active:
        remaining = max(0, count_duration - elapsed_secs)
        ratio = remaining / count_duration if count_duration > 0 else 0
        fill = int(bar_w * ratio)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 200, 60), -1)
        label = f"COUNTING: {int(remaining) + 1}s left"
        label_color = (0, 255, 100)
    else:
        total_elapsed = elapsed_secs - count_duration
        remaining = max(0, wait_duration - total_elapsed)
        ratio = remaining / wait_duration if wait_duration > 0 else 0
        fill = int(bar_w * ratio)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 60, 200), -1)
        label = f"RESULT: {current_count} vehicles | {int(remaining) + 1}s"
        label_color = (80, 80, 255)

    cv2.putText(frame, label, (bar_x + 8, bar_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2)

    return frame


def reset_playback_flash():
    """Reset flash animation state between clips."""
    global _playback_flash_timers, _playback_flash_positions
    _playback_flash_timers.clear()
    _playback_flash_positions.clear()
