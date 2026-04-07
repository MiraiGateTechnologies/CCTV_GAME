"""
UI rendering functions for the CCTV Vehicle Counter.
Extracts drawing logic (dashboard, glow brackets, lines) away from business logic.

Includes:
  - Shared tracking overlay (draw_tracking_overlay) — single source of truth
    for BOTH debug mode and playback mode (dots, brackets, line/ROI)
  - Original real-time rendering (draw_dashboard, draw_glow_bracket, draw_zones)
  - Playback overlay rendering (draw_playback_overlay) for pre-processed clips
"""
import cv2
import time
import math
import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  SHARED TRACKING OVERLAY — Single source of truth for debug + playback
# ═══════════════════════════════════════════════════════════════════

def draw_tracking_overlay(frame, dots, brackets, line_config):
    """
    Single source of truth for ALL tracking visuals.
    Used by BOTH debug mode (render_debug_frame in counting.py) and
    playback mode (draw_playback_overlay below) to guarantee
    pixel-identical rendering.

    Any change here (dot size, color, bracket animation, line style)
    automatically applies to both modes — no dual maintenance.

    Args:
        frame:       OpenCV frame (modified in-place)
        dots:        list of (cx, cy) — white tracking dots (uncounted vehicles only)
        brackets:    list of (cx, cy, half_size, progress)
                     progress: 0.0→1.0 bracket animation progress
        line_config: dict with 'mode', 'line', 'roi_poly' — counting zone config
                     Accepts both list and tuple coordinate formats.
    """
    # ── 1. Draw counting line / ROI zone ──
    if line_config:
        mode = line_config.get("mode", "line")
        line = line_config.get("line")
        roi = line_config.get("roi_poly")

        if mode == "line" and line and len(line) == 2:
            pt1 = (int(line[0][0]), int(line[0][1]))
            pt2 = (int(line[1][0]), int(line[1][1]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "COUNTING LINE", (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        if mode == "roi" and roi and len(roi) >= 3:
            pts = np.array([(int(p[0]), int(p[1])) for p in roi], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 200, 255),
                          thickness=2, lineType=cv2.LINE_AA)
            for pt in roi:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 200, 255), -1)

    # ── 2. White tracking dots (uncounted vehicles only) ──
    for (cx, cy) in dots:
        cv2.circle(frame, (int(cx), int(cy)), 4, (255, 255, 255), -1)

    # ── 3. Green flash brackets + "+1" popup + "10!" checkpoint ──
    for item in brackets:
        fcx, fcy, fhalf, progress = int(item[0]), int(item[1]), int(item[2]), item[3]
        count_at = int(item[4]) if len(item) > 4 else 0
        alpha = abs(math.sin(math.pi * progress))
        if alpha < 0.05:
            continue  # Skip nearly-invisible frame (prevents black bracket flash)
        color = (0, int(255 * alpha), 0)
        size = 20
        th = 2
        bx1, by1 = fcx - fhalf, fcy - fhalf
        bx2, by2 = fcx + fhalf, fcy + fhalf

        # Animated green fill
        hw = int(fhalf * progress)
        hh = int(fhalf * progress)
        overlay_b = frame.copy()
        cv2.rectangle(overlay_b, (fcx - hw, fcy - hh),
                      (fcx + hw, fcy + hh), (0, 180, 0), -1)
        fill_alpha = 0.25 * alpha
        cv2.addWeighted(overlay_b, fill_alpha, frame, 1 - fill_alpha, 0, frame)

        # Corner brackets
        cv2.line(frame, (bx1, by1), (bx1 + size, by1), color, th)
        cv2.line(frame, (bx1, by1), (bx1, by1 + size), color, th)
        cv2.line(frame, (bx2, by1), (bx2 - size, by1), color, th)
        cv2.line(frame, (bx2, by1), (bx2, by1 + size), color, th)
        cv2.line(frame, (bx1, by2), (bx1 + size, by2), color, th)
        cv2.line(frame, (bx1, by2), (bx1, by2 - size), color, th)
        cv2.line(frame, (bx2, by2), (bx2 - size, by2), color, th)
        cv2.line(frame, (bx2, by2), (bx2, by2 - size), color, th)

        # ── "+1" popup / "10!" checkpoint — above the bracket ──
        if count_at > 0 and alpha > 0.2:
            # Float up effect: text moves up as animation progresses
            float_offset = int(15 * progress)
            popup_y = by1 - 10 - float_offset

            if count_at % 10 == 0:
                # ── CHECKPOINT: "10!" / "20!" / "30!" — golden, bigger ──
                label = f"{count_at}!"
                font_scale = 0.8
                font_th = 2
                text_color = (0, int(255 * alpha), int(255 * alpha))   # yellow-green
                bg_color = (0, int(100 * alpha), int(100 * alpha))     # dark yellow-green
            else:
                # ── Normal: "+1" — green, smaller ──
                label = "+1"
                font_scale = 0.55
                font_th = 2
                text_color = (0, int(255 * alpha), 0)    # green
                bg_color = None

            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_th)[0]
            text_x = fcx - t_size[0] // 2
            text_y = max(20, popup_y)

            # Background pill for checkpoint
            if bg_color is not None:
                pad = 5
                overlay_t = frame.copy()
                cv2.rectangle(overlay_t,
                              (text_x - pad, text_y - t_size[1] - pad),
                              (text_x + t_size[0] + pad, text_y + pad),
                              bg_color, -1)
                cv2.addWeighted(overlay_t, 0.5 * alpha, frame, 1 - 0.5 * alpha, 0, frame)

            cv2.putText(frame, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                        font_th, cv2.LINE_AA)


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

    # ── Build dots from stored data — exact replay of debug mode ──
    # show_dot flag was computed during offline_count using same conf >= 0.20 filter.
    # Fallback: if show_dot not present (old data), use counted=False check.
    dots = []
    for det in detections:
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        tid = det.get("track_id", 0)

        # Use show_dot flag (matches debug mode's conf >= 0.20 filter)
        if det.get("show_dot", not det.get("counted", False)):
            dots.append((cx, cy))

        # If vehicle just crossed counting line → start green flash bracket
        if det.get("crossed"):
            if tid not in _playback_flash_timers or _playback_flash_timers.get(tid, 0) <= 0:
                flash_cx = det.get("flash_cx", cx)
                flash_cy = det.get("flash_cy", cy)
                flash_half = det.get("flash_half", max(x2 - x1, y2 - y1) // 2 + 6)
                count_at = det.get("count_at", 0)
                _playback_flash_timers[tid] = PLAYBACK_FLASH_FRAMES
                _playback_flash_positions[tid] = (flash_cx, flash_cy, flash_half, count_at)

    # ── Build brackets list from flash state ──
    brackets = []
    for tid in list(_playback_flash_timers.keys()):
        t = _playback_flash_timers[tid]
        if t <= 0:
            _playback_flash_timers.pop(tid, None)
            _playback_flash_positions.pop(tid, None)
            continue
        if tid in _playback_flash_positions:
            flash_data = _playback_flash_positions[tid]
            fcx, fcy, fhalf = flash_data[0], flash_data[1], flash_data[2]
            count_at = flash_data[3] if len(flash_data) > 3 else 0
            progress = (PLAYBACK_FLASH_FRAMES - t) / PLAYBACK_FLASH_FRAMES
            brackets.append((fcx, fcy, fhalf, progress, count_at))
        _playback_flash_timers[tid] = t - 1

    # ── Draw shared tracking overlay (identical visuals as debug mode) ──
    draw_tracking_overlay(frame, dots, brackets, line_config)

    # ── Dashboard overlay (game data from game_api) ──
    from game.game_api import get_overlay_data
    gd = get_overlay_data()
    radius = 20
    thickness = 3
    margin = 15
    center = (w - margin - radius, margin + radius) # top-right corner
    if counting_active:
        remaining = max(0, count_duration - elapsed_secs)
        ratio = remaining / count_duration if count_duration > 0 else 0
        color = (0, 255, 100)  # green
    else:
        total_elapsed = elapsed_secs - count_duration
        remaining = max(0, wait_duration - total_elapsed)
        ratio = remaining / wait_duration if wait_duration > 0 else 0
        color = (80, 80, 255)  # blue/red

    # Background circle (dark)
    cv2.circle(frame, center, radius, (40, 40, 40), -1)

    # Progress arc (fills clockwise from top)
    if ratio > 0:
        angle = int(360 * ratio)
        cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, color, thickness, cv2.LINE_AA)

    # Count text in center of circle
    count_text = str(current_count)
    t_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.putText(frame, count_text,
                (center[0] - t_size[0] // 2, center[1] + t_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def reset_playback_flash():
    """Reset flash animation state between clips."""
    global _playback_flash_timers, _playback_flash_positions
    _playback_flash_timers = {}
    _playback_flash_positions = {}