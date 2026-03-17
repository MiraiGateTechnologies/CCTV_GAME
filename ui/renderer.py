"""
UI rendering functions for the CCTV Vehicle Counter.
Extracts drawing logic (dashboard, glow brackets, lines) away from business logic.
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
