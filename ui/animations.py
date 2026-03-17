"""
Animations and scheduler visual transitions.
"""
import cv2
import numpy as np
import math
import time
from datetime import datetime, timezone, timedelta

# India Standard Time
IST = timezone(timedelta(hours=5, minutes=30))

def ist_now_str():
    return datetime.now(IST).strftime("%H:%M:%S")

def draw_globe(frame, cx, cy, radius, angle, progress):
    """Draw a rotating wireframe globe on the frame."""
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius + 20, (20, 40, 60), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    cv2.circle(frame, (cx, cy), radius, (100, 200, 255), 2, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (radius, int(radius * 0.15)),
                0, 0, 360, (80, 160, 220), 1, cv2.LINE_AA)

    for lat_frac in [-0.6, -0.3, 0.3, 0.6]:
        y_off = int(radius * lat_frac)
        lat_radius = int(radius * math.cos(math.asin(abs(lat_frac))))
        cv2.ellipse(frame, (cx, cy + y_off), (lat_radius, int(lat_radius * 0.12)),
                    0, 0, 360, (60, 120, 180), 1, cv2.LINE_AA)

    num_meridians = 6
    for i in range(num_meridians):
        meridian_angle = angle + (i * math.pi / num_meridians)
        width_factor = abs(math.sin(meridian_angle))
        if width_factor < 0.05:
            width_factor = 0.05
        ellipse_w = int(radius * width_factor)
        cv2.ellipse(frame, (cx, cy), (ellipse_w, radius),
                    0, 0, 360, (60, 140, 200), 1, cv2.LINE_AA)

    num_dots = 8
    for i in range(num_dots):
        dot_angle = angle * 0.7 + (i * 2 * math.pi / num_dots)
        lat = math.sin(i * 1.3 + 0.5) * 0.6
        dx = int(radius * 0.85 * math.cos(dot_angle) * math.cos(math.asin(abs(lat))))
        dy = int(radius * 0.85 * lat)
        if math.cos(dot_angle) > -0.2:
            pulse = abs(math.sin(time.time() * 3 + i))
            brightness = int(150 + 105 * pulse)
            cv2.circle(frame, (cx + dx, cy + dy), 3, (0, brightness, brightness), -1, cv2.LINE_AA)

    if progress > 0:
        end_angle = int(360 * progress)
        cv2.ellipse(frame, (cx, cy), (radius + 8, radius + 8),
                    -90, 0, end_angle, (0, 255, 200), 3, cv2.LINE_AA)

def show_globe_transition(stream_name, duration_secs, status_text, window_w, window_h, window_name, gui_enabled, web_server):
    """Show globe animation for `duration_secs` in the OpenCV window."""
    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed >= duration_secs:
            break

        frame = np.zeros((window_h, window_w, 3), dtype=np.uint8)

        for y_row in range(window_h):
            val = int(15 + 10 * (y_row / window_h))
            frame[y_row, :] = (val, val + 5, val + 10)

        angle = elapsed * 0.8
        progress = elapsed / duration_secs
        globe_cx, globe_cy = window_w // 2, window_h // 2 - 40
        draw_globe(frame, globe_cx, globe_cy, 120, angle, progress)

        text_size = cv2.getTextSize(stream_name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        tx = (window_w - text_size[0]) // 2
        cv2.putText(frame, stream_name, (tx, globe_cy + 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        dots = "." * (int(elapsed * 2) % 4)
        status = f"{status_text}{dots}"
        st_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        sx = (window_w - st_size[0]) // 2
        cv2.putText(frame, status, (sx, globe_cy + 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 255), 1, cv2.LINE_AA)

        remaining = int(duration_secs - elapsed) + 1
        timer_text = f"{remaining}s"
        cv2.putText(frame, timer_text, (window_w // 2 - 15, globe_cy + 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 150, 200), 2, cv2.LINE_AA)

        ist_str = f"IST {ist_now_str()}"
        cv2.putText(frame, ist_str, (window_w - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 140, 180), 1, cv2.LINE_AA)

        cv2.putText(frame, "CCTV STREAM SCHEDULER", (window_w // 2 - 160, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 180, 255), 2, cv2.LINE_AA)

        web_server.update_frame(frame)
        if not gui_enabled:
            key = cv2.waitKey(30) & 0xFF
        else:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            return True

    return False

def show_results_screen(stream_name, result, duration_secs, window_w, window_h, window_name, gui_enabled, web_server):
    """Show counting results for a few seconds before moving to next stream."""
    start = time.time()

    while time.time() - start < duration_secs:
        frame = np.zeros((window_h, window_w, 3), dtype=np.uint8)

        for y_row in range(window_h):
            val = int(15 + 10 * (y_row / window_h))
            frame[y_row, :] = (val, val + 5, val + 10)

        cx = window_w // 2

        cv2.putText(frame, "COUNTING COMPLETE", (cx - 150, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2, cv2.LINE_AA)

        name_size = cv2.getTextSize(stream_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(frame, stream_name, (cx - name_size[0]//2, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        total_text = str(result.get("total", 0))
        total_size = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 4)[0]
        cv2.putText(frame, total_text, (cx - total_size[0]//2, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 255), 4, cv2.LINE_AA)

        cv2.putText(frame, "VEHICLES", (cx - 55, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 200, 255), 2, cv2.LINE_AA)

        y = 360
        for cls_name, cnt in result.get("class_counts", {}).items():
            text = f"{cls_name}: {cnt}"
            t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.putText(frame, text, (cx - t_size[0]//2, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 240), 1, cv2.LINE_AA)
            y += 35

        remaining = int(duration_secs - (time.time() - start)) + 1
        cv2.putText(frame, f"Next stream in {remaining}s...",
                    (cx - 110, window_h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 150, 200), 1, cv2.LINE_AA)

        cv2.putText(frame, f"IST {ist_now_str()}", (window_w - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 140, 180), 1, cv2.LINE_AA)

        web_server.update_frame(frame)
        if not gui_enabled:
            key = cv2.waitKey(30) & 0xFF
        else:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            return True

    return False
