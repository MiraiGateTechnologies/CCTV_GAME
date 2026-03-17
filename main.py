"""
================================================
  VEHICLE COUNTER - YouTube/HTTP Live Stream
  Uses YOLO11 + OpenCV for real-time counting
================================================

SETUP:
  pip install ultralytics opencv-python yt-dlp numpy supervision

CONTROLS:
  Left-drag        = Draw counting line (LINE mode)
  Right-click x4   = Draw ROI polygon  (switches to ROI mode)
  L key            = Reset to LINE mode (clear ROI)
  R key            = Reset counts + timer
  Z key            = Cancel in-progress polygon
  S key            = Screenshot
  Q key            = Quit
"""

import cv2
import argparse
import sys
import time
import os
import math
from ultralytics import YOLO

# ── Import Refactored Modules ───────────────────────────────────────
from core.counting import VehicleCounter
from ui.renderer import draw_dashboard, draw_glow_bracket, draw_zones
from network.download import get_stream_url
import web_server

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "model":             "yolov8l.pt",
    "confidence":        0.10,          # Low = detect more (incl. far vehicles)
    "min_box_area":      0,            # Far vehicles can be ~7x8=56px area
    "imgsz":             1600,          # KEY: higher res = far vehicles detected!
    "line_color":        (0, 255, 0),
    "roi_color":         (0, 200, 255),
    "bracket_color":     (0, 255, 0),
    "bracket_size":      28,
    "bracket_thickness": 3,
    "flash_frames":      4,
    "count_interval":    35,
    "wait_interval":     15,
    "playback_speed":    1.0,
}

VEHICLE_CLASSES = {
    2: "Car",
    5: "Bus",
    7: "Truck",
}

# ─────────────────────────────────────────────
#  MAIN PROCESS LOOP (Main Logic)
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",        type=str)
    parser.add_argument("--video",      type=str)
    parser.add_argument("--model",      type=str, default=CONFIG["model"])
    parser.add_argument("--conf",       type=float, default=CONFIG["confidence"])
    parser.add_argument("--imgsz",      type=int,   default=CONFIG["imgsz"])
    parser.add_argument("--result-out", type=str, default=None)
    parser.add_argument("--skip-frames", type=int, default=1)
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--web-port", type=int, default=5000)
    args = parser.parse_args()
    args.no_gui = False

    VIDEOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")

    if hasattr(args, 'list') and getattr(args, 'list', False):
        files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(".mp4")] if os.path.exists(VIDEOS_DIR) else []
        if files:
            print("\nAvailable videos in videos/ folder:")
            for f in sorted(files):
                print(f"  python main.py --video {f[:-4]}")
        else:
            print("No videos found. Download one with: python network/download.py --url URL --name video1")
        sys.exit(0)

    if not args.url and not args.video:
        print("[ERROR] --url or --video required!")
        sys.exit(1)

    if args.video:
        v = args.video
        if not os.path.exists(v):
            candidate = os.path.join(VIDEOS_DIR, v if v.endswith(".mp4") else v + ".mp4")
            if os.path.exists(candidate):
                stream_source = candidate
                print(f"[INFO] Using: videos/{os.path.basename(candidate)}")
            else:
                print(f"[ERROR] Video '{v}' not found!")
                sys.exit(1)
        else:
            stream_source = v
    elif "youtube.com" in args.url or "youtu.be" in args.url:
        stream_source = get_stream_url(args.url)
    else:
        stream_source = args.url

    print(f"[INFO] YOLO model loading: {args.model}")
    model = YOLO(args.model, task="detect").to("cuda")
    print("[OK] Model ready!")

    cap = cv2.VideoCapture(stream_source)
    if not cap.isOpened():
        print("[ERROR] Stream nahi khula!")
        sys.exit(1)
    print("[OK] Stream connected!")

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    yolo_imgsz = (round(vid_h / 32) * 32, round(vid_w / 32) * 32)
    print(f"[INFO] Video: {vid_w}x{vid_h} @ {vid_fps:.1f}fps")

    counter = VehicleCounter(
        flash_frames=CONFIG["flash_frames"],
        count_interval=CONFIG["count_interval"],
        wait_interval=CONFIG["wait_interval"]
    )
    
    cycles_completed = 0
    window_name = "Vehicle Counter"

    web_server.start_server(port=args.web_port)

    if not args.no_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    # Mouse callback wraps counter drawing state directly
    def mouse_click(event, x, y, flags, param):
        if counter.last_h is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            counter.drawing_line = True
            counter.temp_line = [(x, y), (x, y)]
        elif event == cv2.EVENT_MOUSEMOVE:
            if counter.drawing_line:
                counter.temp_line[1] = (x, y)
            if counter.poly_points:
                counter.poly_preview_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if counter.drawing_line:
                counter.drawing_line = False
                counter.temp_line[1] = (x, y)
                line_len = math.hypot(x - counter.temp_line[0][0], y - counter.temp_line[0][1])
                if line_len >= 10:
                    counter.line = list(counter.temp_line)
                    counter.mode = "line"
                    counter.roi_poly = None
                    counter.poly_points = []
                    counter.poly_preview_pt = None
                    counter.interval_total = 0
                    counter.interval_class_counts.clear()
                    counter.interval_counted_ids.clear()
                    counter.inside_roi_ids.clear()
                    counter.phase = "COUNT"
                    counter.phase_start = time.time()
                    counter.save_config()
                    print("[INFO] LINE mode activated!")
                counter.temp_line = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            counter.poly_points.append((x, y))
            if len(counter.poly_points) == 4:
                counter.roi_poly = list(counter.poly_points)
                counter.poly_points = []
                counter.poly_preview_pt = None
                counter.mode = "roi"
                counter.interval_total = 0
                counter.interval_class_counts.clear()
                counter.interval_counted_ids.clear()
                counter.inside_roi_ids.clear()
                counter.phase = "COUNT"
                counter.phase_start = time.time()
                counter.save_config()
                print("[OK] ROI mode activated!")
        elif event == cv2.EVENT_MBUTTONDOWN:
            counter.poly_points = []
            counter.poly_preview_pt = None

    if not args.no_gui:
        cv2.setMouseCallback(window_name, mouse_click)

    frame_count = 0
    fps_timer = time.time()
    display_fps = 0
    screenshot_count = 0

    SPEED_STEPS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    playback_speed = CONFIG["playback_speed"]
    speed_idx = min(range(len(SPEED_STEPS)), key=lambda i: abs(SPEED_STEPS[i] - playback_speed))

    last_results = None
    is_live_stream  = bool(args.url)
    reconnect_count = 0
    MAX_RECONNECTS  = 10
    RECONNECT_DELAY = 3
    stream_start_time = time.time()
    URL_REFRESH_SECS  = 3 * 3600
    original_url = args.url

    while True:
        frame_start = time.time()
        ret, frame = cap.read()

        if not ret:
            if not is_live_stream:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            reconnect_count += 1
            if reconnect_count > MAX_RECONNECTS:
                break
            time.sleep(RECONNECT_DELAY)
            cap.release()
            if original_url and ("youtube.com" in original_url or "youtu.be" in original_url):
                try:
                    stream_source = get_stream_url(original_url)
                    stream_start_time = time.time()
                except SystemExit:
                    continue
            cap = cv2.VideoCapture(stream_source)
            if cap.isOpened():
                reconnect_count = 0
            continue

        reconnect_count = 0
        if is_live_stream and original_url and ("youtube.com" in original_url or "youtu.be" in original_url):
            if time.time() - stream_start_time > URL_REFRESH_SECS:
                try:
                    new_url = get_stream_url(original_url)
                    cap.release()
                    cap = cv2.VideoCapture(new_url)
                    stream_source = new_url
                    stream_start_time = time.time()
                except SystemExit:
                    pass

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_timer
            display_fps = 30 / elapsed if elapsed > 0 else 0
            fps_timer = time.time()

        counter.set_frame_size(frame.shape[0], frame.shape[1])

        prev_phase = counter.phase
        counting_active = counter.update_phase()
        if prev_phase == "COUNT" and counter.phase == "WAIT":
             cycles_completed += 1

        if frame_count % args.skip_frames == 0 or last_results is None:
            results = model.track(
                frame, persist=True, classes=list(VEHICLE_CLASSES.keys()),
                conf=args.conf, imgsz=yolo_imgsz, half=True, verbose=False,
                device=0, tracker="bytetrack.yaml", iou=0.5,
            )
            last_results = results
        else:
            results = last_results

        # ── CORE BUSINESS LOGIC ──
        # Process detections returns positions of valid tracked dots
        dots = counter.process_detections(results, counting_active, VEHICLE_CLASSES, CONFIG["min_box_area"], args.conf)

        # ── UI RENDERING ──
        draw_zones(frame, counter, CONFIG["line_color"], CONFIG["roi_color"])
        for (cx, cy) in dots:
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

        for track_id, (fcx, fcy, fbox_half) in list(counter.flash_positions.items()):
            draw_glow_bracket(frame, fcx, fcy, fbox_half, track_id, counter, CONFIG["flash_frames"], CONFIG["bracket_color"], CONFIG["bracket_size"], CONFIG["bracket_thickness"])

        elapsed = time.time() - counter.phase_start
        frame = draw_dashboard(frame, counter, counting_active, elapsed, CONFIG["count_interval"], CONFIG["wait_interval"])

        cv2.putText(frame, f"FPS: {display_fps:.1f}", (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if not is_live_stream:
            speed_label = f"Speed: {playback_speed:.2f}x  [+/-]"
            cv2.putText(frame, speed_label, (frame.shape[1] - 200, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        web_server.update_frame(frame)

        if not args.no_gui:
            cv2.imshow(window_name, frame)
            if not is_live_stream and vid_fps > 0:
                target_ms = int(1000.0 / (vid_fps * playback_speed))
                elapsed_ms = int((time.time() - frame_start) * 1000)
                wait_ms = max(1, target_ms - elapsed_ms)
            else:
                wait_ms = 1
            key = cv2.waitKey(wait_ms) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF
            time.sleep(0.001)

        if key == ord('q'):
            break
        elif key == ord('r'):
            counter.interval_total = 0
            counter.interval_class_counts.clear()
            counter.interval_counted_ids.clear()
            counter.inside_roi_ids.clear()
            counter.flash_timers.clear()
            counter.flash_positions.clear()
            counter.phase = "COUNT"
            counter.phase_start = time.time()
        elif key == ord('s'):
            fname = f"screenshot_{screenshot_count}.jpg"
            cv2.imwrite(fname, frame)
            screenshot_count += 1
        elif key == ord('z'):
            counter.poly_points = []
            counter.poly_preview_pt = None
        elif key == ord('+') or key == ord('='):
            speed_idx = min(len(SPEED_STEPS) - 1, speed_idx + 1)
            playback_speed = SPEED_STEPS[speed_idx]
        elif key == ord('-'):
            speed_idx = max(0, speed_idx - 1)
            playback_speed = SPEED_STEPS[speed_idx]
        elif key == ord('l'):
            counter.mode = "line"
            counter.roi_poly = None
            counter.interval_total = 0
            counter.interval_class_counts.clear()
            counter.interval_counted_ids.clear()
            counter.inside_roi_ids.clear()
            counter.phase = "COUNT"
            counter.phase_start = time.time()
            counter.save_config()

    cap.release()
    cv2.destroyAllWindows()

    if args.result_out:
        import json
        out = {
            "total": counter.interval_total,
            "class_counts": dict(counter.interval_class_counts),
            "cycles": cycles_completed,
        }
        with open(args.result_out, "w") as f:
            json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()