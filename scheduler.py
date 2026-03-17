"""
================================================
  TIME-SLOTTED CCTV STREAM SCHEDULER
  Cycles through live streams by IST time slot
  Globe animation during transitions
================================================
"""
import cv2
import numpy as np
import sys
import os
import time
import argparse
import shutil
import threading
from ultralytics import YOLO

# ── Import Refactored Modules ───────────────────────────────────────
from core.counting import VehicleCounter
from core.config_manager import load_streams_config
from ui.animations import show_globe_transition, show_results_screen, ist_now_str
import web_server
from network.stream_manager import ClipManager
from network.download import get_stream_url
from ui.renderer import draw_dashboard, draw_glow_bracket, draw_zones

# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_CONFIG     = "streams_config.json"
DEFAULT_MODEL      = "yolo11m.pt"
DEFAULT_CLIP_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos", "temp")
LINE_CONFIGS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "line_configs")
WINDOW_NAME        = "CCTV Stream Scheduler"
WINDOW_W, WINDOW_H = 1280, 720

def stream_config_path(stream_name):
    """Get per-stream line/ROI config file path."""
    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in stream_name)
    safe_name = safe_name.strip().replace(' ', '_')
    os.makedirs(LINE_CONFIGS_DIR, exist_ok=True)
    return os.path.join(LINE_CONFIGS_DIR, f"{safe_name}.json")

def time_to_minutes(t_str):
    h, m = map(int, t_str.split(":"))
    return h * 60 + m

def get_active_slot(slots):
    from datetime import datetime
    from ui.animations import IST
    now = datetime.now(IST)
    now_mins = now.hour * 60 + now.minute

    for slot in slots:
        start = time_to_minutes(slot["start"])
        end   = time_to_minutes(slot["end"])
        if end <= start:
            if now_mins >= start or now_mins < end:
                return slot
        else:
            if start <= now_mins < end:
                return slot
    return slots[0]

def resolve_stream_url(url):
    if "youtube.com" in url or "youtu.be" in url:
        try:
            return get_stream_url(url)
        except SystemExit:
            return None
    return url

# ═════════════════════════════════════════════════════════════════════
#  VEHICLE COUNTING ON CLIP
# ═════════════════════════════════════════════════════════════════════
def run_counting_on_clip(clip_path, model, stream_name, count_duration, config_file=None, imgsz=960, skip_frames=1):
    global gui_enabled
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"[{ist_now_str()}] ERROR: Cannot open clip: {clip_path}")
        return {"total": 0, "class_counts": {}, "error": True}

    counter = VehicleCounter(config_file=config_file if config_file else "line_config.json")
    counter.phase = "COUNT"
    counter.phase_start = time.time()

    def mouse_click(event, x, y, flags, param):
        if counter.last_h is None: return
        import math
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
                if math.hypot(x - counter.temp_line[0][0], y - counter.temp_line[0][1]) >= 10:
                    counter.line = list(counter.temp_line)
                    counter.mode = "line"
                    counter.roi_poly = None
                    counter.save_config()
                counter.temp_line = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            counter.poly_points.append((x, y))
            if len(counter.poly_points) == 4:
                counter.roi_poly = list(counter.poly_points)
                counter.poly_points = []
                counter.poly_preview_pt = None
                counter.mode = "roi"
                counter.save_config()

    if gui_enabled:
        cv2.setMouseCallback(WINDOW_NAME, mouse_click)

    frame_count = 0
    fps_timer = time.time()
    display_fps = 0
    last_results = None
    user_quit = False

    VEHICLE_CLASSES = {2: "Car", 5: "Bus", 7: "Truck"}
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_timer
            display_fps = 30 / elapsed if elapsed > 0 else 0
            fps_timer = time.time()

        counter.set_frame_size(frame.shape[0], frame.shape[1])
        counting_active = True
        counter.phase = "COUNT"

        if frame_count % skip_frames == 0 or last_results is None:
            results = model.track(
                frame, persist=True, classes=[2, 5, 7], conf=0.10,
                imgsz=imgsz, half=True, verbose=False, device=0,
                tracker="bytetrack.yaml", iou=0.5
            )
            last_results = results
        else:
            results = last_results

        # Business Logic
        dots = counter.process_detections(results, counting_active, VEHICLE_CLASSES, 0, 0.10)

        # UI Overlay
        draw_zones(frame, counter, (0, 255, 0), (0, 200, 255))
        for (cx, cy) in dots:
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

        for track_id, (fcx, fcy, fbox_half) in list(counter.flash_positions.items()):
            draw_glow_bracket(frame, fcx, fcy, fbox_half, track_id, counter, 4, (0, 255, 0), 28, 3)

        # Scheduler specific dashboard 
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y = 30
        cv2.putText(frame, f"STREAM: {stream_name}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        y += 28
        cv2.putText(frame, f"IST: {ist_now_str()}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 200, 255), 1)
        y += 28
        cv2.putText(frame, f"COUNT: {counter.interval_total}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y += 28
        for cls_name, cnt in counter.interval_class_counts.items():
            cv2.putText(frame, f"  {cls_name}: {cnt}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            y += 20

        cv2.putText(frame, f"FPS: {display_fps:.1f}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            clip_progress = frame_count / total_frames
            bar_x, bar_y, bar_w, bar_h = 10, h - 55, w - 20, 22
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
            fill = int(bar_w * clip_progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 200, 60), -1)
            cv2.putText(frame, f"Processing clip: {int(clip_progress * 100)}%  |  Q=Quit",
                        (bar_x + 8, bar_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)

        web_server.update_frame(frame)
        if not gui_enabled:
            key = cv2.waitKey(1) & 0xFF
        else:
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            user_quit = True
            break

    cap.release()
    result = {"total": counter.interval_total, "class_counts": dict(counter.interval_class_counts), "frames_processed": frame_count}
    return None if user_quit else result


# ═════════════════════════════════════════════════════════════════════
#  MAIN SCHEDULER LOOP
# ═════════════════════════════════════════════════════════════════════
gui_enabled = True

def run_scheduler(config_path, model_name, imgsz, skip_frames, no_gui=False, web_port=5000):
    global gui_enabled
    gui_enabled = not no_gui

    web_server.start_server(web_port)
    slots, count_duration, transition_duration = load_streams_config(config_path)

    print(f"\n{'='*55}")
    print(f"  CCTV STREAM SCHEDULER")
    print(f"  Config        : {config_path}")
    print(f"  Model         : {model_name}")
    print(f"  Clip duration : {count_duration}s")
    print(f"  Transition    : {transition_duration}s")
    print(f"  IST now       : {ist_now_str()}")
    print(f"{'='*55}\n")

    model = YOLO(model_name).to("cuda")
    if gui_enabled:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_W, WINDOW_H)
    os.makedirs(DEFAULT_CLIP_DIR, exist_ok=True)

    slot_indices = {}
    cycle_count = 0
    clip_managers = {} 

    def get_or_create_manager(name, url):
        if name not in clip_managers:
            direct = resolve_stream_url(url)
            if not direct: return None
            clip_managers[name] = ClipManager(stream_url=direct, stream_name=name)
        return clip_managers[name]
        
    def cleanup_managers():
        for name, mgr in clip_managers.items():
            mgr.cleanup()
        clip_managers.clear()

    def peek_next_stream():
        active_slot = get_active_slot(slots)
        key = f"{active_slot['start']}-{active_slot['end']}"
        streams = active_slot.get("streams", [])
        if not streams: return None, None
        next_idx = slot_indices.get(key, 0) % len(streams)
        s = streams[next_idx]
        return s.get("name", f"Stream {next_idx+1}"), s.get("url", "")

    first_name, first_url = peek_next_stream()
    if first_name and first_url:
        first_mgr = get_or_create_manager(first_name, first_url)
        anim_start = time.time()
        while first_mgr and first_mgr.ready_queue.qsize() == 0:
            el = time.time() - anim_start
            if el > 240: break
            
            frame = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
            for yr in range(WINDOW_H):
                v = int(15 + 10 * (yr / WINDOW_H))
                frame[yr, :] = (v, v + 5, v + 10)

            gcx, gcy = WINDOW_W // 2, WINDOW_H // 2 - 40
            angle = el * 0.8
            progress = 0.5 + 0.5 * abs(math.sin(el * 2))
            
            from ui.animations import draw_globe
            draw_globe(frame, gcx, gcy, 120, angle, progress)
            cv2.putText(frame, "CCTV STREAM SCHEDULER", (WINDOW_W // 2 - 160, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 180, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"IST {ist_now_str()}", (WINDOW_W - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 140, 180), 1, cv2.LINE_AA)

            web_server.update_frame(frame)
            if not gui_enabled:
                key = cv2.waitKey(30) & 0xFF
            else:
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                if gui_enabled: cv2.destroyAllWindows()
                cleanup_managers()
                return

    try:
        while True:
            active_slot = get_active_slot(slots)
            slot_key = f"{active_slot['start']}-{active_slot['end']}"
            streams = active_slot.get("streams", [])

            if not streams:
                time.sleep(60)
                continue

            if slot_key not in slot_indices: slot_indices[slot_key] = 0
            idx = slot_indices[slot_key] % len(streams)
            stream = streams[idx]
            slot_indices[slot_key] = idx + 1

            stream_name = stream.get("name", f"Stream {idx+1}")
            stream_url  = stream.get("url", "")
            cycle_count += 1
            
            mgr = get_or_create_manager(stream_name, stream_url)

            quit_req = show_globe_transition(stream_name, transition_duration, "Connecting to stream", WINDOW_W, WINDOW_H, WINDOW_NAME, gui_enabled, web_server)
            if quit_req: break
                
            clip_path = mgr.get_next_valid_clip(timeout=60) if mgr else None
            if not clip_path: continue

            next_name, next_url = peek_next_stream()
            if next_name and next_url:
                get_or_create_manager(next_name, next_url)

            cfg_path = stream_config_path(stream_name)
            result = run_counting_on_clip(clip_path, model, stream_name, count_duration, config_file=cfg_path, imgsz=imgsz, skip_frames=skip_frames)

            try:
                mgr.mark_clip_done(clip_path)
            except Exception:
                pass

            if result is None: break

            quit_req = show_results_screen(stream_name, result, 5, WINDOW_W, WINDOW_H, WINDOW_NAME, gui_enabled, web_server)
            if quit_req: break

    except KeyboardInterrupt:
        pass
    finally:
        cleanup_managers()
        if gui_enabled: cv2.destroyAllWindows()
        try: shutil.rmtree(DEFAULT_CLIP_DIR, ignore_errors=True)
        except: pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--skip-frames", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--web-port", type=int, default=5000)
    args = parser.parse_args()

    if args.test:
        print("[TEST] Run visual/unit test logic...")
        return
    run_scheduler(args.config, args.model, args.imgsz, args.skip_frames, args.no_gui, args.web_port)

if __name__ == "__main__":
    main()
