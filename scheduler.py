"""
================================================================
  CCTV CASINO GAME — Main Scheduler
  Parallel Download + Single YOLO + Provably Fair + Round Cycle
================================================================

Architecture:
  StreamDownloader (per stream, parallel) → download_queue
  YOLOWorker (single, GPU-safe) → ready_queue
  Scheduler picks from ready_queue (fastest stream wins)

Round Cycle (56 seconds):
  1. BETTING  (15 sec) — No stream, hash published, players bet
  2. COUNTING (35 sec) — Pre-recorded clip plays with detection overlay
  3. WAITING  (6 sec)  — Stream continues, count frozen, result reveal

Modes:
  python scheduler.py --config streams_config.json
      → Production: web browser only (http://localhost:5000)

  python scheduler.py --config streams_config.json --gui
      → Development: web + debug GUI showing live YOLO processing
        with mouse support for drawing counting line/ROI
"""

import cv2
import numpy as np
import sys
import os
import time
import argparse
import math
from ultralytics import YOLO

# ── Import Modules ─────────────────────────────────────────────────
from core.config_manager import load_streams_config, load_line_config, save_line_config
from ui.animations import ist_now_str, IST
from ui.renderer import draw_playback_overlay, reset_playback_flash
import web_server
from network.stream_manager import Pipeline, PreProcessedClip
from network import livekit_publisher
from game.round_manager import (
    prepare_round, play_clip_with_overlay, finalize_round,
    get_betting_phase_data, get_counting_phase_data, get_waiting_phase_data,
    BETTING_DURATION, COUNTING_DURATION, WAITING_DURATION, RoundData,
)
from game.history_tracker import HistoryTracker
import game.game_api as game_api

# ── Defaults ───────────────────────────────────────────────────────
DEFAULT_CONFIG = "streams_config.json"
DEFAULT_MODEL = "yolo11x.engine"
LINE_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "line_configs")
DEBUG_WINDOW = "DEBUG: Background YOLO Processing"
WINDOW_W, WINDOW_H = 1280, 720

# ── Global State ───────────────────────────────────────────────────
debug_gui = False
current_round_data: RoundData | None = None


def push_frame(frame):
    """Push frame to BOTH MJPEG (web_server) and LiveKit (if running).
    Single call replaces all web_server.update_frame(frame) calls."""
    web_server.update_frame(frame)
    livekit_publisher.update_frame(frame)

# ── Debug Drawing State (mouse callbacks in --gui mode) ────────────
_debug_draw = {
    "drawing_line": False,
    "temp_line": None,           # [(x,y),(x,y)] in display coords (WINDOW_W x WINDOW_H)
    "poly_points": [],           # [(x,y),...] in display coords
    "poly_preview_pt": None,     # (x,y) in display coords
    "config_path": None,
    "stream_name": "",
    "last_frame": None,          # last displayed frame (WINDOW_W x WINDOW_H)
    "original_shape": None,      # (h, w) of original frame before resize
}


def stream_config_path(stream_name: str) -> str:
    """Get per-stream line/ROI config file path."""
    safe = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in stream_name)
    safe = safe.strip().replace(' ', '_')
    os.makedirs(LINE_CONFIGS_DIR, exist_ok=True)
    return os.path.join(LINE_CONFIGS_DIR, f"{safe}.json")


def time_to_minutes(t_str: str) -> int:
    h, m = map(int, t_str.split(":"))
    return h * 60 + m


def get_active_slot(slots: list) -> dict:
    """Get the currently active time slot based on IST."""
    from datetime import datetime
    now = datetime.now(IST)
    now_mins = now.hour * 60 + now.minute

    for slot in slots:
        start = time_to_minutes(slot["start"])
        end = time_to_minutes(slot["end"])
        if end <= start:
            if now_mins >= start or now_mins < end:
                return slot
        else:
            if start <= now_mins < end:
                return slot

    for slot in slots:
        if slot.get("streams"):
            return slot
    return slots[0]


# ═══════════════════════════════════════════════════════════════════
#  DEBUG GUI — Mouse callbacks + window (--gui mode only)
# ═══════════════════════════════════════════════════════════════════

def _mouse_to_original_coords(mx, my):
    """
    Convert mouse coords (WINDOW_W x WINDOW_H space) to ORIGINAL frame coords.

    Since we resize every frame to WINDOW_W x WINDOW_H before imshow,
    mouse coords are ALWAYS in WINDOW_W x WINDOW_H space — guaranteed.
    No dependency on cv2.getWindowImageRect() (unreliable on Windows DPI).

    We scale to original frame resolution for saving to line_configs JSON
    (counting logic needs original-resolution coordinates).
    """
    orig = _debug_draw.get("original_shape")  # (h, w) of original frame
    if orig is None:
        return mx, my
    orig_h, orig_w = orig
    fx = int(mx * orig_w / WINDOW_W)
    fy = int(my * orig_h / WINDOW_H)
    return fx, fy


def _debug_mouse_callback(event, x, y, flags, param):
    """Mouse callback for debug window — draw line or ROI polygon.

    Mouse coords (x, y) are in WINDOW_W x WINDOW_H space because we
    resize every frame to that size before imshow. So:
      - Drawing preview: use (x, y) directly on the display frame
      - Saving to config: convert to original resolution via _mouse_to_original_coords
    """
    state = _debug_draw
    cfg_path = state.get("config_path")
    if not cfg_path or state.get("original_shape") is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        state["drawing_line"] = True
        state["temp_line"] = [(x, y), (x, y)]

    elif event == cv2.EVENT_MOUSEMOVE:
        if state["drawing_line"] and state["temp_line"]:
            state["temp_line"][1] = (x, y)
        if state["poly_points"]:
            state["poly_preview_pt"] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if state["drawing_line"] and state["temp_line"]:
            state["drawing_line"] = False
            state["temp_line"][1] = (x, y)
            # Convert display coords → original frame coords for config
            fp1 = _mouse_to_original_coords(*state["temp_line"][0])
            fp2 = _mouse_to_original_coords(*state["temp_line"][1])
            line_len = math.hypot(fp2[0] - fp1[0], fp2[1] - fp1[1])
            if line_len >= 10:
                data = {"line": [list(fp1), list(fp2)], "roi_poly": None, "mode": "line"}
                save_line_config(cfg_path, data)
                print(f"[DEBUG GUI] Line saved: {state['stream_name']} → {[fp1, fp2]}")
            state["temp_line"] = None

    elif event == cv2.EVENT_RBUTTONDOWN:
        state["poly_points"].append((x, y))

        if len(state["poly_points"]) == 4:
            # Convert all 4 display points → original frame coords
            frame_pts = [_mouse_to_original_coords(*pt) for pt in state["poly_points"]]
            data = {"line": None, "roi_poly": [list(p) for p in frame_pts], "mode": "roi"}
            save_line_config(cfg_path, data)
            print(f"[DEBUG GUI] ROI saved: {state['stream_name']} → {frame_pts}")
            state["poly_points"] = []
            state["poly_preview_pt"] = None

    elif event == cv2.EVENT_MBUTTONDOWN:
        state["poly_points"] = []
        state["poly_preview_pt"] = None


def _draw_debug_mouse_overlay(frame):
    """Draw in-progress line/ROI on the debug frame.

    Frame is ALREADY resized to WINDOW_W x WINDOW_H before this is called.
    Mouse coords are in the same space. So draw DIRECTLY — no conversion.
    """
    state = _debug_draw

    if state["drawing_line"] and state["temp_line"]:
        pt1 = tuple(state["temp_line"][0])
        pt2 = tuple(state["temp_line"][1])
        cv2.line(frame, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

    if state["poly_points"]:
        for pt in state["poly_points"]:
            cv2.circle(frame, tuple(pt), 6, (0, 100, 255), -1)
        for i in range(1, len(state["poly_points"])):
            cv2.line(frame, tuple(state["poly_points"][i - 1]),
                     tuple(state["poly_points"][i]), (0, 100, 255), 2)
        if state["poly_preview_pt"]:
            cv2.line(frame, tuple(state["poly_points"][-1]),
                     tuple(state["poly_preview_pt"]), (0, 100, 255), 1)
        n = len(state["poly_points"])
        cv2.putText(frame, f"ROI: {n}/4 corners (RClick {4 - n} more | MidBtn=cancel)",
                    (10, WINDOW_H - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)


def setup_debug_window():
    cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEBUG_WINDOW, WINDOW_W, WINDOW_H)
    cv2.setMouseCallback(DEBUG_WINDOW, _debug_mouse_callback)
    print("[DEBUG GUI] Window created — draw lines/ROI with mouse")


def update_debug_window(pipeline: Pipeline) -> int:
    """
    Poll latest debug frame from YOLO worker and display.
    Updates mouse callback state for the currently-processing stream.
    Returns key press (0xFF masked) or 0.

    CRITICAL: We resize every frame to WINDOW_W x WINDOW_H before imshow.
    This guarantees mouse coords = display coords (1:1 mapping).
    No dependency on cv2.getWindowImageRect() (unreliable on Windows DPI).
    Original resolution stored in _debug_draw["original_shape"] for config saving.
    """
    frame = pipeline.get_debug_frame()

    if frame is not None:
        # Update which stream the mouse callback saves to
        info = pipeline.get_debug_stream_info()
        _debug_draw["config_path"] = info.get("config_path") or ""
        _debug_draw["stream_name"] = info.get("current_stream") or ""

        # Store ORIGINAL resolution (for converting mouse→frame coords when saving config)
        _debug_draw["original_shape"] = frame.shape[:2]  # (h, w)

        # Resize to fixed display size BEFORE any drawing
        # This makes mouse coords = display coords (1:1, always correct)
        display = cv2.resize(frame, (WINDOW_W, WINDOW_H))

        # Draw mouse overlay on the resized display frame (coords match directly)
        _draw_debug_mouse_overlay(display)

        # Stats overlay
        stats_text = pipeline.get_stats_summary()
        cv2.putText(display, stats_text, (WINDOW_W - 500, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1, cv2.LINE_AA)

        _debug_draw["last_frame"] = display
        cv2.imshow(DEBUG_WINDOW, display)

    elif _debug_draw["last_frame"] is not None:
        cv2.imshow(DEBUG_WINDOW, _debug_draw["last_frame"])

    return cv2.waitKey(1) & 0xFF


# ═══════════════════════════════════════════════════════════════════
#  BETTING PHASE — 15 seconds, no stream
# ═══════════════════════════════════════════════════════════════════

def show_betting_phase(rd: RoundData, duration: float,
                       pipeline: Pipeline = None) -> bool:
    global current_round_data
    current_round_data = rd
    rd.phase = "BETTING"
    rd.phase_start_time = time.time()
    web_server.update_game_state(get_betting_phase_data(rd))

    # Update game_api — single source of truth for all data
    game_api.update_phase(
        "BETTING", round_id=rd.round_id, stream_name=rd.stream_name,
        commitment_hash=rd.commitment_hash, boundaries=rd.boundaries,
        odds=rd.odds, win_chances=rd.win_chances,
        under_threshold=rd.under_threshold,
        over_threshold=rd.over_threshold,
        exact_numbers=rd.exact_numbers,
    )

    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed >= duration:
            break

        frame = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
        for y_row in range(WINDOW_H):
            val = int(15 + 12 * (y_row / WINDOW_H))
            frame[y_row, :] = (val, val + 3, val + 8)

        cx = WINDOW_W // 2
        remaining = int(duration - elapsed) + 1

        # Just countdown timer — clean, no data
        cv2.putText(frame, f"{remaining}", (cx - 30, WINDOW_H // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 200), 4, cv2.LINE_AA)
        push_frame(frame)

        # CRITICAL: Always call cv2.waitKey to process Windows messages
        # Without this → "NOT RESPONDING" error
        if debug_gui and pipeline:
            key = update_debug_window(pipeline)
        else:
            key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            return True

    return False


# ═══════════════════════════════════════════════════════════════════
#  COUNTING + WAITING PHASE — 35s + 6s, clip plays
# ═══════════════════════════════════════════════════════════════════

def play_round_clip(rd: RoundData, pipeline: Pipeline = None) -> bool:
    global current_round_data
    current_round_data = rd
    rd.phase = "COUNTING"
    rd.phase_start_time = time.time()

    line_config = load_line_config(stream_config_path(rd.stream_name))
    reset_playback_flash()
    user_quit = False

    def frame_callback(frame, frame_no, current_count, counting_active,
                       elapsed_secs, detections):
        nonlocal user_quit

        if counting_active:
            rd.phase = "COUNTING"
            web_server.update_game_state(
                get_counting_phase_data(rd, current_count, elapsed_secs))
            # Update game_api with live count
            if frame_no == 0:
                game_api.update_phase("COUNTING")
            game_api.update_count(current_count)
        else:
            rd.phase = "WAITING"
            if frame_no == int(COUNTING_DURATION * (rd.clip.clip_fps or 25)):
                web_server.update_game_state(get_waiting_phase_data(rd))
                game_api.update_phase(
                    "WAITING", server_seed=rd.server_seed,
                    result=rd.result, vehicle_count=rd.result,
                    bet_outcomes=rd.bet_outcomes,
                )

        draw_playback_overlay(
            frame, frame_no, current_count, counting_active, elapsed_secs,
            detections, stream_name=rd.stream_name,
            count_duration=COUNTING_DURATION, wait_duration=WAITING_DURATION,
            line_config=line_config,
        )

        push_frame(frame)

        if debug_gui and pipeline:
            key = update_debug_window(pipeline)
            if key == ord('q'):
                user_quit = True

    play_clip_with_overlay(
        rd.clip, frame_callback, lambda: user_quit,
        count_duration=COUNTING_DURATION, wait_duration=WAITING_DURATION,
    )
    return user_quit


# ═══════════════════════════════════════════════════════════════════
#  MAIN SCHEDULER
# ═══════════════════════════════════════════════════════════════════

def run_scheduler(config_path: str, model_name: str, imgsz: int,
                  gui: bool = False, web_port: int = 5000):
    global debug_gui, current_round_data
    debug_gui = gui

    web_server.start_server(web_port)
    slots, count_duration, transition_duration = load_streams_config(config_path)
    history = HistoryTracker()

    # Initialize LiveKit publisher (connects to Docker container)
    try:
        livekit_publisher.init_publisher()
    except Exception as e:
        print(f"[LIVEKIT] Init failed (MJPEG fallback active): {e}")

    mode_str = "DEVELOPMENT (web + debug GUI)" if debug_gui else "PRODUCTION (web only)"
    print(f"\n{'=' * 60}")
    print(f"  CCTV CASINO GAME — Scheduler")
    print(f"  Mode           : {mode_str}")
    print(f"  Config         : {config_path}")
    print(f"  Model          : {model_name}")
    print(f"  Round cycle    : {BETTING_DURATION}s bet + {COUNTING_DURATION}s count + {WAITING_DURATION}s wait = 56s")
    print(f"  Web            : http://localhost:{web_port}")
    print(f"  LiveKit        : ws://localhost:7880 (room: cctv-game)")
    print(f"  IST now        : {ist_now_str()}")
    print(f"{'=' * 60}\n")

    print("[INIT] Loading YOLO model...")
    model = YOLO(model_name)
    if model_name.endswith(".pt"):
        model = model.to("cuda")
    print("[INIT] Model ready!")

    if debug_gui:
        setup_debug_window()

    # Create pipeline (parallel downloads + single YOLO worker)
    pipeline = Pipeline(model=model, debug_mode=debug_gui)

    round_counter = 0
    prev_slot_key = None

    def add_slot_streams(slot):
        """Add all streams in a time slot to the pipeline."""
        streams = slot.get("streams", [])
        for s in streams:
            name = s.get("name", "")
            url = s.get("url", "")
            if name and url:
                cfg_path = stream_config_path(name)
                pipeline.add_stream(name, url, cfg_path, imgsz=imgsz)

    # ── Cold start: add all streams from current slot ──
    active_slot = get_active_slot(slots)
    add_slot_streams(active_slot)
    COLD_START_MIN_CLIPS = 5  # Wait for 5 processed clips before first round
    print(f"[STARTUP] {len(pipeline.stream_names)} streams in round-robin rotation (1 download at a time)")
    print(f"[STARTUP] Waiting for {COLD_START_MIN_CLIPS} approved clips...")

    # Wait for 5 clips with loading animation
    anim_start = time.time()
    while pipeline.ready_count() < COLD_START_MIN_CLIPS and time.time() - anim_start < 600:
        el = time.time() - anim_start
        frame = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
        for yr in range(WINDOW_H):
            v = int(15 + 10 * (yr / WINDOW_H))
            frame[yr, :] = (v, v + 5, v + 10)

        from ui.animations import draw_globe
        gcx, gcy = WINDOW_W // 2, WINDOW_H // 2 - 40
        draw_globe(frame, gcx, gcy, 120, el * 0.8, 0.5 + 0.5 * abs(math.sin(el * 2)))

        cv2.putText(frame, "CCTV CASINO GAME", (WINDOW_W // 2 - 140, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 180, 255), 2, cv2.LINE_AA)
        ready = pipeline.ready_count()
        cv2.putText(frame, f"Preparing: {ready}/{COLD_START_MIN_CLIPS} clips ready...",
                    (WINDOW_W // 2 - 150, WINDOW_H // 2 + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"IST {ist_now_str()}", (WINDOW_W - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 140, 180), 1, cv2.LINE_AA)

        stats = pipeline.get_stats()
        yolo = stats["yolo"]
        dl_count = stats["downloaders"]["downloaded"]
        cv2.putText(frame,
                    f"Downloads: {dl_count} | YOLO: {yolo['current'] or 'waiting'} | "
                    f"Approved: {yolo['approved']} | Ready: {stats['ready_queue']}",
                    (WINDOW_W // 2 - 250, WINDOW_H // 2 + 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 150, 200), 1, cv2.LINE_AA)

        push_frame(frame)

        # CRITICAL: cv2.waitKey processes Windows messages — prevents "NOT RESPONDING"
        if debug_gui:
            key = update_debug_window(pipeline)
        else:
            key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            pipeline.cleanup()
            if debug_gui:
                cv2.destroyAllWindows()
            return

    # ── Main round loop ──
    try:
        while True:
            # Check for time slot change
            active_slot = get_active_slot(slots)
            slot_key = f"{active_slot['start']}-{active_slot['end']}"

            if slot_key != prev_slot_key and prev_slot_key is not None:
                print(f"[SCHEDULER] Time slot changed: {prev_slot_key} → {slot_key}")
                new_names = [s.get("name", "") for s in active_slot.get("streams", [])]
                pipeline.remove_streams_not_in(new_names)
                add_slot_streams(active_slot)
            prev_slot_key = slot_key

            # Get next clip — non-blocking poll with animation
            # NEVER block main thread > 30ms (prevents "NOT RESPONDING")
            clip = pipeline.get_next_clip(timeout=0.01)
            if clip is None:
                # Show "preparing" screen while polling — keeps GUI responsive
                poll_start = time.time()
                while clip is None and time.time() - poll_start < 180:
                    el = time.time() - poll_start
                    frame = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
                    for yr in range(WINDOW_H):
                        v = int(15 + 10 * (yr / WINDOW_H))
                        frame[yr, :] = (v, v + 5, v + 10)

                    from ui.animations import draw_globe
                    gcx, gcy = WINDOW_W // 2, WINDOW_H // 2 - 40
                    draw_globe(frame, gcx, gcy, 80, el * 0.8, min(1.0, el / 10.0))

                    cv2.putText(frame, "Preparing next round...",
                                (WINDOW_W // 2 - 130, WINDOW_H // 2 + 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 255), 1, cv2.LINE_AA)
                    stats = pipeline.get_stats()
                    cv2.putText(frame,
                                f"Ready: {stats['ready_queue']} | "
                                f"YOLO: {stats['yolo']['current'] or 'idle'}",
                                (WINDOW_W // 2 - 150, WINDOW_H // 2 + 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 150, 200), 1, cv2.LINE_AA)

                    push_frame(frame)
                    if debug_gui:
                        key = update_debug_window(pipeline)
                    else:
                        key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        raise KeyboardInterrupt

                    # Non-blocking poll (never blocks > 10ms)
                    clip = pipeline.get_next_clip(timeout=0.01)

                if clip is None:
                    print("[SCHEDULER] No clip ready after 180s, retrying...")
                    continue

            # ── PREPARE ROUND ──
            round_counter += 1
            rd = prepare_round(clip, round_counter, history)

            b = rd.boundaries
            print(f"\n{'─' * 60}")
            print(f"  ROUND #{round_counter}")
            print(f"  Stream : {clip.stream_name}")
            print(f"  Result : {rd.result} vehicles (pre-counted)")
            print(f"  Hash   : {rd.commitment_hash[:24]}...")
            print(f"  Lambda : {rd.lambda_mean} (historical mean)")
            print(f"  Under  : < {b.get('under', '?')}  |  "
                  f"Range : {b.get('range_low', '?')}-{b.get('range_high', '?')}  |  "
                  f"Over : > {b.get('over', '?')}")
            print(f"  Exact  : {b.get('exact_1', '?')} or {b.get('exact_2', '?')}")
            print(f"  Mult   : Under {rd.odds.get('under', '?')}x | "
                  f"Range {rd.odds.get('range', '?')}x | "
                  f"Over {rd.odds.get('over', '?')}x | "
                  f"Exact {rd.odds.get('exact', '?')}x (constant)")
            print(f"{'─' * 60}")

            # ── PHASE 1: BETTING (15 sec) ──
            if show_betting_phase(rd, BETTING_DURATION, pipeline):
                break

            # ── PHASE 2+3: COUNTING (35 sec) + WAITING (6 sec) ──
            if play_round_clip(rd, pipeline):
                break

            # ── FINALIZE ──
            finalize_round(rd, history)
            pipeline.mark_clip_done(clip)

            web_server.update_verification({
                "round_id": rd.round_id,
                "server_seed": rd.server_seed,
                "result": rd.result,
                "boundaries": rd.boundaries,
                "commitment_hash": rd.commitment_hash,
                "verification_string": rd.verification_data.get(
                    "verification_string", ""),
                "bet_outcomes": rd.bet_outcomes,
            })

            print(f"  Round #{round_counter} complete | Result: {rd.result} | "
                  f"Ready: {pipeline.ready_count()} clips")

    except KeyboardInterrupt:
        print("\n[SCHEDULER] Interrupted by user")
    finally:
        pipeline.cleanup()
        if debug_gui:
            cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CCTV Casino Game Scheduler")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="Path to streams_config.json")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="YOLO model file")
    parser.add_argument("--imgsz", type=int, default=1600,
                        help="YOLO input image size")
    parser.add_argument("--gui", action="store_true",
                        help="Development mode: debug GUI window showing live "
                             "YOLO processing + mouse line/ROI drawing")
    parser.add_argument("--web-port", type=int, default=5000,
                        help="Flask web server port")
    parser.add_argument("--test", action="store_true",
                        help="Validate config and exit")
    args = parser.parse_args()

    if args.test:
        slots, cd, td = load_streams_config(args.config)
        print(f"[TEST] Config OK: {len(slots)} time slots, "
              f"count_duration={cd}, transition_duration={td}")
        total_streams = sum(len(s.get("streams", [])) for s in slots)
        print(f"[TEST] Total streams: {total_streams}")
        return

    run_scheduler(args.config, args.model, args.imgsz, args.gui, args.web_port)


if __name__ == "__main__":
    main()
