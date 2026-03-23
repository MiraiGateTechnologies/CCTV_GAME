"""
================================================================
  WEB SERVER — Browser live video + Game State API
  Flask MJPEG streaming + JSON endpoints for frontend
  http://localhost:5000
================================================================

Endpoints:
  GET /              — HTML page with video stream
  GET /video_feed    — MJPEG live video feed
  GET /game_state    — Current round state JSON (phase, hash, odds, result)
  GET /verify        — Verification data for completed round
"""

import cv2
import json
import threading
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# ── Frame buffer (thread-safe) ──
output_frame = None
lock = threading.Lock()

# ── Game state (set by scheduler.py) ──
_game_state = {
    "phase": "IDLE",
    "round_id": 0,
    "stream_name": "",
    "commitment_hash": "",
    "bet_options": ["UNDER", "RANGE", "OVER", "EXACT"],
}
_game_state_lock = threading.Lock()

# ── Verification data (revealed after round) ──
_verification = {}
_verification_lock = threading.Lock()


def update_frame(frame):
    """Updates the global frame with the latest processed frame."""
    global output_frame
    with lock:
        output_frame = frame.copy()


def update_game_state(state: dict):
    """Update the current game state (called by scheduler/round_manager)."""
    with _game_state_lock:
        _game_state.update(state)


def update_verification(data: dict):
    """Update verification data (called after round ends)."""
    with _verification_lock:
        _verification.update(data)


def generate_frames():
    """Generator function that yields MJPEG chunks."""
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encoded_image) + b'\r\n')


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/game_state")
def game_state():
    """
    Current round state for frontend/backend consumption.

    During BETTING:  {phase, round_id, stream_name, commitment_hash, bet_options}
    During COUNTING: {phase, round_id, current_count, odds, thresholds, remaining}
    During WAITING:  {phase, round_id, result, verification, bet_outcomes}
    """
    with _game_state_lock:
        return jsonify(_game_state)


@app.route("/verify")
def verify():
    """
    Verification data for the last completed round.
    Players use this to verify provably fair commitment.
    """
    with _verification_lock:
        return jsonify(_verification)


@app.route("/api/round")
def api_round():
    """
    PRIMARY game data API — single source of truth.
    All data goes through game/game_api.py only.

    Returns: {phase, round_id, phase_timer, round_timer, vehicle_count,
              commitment_hash, odds (if COUNTING/WAITING),
              server_seed + result (if WAITING)}
    """
    from game.game_api import get_api_response
    return jsonify(get_api_response())


def start_server(port=5000, host="0.0.0.0"):
    """Starts the Flask server in a daemon thread."""
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    t.start()
    print(f"[WEB] Server started at http://localhost:{port}")
    print(f"[WEB] API endpoint:  http://localhost:{port}/api/round")
    print(f"[WEB] Verification:  http://localhost:{port}/verify")
