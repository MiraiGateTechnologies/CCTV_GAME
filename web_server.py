"""
================================================================
  WEB SERVER — FastAPI + Uvicorn
  MJPEG video stream + REST API + WebSocket real-time updates
  http://localhost:5000
================================================================

Endpoints:
  GET  /              — HTML page with video stream
  GET  /video_feed    — MJPEG live video feed
  GET  /api/round     — PRIMARY game data (timer, count, odds, hash)
  GET  /api/state     — Raw game state (phase, round_id, options)
  GET  /api/verify    — Verification data for completed round
  WS   /ws/game       — WebSocket: real-time game state push
  GET  /docs          — Auto-generated Swagger API docs (FREE)

Data flow:
  scheduler.py → web_server.update_frame() / update_game_state()
             → REST API returns data on poll
             → WebSocket pushes data instantly to connected clients
"""

import os
import cv2
import asyncio
import threading
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="CCTV Casino Game API",
    description="Real-time vehicle counting game — stream + game data",
    version="1.0.0",
)

# ── Frame buffer (thread-safe) ────────────────────────────────
_output_frame = None
_frame_lock = threading.Lock()

# ── Game state (set by scheduler.py) ─────────────────────────
_game_state = {
    "phase": "IDLE",
    "round_id": 0,
    "stream_name": "",
    "commitment_hash": "",
    "bet_options": ["UNDER", "RANGE", "OVER", "EXACT"],
}
_game_state_lock = threading.Lock()

# ── Verification data (revealed after round) ──────────────────
_verification = {}
_verification_lock = threading.Lock()

# ── WebSocket connection manager ──────────────────────────────
_event_loop: Optional[asyncio.AbstractEventLoop] = None


class _WSManager:
    """Manages WebSocket connections and broadcasts game state."""

    def __init__(self):
        self.connections: list[WebSocket] = []
        self._lock = threading.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        with self._lock:
            self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        with self._lock:
            if ws in self.connections:
                self.connections.remove(ws)

    async def broadcast(self, data: dict):
        with self._lock:
            stale = []
            for ws in self.connections:
                try:
                    await ws.send_json(data)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                self.connections.remove(ws)

    @property
    def count(self):
        return len(self.connections)


_ws_manager = _WSManager()


def _broadcast_sync(data: dict):
    """Thread-safe broadcast — called from scheduler thread, runs on server event loop."""
    if _event_loop is not None and not _event_loop.is_closed():
        try:
            asyncio.run_coroutine_threadsafe(_ws_manager.broadcast(data), _event_loop)
        except RuntimeError:
            pass


# ═══════════════════════════════════════════════════════════════
#  PUBLIC API — called by scheduler.py (same interface as before)
# ═══════════════════════════════════════════════════════════════

def update_frame(frame):
    """Updates the global frame with the latest processed frame."""
    global _output_frame
    with _frame_lock:
        _output_frame = frame.copy()


def update_game_state(state: dict):
    """Update the current game state (called by scheduler/round_manager)."""
    with _game_state_lock:
        _game_state.update(state)
    # Push to all WebSocket clients instantly
    _broadcast_sync({"type": "game_state", "data": state})


def update_verification(data: dict):
    """Update verification data (called after round ends)."""
    with _verification_lock:
        _verification.update(data)
    _broadcast_sync({"type": "verification", "data": data})


# ═══════════════════════════════════════════════════════════════
#  MJPEG STREAM
# ═══════════════════════════════════════════════════════════════

def _generate_frames():
    """Generator that yields MJPEG chunks for StreamingResponse."""
    while True:
        with _frame_lock:
            if _output_frame is None:
                time.sleep(0.01)
                continue
            ok, encoded = cv2.imencode(".jpg", _output_frame)
            if not ok:
                continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               bytearray(encoded) + b"\r\n")

        time.sleep(0.03)  # ~30fps cap — prevents CPU spinning


# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the game viewer HTML page."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/video_feed")
def video_feed():
    """MJPEG live video stream — <img src='/video_feed'> in browser."""
    return StreamingResponse(
        _generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/round")
def api_round():
    """
    PRIMARY game data endpoint — single source of truth.

    Returns timer, round_id, vehicle_count, odds, hash.
    Data comes from game/game_api.py only.
    """
    from game.game_api import get_api_response
    return get_api_response()


@app.get("/api/state")
def api_state():
    """
    Raw game state set by scheduler.

    During BETTING:  {phase, round_id, stream_name, commitment_hash, bet_options}
    During COUNTING: {phase, round_id, current_count, odds, thresholds, remaining}
    During WAITING:  {phase, round_id, result, verification, bet_outcomes}
    """
    with _game_state_lock:
        return dict(_game_state)


@app.get("/api/verify")
def api_verify():
    """
    Verification data for the last completed round.
    Players use this to verify provably fair commitment.

    Returns: {round_id, server_seed, result, under_threshold,
              over_threshold, commitment_hash}
    """
    with _verification_lock:
        return dict(_verification)


# ═══════════════════════════════════════════════════════════════
#  LIVEKIT — Token generation for viewers
# ═══════════════════════════════════════════════════════════════

@app.get("/api/livekit/token")
def livekit_token(identity: str = "viewer"):
    """
    Generate a LiveKit JWT token for a viewer to join the game stream room.

    Frontend calls this, then uses the token to connect via LiveKit JS SDK.
    Query param: ?identity=player123 (optional, defaults to "viewer")

    Returns: {"token": "eyJ...", "url": "ws://localhost:7880", "room": "cctv-game"}
    """
    try:
        from network.livekit_publisher import get_publisher
        pub = get_publisher()
        if pub is None:
            return {"error": "LiveKit publisher not initialized"}
        token = pub.generate_viewer_token(identity)
        return {
            "token": token,
            "url": pub.livekit_url,
            "room": pub.room_name,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/livekit/status")
def livekit_status():
    """Check LiveKit publisher connection status."""
    try:
        from network.livekit_publisher import get_publisher
        pub = get_publisher()
        if pub is None:
            return {"connected": False, "reason": "Publisher not initialized"}
        return {
            "connected": pub.is_connected,
            "url": pub.livekit_url,
            "room": pub.room_name,
        }
    except Exception as e:
        return {"connected": False, "reason": str(e)}


# ═══════════════════════════════════════════════════════════════
#  WEBSOCKET — Real-time game state push
# ═══════════════════════════════════════════════════════════════

@app.websocket("/ws/game")
async def ws_game(ws: WebSocket):
    """
    WebSocket endpoint for real-time game updates.

    Connect: ws://localhost:5000/ws/game

    Server pushes messages:
      {"type": "game_state", "data": {...}}
      {"type": "verification", "data": {...}}
      {"type": "count_update", "data": {"vehicle_count": N}}

    Client can send:
      {"type": "ping"} → server replies {"type": "pong"}
    """
    await _ws_manager.connect(ws)
    try:
        # Send current state on connect
        from game.game_api import get_api_response
        await ws.send_json({"type": "initial_state", "data": get_api_response()})

        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _ws_manager.disconnect(ws)

# NEW: Timer broadcast thread — pushes timer every 1 sec to all WebSocket clients
def _start_timer_broadcast():
    def _timer_loop():
        while True:
            time.sleep(1)
            try:
                from game.game_api import get_timer_update
                data = get_timer_update()
                _broadcast_sync(data)
            except Exception:
                pass
    
    t = threading.Thread(target=_timer_loop, daemon=True, name="WS-Timer")
    t.start()

# ═══════════════════════════════════════════════════════════════
#  SERVER START — runs uvicorn in daemon thread
# ═══════════════════════════════════════════════════════════════

def start_server(port=5000, host="0.0.0.0"):
    """Starts the FastAPI server in a daemon thread. Same signature as before."""

    config = uvicorn.Config(
        app, host=host, port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _run():
        global _event_loop
        _event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_event_loop)
        _event_loop.run_until_complete(server.serve())

    t = threading.Thread(target=_run, daemon=True, name="FastAPI-Server")
    t.start()

    # Give server a moment to bind
    time.sleep(0.5)

    print(f"[WEB] FastAPI server at http://localhost:{port}")
    print(f"[WEB] API docs:     http://localhost:{port}/docs")
    print(f"[WEB] Game data:    http://localhost:{port}/api/round")
    print(f"[WEB] WebSocket:    ws://localhost:{port}/ws/game")
    print(f"[WEB] Verify:       http://localhost:{port}/api/verify")

    _start_timer_broadcast()