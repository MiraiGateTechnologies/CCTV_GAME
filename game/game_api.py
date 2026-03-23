"""
================================================================
  GAME API — Single source of truth for all game data
================================================================

All game data flows through this module ONLY:
  - Timer state (phase + countdown)
  - Round ID
  - Odds (Under, Range, Over, Exact)
  - Vehicle count
  - Provably fair hash

Data is:
  1. Overlaid on the web stream (via get_overlay_data)
  2. Available via API endpoint /api/round (via get_api_response)
  3. Updated by the scheduler each frame

Timer format per round (56 sec total):
  Phase BETTING:  timer counts 15 → 0  (phase_elapsed 0-15)
  Phase COUNTING: timer counts 35 → 0  (phase_elapsed 0-35)
  Phase WAITING:  timer counts 6 → 0   (phase_elapsed 0-6)
"""

import time
import threading

# ── Round durations ──
BETTING_SECS = 15
COUNTING_SECS = 35
WAITING_SECS = 6
TOTAL_ROUND_SECS = BETTING_SECS + COUNTING_SECS + WAITING_SECS  # 56

# ── Thread-safe game state ──
_lock = threading.Lock()
_state = {
    "phase": "IDLE",          # BETTING / COUNTING / WAITING / IDLE
    "round_id": 0,
    "stream_name": "",

    # Timer
    "phase_timer": 0,         # countdown seconds remaining in current phase
    "round_timer": 0,         # countdown seconds remaining in entire round (56→0)
    "round_elapsed": 0.0,     # seconds elapsed since round started

    # Counting
    "vehicle_count": 0,

    # Odds (shown during COUNTING phase, generated from provably fair)
    "odds": {
        "under": 0.0,
        "range": 0.0,
        "over": 0.0,
    },
    "under_threshold": 0,
    "over_threshold": 0,
    "exact_odds": {},         # {number: odds_multiplier}

    # Provably fair
    "commitment_hash": "",
    "server_seed": "",        # only revealed in WAITING phase
    "result": None,           # only revealed in WAITING phase

    # Internal
    "_round_start_time": 0.0,
    "_phase_start_time": 0.0,
}


def update_phase(phase: str, round_id: int = None, stream_name: str = None,
                 commitment_hash: str = None, odds: dict = None,
                 under_threshold: int = None, over_threshold: int = None,
                 exact_odds: dict = None, vehicle_count: int = None,
                 server_seed: str = None, result: int = None):
    """
    Called by scheduler at each phase transition or count update.
    This is the ONLY way game state gets updated.
    """
    with _lock:
        now = time.time()
        _state["phase"] = phase

        if round_id is not None:
            _state["round_id"] = round_id
        if stream_name is not None:
            _state["stream_name"] = stream_name
        if commitment_hash is not None:
            _state["commitment_hash"] = commitment_hash
        if odds is not None:
            _state["odds"] = odds
        if under_threshold is not None:
            _state["under_threshold"] = under_threshold
        if over_threshold is not None:
            _state["over_threshold"] = over_threshold
        if exact_odds is not None:
            _state["exact_odds"] = exact_odds
        if vehicle_count is not None:
            _state["vehicle_count"] = vehicle_count
        if server_seed is not None:
            _state["server_seed"] = server_seed
        if result is not None:
            _state["result"] = result

        # Reset timers on phase change
        if phase == "BETTING":
            _state["_round_start_time"] = now
            _state["_phase_start_time"] = now
            _state["vehicle_count"] = 0
            _state["server_seed"] = ""
            _state["result"] = None
        elif phase == "COUNTING":
            _state["_phase_start_time"] = now
        elif phase == "WAITING":
            _state["_phase_start_time"] = now


def update_count(count: int):
    """Called every frame during COUNTING phase to update vehicle count."""
    with _lock:
        _state["vehicle_count"] = count


def _calculate_timers() -> dict:
    """Calculate current timer values from timestamps."""
    now = time.time()
    phase = _state["phase"]
    phase_elapsed = now - _state["_phase_start_time"] if _state["_phase_start_time"] else 0
    round_elapsed = now - _state["_round_start_time"] if _state["_round_start_time"] else 0

    if phase == "BETTING":
        phase_remaining = max(0, BETTING_SECS - phase_elapsed)
        round_remaining = max(0, TOTAL_ROUND_SECS - round_elapsed)
    elif phase == "COUNTING":
        phase_remaining = max(0, COUNTING_SECS - phase_elapsed)
        round_remaining = max(0, TOTAL_ROUND_SECS - round_elapsed)
    elif phase == "WAITING":
        phase_remaining = max(0, WAITING_SECS - phase_elapsed)
        round_remaining = max(0, TOTAL_ROUND_SECS - round_elapsed)
    else:
        phase_remaining = 0
        round_remaining = 0

    return {
        "phase_timer": round(phase_remaining, 1),
        "round_timer": round(round_remaining, 1),
        "round_elapsed": round(round_elapsed, 1),
    }


def get_api_response() -> dict:
    """
    Full game state for API endpoint /api/round.
    Called by web_server — returns everything frontend needs.
    """
    with _lock:
        timers = _calculate_timers()
        phase = _state["phase"]

        resp = {
            "phase": phase,
            "round_id": _state["round_id"],
            "stream_name": _state["stream_name"],

            # Timers
            "phase_timer": timers["phase_timer"],
            "round_timer": timers["round_timer"],
            "round_elapsed": timers["round_elapsed"],

            # Always show
            "commitment_hash": _state["commitment_hash"],
            "vehicle_count": _state["vehicle_count"],
        }

        # Odds only shown during COUNTING and WAITING (not during BETTING)
        if phase in ("COUNTING", "WAITING"):
            resp["odds"] = _state["odds"]
            resp["under_threshold"] = _state["under_threshold"]
            resp["over_threshold"] = _state["over_threshold"]

        # Seed + result only revealed during WAITING
        if phase == "WAITING":
            resp["server_seed"] = _state["server_seed"]
            resp["result"] = _state["result"]

        return resp


def get_overlay_data() -> dict:
    """
    Lightweight data for drawing on the web stream overlay.
    Called every frame by the renderer.
    """
    with _lock:
        timers = _calculate_timers()
        return {
            "phase": _state["phase"],
            "round_id": _state["round_id"],
            "phase_timer": int(timers["phase_timer"]) + 1,  # ceiling for display
            "round_timer": int(timers["round_timer"]) + 1,
            "vehicle_count": _state["vehicle_count"],
            "stream_name": _state["stream_name"],
            "odds": _state["odds"],
            "under_threshold": _state["under_threshold"],
            "over_threshold": _state["over_threshold"],
        }
