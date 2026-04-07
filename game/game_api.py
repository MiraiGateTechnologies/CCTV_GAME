"""
================================================================
  GAME API — Single source of truth for all game data
================================================================

All game data flows through this module ONLY:
  - Timer state (phase + countdown)
  - Round ID
  - Boundaries / Odds (Under, Range, Over, Exact numbers — change per round)
  - Multipliers (CONSTANT: Under 3.13x, Range 2.35x, Over 3.76x, Exact 18.8x)
  - Win chances (CONSTANT: Under 30%, Range 40%, Over 25%, Exact 5%)
  - Vehicle count
  - Provably fair (commitment hash, server seed)
  - Bet outcomes

Data is:
  1. Overlaid on the web stream (via get_overlay_data)
  2. Available via REST API /api/round (via get_api_response)
  3. Pushed via WebSocket /ws/game (real-time, no polling)
  4. Updated by the scheduler each frame

Timer format per round (56 sec total):
  Phase BETTING:  timer counts 15 -> 0  (phase_elapsed 0-15)
  Phase COUNTING: timer counts 35 -> 0  (phase_elapsed 0-35)
  Phase WAITING:  timer counts 6 -> 0   (phase_elapsed 0-6)
"""

import time
import threading
from typing import Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════
#  PYDANTIC MODELS — typed API responses (for Swagger docs)
# ═══════════════════════════════════════════════════════════════

class BoundariesData(BaseModel):
    """Odds = boundary numbers for each bet option (change every round)."""
    under: int = Field(0, description="Count < under → Under wins")
    range_low: int = Field(0, description="range_low <= count <= range_high → Range wins")
    range_high: int = Field(0, description="range_low <= count <= range_high → Range wins")
    over: int = Field(0, description="Count > over → Over wins")
    exact_1: int = Field(0, description="Count == exact_1 or exact_2 → Exact wins")
    exact_2: int = Field(0, description="Count == exact_1 or exact_2 → Exact wins")


class MultipliersData(BaseModel):
    """Payout multipliers for each bet type (CONSTANT every round)."""
    under: float = 3.13
    range: float = Field(2.35, alias="range")
    over: float = 3.76
    exact: float = 18.8

    model_config = {"populate_by_name": True}


class WinChancesData(BaseModel):
    """Display-friendly win chances (CONSTANT every round)."""
    under: int = 30
    range: int = Field(40, alias="range")
    over: int = 25
    exact: int = 5

    model_config = {"populate_by_name": True}


class TimerData(BaseModel):
    phase_timer: float = Field(description="Seconds remaining in current phase")
    round_timer: float = Field(description="Seconds remaining in entire round (56->0)")
    round_elapsed: float = Field(description="Seconds elapsed since round started")


class RoundResponse(BaseModel):
    """Response for GET /api/round — primary game data endpoint."""
    phase: str = Field(description="Current phase: IDLE/BETTING/COUNTING/WAITING")
    round_id: int = 0
    stream_name: str = ""

    # Timers
    phase_timer: float = 0
    round_timer: float = 0
    round_elapsed: float = 0

    # Always visible
    commitment_hash: str = ""
    vehicle_count: int = 0

    # Shown during BETTING, COUNTING, and WAITING
    boundaries: Optional[BoundariesData] = None
    odds: Optional[MultipliersData] = None
    win_chances: Optional[WinChancesData] = None
    under_threshold: Optional[int] = None
    over_threshold: Optional[int] = None
    exact_numbers: Optional[list[int]] = None

    # Revealed during WAITING only
    server_seed: Optional[str] = None
    result: Optional[int] = None
    bet_outcomes: Optional[dict] = None


class VerificationResponse(BaseModel):
    """Response for GET /api/verify."""
    round_id: int = 0
    server_seed: str = ""
    result: int = 0
    boundaries: Optional[dict] = None
    commitment_hash: str = ""
    verification_string: str = ""
    bet_outcomes: dict = {}


class WSMessage(BaseModel):
    """WebSocket message format."""
    type: str = Field(description="Message type: game_state/count_update/verification")
    data: dict = {}


# ═══════════════════════════════════════════════════════════════
#  ROUND DURATIONS
# ═══════════════════════════════════════════════════════════════

BETTING_SECS = 15
COUNTING_SECS = 35
WAITING_SECS = 6
TOTAL_ROUND_SECS = BETTING_SECS + COUNTING_SECS + WAITING_SECS  # 56


# ═══════════════════════════════════════════════════════════════
#  THREAD-SAFE GAME STATE
# ═══════════════════════════════════════════════════════════════

_lock = threading.Lock()
_state = {
    "phase": "IDLE",
    "round_id": 0,
    "stream_name": "",

    # Timer
    "phase_timer": 0,
    "round_timer": 0,
    "round_elapsed": 0.0,

    # Counting
    "vehicle_count": 0,

    # Boundaries (odds numbers — change every round)
    "boundaries": {},

    # Multipliers (CONSTANT every round)
    "odds": {"under": 3.13, "range": 2.35, "over": 3.76, "exact": 18.8},

    # Win chances (CONSTANT every round)
    "win_chances": {"under": 30, "range": 40, "over": 25, "exact": 5},

    # Thresholds (extracted from boundaries for convenience)
    "under_threshold": 0,
    "over_threshold": 0,

    # Exact bet numbers (extracted from boundaries for convenience)
    "exact_numbers": [],

    # Provably fair
    "commitment_hash": "",
    "server_seed": "",
    "result": None,

    # Bet outcomes (populated during WAITING)
    "bet_outcomes": {},

    # Internal timestamps (not exposed in API)
    "_round_start_time": 0.0,
    "_phase_start_time": 0.0,
}

# Previous count — used to detect changes and push WebSocket updates
_prev_count = 0


def update_phase(phase: str, round_id: int = None, stream_name: str = None,
                 commitment_hash: str = None, boundaries: dict = None,
                 odds: dict = None, win_chances: dict = None,
                 under_threshold: int = None, over_threshold: int = None,
                 exact_numbers: list = None, vehicle_count: int = None,
                 server_seed: str = None, result: int = None,
                 bet_outcomes: dict = None):
    """
    Called by scheduler at each phase transition.
    This is the ONLY way game state gets updated.
    Also pushes phase change to WebSocket clients.
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
        if boundaries is not None:
            _state["boundaries"] = boundaries
        if odds is not None:
            _state["odds"] = odds
        if win_chances is not None:
            _state["win_chances"] = win_chances
        if under_threshold is not None:
            _state["under_threshold"] = under_threshold
        if over_threshold is not None:
            _state["over_threshold"] = over_threshold
        if exact_numbers is not None:
            _state["exact_numbers"] = exact_numbers
        if vehicle_count is not None:
            _state["vehicle_count"] = vehicle_count
        if server_seed is not None:
            _state["server_seed"] = server_seed
        if result is not None:
            _state["result"] = result
        if bet_outcomes is not None:
            _state["bet_outcomes"] = bet_outcomes

        # Phase-specific resets
        if phase == "BETTING":
            _state["_round_start_time"] = now
            _state["_phase_start_time"] = now
            _state["vehicle_count"] = 0
            _state["server_seed"] = ""
            _state["result"] = None
            _state["bet_outcomes"] = {}
        elif phase == "COUNTING":
            _state["_phase_start_time"] = now
        elif phase == "WAITING":
            _state["_phase_start_time"] = now


def update_count(count: int):
    """Called every frame during COUNTING phase to update vehicle count."""
    global _prev_count
    with _lock:
        _state["vehicle_count"] = count

    # Push count update via WebSocket only when count changes
    if count != _prev_count:
        _prev_count = count
        try:
            import web_server
            web_server._broadcast_sync(
                {"type": "count_update", "data": {"vehicle_count": count}}
            )
        except Exception:
            pass


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
    Full game state for GET /api/round.
    Returns everything frontend/betting backend needs.

    Data visibility by phase:
      IDLE:     basic info only
      BETTING:  + odds, thresholds, exact_numbers, win_chances, commitment_hash
      COUNTING: + live vehicle_count
      WAITING:  + server_seed, result, bet_outcomes (for verification)
    """
    with _lock:
        timers = _calculate_timers()
        phase = _state["phase"]

        raw_id = _state["round_id"]
        round_id_str = f"#RId{raw_id:07d}"

        # IST datetime
        from datetime import datetime, timezone, timedelta
        ist = timezone(timedelta(hours=5, minutes=30))
        created = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")

        # Status: True when clip is playing (COUNTING or WAITING)
        status = phase in ("COUNTING", "WAITING")
        
        # Betting status
        betting_status = phase == "BETTING"

        resp = {
            # 1. Odds (boundaries)
            "odds": _state["boundaries"],
            
            # 2. Round timer (56 sec countdown) 
            "round_timer": timers["round_timer"],
            
            # 3. Unique round ID
            "round_id": round_id_str,
            
            # 4. Created datetime (IST)
            "created": created,
            
            # 5. Status (clip playing?)
            "status": status,
            
            # 6. Waiting timer (6 sec, only during WAITING)
            "waiting_timer": timers["phase_timer"] if phase == "WAITING" else 0,
            
            # 7. Vehicle count
            "vehicle_count": _state["vehicle_count"],
            
            # 8. Result (final count, revealed in WAITING)
            "result": _state["result"] if phase == "WAITING" else None,
            
            # 9. Betting status
            "betting_status": betting_status,
            
            # 10. Betting timer (15 sec, only during BETTING)
            "betting_timer": timers["phase_timer"] if phase == "BETTING" else 0,
            
            "phase": phase,
            "commitment_hash": _state["commitment_hash"],

            # Extra fields (revealed during WAITING only)
            "server_seed": _state["server_seed"] if phase == "WAITING" else None,
            "bet_outcomes": _state["bet_outcomes"] if phase == "WAITING" else None,
        }

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
            "phase_timer": int(timers["phase_timer"]) + 1,
            "round_timer": int(timers["round_timer"]) + 1,
            "vehicle_count": _state["vehicle_count"],
            "stream_name": _state["stream_name"],
            "boundaries": _state["boundaries"],
            "odds": _state["odds"],
            "win_chances": _state["win_chances"],
            "under_threshold": _state["under_threshold"],
            "over_threshold": _state["over_threshold"],
            "exact_numbers": _state["exact_numbers"],
        }

        
def get_timer_update() -> dict:
    """Lightweight timer data for WebSocket push every 1 sec."""
    with _lock:
        timers = _calculate_timers()
        phase = _state["phase"]
        return {
            "type": "timer_update",
            "data": {
                "round_timer": timers["round_timer"],
                "betting_timer": timers["phase_timer"] if phase == "BETTING" else 0,
                "waiting_timer": timers["phase_timer"] if phase == "WAITING" else 0,
                "phase": phase,
            }
        }