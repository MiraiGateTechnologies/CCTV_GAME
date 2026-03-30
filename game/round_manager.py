"""
================================================================
  ROUND MANAGER — Orchestrates the 15s→35s→6s game cycle
================================================================

One round = 56 seconds:
  Phase 1: BETTING  (15 sec) — No stream, players bet, hash published
  Phase 2: COUNTING (35 sec) — Stream plays, counting overlay, odds shown
  Phase 3: WAITING  (6 sec)  — Stream continues, count frozen, result reveal

This module provides RoundData and the playback logic for Phase 2+3.
The betting phase (Phase 1) is handled by the scheduler/frontend.

Data Flow:
  prepare_round()  →  RoundData (immutable once created)
    ├── odds_engine.generate_round_odds()     →  boundaries, multipliers, win_chances
    └── provably_fair.generate_round_fairness()  →  seed, hash

Phase transitions expose progressive data:
  BETTING:  commitment_hash + boundaries + multipliers + win_chances
  COUNTING: same + live vehicle count
  WAITING:  same + result + server_seed + verification
"""

import time
import cv2
import numpy as np
from dataclasses import dataclass, field

from network.stream_manager import PreProcessedClip
from core.provably_fair import generate_round_fairness, get_verification_data
from core.odds_engine import generate_round_odds, settle_bets
from game.history_tracker import HistoryTracker


# Phase durations (seconds)
BETTING_DURATION = 15
COUNTING_DURATION = 35
WAITING_DURATION = 6
TOTAL_ROUND_DURATION = BETTING_DURATION + COUNTING_DURATION + WAITING_DURATION


@dataclass
class RoundData:
    """
    All data for a single game round — populated once during prepare_round().

    After creation, only 'phase' and 'phase_start_time' change during the round.
    All odds, boundaries, and provably fair data are IMMUTABLE once set.
    """
    round_id: int
    stream_name: str

    # Pre-processed clip data
    clip: PreProcessedClip = None

    # Provably fair
    server_seed: str = ""
    commitment_hash: str = ""
    result: int = 0

    # Boundaries (odds) — the numbers for each bet option
    # {"under": 14, "range_low": 20, "range_high": 21,
    #  "over": 28, "exact_1": 20, "exact_2": 21}
    boundaries: dict = field(default_factory=dict)

    # Convenience accessors (extracted from boundaries for backward compat)
    under_threshold: int = 0
    over_threshold: int = 0
    exact_numbers: list = field(default_factory=list)

    # Multipliers — CONSTANT every round
    # {"under": 3.13, "range": 2.35, "over": 3.76, "exact": 18.8}
    odds: dict = field(default_factory=dict)

    # Win chances — CONSTANT every round
    # {"under": 30, "range": 40, "over": 25, "exact": 5}
    win_chances: dict = field(default_factory=dict)

    # Historical data info
    lambda_mean: float = 0.0
    has_enough_data: bool = False
    rounds_tracked: int = 0

    # Phase tracking
    phase: str = "BETTING"  # BETTING, COUNTING, WAITING
    phase_start_time: float = 0.0

    # Verification (populated during WAITING phase)
    verification_data: dict = field(default_factory=dict)
    bet_outcomes: dict = field(default_factory=dict)


def prepare_round(clip: PreProcessedClip, round_id: int,
                  history: HistoryTracker) -> RoundData:
    """
    Prepare all data for a new round BEFORE betting phase starts.

    This is called ONCE when a clip is picked from the ready queue.
    All data is pre-calculated — no heavy computation during the round.

    Flow:
      1. Generate odds (boundaries) from lambda_mean (historical average)
      2. Generate provably fair data (server_seed → commitment hash)
      3. Package everything into immutable RoundData
    """
    stream_name = clip.stream_name
    result = clip.result

    # Step 1: Generate odds — boundaries from historical lambda_mean
    odds_data = generate_round_odds(history, stream_name)
    boundaries = odds_data["boundaries"]

    # Step 2: Generate provably fair data (commitment locks result + boundaries)
    pf_data = generate_round_fairness(
        result=result,
        round_id=round_id,
        boundaries=boundaries,
    )

    # Step 3: Log round info
    print(
        f"[ODDS] Round #{round_id}: "
        f"lambda_mean={odds_data['lambda_mean']} | "
        f"data={odds_data['rounds_tracked']} rounds | "
        f"enough_data={odds_data['has_enough_data']}"
    )

    # Step 4: Create immutable round data
    rd = RoundData(
        round_id=round_id,
        stream_name=stream_name,
        clip=clip,
        # Provably fair
        server_seed=pf_data["server_seed"],
        commitment_hash=pf_data["commitment_hash"],
        result=result,
        # Boundaries (odds numbers)
        boundaries=boundaries,
        under_threshold=boundaries["under"],
        over_threshold=boundaries["over"],
        exact_numbers=[boundaries["exact_1"], boundaries["exact_2"]],
        # Multipliers + win chances (constant)
        odds=odds_data["multipliers"],
        win_chances=odds_data["win_chances"],
        # Historical info
        lambda_mean=odds_data["lambda_mean"],
        has_enough_data=odds_data["has_enough_data"],
        rounds_tracked=odds_data["rounds_tracked"],
        # Phase
        phase="BETTING",
        phase_start_time=time.time(),
    )

    return rd


# ═══════════════════════════════════════════════════════════════
#  PHASE DATA — What to expose at each phase transition
# ═══════════════════════════════════════════════════════════════

def get_betting_phase_data(rd: RoundData) -> dict:
    """
    Data to expose to frontend during BETTING phase (15 sec).

    Players see: odds (boundary numbers), multipliers, win chances, hash.
    Players do NOT see: result, server_seed.
    """
    return {
        "round_id": rd.round_id,
        "stream_name": rd.stream_name,
        "commitment_hash": rd.commitment_hash,
        "phase": "BETTING",
        "bet_options": ["UNDER", "RANGE", "OVER", "EXACT"],
        # Odds = boundary numbers (change every round)
        "boundaries": rd.boundaries,
        "under_threshold": rd.under_threshold,
        "over_threshold": rd.over_threshold,
        "exact_numbers": rd.exact_numbers,
        # Multipliers + win chances (constant)
        "odds": rd.odds,
        "win_chances": rd.win_chances,
    }


def get_counting_phase_data(rd: RoundData, current_count: int,
                            elapsed: float) -> dict:
    """
    Data to expose during COUNTING phase (35 sec).
    Same as betting + live vehicle count + timing.
    """
    return {
        "round_id": rd.round_id,
        "stream_name": rd.stream_name,
        "commitment_hash": rd.commitment_hash,
        "phase": "COUNTING",
        "current_count": current_count,
        "elapsed": round(elapsed, 1),
        "remaining": round(max(0, COUNTING_DURATION - elapsed), 1),
        # Odds + multipliers still visible
        "boundaries": rd.boundaries,
        "under_threshold": rd.under_threshold,
        "over_threshold": rd.over_threshold,
        "exact_numbers": rd.exact_numbers,
        "odds": rd.odds,
        "win_chances": rd.win_chances,
    }


def get_waiting_phase_data(rd: RoundData) -> dict:
    """
    Data to expose during WAITING phase (6 sec).
    Result + verification revealed — the "big reveal" moment.
    """
    # Generate verification data for this round
    rd.verification_data = get_verification_data({
        "server_seed": rd.server_seed,
        "result": rd.result,
        "boundaries": rd.boundaries,
        "round_id": rd.round_id,
        "commitment_hash": rd.commitment_hash,
    })

    # Settle all bet types
    rd.bet_outcomes = settle_bets(rd.result, rd.boundaries)

    return {
        "round_id": rd.round_id,
        "stream_name": rd.stream_name,
        "phase": "WAITING",
        "result": rd.result,
        "verification": rd.verification_data,
        "bet_outcomes": rd.bet_outcomes,
        # Odds + multipliers still visible for reference
        "boundaries": rd.boundaries,
        "under_threshold": rd.under_threshold,
        "over_threshold": rd.over_threshold,
        "exact_numbers": rd.exact_numbers,
        "odds": rd.odds,
        "win_chances": rd.win_chances,
    }


# ═══════════════════════════════════════════════════════════════
#  CLIP PLAYBACK — Phase 2 + 3 (35s counting + 6s waiting)
# ═══════════════════════════════════════════════════════════════

def play_clip_with_overlay(clip: PreProcessedClip, frame_callback,
                           should_stop=None, count_duration: float = 35.0,
                           wait_duration: float = 6.0):
    """
    Play a pre-processed clip with detection overlays.
    NO YOLO runs during playback — uses stored detection data.

    Args:
        clip:           PreProcessedClip with pre-computed detections
        frame_callback: Called for each frame: callback(frame, frame_no, current_count,
                            counting_active, elapsed_secs, detections_this_frame)
        should_stop:    Callable returning True to abort playback
        count_duration: Seconds of counting phase (35)
        wait_duration:  Seconds of waiting phase (6)
    """
    cap = cv2.VideoCapture(clip.clip_path)
    if not cap.isOpened():
        print(f"[PLAYBACK] ERROR: Cannot open clip: {clip.clip_path}")
        return

    fps = clip.clip_fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_time = 1.0 / fps
    max_count_frame = int(count_duration * fps)
    max_total_frame = int((count_duration + wait_duration) * fps)

    # Build frame→count lookup from counting_events
    count_at_frame = {}
    for event in clip.counting_events:
        count_at_frame[event["frame"]] = event["count_at"]

    current_count = 0
    frame_no = 0

    while frame_no < max_total_frame:
        loop_start = time.perf_counter()

        if should_stop and should_stop():
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Update displayed count
        if frame_no in count_at_frame:
            current_count = count_at_frame[frame_no]

        counting_active = frame_no < max_count_frame
        elapsed_secs = frame_no / fps

        # Get stored detections for this frame
        dets = clip.detections.get(frame_no, [])
        if not dets:
            dets = clip.detections.get(str(frame_no), [])

        frame_callback(frame, frame_no, current_count, counting_active,
                       elapsed_secs, dets)

        frame_no += 1

        # Frame pacing
        elapsed_loop = time.perf_counter() - loop_start
        sleep_time = frame_time - elapsed_loop
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()


def finalize_round(rd: RoundData, history: HistoryTracker):
    """
    Called after a round ends. Records result in history for future odds.
    """
    history.add_result(rd.stream_name, rd.result)
