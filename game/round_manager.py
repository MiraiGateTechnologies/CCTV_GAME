"""
Round Manager — Orchestrates the 15s→35s→6s game cycle.

One round = 56 seconds:
  Phase 1: BETTING  (15 sec) — No stream, players bet, hash published
  Phase 2: COUNTING (35 sec) — Stream plays, counting overlay, odds shown
  Phase 3: WAITING  (6 sec)  — Stream continues, count frozen, result reveal

This module provides RoundData and the playback logic for Phase 2+3.
The betting phase (Phase 1) is handled by the scheduler/frontend.
"""

import time
import cv2
import numpy as np
from dataclasses import dataclass, field

from network.stream_manager import PreProcessedClip
from core.provably_fair import generate_round_commitment, get_verification_data
from core.odds_engine import generate_round_odds, settle_bets
from game.history_tracker import HistoryTracker


# Phase durations (seconds)
BETTING_DURATION = 15
COUNTING_DURATION = 35
WAITING_DURATION = 6
TOTAL_ROUND_DURATION = BETTING_DURATION + COUNTING_DURATION + WAITING_DURATION


@dataclass
class RoundData:
    """All data for a single game round."""
    round_id: int
    stream_name: str

    # Pre-processed clip data
    clip: PreProcessedClip = None

    # Provably fair
    server_seed: str = ""
    commitment_hash: str = ""
    result: int = 0

    # Odds
    under_threshold: int = 4
    over_threshold: int = 7
    odds: dict = field(default_factory=dict)
    probabilities: dict = field(default_factory=dict)
    exact_odds: dict = field(default_factory=dict)

    # Phase tracking
    phase: str = "BETTING"  # BETTING, COUNTING, WAITING
    phase_start_time: float = 0.0

    # Verification (revealed after round)
    verification_data: dict = field(default_factory=dict)
    bet_outcomes: dict = field(default_factory=dict)


def prepare_round(clip: PreProcessedClip, round_id: int,
                  history: HistoryTracker) -> RoundData:
    """
    Prepare all data for a new round BEFORE betting phase starts.

    This is called ONCE when a clip is picked from the ready queue.
    All data is pre-calculated — no heavy computation during the round.
    """
    stream_name = clip.stream_name
    result = clip.result

    # Generate odds from historical data
    odds_data = generate_round_odds(history, stream_name)
    under_threshold = odds_data["under_threshold"]
    over_threshold = odds_data["over_threshold"]

    # Generate provably fair commitment
    pf_data = generate_round_commitment(result, under_threshold, over_threshold)

    # Create round data
    rd = RoundData(
        round_id=round_id,
        stream_name=stream_name,
        clip=clip,
        server_seed=pf_data["server_seed"],
        commitment_hash=pf_data["commitment_hash"],
        result=result,
        under_threshold=under_threshold,
        over_threshold=over_threshold,
        odds=odds_data["odds"],
        probabilities=odds_data["probabilities"],
        exact_odds=odds_data["exact_odds"],
        phase="BETTING",
        phase_start_time=time.time(),
    )

    return rd


def get_betting_phase_data(rd: RoundData) -> dict:
    """
    Data to expose to frontend during BETTING phase (15 sec).
    NO odds shown — only hash and 4 bet options.
    """
    return {
        "round_id": rd.round_id,
        "stream_name": rd.stream_name,
        "commitment_hash": rd.commitment_hash,
        "phase": "BETTING",
        "bet_options": ["UNDER", "RANGE", "OVER", "EXACT"],
        # NO odds, NO thresholds during betting
    }


def get_counting_phase_data(rd: RoundData, current_count: int, elapsed: float) -> dict:
    """
    Data to expose to frontend during COUNTING phase (35 sec).
    NOW show odds + thresholds.
    """
    return {
        "round_id": rd.round_id,
        "stream_name": rd.stream_name,
        "commitment_hash": rd.commitment_hash,
        "phase": "COUNTING",
        "current_count": current_count,
        "elapsed": round(elapsed, 1),
        "remaining": max(0, COUNTING_DURATION - elapsed),
        # Odds NOW visible
        "under_threshold": rd.under_threshold,
        "over_threshold": rd.over_threshold,
        "odds": rd.odds,
    }


def get_waiting_phase_data(rd: RoundData) -> dict:
    """
    Data to expose during WAITING phase (6 sec).
    Result + verification revealed.
    """
    rd.verification_data = get_verification_data({
        "server_seed": rd.server_seed,
        "result": rd.result,
        "under_threshold": rd.under_threshold,
        "over_threshold": rd.over_threshold,
        "commitment_hash": rd.commitment_hash,
    })

    rd.bet_outcomes = settle_bets(rd.result, rd.under_threshold, rd.over_threshold)

    return {
        "round_id": rd.round_id,
        "stream_name": rd.stream_name,
        "phase": "WAITING",
        "result": rd.result,
        "verification": rd.verification_data,
        "bet_outcomes": rd.bet_outcomes,
        "odds": rd.odds,
        "under_threshold": rd.under_threshold,
        "over_threshold": rd.over_threshold,
    }


def play_clip_with_overlay(clip: PreProcessedClip, frame_callback,
                           should_stop=None, count_duration: float = 35.0,
                           wait_duration: float = 6.0):
    """
    Play a pre-processed clip with detection overlays.
    NO YOLO runs during playback — uses stored detection data.

    Args:
        clip:           PreProcessedClip with detections
        frame_callback: Called for each frame: callback(frame, frame_no, current_count,
                            counting_active, elapsed_secs, detections_this_frame)
        should_stop:    Callable returning True to abort playback
        count_duration: Seconds of counting phase (35)
        wait_duration:  Seconds of waiting phase (6)

    The callback receives:
        frame:               raw OpenCV frame from clip
        frame_no:            current frame number
        current_count:       count up to this frame (0→1→3→5→7)
        counting_active:     True during first 35s, False in last 6s
        elapsed_secs:        seconds since clip started playing
        detections_this_frame: list of detection dicts for this frame
    """
    cap = cv2.VideoCapture(clip.clip_path)
    if not cap.isOpened():
        print(f"[PLAYBACK] ERROR: Cannot open clip: {clip.clip_path}")
        return

    fps = clip.clip_fps or cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_time = 1.0 / fps  # target time per frame
    max_count_frame = int(count_duration * fps)
    max_total_frame = int((count_duration + wait_duration) * fps)

    # Build frame→count lookup from counting_events
    # counting_events: [{frame: 115, ts: 4.6, count_at: 1}, ...]
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
        # Also check string keys (JSON deserialization may produce string keys)
        if not dets:
            dets = clip.detections.get(str(frame_no), [])

        # Call the frame callback (renderer will overlay everything)
        frame_callback(frame, frame_no, current_count, counting_active,
                       elapsed_secs, dets)

        frame_no += 1

        # Frame pacing — sleep to maintain exact fps
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
