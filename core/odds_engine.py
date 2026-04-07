"""
================================================================
  ODDS ENGINE — CCTV Casino Betting Game
================================================================

Generates the round's betting boundaries from the CURRENT round's
vehicle count (result). The result randomly falls into any category
(Under/Range/Over/Exact) due to a random offset.

Terminology:
  "Odds" = boundary NUMBERS for each bet option (change every round)
      Under < 17  |  Range 18-20  |  Over > 21  |  Exact 16 or 22

  "Multipliers" = payout multipliers (CONSTANT, same every round)
      Under: 3.03x  |  Range: 2.27x  |  Over: 3.63x  |  Exact: 18.18x

How it works:
  1. Take the actual vehicle count (result) from YOLO
  2. Add a random offset (-4 to +4) to create a "center" point
  3. Build boundaries around that center
  4. Result randomly lands in Under, Range, Over, Exact, or gap zone
"""

import random

from game.history_tracker import HistoryTracker


# ═══════════════════════════════════════════════════════════════
#  CONSTANT MULTIPLIERS & WIN CHANCES (never change per round)
# ═══════════════════════════════════════════════════════════════

MULTIPLIERS = {
    "under": 3.03,
    "range": 2.27,
    "over":  3.63,
    "exact": 18.18,
}

WIN_CHANCES = {
    "under": 30,
    "range": 40,
    "over":  25,
    "exact": 5,
}


# ═══════════════════════════════════════════════════════════════
#  ODDS (BOUNDARY) GENERATION — Random offset from result
# ═══════════════════════════════════════════════════════════════

def generate_boundaries(result: int) -> dict:
    """
    Generate betting boundaries from the current round's result.

    Uses a random offset so the result can land in ANY category:
      - Under, Range, Over, Exact, or gap (house wins)

    The offset shifts the center away from the result, so players
    cannot predict which category will win just by watching the count.

    Args:
        result: Actual vehicle count for this round (from YOLO)

    Returns:
        {
            "under": 16,           # count <= under → Under wins
            "range_low": 18,       # range_low <= count <= range_high → Range wins
            "range_high": 20,
            "over": 22,            # count >= over → Over wins
            "exact_1": 16,         # count == exact_1 or exact_2 → Exact wins
            "exact_2": 22,
        }
    """
    # Random offset — result randomly lands in any category
    offset = random.randint(-4, 4)
    center = result + offset

    # Build boundaries around the shifted center (exact original logic)
    under_max  = center - 3
    exact_low  = center - 2
    range_low  = center - 1
    range_high = center + 1
    exact_high = center + 2
    over_min   = center + 3

    return {
        "under": under_max,
        "range_low": range_low,
        "range_high": range_high,
        "over": over_min,
        "exact_1": exact_low,
        "exact_2": exact_high,
    }


# ═══════════════════════════════════════════════════════════════
#  BET SETTLEMENT — Determine winners after round
# ═══════════════════════════════════════════════════════════════

def settle_bets(result: int, boundaries: dict) -> dict:
    """
    Determine which bet types won for a given result.

    Rules:
      Under: result <= under               → Under wins
      Range: range_low <= result <= range_high  → Range wins
      Over:  result >= over                 → Over wins
      Exact: result == exact_1 or exact_2   → Exact wins
      Gap:   anything else                  → House wins all main bets

    Args:
        result:     Final vehicle count
        boundaries: Dict from generate_boundaries()

    Returns:
        {"under": True/False, "range": True/False,
         "over": True/False, "exact": True/False,
         "winning_option": "RANGE" or "NONE" etc.}
    """
    # Exact check FIRST (highest priority — matches your original logic)
    exact_win = result in (boundaries["exact_1"], boundaries["exact_2"])
    under_win = result <= boundaries["under"]
    over_win  = result >= boundaries["over"]
    range_win = boundaries["range_low"] <= result <= boundaries["range_high"]

    # Winner priority: Exact → Under → Over → Range (fallback)
    if exact_win:
        winning_option = "EXACT"
    elif under_win:
        winning_option = "UNDER"
    elif over_win:
        winning_option = "OVER"
    elif range_win:
        winning_option = "RANGE"
    else:
        winning_option = "RANGE"  # Fallback — matches your original logic

    return {
        "under": under_win,
        "range": range_win,
        "over": over_win,
        "exact": exact_win,
        "winning_option": winning_option,
    }


# ═══════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT — Generate complete odds for a round
# ═══════════════════════════════════════════════════════════════

def generate_round_odds(result: int, history: HistoryTracker = None,
                        stream_name: str = "") -> dict:
    """
    Generate complete odds data for a round.

    This is the SINGLE ENTRY POINT called by round_manager.prepare_round().
    Uses the CURRENT round's result (not historical average) to generate
    boundaries with a random offset.

    Args:
        result:         Actual vehicle count for this round
        history:        HistoryTracker instance (kept for API compat, used for stats only)
        stream_name:    Current stream identifier (used for stats only)

    Returns:
        {
            "boundaries": {
                "under": 16, "range_low": 18, "range_high": 20,
                "over": 22, "exact_1": 16, "exact_2": 22
            },
            "multipliers": {"under": 3.03, "range": 2.27, "over": 3.63, "exact": 18.18},
            "win_chances": {"under": 30, "range": 40, "over": 25, "exact": 5},
            "result_used": 19,
            "rounds_tracked": 250,
        }
    """
    rounds_tracked = 0
    if history is not None and stream_name:
        rounds_tracked = history.get_round_count(stream_name)

    # Generate boundaries from CURRENT result (not historical mean)
    boundaries = generate_boundaries(result)

    return {
        "boundaries": boundaries,
        "multipliers": MULTIPLIERS,
        "win_chances": WIN_CHANCES,
        "result_used": result,
        "rounds_tracked": rounds_tracked,
    }
