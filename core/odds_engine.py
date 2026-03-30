"""
================================================================
  ODDS ENGINE — CCTV Casino Betting Game
================================================================

Generates the round's betting numbers (odds) for all four options
based on the historical mean vehicle count (lambda_mean) per stream.

Terminology in THIS project:
  "Odds" = the NUMBERS/THRESHOLDS for each bet option (change every round)
      Under < 14  |  Range 16-18  |  Over > 22  |  Exact 20 or 21

  "Multipliers" = payout multipliers (CONSTANT, same every round)
      Under: 3.13x  |  Range: 2.35x  |  Over: 3.76x  |  Exact: 18.8x

  "Win Chances" = display percentages (CONSTANT, same every round)
      Under: 30%  |  Range: 40%  |  Over: 25%  |  Exact: 5%

Odds Formula (lambda_mean-based boundary generation):
  under         = floor(lambda_mean * 0.75 * adjust_factor) - 1
  range_low     = round(lambda_mean - 0.5)
  range_high    = round(lambda_mean + 0.8)
  over          = ceil(lambda_mean * 1.35 * adjust_factor) + 1
  exact_number1 = round(lambda_mean)
  exact_number2 = exact_number1 + 1

  lambda_mean = historical average vehicle count for this stream
  adjust_factor = tunable parameter (default 1.0)

The gaps between Under/Range/Over boundaries are INTENTIONAL —
counts falling in gap zones mean all main bets lose (house wins).
"""

import math
import numpy as np
from game.history_tracker import HistoryTracker


# ═══════════════════════════════════════════════════════════════
#  CONSTANT MULTIPLIERS & WIN CHANCES (never change per round)
# ═══════════════════════════════════════════════════════════════

MULTIPLIERS = {
    "under": 3.13,
    "range": 2.35,
    "over":  3.76,
    "exact": 18.8,
}

WIN_CHANCES = {
    "under": 30,
    "range": 40,
    "over":  25,
    "exact": 5,
}

# Minimum historical rounds before using lambda_mean-based odds
MIN_DATA_ROUNDS = 20

# Default lambda_mean when no history exists
DEFAULT_LAMBDA_MEAN = 10.0

# Adjustment factor for boundary tuning (1.0 = standard)
DEFAULT_ADJUST_FACTOR = 1.0


# ═══════════════════════════════════════════════════════════════
#  ODDS (BOUNDARY) GENERATION — Core formula
# ═══════════════════════════════════════════════════════════════

def generate_boundaries(lambda_mean: float,
                        adjust_factor: float = DEFAULT_ADJUST_FACTOR) -> dict:
    """
    Generate betting boundaries (odds) for all four options from lambda_mean.

    Uses lambda_mean-based boundary generation formula:
      under       = floor(lambda_mean * 0.75 * adjust_factor) - 1
      range_low   = round(lambda_mean - 0.5)
      range_high  = round(lambda_mean + 0.8)
      over        = ceil(lambda_mean * 1.35 * adjust_factor) + 1
      exact_num1  = round(lambda_mean)
      exact_num2  = exact_num1 + 1

    Args:
        lambda_mean:    Historical average vehicle count for the stream
        adjust_factor:  Tunable multiplier for Under/Over boundaries (default 1.0)

    Returns:
        {
            "under": 14,           # count < 14 → Under wins
            "range_low": 20,       # range_low <= count <= range_high → Range wins
            "range_high": 21,
            "over": 28,            # count > 28 → Over wins
            "exact_1": 20,         # count == 20 or count == 21 → Exact wins
            "exact_2": 21,
        }
    """
    # Ensure lambda_mean is positive and reasonable
    lambda_mean = max(1.0, float(lambda_mean))

    under = int(np.floor(lambda_mean * 0.75 * adjust_factor) - 1)
    range_low = int(np.round(lambda_mean - 0.5))
    range_high = int(np.round(lambda_mean + 0.8))
    over = int(np.ceil(lambda_mean * 1.35 * adjust_factor) + 1)
    exact_1 = int(np.round(lambda_mean))
    exact_2 = exact_1 + 1

    # Safety clamps: under must be >= 0, range must be valid
    under = max(0, under)
    range_low = max(under + 1, range_low)  # range must be above under
    range_high = max(range_low, range_high)  # range_high >= range_low
    over = max(range_high + 1, over)  # over must be above range
    exact_1 = max(0, exact_1)
    exact_2 = max(exact_1 + 1, exact_2)

    return {
        "under": under,
        "range_low": range_low,
        "range_high": range_high,
        "over": over,
        "exact_1": exact_1,
        "exact_2": exact_2,
    }


# ═══════════════════════════════════════════════════════════════
#  BET SETTLEMENT — Determine winners after round
# ═══════════════════════════════════════════════════════════════

def settle_bets(result: int, boundaries: dict) -> dict:
    """
    Determine which bet types won for a given result.

    Rules:
      Under: result < under                → Under wins
      Range: range_low <= result <= range_high  → Range wins
      Over:  result > over                  → Over wins
      Exact: result == exact_1 or exact_2   → Exact wins

    Gaps between boundaries are intentional — if result falls in a gap,
    ALL main bets (Under/Range/Over) LOSE. Only Exact can potentially
    still win if the result matches one of the exact numbers.

    Args:
        result:     Final vehicle count
        boundaries: Dict from generate_boundaries()

    Returns:
        {"under": True/False, "range": True/False,
         "over": True/False, "exact": True/False,
         "winning_option": "RANGE" or "NONE" etc.}
    """
    under_win = result < boundaries["under"]
    range_win = boundaries["range_low"] <= result <= boundaries["range_high"]
    over_win = result > boundaries["over"]
    exact_win = result in (boundaries["exact_1"], boundaries["exact_2"])

    # Determine which main option won (if any)
    if under_win:
        winning_option = "UNDER"
    elif range_win:
        winning_option = "RANGE"
    elif over_win:
        winning_option = "OVER"
    else:
        winning_option = "NONE"  # Result fell in a gap — house wins

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

def generate_round_odds(history: HistoryTracker, stream_name: str,
                        adjust_factor: float = DEFAULT_ADJUST_FACTOR) -> dict:
    """
    Generate complete odds data for a round.

    This is the SINGLE ENTRY POINT called by round_manager.prepare_round().
    Everything needed for betting is calculated here in one call.

    Args:
        history:        HistoryTracker instance with per-stream count data
        stream_name:    Current stream identifier (e.g., "Stream 16")
        adjust_factor:  Boundary tuning parameter (default 1.0)

    Returns:
        {
            "boundaries": {
                "under": 14, "range_low": 20, "range_high": 21,
                "over": 28, "exact_1": 20, "exact_2": 21
            },
            "multipliers": {"under": 3.13, "range": 2.35, "over": 3.76, "exact": 18.8},
            "win_chances": {"under": 30, "range": 40, "over": 25, "exact": 5},
            "lambda_mean": 20.0,
            "has_enough_data": True,
            "rounds_tracked": 250,
        }
    """
    has_data = history.has_enough_data(stream_name, minimum=MIN_DATA_ROUNDS)
    rounds_tracked = history.get_round_count(stream_name)

    # Get lambda_mean from historical data
    if has_data:
        lambda_mean = history.get_mean(stream_name)
    else:
        lambda_mean = DEFAULT_LAMBDA_MEAN

    # Generate boundaries using the formula
    boundaries = generate_boundaries(lambda_mean, adjust_factor)

    return {
        "boundaries": boundaries,
        "multipliers": MULTIPLIERS,
        "win_chances": WIN_CHANCES,
        "lambda_mean": round(lambda_mean, 2),
        "has_enough_data": has_data,
        "rounds_tracked": rounds_tracked,
    }
