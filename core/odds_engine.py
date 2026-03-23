"""
Odds Engine for CCTV Casino Betting Game.

Calculates thresholds and payout multipliers based on
per-stream historical vehicle count distributions.

Bet types:
  1. Under  — count < under_threshold
  2. Range  — under_threshold <= count <= over_threshold
  3. Over   — count > over_threshold
  4. Exact  — count == player's chosen number
"""

from game.history_tracker import HistoryTracker

# House edge — the casino's profit margin (5%)
HOUSE_EDGE = 0.05

# Minimum odds cap (never pay less than this)
MIN_ODDS = 1.10

# Maximum odds cap (prevent absurd payouts on ultra-rare outcomes)
MAX_ODDS = 100.0

# Default odds when not enough historical data (< 20 rounds)
DEFAULT_ODDS = {
    "under": 3.50,
    "range": 1.80,
    "over": 3.50,
}

# Default thresholds when no history
DEFAULT_UNDER_THRESHOLD = 4
DEFAULT_OVER_THRESHOLD = 7


def calculate_thresholds(history: HistoryTracker, stream_name: str) -> tuple[int, int]:
    """
    Calculate Under/Over thresholds from historical median.

    Returns: (under_threshold, over_threshold)
        Under bet wins if count < under_threshold
        Over bet wins if count > over_threshold
        Range bet wins if under_threshold <= count <= over_threshold
    """
    if not history.has_enough_data(stream_name):
        return DEFAULT_UNDER_THRESHOLD, DEFAULT_OVER_THRESHOLD

    median = history.get_median(stream_name)

    # Thresholds centered around median
    # Ensures all 4 bet types have meaningful probability
    under_threshold = max(2, median - 1)
    over_threshold = max(under_threshold + 2, median + 2)

    return under_threshold, over_threshold


def calculate_probability(history: HistoryTracker, stream_name: str,
                          under_threshold: int, over_threshold: int) -> dict:
    """
    Calculate probability of each bet type winning based on historical data.

    Returns: {"under": 0.236, "range": 0.570, "over": 0.194}
    """
    distribution = history.get_distribution(stream_name)

    if not distribution:
        # No data — return equal-ish probabilities
        return {"under": 0.25, "range": 0.45, "over": 0.30}

    p_under = sum(prob for count, prob in distribution.items() if count < under_threshold)
    p_over = sum(prob for count, prob in distribution.items() if count > over_threshold)
    p_range = sum(prob for count, prob in distribution.items()
                  if under_threshold <= count <= over_threshold)

    # Ensure no zero probabilities (add small epsilon)
    eps = 0.005
    p_under = max(eps, p_under)
    p_over = max(eps, p_over)
    p_range = max(eps, p_range)

    # Normalize to sum to 1
    total = p_under + p_range + p_over
    p_under /= total
    p_range /= total
    p_over /= total

    return {"under": p_under, "range": p_range, "over": p_over}


def probability_to_odds(probability: float) -> float:
    """Convert probability to payout multiplier with house edge."""
    if probability <= 0:
        return MAX_ODDS

    fair_odds = 1.0 / probability
    payout = fair_odds * (1.0 - HOUSE_EDGE)

    return max(MIN_ODDS, min(MAX_ODDS, round(payout, 2)))


def calculate_exact_odds(history: HistoryTracker, stream_name: str, number: int) -> float:
    """
    Calculate odds for an Exact bet (count == number).
    """
    distribution = history.get_distribution(stream_name)

    if not distribution:
        # Default: assume ~10% chance for any single number
        return probability_to_odds(0.10)

    prob = distribution.get(number, 0.0)

    # Minimum probability for exact (very rare outcomes still payable)
    prob = max(0.005, prob)

    return probability_to_odds(prob)


def generate_round_odds(history: HistoryTracker, stream_name: str) -> dict:
    """
    Generate complete odds data for a round.

    Returns:
        {
            "under_threshold": 4,
            "over_threshold": 7,
            "odds": {
                "under": 4.03,
                "range": 1.66,
                "over": 4.89
            },
            "probabilities": {
                "under": 0.236,
                "range": 0.570,
                "over": 0.194
            },
            "exact_odds": {0: 59.37, 1: 23.75, ..., 15: 47.50},
            "has_enough_data": True,
            "rounds_tracked": 500
        }
    """
    under_threshold, over_threshold = calculate_thresholds(history, stream_name)
    probabilities = calculate_probability(history, stream_name, under_threshold, over_threshold)

    odds = {
        "under": probability_to_odds(probabilities["under"]),
        "range": probability_to_odds(probabilities["range"]),
        "over": probability_to_odds(probabilities["over"]),
    }

    # Calculate exact odds for common count values (0-30)
    exact_odds = {}
    for n in range(31):
        exact_odds[n] = calculate_exact_odds(history, stream_name, n)

    has_data = history.has_enough_data(stream_name)
    if not has_data:
        odds = dict(DEFAULT_ODDS)

    return {
        "under_threshold": under_threshold,
        "over_threshold": over_threshold,
        "odds": odds,
        "probabilities": probabilities,
        "exact_odds": exact_odds,
        "has_enough_data": has_data,
        "rounds_tracked": history.get_round_count(stream_name),
    }


def settle_bets(result: int, under_threshold: int, over_threshold: int) -> dict:
    """
    Determine which bet types won for a given result.

    Returns: {"under": False, "range": True, "over": False, "exact": {7: True, 5: False, ...}}
    """
    return {
        "under": result < under_threshold,
        "range": under_threshold <= result <= over_threshold,
        "over": result > over_threshold,
    }
