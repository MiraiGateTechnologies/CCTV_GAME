"""
Per-stream historical count tracker for odds calculation.

Tracks past round results per stream to build probability distributions.
Used by odds_engine.py to calculate thresholds and payout multipliers.
"""

import json
import os
import threading
from collections import defaultdict

# Default file where history is persisted
DEFAULT_HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "round_history.json"
)

# How many past rounds to keep per stream for probability calculation
MAX_HISTORY_PER_STREAM = 500


class HistoryTracker:
    """
    Stores and retrieves per-stream vehicle count history.
    Thread-safe for use with background pipeline.
    """

    def __init__(self, history_file: str = None):
        self.history_file = history_file or DEFAULT_HISTORY_FILE
        self.lock = threading.Lock()
        # {stream_name: [count1, count2, count3, ...]}
        self.history: dict[str, list[int]] = defaultdict(list)
        self._load()

    def _load(self):
        """Load history from disk if exists."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                with self.lock:
                    for stream, counts in data.items():
                        self.history[stream] = counts[-MAX_HISTORY_PER_STREAM:]
            except Exception as e:
                print(f"[HISTORY] Warning: could not load history: {e}")

    def _save(self):
        """Persist history to disk."""
        try:
            os.makedirs(os.path.dirname(self.history_file) or ".", exist_ok=True)
            with self.lock:
                data = {k: v[-MAX_HISTORY_PER_STREAM:] for k, v in self.history.items()}
            with open(self.history_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[HISTORY] Warning: could not save history: {e}")

    def add_result(self, stream_name: str, count: int):
        """Record a round result for a stream."""
        with self.lock:
            self.history[stream_name].append(count)
            # Trim if over limit
            if len(self.history[stream_name]) > MAX_HISTORY_PER_STREAM:
                self.history[stream_name] = self.history[stream_name][-MAX_HISTORY_PER_STREAM:]
        self._save()

    def get_history(self, stream_name: str) -> list[int]:
        """Get all past counts for a stream."""
        with self.lock:
            return list(self.history.get(stream_name, []))

    def get_distribution(self, stream_name: str) -> dict[int, float]:
        """
        Get probability distribution for a stream.

        Returns: {count_value: probability}
            e.g., {0: 0.02, 1: 0.04, 2: 0.07, ...}
        """
        counts = self.get_history(stream_name)
        if not counts:
            return {}

        total = len(counts)
        freq: dict[int, int] = defaultdict(int)
        for c in counts:
            freq[c] += 1

        return {k: v / total for k, v in sorted(freq.items())}

    def get_median(self, stream_name: str) -> int:
        """Get median count for a stream. Returns 5 as default if no data."""
        counts = self.get_history(stream_name)
        if not counts:
            return 5  # sensible default for new streams
        sorted_counts = sorted(counts)
        mid = len(sorted_counts) // 2
        return sorted_counts[mid]

    def get_mean(self, stream_name: str) -> float:
        """Get mean count for a stream."""
        counts = self.get_history(stream_name)
        if not counts:
            return 5.0
        return sum(counts) / len(counts)

    def get_round_count(self, stream_name: str) -> int:
        """How many rounds of data we have for this stream."""
        with self.lock:
            return len(self.history.get(stream_name, []))

    def has_enough_data(self, stream_name: str, minimum: int = 20) -> bool:
        """Check if we have enough historical data for reliable odds."""
        return self.get_round_count(stream_name) >= minimum
