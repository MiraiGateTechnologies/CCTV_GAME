# Smart Stream Scheduler — Plan Document

## Overview

Intelligent stream selection system that:
1. Auto-blocks failing/timeout streams for 30 min (with exponential backoff)
2. Prioritizes high-traffic streams (count > 10) for more frequent play
3. Maintains stream health metrics for monitoring

## Why Not ML

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| Deep ML (neural net) | Learns patterns | Overkill, needs training data, maintenance | NO |
| Reinforcement Learning | Adapts over time | Complex, unstable, slow convergence | NO |
| Rule-based scoring | Simple, fast, predictable, debuggable | Less "smart" | YES |
| Multi-armed bandit | Good for exploration/exploitation | Unnecessary complexity | MAYBE later |

**Decision: Rule-based scoring with weighted selection.**
Simple, predictable, zero training, works immediately.

---

## Architecture

```
stream_manager.py
    |
    v
StreamHealthTracker (NEW)          HistoryTracker (EXISTING)
    |                                   |
    | health scores                     | avg vehicle counts
    v                                   v
StreamPriorityScheduler (NEW)
    |
    | picks next stream(s) based on health + priority
    v
_download_loop() (EXISTING — uses scheduler instead of simple round-robin)
```

## Component 1: Stream Health Tracker

### Data Structure (per stream)

```python
stream_health = {
    "Stream 7": {
        "status": "active",           # active / blocked / cooldown
        "timeout_count": 0,           # timeouts in current window
        "failure_count": 0,           # consecutive failures
        "success_count": 15,          # total successful downloads
        "last_success": 1712000000,   # timestamp
        "last_failure": 0,            # timestamp
        "blocked_until": 0,           # timestamp (0 = not blocked)
        "block_duration": 1800,       # current block duration (30 min default)
        "total_timeouts": 2,          # lifetime timeout count
        "avg_download_time": 48.5,    # seconds
    }
}
```

### Rules

```
RULE 1: Timeout Tracking
  On timeout:
    timeout_count += 1
    failure_count += 1

  On success:
    failure_count = 0
    timeout_count = max(0, timeout_count - 1)  # slowly recover

RULE 2: Auto-Block
  IF timeout_count >= 3 in last 60 min:
    status = "blocked"
    blocked_until = now + block_duration
    block_duration = min(block_duration * 2, 7200)  # max 2 hours
    Log: "[HEALTH] Stream X BLOCKED for {block_duration/60} min (3 timeouts)"

RULE 3: Auto-Unblock
  IF now > blocked_until:
    status = "cooldown"  (try once, if success → active, if fail → re-block)
    timeout_count = 0
    Log: "[HEALTH] Stream X UNBLOCKED — testing..."

RULE 4: Cooldown Success
  IF status == "cooldown" AND download succeeds:
    status = "active"
    block_duration = 1800  # reset to 30 min
    Log: "[HEALTH] Stream X RECOVERED"

RULE 5: Cooldown Failure
  IF status == "cooldown" AND download fails:
    status = "blocked"
    blocked_until = now + block_duration
    block_duration = min(block_duration * 2, 7200)
    Log: "[HEALTH] Stream X RE-BLOCKED for {block_duration/60} min"
```

## Component 2: Stream Priority Scorer

### Scoring Formula

```
priority_score = base_score * health_multiplier * traffic_multiplier

WHERE:
  base_score = 1.0 (all streams start equal)

  health_multiplier:
    active:   1.0
    cooldown: 0.5 (try less often)
    blocked:  0.0 (never picked)

  traffic_multiplier (from HistoryTracker avg count):
    avg_count > 15:  3.0  (HIGH traffic — pick 3x more)
    avg_count > 10:  2.0  (GOOD traffic — pick 2x more)
    avg_count >= 5:  1.0  (NORMAL traffic)
    avg_count < 5:   0.5  (LOW traffic — pick less)
    no history:      1.0  (NEW stream — normal priority)
```

### Example

```
Stream 7:  avg_count=23, active   → score = 1.0 * 1.0 * 3.0 = 3.0
Stream 16: avg_count=12, active   → score = 1.0 * 1.0 * 2.0 = 2.0
Stream 19: avg_count=7,  active   → score = 1.0 * 1.0 * 1.0 = 1.0
Stream 28: avg_count=3,  blocked  → score = 1.0 * 0.0 * 0.5 = 0.0 (NEVER)
Stream 29: avg_count=15, cooldown → score = 1.0 * 0.5 * 3.0 = 1.5

Selection probability (normalized):
  Stream 7:  3.0 / 7.5 = 40%  ← picked most often
  Stream 16: 2.0 / 7.5 = 27%
  Stream 19: 1.0 / 7.5 = 13%
  Stream 29: 1.5 / 7.5 = 20%
  Stream 28: 0.0 / 7.5 =  0%  ← never picked (blocked)
```

## Component 3: Weighted Stream Selection

### How _download_loop() Changes

```
CURRENT (simple round-robin):
  index = 0
  stream = self.streams[index % len(self.streams)]
  index += 1
  → Equal chance for all streams, no health check

NEW (weighted selection):
  scores = calculate_all_scores()
  stream = weighted_random_pick(scores)
  → High-count streams picked more, blocked streams skipped
```

### Weighted Random Pick Algorithm

```python
import random

def pick_next_stream(streams, health_tracker, history_tracker):
    """Pick next stream using weighted random selection."""
    weights = []
    eligible = []

    for stream in streams:
        name = stream["name"]
        health = health_tracker.get_health(name)

        # Blocked streams get weight 0
        if health["status"] == "blocked" and time.time() < health["blocked_until"]:
            continue

        # Calculate weight
        avg_count = history_tracker.get_mean(name)

        if avg_count > 15:
            traffic_mult = 3.0
        elif avg_count > 10:
            traffic_mult = 2.0
        elif avg_count >= 5:
            traffic_mult = 1.0
        else:
            traffic_mult = 0.5

        health_mult = 0.5 if health["status"] == "cooldown" else 1.0

        weight = traffic_mult * health_mult
        weights.append(weight)
        eligible.append(stream)

    if not eligible:
        return None  # all streams blocked!

    # Weighted random selection
    return random.choices(eligible, weights=weights, k=1)[0]
```

## Implementation Plan

### Files to Create/Modify

```
NEW:    core/stream_health.py     — StreamHealthTracker class
MODIFY: network/stream_manager.py — _download_loop() uses weighted selection
MODIFY: network/stream_manager.py — report_timeout/success to health tracker
```

### stream_health.py — Class Design

```python
class StreamHealthTracker:
    def __init__(self):
        self._health = {}  # {stream_name: health_dict}

    def report_success(self, name, download_time):
        """Called after successful download."""

    def report_failure(self, name, error_type="generic"):
        """Called after failed download. error_type: "timeout" / "generic" / "429"."""

    def report_timeout(self, name):
        """Called specifically on download timeout."""

    def is_blocked(self, name) -> bool:
        """Check if stream is currently blocked."""

    def get_health(self, name) -> dict:
        """Get full health status for a stream."""

    def get_all_health(self) -> dict:
        """Get health status for all streams (for API/monitoring)."""

    def pick_stream(self, streams, history_tracker) -> dict | None:
        """Weighted random selection considering health + traffic priority."""
```

### API Endpoint (monitoring)

```
GET /api/stream_health

Response:
{
  "streams": [
    {
      "name": "Stream 7",
      "status": "active",
      "avg_count": 23,
      "priority_score": 3.0,
      "timeouts_last_hour": 0,
      "blocked_until": null,
      "avg_download_time": 48.5
    },
    {
      "name": "Stream 28",
      "status": "blocked",
      "avg_count": 3,
      "priority_score": 0.0,
      "timeouts_last_hour": 4,
      "blocked_until": "2026-04-07T15:30:00",
      "avg_download_time": null
    }
  ]
}
```

## Testing Strategy

```
TEST 1: Timeout blocking
  - Force 3 timeouts on Stream X
  - Verify: Stream X blocked for 30 min
  - After 30 min: verify unblock + retry

TEST 2: Priority selection
  - Run 100 picks with known avg_counts
  - Verify: high-count streams picked ~3x more
  - Verify: blocked streams NEVER picked

TEST 3: Exponential backoff
  - Block → unblock → fail again
  - Verify: block duration doubles (30→60→120 min)
  - Verify: max 2 hours

TEST 4: Recovery
  - Blocked stream unblocks → success
  - Verify: status → active, block_duration reset to 30 min
```

## Timeline

```
STEP 1: Create core/stream_health.py          — 1-2 hours
STEP 2: Modify _download_loop() for weighted   — 1 hour
STEP 3: Add /api/stream_health endpoint        — 30 min
STEP 4: Test with real streams                  — 1 hour
TOTAL: ~4-5 hours
```
