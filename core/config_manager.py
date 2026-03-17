"""
JSON Configuration handling for CCTV Game.
"""
import json
import os
import sys

def load_streams_config(path):
    """Load and validate streams_config.json.
    
    Expected JSON structure:
    {
        "time_slots": [
            {
                "start": "06:00", "end": "12:00",
                "streams": [{"name": "...", "url": "..."}]
            }
        ],
        "count_duration": 35,
        "transition_duration": 15
    }
    
    Returns: (time_slots list, count_duration int, transition_duration int)
    """
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not read config {path}: {e}")
        sys.exit(1)

    slots = cfg.get("time_slots", [])
    if not slots:
        print("[ERROR] No time_slots found in config!")
        sys.exit(1)

    count_dur = cfg.get("count_duration", 35)
    trans_dur = cfg.get("transition_duration", 15)

    return slots, count_dur, trans_dur

def load_line_config(config_file):
    """Safely lead a line config JSON."""
    if not os.path.exists(config_file):
         return None
    try:
        with open(config_file) as f:
            return json.load(f)
    except Exception:
        return None

def save_line_config(config_file, data):
    """Safely save a line config JSON."""
    try:
        os.makedirs(os.path.dirname(config_file) or ".", exist_ok=True)
        with open(config_file, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception:
        return False
