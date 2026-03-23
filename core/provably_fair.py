"""
Provably Fair System for CCTV Casino Betting Game.

Flow:
  1. Before betting: generate server_seed, compute commitment hash
  2. Publish hash to players (result hidden inside)
  3. After round: reveal server_seed + result
  4. Anyone can verify: SHA256(seed + result + thresholds) == hash
"""

import hashlib
import secrets
import json


def generate_server_seed() -> str:
    """Generate a cryptographically secure random 32-byte hex seed."""
    return secrets.token_hex(32)


def create_commitment(server_seed: str, result: int, under_threshold: int, over_threshold: int) -> str:
    """
    Create a SHA-256 commitment hash that locks in the result + thresholds.

    The commitment string format:
        server_seed:result:under_threshold:over_threshold

    This ensures the operator cannot change result OR thresholds after betting.
    """
    commitment_string = f"{server_seed}:{result}:{under_threshold}:{over_threshold}"
    return hashlib.sha256(commitment_string.encode("utf-8")).hexdigest()


def verify_commitment(server_seed: str, result: int, under_threshold: int, over_threshold: int, expected_hash: str) -> bool:
    """
    Verify that a commitment hash matches the revealed data.
    Players call this after the round to confirm fairness.
    """
    computed = create_commitment(server_seed, result, under_threshold, over_threshold)
    return computed == expected_hash


def generate_round_commitment(result: int, under_threshold: int, over_threshold: int) -> dict:
    """
    Generate complete provably fair data for one round.

    Returns:
        {
            "server_seed": "a7f3b2c9...",
            "commitment_hash": "e4d2f1a8...",
            "result": 7,
            "under_threshold": 4,
            "over_threshold": 7
        }
    """
    server_seed = generate_server_seed()
    commitment_hash = create_commitment(server_seed, result, under_threshold, over_threshold)

    return {
        "server_seed": server_seed,
        "commitment_hash": commitment_hash,
        "result": result,
        "under_threshold": under_threshold,
        "over_threshold": over_threshold,
    }


def get_verification_data(round_data: dict) -> dict:
    """
    Extract data that should be revealed to players AFTER the round.
    This is what players use to verify fairness.
    """
    return {
        "server_seed": round_data["server_seed"],
        "result": round_data["result"],
        "under_threshold": round_data["under_threshold"],
        "over_threshold": round_data["over_threshold"],
        "commitment_hash": round_data["commitment_hash"],
        "verification_string": f"{round_data['server_seed']}:{round_data['result']}:{round_data['under_threshold']}:{round_data['over_threshold']}",
    }
