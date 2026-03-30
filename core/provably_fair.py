"""
================================================================
  PROVABLY FAIR SYSTEM — Commitment Scheme for Real-World Events
================================================================

Mechanism: SHA-256 Commitment Scheme
  (NOT traditional seed→result derivation — result is a real-world measurement)

Why Commitment Scheme:
  - Result = vehicle count from CCTV = physical measurement
  - Cannot derive a physical count from HMAC
  - Commitment proves result was LOCKED before betting opened
  - Same mechanism used by live casino games (live dealer, crash games)

What gets committed (ALL round parameters):
  server_seed : random 256-bit hex — makes hash unpredictable
  result      : actual vehicle count
  under       : Under boundary number
  range_low   : Range lower boundary
  range_high  : Range upper boundary
  over        : Over boundary number
  exact_1     : First exact number
  exact_2     : Second exact number
  round_id    : Global round counter (nonce, prevents replay)

Flow:
  1. YOLO processes clip → result = vehicle count (pre-determined)
  2. Server generates cryptographic server_seed (256-bit)
  3. Odds engine generates boundaries from lambda_mean (deterministic)
  4. Commitment hash = SHA-256(server_seed:result:under:range_low:range_high:over:exact_1:exact_2:round_id)
  5. Commitment hash published to players BEFORE betting opens
  6. After round ends: reveal server_seed + all values
  7. Anyone can verify: recompute SHA-256 → must match published commitment

Security:
  - server_seed prevents brute-force guessing of result
    (vehicle count range 0-100 is tiny — without seed, SHA-256 is cracked in <1ms)
  - 256-bit server_seed → 2^256 possibilities → computationally impossible to brute-force
  - All game parameters in commitment → nothing can be changed post-betting
  - Boundaries derived from published formula + historical data → deterministic & auditable
  - secrets module (CSPRNG) for seed generation — never math.random()
"""

import hashlib
import secrets


# ═══════════════════════════════════════════════════════════════
#  SERVER SEED GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_server_seed() -> str:
    """
    Generate a cryptographically secure random 32-byte (256-bit) hex seed.

    Uses Python's `secrets` module which provides CSPRNG
    (Cryptographically Secure Pseudo-Random Number Generator).

    Returns:
        64-character hex string (32 bytes)
        e.g., "a7f3b2c9d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1"
    """
    return secrets.token_hex(32)


# ═══════════════════════════════════════════════════════════════
#  COMMITMENT HASH CREATION
# ═══════════════════════════════════════════════════════════════

def create_commitment(server_seed: str, result: int,
                      boundaries: dict, round_id: int) -> str:
    """
    Create SHA-256 commitment hash that locks ALL round parameters.

    Commitment string format:
        server_seed:result:under:range_low:range_high:over:exact_1:exact_2:round_id

    Every game parameter is included to prevent ANY post-betting manipulation:
      - server_seed:    Makes hash unpredictable (anti-brute-force)
      - result:         The actual vehicle count — core commitment
      - under:          Under boundary number
      - range_low:      Range lower boundary
      - range_high:     Range upper boundary
      - over:           Over boundary number
      - exact_1:        First exact number
      - exact_2:        Second exact number
      - round_id:       Prevents replay attacks across rounds

    Args:
        server_seed: 64-char hex string from generate_server_seed()
        result:      Vehicle count (integer)
        boundaries:  Dict with under, range_low, range_high, over, exact_1, exact_2
        round_id:    Global round counter (nonce)

    Returns:
        64-character hex SHA-256 hash string
    """
    commitment_string = (
        f"{server_seed}:{result}"
        f":{boundaries['under']}:{boundaries['range_low']}"
        f":{boundaries['range_high']}:{boundaries['over']}"
        f":{boundaries['exact_1']}:{boundaries['exact_2']}"
        f":{round_id}"
    )
    return hashlib.sha256(commitment_string.encode("utf-8")).hexdigest()


def _build_verification_string(server_seed: str, result: int,
                               boundaries: dict, round_id: int) -> str:
    """Build the raw string that gets hashed — used for verification display."""
    return (
        f"{server_seed}:{result}"
        f":{boundaries['under']}:{boundaries['range_low']}"
        f":{boundaries['range_high']}:{boundaries['over']}"
        f":{boundaries['exact_1']}:{boundaries['exact_2']}"
        f":{round_id}"
    )


# ═══════════════════════════════════════════════════════════════
#  VERIFICATION
# ═══════════════════════════════════════════════════════════════

def verify_commitment(server_seed: str, result: int,
                      boundaries: dict, round_id: int,
                      expected_hash: str) -> bool:
    """
    Verify that a commitment hash matches the revealed data.

    Players/auditors call this after the round to confirm fairness.
    Can be run client-side (JS), server-side (Python), or via third-party tools.

    Uses constant-time comparison (secrets.compare_digest) to prevent
    timing attacks — standard practice in cryptographic verification.

    Returns:
        True if hash matches → round was provably fair
        False if mismatch → potential manipulation (should NEVER happen)
    """
    computed = create_commitment(server_seed, result, boundaries, round_id)
    return secrets.compare_digest(computed, expected_hash)


# ═══════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT — Generate complete PF data for one round
# ═══════════════════════════════════════════════════════════════

def generate_round_fairness(result: int, round_id: int,
                            boundaries: dict) -> dict:
    """
    Generate complete provably fair data for one round.

    Called ONCE during prepare_round(), BEFORE betting opens.
    All values are locked and immutable for the entire round.

    Args:
        result:      Vehicle count (from YOLO pre-processing)
        round_id:    Global round counter (nonce)
        boundaries:  Dict from odds_engine.generate_boundaries()

    Returns:
        {
            "server_seed": "a7f3b2c9...",
            "commitment_hash": "e4d2f1a8...",
            "round_id": 42,
            "result": 7,
            "boundaries": {...},
        }
    """
    # Step 1: Generate cryptographic seed
    server_seed = generate_server_seed()

    # Step 2: Create commitment hash locking ALL parameters
    commitment_hash = create_commitment(
        server_seed, result, boundaries, round_id,
    )

    return {
        "server_seed": server_seed,
        "commitment_hash": commitment_hash,
        "round_id": round_id,
        "result": result,
        "boundaries": boundaries,
    }


def get_verification_data(pf_data: dict) -> dict:
    """
    Extract data to reveal to players AFTER the round ends.

    This is everything needed to independently verify the commitment:
      1. Take the verification_string
      2. Compute SHA-256(verification_string)
      3. Compare with commitment_hash published before betting
      4. If they match → round was provably fair

    Returns:
        Complete verification package with instructions.
    """
    server_seed = pf_data["server_seed"]
    result = pf_data["result"]
    boundaries = pf_data["boundaries"]
    round_id = pf_data["round_id"]

    verification_string = _build_verification_string(
        server_seed, result, boundaries, round_id,
    )

    return {
        "server_seed": server_seed,
        "result": result,
        "boundaries": boundaries,
        "round_id": round_id,
        "commitment_hash": pf_data["commitment_hash"],
        "verification_string": verification_string,
        "algorithm": "SHA-256",
        "how_to_verify": (
            "Compute SHA-256 of the verification_string. "
            "The result must match the commitment_hash "
            "that was published before betting opened."
        ),
    }
