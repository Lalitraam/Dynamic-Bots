"""
Classifier service for Lichess time control strings.

Provides parse_time_control, meets_min_time_control, and classify_time_control.
"""

from .. import config

# ---------------------------------------------------------------------------
# parse_time_control
# ---------------------------------------------------------------------------

def parse_time_control(tc: str) -> tuple[int, int] | None:
    """
    Parse a Lichess time control string into (base_seconds, increment_seconds).

    Accepts:
      - "180+2"  → (180, 2)   (already in seconds)
      - "600"    → (600, 0)   (no increment)
      - "60+0"   → (60, 0)    (already in seconds)
      - "1+0"    → (1, 0)     (literal 1 second base)
      - "3+2"    → (3, 2)     (literal 3 seconds base)

    Returns None for "-", "?", empty, or garbage.
    """
    if not tc or tc.strip() in ("-", "?"):
        return None

    tc = tc.strip()

    if "+" in tc:
        parts = tc.split("+")
        if len(parts) != 2:
            return None
        try:
            base = int(parts[0])
            inc = int(parts[1])
            return base, inc
        except ValueError:
            return None
    else:
        try:
            base = int(tc)
            return base, 0
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# meets_min_time_control
# ---------------------------------------------------------------------------

def meets_min_time_control(tc: str) -> bool:
    """
    Return True only if the time control parses and
    base_seconds >= config.MIN_BASE_SECONDS.
    """
    parsed = parse_time_control(tc)
    if parsed is None:
        return False
    base, _ = parsed
    return base >= config.MIN_BASE_SECONDS


# ---------------------------------------------------------------------------
# classify_time_control
# ---------------------------------------------------------------------------

def classify_time_control(tc: str) -> str:
    """
    Return the accepted category name for tc, or "other" if:
      - the string is unparseable
      - the derived category is not in config.INCLUDED_CATEGORIES

    Uses Lichess's rule: estimated = base + 40 * increment.
    Thresholds:
      < 30   -> "ultrabullet"
      < 180  -> "bullet"
      < 480  -> "blitz"
      < 1500 -> "rapid"
      else   -> "classical"
    """
    parsed = parse_time_control(tc)
    if parsed is None:
        return "other"

    base, inc = parsed
    estimated = base + 40 * inc

    if estimated < 30:
        cat = "ultrabullet"
    elif estimated < 180:
        cat = "bullet"
    elif estimated < 480:
        cat = "blitz"
    elif estimated < 1500:
        cat = "rapid"
    else:
        cat = "classical"

    if cat not in config.INCLUDED_CATEGORIES:
        return "other"

    return cat
