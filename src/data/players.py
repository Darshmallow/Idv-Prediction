"""
src/players.py
==============
Player ID normalization for IDV esports data (JSON-data version).

Player alias source: data/raw_json/id.json
  Structure: {canonical_id: [canonical_id, alias_1, alias_2, ...]}
  Provides ~96 alias mappings (much more comprehensive than what we could
  derive from the legacy `曾用id` Excel tab).

Algorithm
---------
1. Load id.json → flat alias_map: {alias_id → canonical_id}
2. Merge MANUAL_OVERRIDES (corrections + post-id.json drift) on top
3. normalize_player_id() lowercases + strips, then applies ALIAS_MAP

Public API
----------
    ALIAS_MAP: dict[str, str]
        Flat old_id → canonical_id mapping (id.json + manual overrides).

    normalize_player_id(raw_id: str) -> str
        Lowercase/strip + alias lookup.  Returns the cleaned ID unchanged
        if not in the map.

    normalize_player_ids(df: pd.DataFrame) -> pd.DataFrame
        Applies normalize_player_id to all five player ID columns.
        Returns a copy; null cells stay null.

Adding a manual override
------------------------
Edit MANUAL_OVERRIDES below.  Both keys and values should be lowercase;
the *right-hand side* is the canonical name kept in the database.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ingest_json import load_id_alias_map


_HERE    = Path(__file__).parent
_ROOT    = _HERE.parent.parent

# ---------------------------------------------------------------------------
# ✏️  MANUAL OVERRIDES — corrections layered on top of id.json
# ---------------------------------------------------------------------------
# Format: "old_id": "canonical_id"   (both lowercase)
# Used for:
#   • post-id.json ID changes not yet reflected in the JSON
#   • suspected typo-duplicates uncovered via fuzzy matching
#
MANUAL_OVERRIDES: dict[str, str] = {
    # Suspected typo-duplicates — uncomment / extend as confirmed
    "ppicha": "pipicha",
    "gua":    "guag",
}

# ---------------------------------------------------------------------------
# Player columns to normalise in a DataFrame
# ---------------------------------------------------------------------------
PLAYER_COLUMNS = [
    "hunter_player",
    "survivor1_player",
    "survivor2_player",
    "survivor3_player",
    "survivor4_player",
]


# ---------------------------------------------------------------------------
# Build the alias map at import time
# ---------------------------------------------------------------------------

def _build_alias_map(manual_overrides: dict[str, str]) -> dict[str, str]:
    """
    Build the flat alias → canonical mapping from id.json, then layer
    manual overrides on top.

    When an override declares X → Y, Y becomes canonical, so any existing
    entry pointing *to* X is rewritten (or removed if X was previously
    canonical itself).
    """
    alias_map = load_id_alias_map()       # from id.json

    for old, canon in manual_overrides.items():
        old   = old.lower().strip()
        canon = canon.lower().strip()
        # If `canon` is currently mapped elsewhere, redirect it to itself
        # (i.e. make `canon` the canonical for its own group).
        alias_map.pop(canon, None)
        # If anything was pointing at `old`, redirect it to `canon` instead.
        for k, v in list(alias_map.items()):
            if v == old:
                alias_map[k] = canon
        alias_map[old] = canon
    return alias_map


ALIAS_MAP: dict[str, str] = _build_alias_map(MANUAL_OVERRIDES)


# ---------------------------------------------------------------------------
# Public normalization functions
# ---------------------------------------------------------------------------

def normalize_player_id(raw_id: str) -> str:
    """
    Normalize a single player ID:
      1. Lowercase + strip whitespace
      2. Look up in ALIAS_MAP; return canonical if found, else the cleaned ID

    Returns empty string for null-like inputs ("", "none").
    Note: the string "nan" is a real player ID (player 楠) and is NOT
    treated as null.
    """
    if not isinstance(raw_id, str):
        raw_id = "" if raw_id != raw_id else str(raw_id)   # NaN-safe
    cleaned = raw_id.lower().strip()
    if cleaned in ("", "none"):
        return ""
    return ALIAS_MAP.get(cleaned, cleaned)


def normalize_player_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply normalize_player_id to every player ID column present in df.
    Returns a copy; original is not modified.  Null cells stay null.
    """
    df = df.copy()
    for col in PLAYER_COLUMNS:
        if col not in df.columns:
            continue
        mask = df[col].notna()
        df.loc[mask, col] = df.loc[mask, col].astype(str).map(normalize_player_id)
        df.loc[df[col] == "", col] = pd.NA
    return df
