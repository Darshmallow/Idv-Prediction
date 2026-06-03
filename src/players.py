"""
src/players.py
==============
Player ID normalization for IDV esports data.

The 曾用id tab in 2020-2023.xlsx tracks historical IDs per player, but
only covers up to ~end of 2023. Post-2023 ID changes must be added
manually to MANUAL_OVERRIDES below.

Algorithm
---------
1. Parse 曾用id tab → groups of (role, nickname, [id1, id2, ...]) + game counts
2. Canonical ID = the ID with the most recorded game appearances in that group
   (most-played is usually the most recognisable / current name)
3. All other IDs in the group → ALIAS_MAP entries pointing to canonical
4. MANUAL_OVERRIDES is merged on top (takes precedence over everything)
5. normalize_player_id() lowercases + strips, then applies ALIAS_MAP

Public API
----------
    ALIAS_MAP: dict[str, str]
        All old_id → canonical_id mappings (tab-derived + manual).

    normalize_player_id(raw_id: str) -> str
        Lowercase/strip + alias lookup.  Returns raw_id if not in map.

    normalize_player_ids(df: pd.DataFrame) -> pd.DataFrame
        Applies normalize_player_id to all five player ID columns.
        Returns a copy.

Adding a post-2023 ID change
-----------------------------
Edit MANUAL_OVERRIDES below:

    MANUAL_OVERRIDES: dict[str, str] = {
        "old_id": "new_canonical_id",
    }

Both IDs should be lowercase.  The *right-hand side* is the canonical
(the name you want to keep in the database going forward).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_HERE    = Path(__file__).parent      # src/
_ROOT    = _HERE.parent               # project root
DATA_DIR = str(_ROOT / "data" / "raw")
LEGACY_FILE = "2020-2023.xlsx"

# ---------------------------------------------------------------------------
# ✏️  MANUAL OVERRIDES — add post-2023 ID changes here
# ---------------------------------------------------------------------------
# Format: "old_id": "canonical_id"   (both lowercase)
# The right-hand value is the canonical name kept in the database.
#
# Example:
#   "formerName": "currentName",
#
MANUAL_OVERRIDES: dict[str, str] = {
    # --- Within-tab corrections ---
    # The algorithm picks the ID with the most legacy appearances as canonical,
    # but sometimes a player switched permanently to a new ID in 2024+ that
    # has fewer legacy games. Add those corrections here.
    "taoxing": "tx",    # 逃行: 36 legacy games as taoxing, but tx is used in 2024+

    # --- Post-2023 ID changes ---
    # (none recorded yet)

    # --- Suspected typo-duplicates (fuzzy match > 0.85) ---
    # Uncomment once confirmed to be the same person.
    "ppicha": "pipicha",   # pipicha ~ ppicha — likely same player
    "gua":    "guag",      # gua ~ guag
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
# Tab parsing
# ---------------------------------------------------------------------------

def _parse_alias_tab(path: str) -> list[tuple[str, str, list[str], list[float]]]:
    """
    Parse the 曾用id tab from the legacy xlsx.

    Layout of the sheet (header=None):
      Row 0: section labels  — "求生者" (cols 0-7) | "监管者" (cols 8-15)
      Row 1: column headers  — 昵称 id1 id2 id3 id1上场 id2上场 id3上场 总上场  × 2
      Rows 2+: data

    Returns list of (role, nickname, [ids], [game_counts]) tuples.
    role is "survivor" or "hunter".
    ids and game_counts are parallel lists; counts may be 0 for absent IDs.
    """
    raw = pd.read_excel(path, sheet_name="曾用id", header=None)
    data_rows = raw.iloc[2:]

    entries: list[tuple[str, str, list[str], list[float]]] = []

    for _, r in data_rows.iterrows():
        for role, nick_col, id_cols, ct_cols in [
            ("survivor", 0,  [1, 2, 3],  [4, 5, 6]),
            ("hunter",   8,  [9, 10, 11], [12, 13, 14]),
        ]:
            nick = str(r[nick_col]).strip() if pd.notna(r[nick_col]) else None
            if not nick or nick == "nan":
                continue

            ids: list[str] = []
            counts: list[float] = []
            for ic, cc in zip(id_cols, ct_cols):
                raw_id = str(r[ic]).strip() if pd.notna(r[ic]) else ""
                raw_ct = float(r[cc]) if pd.notna(r[cc]) else 0.0
                if raw_id and raw_id != "nan":
                    ids.append(raw_id.lower())
                    counts.append(raw_ct)

            if ids:
                entries.append((role, nick, ids, counts))

    return entries


def _build_alias_map(
    entries: list[tuple[str, str, list[str], list[float]]],
    manual_overrides: dict[str, str],
) -> dict[str, str]:
    """
    Build the flat old_id → canonical_id mapping.

    Canonical selection:
      The ID with the highest game-appearance count in the tab wins.
      Ties broken by taking the first listed ID.

    MANUAL_OVERRIDES are merged last and take precedence over everything,
    so any tab-derived mapping can be corrected there.
    """
    alias_map: dict[str, str] = {}

    for _role, _nick, ids, counts in entries:
        if len(ids) == 1:
            continue  # nothing to alias

        # Canonical = highest game count (tie-break: first listed)
        best_idx = max(range(len(ids)), key=lambda i: (counts[i], -i))
        canonical = ids[best_idx]

        for id_ in ids:
            if id_ != canonical:
                alias_map[id_] = canonical

    # Manual overrides take absolute precedence.
    # When an override declares X → Y, Y becomes canonical — so any existing
    # entry pointing *to* X (i.e. X was the tab-derived canonical) must be
    # removed to avoid cycles like tx→taoxing→tx.
    for old, canon in manual_overrides.items():
        old   = old.lower().strip()
        canon = canon.lower().strip()
        alias_map.pop(canon, None)  # canon is now canonical — remove if it was an alias
        alias_map[old] = canon

    return alias_map


# ---------------------------------------------------------------------------
# Module-level alias map (built once at import)
# ---------------------------------------------------------------------------

_ENTRIES = _parse_alias_tab(os.path.join(DATA_DIR, LEGACY_FILE))
ALIAS_MAP: dict[str, str] = _build_alias_map(_ENTRIES, MANUAL_OVERRIDES)

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def normalize_player_id(raw_id: str) -> str:
    """
    Normalize a single player ID:
      1. Lowercase + strip whitespace
      2. Look up in ALIAS_MAP; return canonical if found, else the cleaned ID

    Returns empty string for null-like inputs ("", "nan", "none").
    """
    if not isinstance(raw_id, str):
        raw_id = "" if raw_id != raw_id else str(raw_id)  # handles NaN
    cleaned = raw_id.lower().strip()
    if cleaned in ("", "none"):
        return ""
    return ALIAS_MAP.get(cleaned, cleaned)


def normalize_player_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply normalize_player_id to every player ID column present in df.
    Returns a copy; original is not modified.
    Null cells (NaN) are left as NaN.
    """
    df = df.copy()
    for col in PLAYER_COLUMNS:
        if col not in df.columns:
            continue
        mask = df[col].notna()
        df.loc[mask, col] = df.loc[mask, col].astype(str).map(normalize_player_id)
        # Restore empty-string results (null-like) back to NaN
        df.loc[df[col] == "", col] = pd.NA
    return df


# ---------------------------------------------------------------------------
# Diagnostic helper (not used in the pipeline)
# ---------------------------------------------------------------------------

def print_alias_summary() -> None:
    """Print the full alias map and a breakdown by player."""
    print(f"Total alias entries: {len(ALIAS_MAP)}\n")
    print("Alias groups (old → canonical):")
    # Group by canonical for readability
    from collections import defaultdict
    by_canonical: dict[str, list[str]] = defaultdict(list)
    for old, canon in ALIAS_MAP.items():
        by_canonical[canon].append(old)
    for canon in sorted(by_canonical):
        aliases = by_canonical[canon]
        print(f"  {canon!r:20s} ← {aliases}")
