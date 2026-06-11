"""
src/db.py
=========
Build the IDV match-outcome dataset from JSON sources and write to SQLite.

Data sources
------------
JSON files under data/raw_json/  (loaded via ingest_json.read_all()):

  IVL_COA_2020-2024.json    IVL + COA matches 2020-2024
  IVL_COA_2025-2026.json    IVL + COA matches 2025-2026  (incl. COA9)
  oversea_2020-2024.json    Non-IVL/COA: IJL, IVS, IVT, IVC 2020-2024
  oversea_2025.json         Non-IVL/COA matches in 2025

All non-IVL/IJL data is included for **training** (helps calibrate IVL/IJL
players who face overseas opponents at COA), but **out-of-sample tests**
typically filter to the IVL+IJL tiers — see eval.temporal_cv(..., test_tiers=).

Schema
------
  matches  — one row per game half
    Adds tournament_tier ∈ {IVL, IJL, COA, IVS, IVT, IVC, other}
  players  — one row per canonical player ID

Outcome (n_escaped)
-------------------
Derived by source priority:
  1. Count per-survivor results (出门 = escaped, 淘汰 = eliminated)
  2. Fallback for 2020 (no per-survivor results): derive from winner_side
     + min(home_score, away_score)

Public API
----------
  build_matches(verbose=True)            → matches_df
  build_players_table(matches_df)        → players_df
  write_to_db(matches_df, players_df, db_path=DEFAULT_DB_PATH)
  build_database(db_path=DEFAULT_DB_PATH)
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from data.ingest_json import read_all as read_all_json
from data.players import normalize_player_ids

_HERE           = Path(__file__).parent
_ROOT           = _HERE.parent.parent
DEFAULT_DB_PATH = str(_ROOT / "data" / "processed" / "idv.db")

PLAYER_ID_COLS = [
    "hunter_player",
    "survivor1_player",
    "survivor2_player",
    "survivor3_player",
    "survivor4_player",
]
RESULT_COLS = [f"survivor{i}_result" for i in range(1, 5)]


# ---------------------------------------------------------------------------
# Outcome parsing
# ---------------------------------------------------------------------------

def _escaped_flag(result) -> int | None:
    """出门 → 1 (escaped), 淘汰* → 0 (eliminated), anything else → None."""
    if result is None or (isinstance(result, float) and pd.isna(result)):
        return None
    s = str(result).strip()
    if s.startswith("出门"):
        return 1
    if s.startswith("淘汰"):       # catches '淘汰', '淘汰·', etc.
        return 0
    if s == "逃脱":                # rare alternate spelling for "escaped"
        return 1
    return None                    # '0', '58', junk → unknown


def _hunter_wins(winner) -> int | None:
    """屠 → 1, 人 → 0, 平 (draw) / unknown → None."""
    if winner is None or (isinstance(winner, float) and pd.isna(winner)):
        return None
    s = str(winner).strip()
    if s.startswith("屠"):
        return 1
    if s.startswith("人"):
        return 0
    return None


def _hw_from_escapes(n) -> int | None:
    if pd.isna(n):
        return None
    return 1 if n <= 1 else (0 if n >= 3 else None)


def _escapes_from_score(df: pd.DataFrame) -> pd.Series:
    """
    2020 fallback: derive n_escaped from winner_side + half score.
    Loser's score (min of home/away) = margin: 2:2 → 2, 1:3 → 1, 0:5 → 0.
    Direction: 平→2, 屠→margin, 人→4-margin.
    """
    w      = _null_series(df, "winner_side").astype(str).str.strip()
    home   = pd.to_numeric(_null_series(df, "home_score"), errors="coerce")
    away   = pd.to_numeric(_null_series(df, "away_score"), errors="coerce")
    margin = pd.concat([home, away], axis=1).min(axis=1)

    out = pd.Series(pd.NA, index=df.index, dtype="Int64")
    out[w.str.startswith("平")] = 2
    hunt = w.str.startswith("屠") & margin.notna()
    surv = w.str.startswith("人") & margin.notna()
    out[hunt] = margin[hunt].astype("Int64")
    out[surv] = (4 - margin[surv]).astype("Int64")
    return out


def _null_series(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series([None] * len(df), index=df.index)


def _outcome_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Compute n_escaped + hunter_wins from a normalized source frame."""
    flags = pd.DataFrame(index=df.index)
    for i, rcol in enumerate(RESULT_COLS, start=1):
        flags[i] = _null_series(df, rcol).map(_escaped_flag).astype("Int64")

    # n_escaped from per-survivor results when all 4 are known
    n_esc = flags.sum(axis=1).astype("Int64").where(
        flags.notna().all(axis=1), other=pd.NA
    )

    # 2020 fallback: derive from winner + score
    if n_esc.isna().any():
        n_esc = n_esc.fillna(_escapes_from_score(df))

    out = pd.DataFrame(index=df.index)
    out["n_escaped"] = n_esc

    # hunter_wins: prefer escape-derived; fall back to winner_side
    hw     = n_esc.map(_hw_from_escapes)
    win_hw = _null_series(df, "winner_side").map(_hunter_wins)
    hw     = hw.where(~n_esc.isna(), win_hw)
    out["hunter_wins"] = pd.array(hw, dtype="Int64")
    return out


# ---------------------------------------------------------------------------
# Match assembly
# ---------------------------------------------------------------------------

_MATCH_COLS = [
    "date", "tournament", "tournament_tier",
    *PLAYER_ID_COLS,
    "map_name",
    "n_escaped", "hunter_wins",
    "source_file", "source_row",
]


def build_matches(verbose: bool = True) -> pd.DataFrame:
    """Read JSON files → normalized matches DataFrame."""
    if verbose:
        print("Reading JSON files …")
    df = read_all_json()
    if verbose:
        print(f"  loaded {len(df):,} game halves")

    # Normalize player IDs
    df = normalize_player_ids(df)

    # Compute outcomes
    outcome = _outcome_frame(df)

    matches = pd.DataFrame({
        "date":            df["date"].values,
        "tournament":      df["tournament"].values,
        "tournament_tier": df["tournament_tier"].values,
        "map_name":        df["map_name"].values,
        "source_file":     df["source_file"].values,
        "source_row":      df["source_row"].values,
    })
    for col in PLAYER_ID_COLS:
        matches[col] = df[col].values
    matches["n_escaped"]   = outcome["n_escaped"].values
    matches["hunter_wins"] = outcome["hunter_wins"].values

    # Drop rows missing hunter (unusable for the model)
    n_before = len(matches)
    matches = matches[matches["hunter_player"].notna()].reset_index(drop=True)
    if verbose and (n_before - len(matches)):
        print(f"  dropped {n_before - len(matches):,} rows with null hunter_player")

    matches.insert(0, "match_id", range(1, len(matches) + 1))

    if verbose:
        _print_qc(matches)
    return matches[["match_id"] + _MATCH_COLS]


def _print_qc(m: pd.DataFrame) -> None:
    n = len(m)
    decided = m["hunter_wins"].notna().sum()
    hwins   = (m["hunter_wins"] == 1).sum()
    draws   = m["hunter_wins"].isna().sum()
    esc_cov = m["n_escaped"].notna().sum()
    dates   = m["date"].dropna()
    print(f"\nMatches assembled: {n:,}")
    print(f"  decided (non-draw): {decided:,}")
    print(f"  draws:              {draws:,}  ({draws/n*100:.1f}%)")
    if decided:
        print(f"  hunter win rate:    {hwins/decided*100:.1f}%")
    print(f"  n_escaped present:  {esc_cov:,}  ({esc_cov/n*100:.1f}%)")
    if len(dates):
        print(f"  date range:         {dates.min()} → {dates.max()}")
    print(f"  tier distribution:")
    for tier, n_t in m["tournament_tier"].value_counts().items():
        print(f"      {tier:<5s} {n_t:>6,}")


# ---------------------------------------------------------------------------
# Players table
# ---------------------------------------------------------------------------

def build_players_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Derive the players dimension (roles, first/last seen, n_games)."""
    frames = []
    h = matches[["match_id", "date", "hunter_player"]].rename(
        columns={"hunter_player": "pid"})
    h["role"] = "hunter"
    frames.append(h)
    for i in range(1, 5):
        s = matches[["match_id", "date", f"survivor{i}_player"]].rename(
            columns={f"survivor{i}_player": "pid"})
        s["role"] = "survivor"
        frames.append(s)

    long = pd.concat(frames, ignore_index=True)
    long = long[long["pid"].notna() & (long["pid"] != "")]

    roles  = long.groupby("pid")["role"].agg(lambda s: set(s))
    ngames = long.groupby("pid")["match_id"].nunique()

    dated = long.dropna(subset=["date"])
    first_seen = dated.groupby("pid")["date"].min()
    last_seen  = dated.groupby("pid")["date"].max()

    def _role_label(rset: set) -> str:
        if {"hunter", "survivor"} <= rset:
            return "both"
        return "hunter" if "hunter" in rset else "survivor"

    players = pd.DataFrame({"canonical_id": ngames.index})
    players["known_roles"] = players["canonical_id"].map(lambda p: _role_label(roles[p]))
    players["first_seen"]  = players["canonical_id"].map(first_seen)
    players["last_seen"]   = players["canonical_id"].map(last_seen)
    players["n_games"]     = players["canonical_id"].map(ngames).astype(int)
    return players.sort_values("canonical_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# SQLite write
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id          INTEGER PRIMARY KEY,
    date              TEXT,
    tournament        TEXT,
    tournament_tier   TEXT,
    hunter_player     TEXT,
    survivor1_player  TEXT,
    survivor2_player  TEXT,
    survivor3_player  TEXT,
    survivor4_player  TEXT,
    map_name          TEXT,
    n_escaped         INTEGER,
    hunter_wins       INTEGER,
    source_file       TEXT,
    source_row        INTEGER
);

CREATE TABLE IF NOT EXISTS players (
    canonical_id TEXT PRIMARY KEY,
    known_roles  TEXT,
    first_seen   TEXT,
    last_seen    TEXT,
    n_games      INTEGER
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_matches_date            ON matches(date);",
    "CREATE INDEX IF NOT EXISTS idx_matches_hunter_player   ON matches(hunter_player);",
    "CREATE INDEX IF NOT EXISTS idx_matches_tournament      ON matches(tournament);",
    "CREATE INDEX IF NOT EXISTS idx_matches_tournament_tier ON matches(tournament_tier);",
    "CREATE INDEX IF NOT EXISTS idx_matches_map_name        ON matches(map_name);",
]


def write_to_db(matches: pd.DataFrame, players: pd.DataFrame,
                db_path: str = DEFAULT_DB_PATH) -> None:
    """Write both tables to SQLite and create indexes. Replaces any existing db."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA)
        matches.to_sql("matches", conn, if_exists="append", index=False)
        players.to_sql("players", conn, if_exists="append", index=False)
        for idx in _INDEXES:
            conn.execute(idx)
        conn.commit()
        print(f"\nDatabase written → {db_path}")
        for tbl in ("matches", "players"):
            n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            print(f"  {tbl:<10} {n:>7,} rows")
    finally:
        conn.close()


def build_database(db_path: str = DEFAULT_DB_PATH) -> None:
    """Build matches + players from JSON and write to SQLite."""
    print("Building matches from JSON …\n")
    matches = build_matches()
    print("\nBuilding players …")
    players = build_players_table(matches)
    print(f"  players: {len(players):,}")
    write_to_db(matches, players, db_path)


if __name__ == "__main__":
    build_database()
