"""
src/db.py
=========
Build the IDV match-outcome dataset and write it to SQLite.

Scope (skill-decay project, see PROJECT_HANDOFF.md)
--------------------------------------------------
Two tables only:
  matches  — one row per game half
  players  — one row per canonical player ID

The outcome is stored at **escape-count resolution** (0-4 survivors escaped),
not collapsed to binary win/loss, so draws (2-2 halves) are retained:

  margin = n_escaped - 2  ∈ {-2,-1,0,+1,+2}
           -2 = hunter 4-0   0 = draw (2-2)   +2 = survivor 0-4

Source layout (verified)
------------------------
  Legacy  2020-2023.xlsx tabs
      hunter_player, 4 survivor IDs, winner_side, per-survivor results inline
      (EXCEPT 2020原始, which has no per-survivor results → escapes NULL)

  Modern  one file per season — 赛后数据 is the primary sheet:
      赛后数据 : 4 survivor IDs + per-survivor results + date (month/day).
                 Has the hunter ID in 13/18 files.
      原始数据 : consulted ONLY to backfill the hunter ID for the 5 early-2024
                 files whose 赛后数据 omits it. Backfill aligns on
                 (home_team, away_team, half) sequential order and is validated
                 against winner_side — rows whose winner disagrees are left NULL
                 rather than risk assigning the wrong hunter (8 of 910 rows).

Outcome fields per half
-----------------------
  n_escaped    : 0-4 survivors escaped. The graded outcome; the model uses
                 margin = n_escaped - 2. Derived by source priority:
                   1. total_escapes column (总逃脱) — every modern 赛后数据
                   2. sum of per-survivor results (出门) — legacy 2021-2023, COA
                   3. winner_side + half score — 2020 (no results, no total col)
  hunter_wins  : 1 hunter, 0 survivor, NULL draw. Derived from n_escaped where
                 present (consistent with the margin), falling back to
                 winner_side only if n_escaped is itself uncomputable.

Public API
----------
  build_matches(verbose=True)            -> matches_df
  build_players_table(matches_df)        -> players_df
  write_to_db(matches_df, players_df, db_path=DEFAULT_DB_PATH)
  build_database(db_path=DEFAULT_DB_PATH)   # build + write convenience
  tournament_id(source, stage=None)      -> str
"""

from __future__ import annotations

import os
import re
import sqlite3
import warnings
from pathlib import Path

import pandas as pd

from ingest import (
    read_legacy_tab,
    read_modern_sheet,
    LEGACY_RAW_TABS,
    LEGACY_COA_TABS,
    ALL_MODERN_FILES,
    MODERN_RAW_SHEET,
    MODERN_PLAYER_SHEET,
)
from players import normalize_player_ids

_HERE           = Path(__file__).parent
_ROOT           = _HERE.parent
DEFAULT_DB_PATH = str(_ROOT / "data" / "processed" / "idv.db")

# COA editions carry no year in their filename/tab name. Pinned by COA9 = 2026
# (its world finals were held in spring 2026); the sequence steps back one year
# per edition, with COA7 = 2024 the missing one.
COA_YEAR = {4: 2021, 5: 2022, 6: 2023, 7: 2024, 8: 2025, 9: 2026}

PLAYER_ID_COLS = [
    "hunter_player",
    "survivor1_player",
    "survivor2_player",
    "survivor3_player",
    "survivor4_player",
]


def _coa_year(source: str) -> int | None:
    m = re.search(r"COA(\d+)", source)
    return COA_YEAR.get(int(m.group(1))) if m else None


def _source_year(source: str) -> int | None:
    """Year from a 4-digit filename/tab prefix, falling back to the COA map."""
    m = re.match(r"(\d{4})", source)
    if m:
        return int(m.group(1))
    return _coa_year(source)
RESULT_COLS = [f"survivor{i}_result" for i in range(1, 5)]


# ---------------------------------------------------------------------------
# Tournament identifiers  (clean English strings)
# ---------------------------------------------------------------------------

def _season_str(text: str) -> str:
    return "summer" if "夏" in text else "fall" if "秋" in text else "unk"


def _stage_str(text: str) -> str:
    return "regular" if "常规" in text else "playoffs" if "季后" in text else "unk"


def tournament_id(source: str, stage: str | None = None) -> str:
    """
    Build a clean tournament identifier.

    Modern filename : '2024IVL夏季赛常规赛.xlsx' -> 'IVL_2024_summer_regular'
                      'COA9 全球总决赛小组赛.xlsx' -> 'COA9'   (groups/knockout merged)
    Legacy raw tab  : tab='2020原始', stage='夏季赛常规赛' -> 'IVL_2020_summer_regular'
    Legacy COA tab  : 'COA4' -> 'COA4'
    """
    # Legacy COA tabs are passed bare (no digits-year prefix, start with COA)
    if re.fullmatch(r"COA\d+", source):
        return source

    name = source.replace(".xlsx", "")

    # Any COA event (modern files) -> 'COA{n}', no groups/knockout split
    coa = re.search(r"COA(\d+)", name)
    if coa:
        return f"COA{coa.group(1)}"

    year_m = re.match(r"(\d{4})", name)
    year   = year_m.group(1) if year_m else "????"

    league = "IVL" if "IVL" in name else "IJL" if "IJL" in name else "IVL"

    # Modern files carry season/stage in the filename; legacy raw tabs pass it
    # via the per-row `stage` column.
    src_for_season = stage if stage is not None else name
    return f"{league}_{year}_{_season_str(src_for_season)}_{_stage_str(src_for_season)}"


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _parse_legacy_dates(date_series: pd.Series, year: int | None) -> list[str | None]:
    """
    Parse legacy date cells into ISO 'YYYY-MM-DD' strings.

    Handles three formats found in the source files:
      Separator  : '6.25', '6*9', '4/16'  → month.day
      4-digit int: 1104, 1227             → MMDD (November 4, December 27)
      3-digit int: 108, 109               → MDD  (January 8, January 9)
                                            Note: 3-digit months are 1-9 so
                                            day is the last 2 digits.

    For the integer formats the month/year may cross a calendar year boundary
    (IVL fall season extends to Jan-Feb of the following year).  Months 1-5
    in a tab named for year Y are assigned to year Y+1.
    """
    out: list[str | None] = []
    for val in date_series:
        parsed = None
        if not pd.isna(val) and year:
            s = str(val).strip()

            # --- Try separator-based parsing first ---
            # This handles '6.25', '6*9', '4/16'.
            # Note: integers like 108 become '108.0' after float-string
            # conversion and do contain a '.', but yield m=108 which fails
            # the validity check and falls through to the integer path below.
            for sep in (".", "*", "/"):
                if sep in s:
                    try:
                        m_str, d_str = s.split(sep, 1)
                        m, d = int(float(m_str)), int(float(d_str))
                        if 1 <= m <= 12 and 1 <= d <= 31:
                            parsed = f"{year}-{m:02d}-{d:02d}"
                    except (ValueError, TypeError):
                        pass
                    break   # only try one separator; fall through if invalid

            # --- Integer MMDD / MDD fallback ---
            # Handles cells stored as bare integers: 1104 → Nov 4, 108 → Jan 8.
            # Triggered when separator parsing found no valid date.
            if parsed is None:
                try:
                    n = int(float(s))
                    if 1000 <= n <= 1231:           # MMDD: e.g. 1104 → Nov 4
                        m, d = n // 100, n % 100
                    elif 100 <= n <= 931:            # MDD:  e.g. 108 → Jan 8
                        m, d = n // 100, n % 100
                    else:
                        m, d = 0, 0
                    if 1 <= m <= 12 and 1 <= d <= 31:
                        # Months Jan–May likely belong to the following year
                        # (IVL fall season sometimes extends into early spring)
                        y = year + 1 if m <= 5 else year
                        parsed = f"{y}-{m:02d}-{d:02d}"
                except (ValueError, TypeError):
                    pass

        out.append(parsed)
    return out


def _make_modern_dates(month_col: pd.Series, day_col: pd.Series,
                       year: int | None) -> list[str | None]:
    """Build ISO date strings from separate month/day integer columns."""
    out: list[str | None] = []
    for m, d in zip(month_col, day_col):
        if pd.isna(m) or pd.isna(d) or not year:
            out.append(None)
        else:
            try:
                out.append(f"{year}-{int(m):02d}-{int(d):02d}")
            except (ValueError, TypeError):
                out.append(None)
    return out


# ---------------------------------------------------------------------------
# Outcome parsing
# ---------------------------------------------------------------------------

def _escaped_flag(result) -> int | None:
    """出门 → 1 (escaped), 淘汰* → 0 (eliminated), anything else → None."""
    if result is None or pd.isna(result):
        return None
    s = str(result).strip()
    if s == "出门":
        return 1
    if s.startswith("淘汰"):       # also catches the '淘汰·' variant
        return 0
    return None                    # '0', '58', junk → unknown


def _hunter_wins(winner) -> int | None:
    """屠 → 1, 人 → 0, 平 (draw) / unknown → None."""
    if winner is None or pd.isna(winner):
        return None
    s = str(winner).strip()
    if s.startswith("屠"):
        return 1
    if s.startswith("人"):
        return 0
    return None


def _hw_from_escapes(n) -> int | None:
    """Escape count → hunter_wins: ≤1 escaped → hunter (1), ≥3 → survivor (0),
    2 (draw) or unknown → None."""
    if pd.isna(n):
        return None
    return 1 if n <= 1 else (0 if n >= 3 else None)


def _escapes_from_score(df: pd.DataFrame) -> pd.Series:
    """
    Derive n_escaped from winner_side + the half score (home_score:away_score).
    Used for 2020, which has no per-survivor results and no total_escapes column.

    The loser's score — min(home, away) — is the margin:
        2:2 (draw)  → 2 escaped
        1:3 / 3:1   → margin 1
        0:5 / 5:0   → margin 0 (shutout)

    Direction comes from the winner:
        draw (平)         → 2
        hunter won (屠)   → min(home, away)        # 0:5→0, 1:3→1
        survivor won (人) → 4 - min(home, away)    # 0:5→4, 1:3→3
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
    """
    Compute the per-half outcome columns from a normalized source frame.
    Returns a frame with ESCAPED_COLS + n_escaped + hunter_wins, aligned to df.index.
    """
    # n_escaped, by source priority:
    #   1) total_escapes column (总逃脱) — present in every modern 赛后数据
    n_esc = pd.to_numeric(_null_series(df, "total_escapes"), errors="coerce").astype("Int64")

    #   2) sum per-survivor results where no total column (legacy 2021-2023, COA)
    if n_esc.isna().any():
        flags = pd.DataFrame(index=df.index)
        for i, rcol in enumerate(RESULT_COLS, start=1):
            flags[i] = _null_series(df, rcol).map(_escaped_flag).astype("Int64")
        summed = flags.sum(axis=1).astype("Int64").where(flags.notna().all(axis=1), other=pd.NA)
        n_esc = n_esc.fillna(summed)

    #   3) derive from winner_side + score where still missing (2020)
    if n_esc.isna().any():
        n_esc = n_esc.fillna(_escapes_from_score(df))

    out = pd.DataFrame(index=df.index)
    out["n_escaped"] = n_esc

    # hunter_wins: prefer the escape-derived outcome (consistent with margin);
    # fall back to winner_side only where no escape data exists (2020).
    hw      = n_esc.map(_hw_from_escapes)
    win_hw  = _null_series(df, "winner_side").map(_hunter_wins)
    hw      = hw.where(~n_esc.isna(), win_hw)        # fill only the no-escape rows
    out["hunter_wins"] = pd.array(hw, dtype="Int64")
    return out


# ---------------------------------------------------------------------------
# Modern hunter-ID backfill (5 early-2024 files lack 屠ID in 赛后数据)
# ---------------------------------------------------------------------------

_BACKFILL_KEYS = ["home_team", "away_team", "half"]


def _backfill_hunter_from_raw(filename: str, player: pd.DataFrame) -> pd.DataFrame:
    """
    Fill hunter_player from 原始数据 for a 赛后数据 frame that lacks it.

    Alignment: within each (home_team, away_team, half) group the Nth 赛后数据
    row maps to the Nth 原始数据 row (series order). Validated against
    winner_side — both sheets carry it for these files — and any row whose
    winner disagrees is left NULL rather than risk a wrong hunter.
    """
    raw  = read_modern_sheet(filename, MODERN_RAW_SHEET)
    keys = [k for k in _BACKFILL_KEYS if k in player.columns and k in raw.columns]

    p = player.copy()
    r = raw.copy()
    p["_seq"] = p.groupby(keys).cumcount()
    r["_seq"] = r.groupby(keys).cumcount()

    bring = (
        r[keys + ["_seq", "hunter_player"]]
        .rename(columns={"hunter_player": "_hunter_bf"})
    )
    if "winner_side" in raw.columns:
        bring["_ws_raw"] = r["winner_side"].values

    p = p.merge(bring, on=keys + ["_seq"], how="left")

    hunter = p["_hunter_bf"]
    if "winner_side" in player.columns and "_ws_raw" in p.columns:
        disagree = (p["winner_side"].astype(str).str.strip()
                    != p["_ws_raw"].astype(str).str.strip())
        hunter = hunter.where(~disagree)        # null the unverifiable ones

    p["hunter_player"] = hunter
    p = p.drop(columns=[c for c in ("_seq", "_hunter_bf", "_ws_raw") if c in p.columns])

    n_null = p["hunter_player"].isna().sum()
    if n_null:
        warnings.warn(
            f"{filename}: {n_null} hunter ID(s) unverifiable after backfill "
            f"— left NULL",
            stacklevel=2,
        )
    return p


# ---------------------------------------------------------------------------
# Per-source processors → matches rows
# ---------------------------------------------------------------------------

_MATCH_COLS = (
    ["date", "tournament"]
    + PLAYER_ID_COLS
    + ["n_escaped", "hunter_wins", "source_file", "source_row"]
)


def _assemble(df: pd.DataFrame, dates, tournaments, source_file) -> pd.DataFrame:
    """Build the matches frame slice from a processed source frame."""
    df = normalize_player_ids(df)
    outcome = _outcome_frame(df)

    out = pd.DataFrame({
        "date":          dates,
        "tournament":    tournaments,
        "source_file":   source_file,
        "source_row":    _null_series(df, "source_row").values,
    })
    for col in PLAYER_ID_COLS:
        out[col] = _null_series(df, col).values
    for col in ["n_escaped", "hunter_wins"]:
        out[col] = outcome[col].values
    return out[_MATCH_COLS]


def _process_legacy_tab(tab_name: str) -> pd.DataFrame:
    df = read_legacy_tab(tab_name).reset_index(drop=True)
    df["source_row"] = df.index

    year = _source_year(tab_name)

    dates = _parse_legacy_dates(_null_series(df, "date"), year)

    if tab_name.startswith("COA"):
        tournaments = [tournament_id(tab_name)] * len(df)
    else:
        tournaments = [
            tournament_id(tab_name, stage=str(s) if not pd.isna(s) else "")
            for s in _null_series(df, "stage")
        ]
    return _assemble(df, dates, tournaments, source_file="2020-2023.xlsx")


def _process_modern_file(filename: str) -> pd.DataFrame:
    # 赛后数据 is the base: survivor IDs, per-survivor results, date.
    df = read_modern_sheet(filename, MODERN_PLAYER_SHEET).reset_index(drop=True)
    df["source_row"] = df.index

    # Backfill hunter ID from 原始数据 only when 赛后数据 omits it.
    if "hunter_player" not in df.columns or df["hunter_player"].isna().all():
        df = _backfill_hunter_from_raw(filename, df)

    year = _source_year(filename)

    dates       = _make_modern_dates(_null_series(df, "month"), _null_series(df, "day"), year)
    tournaments = tournament_id(filename)
    return _assemble(df, dates, tournaments, source_file=filename)


# ---------------------------------------------------------------------------
# Build matches
# ---------------------------------------------------------------------------

def build_matches(verbose: bool = True) -> pd.DataFrame:
    """Assemble the matches table from all sources, in chronological-ish order."""
    parts: list[pd.DataFrame] = []

    for tab in LEGACY_RAW_TABS + LEGACY_COA_TABS:
        if verbose:
            print(f"  [legacy]  {tab}")
        parts.append(_process_legacy_tab(tab))

    for f in ALL_MODERN_FILES:
        if verbose:
            print(f"  [modern]  {f}")
        parts.append(_process_modern_file(f))

    matches = pd.concat(parts, ignore_index=True)

    # Drop rows where the hunter player is unknown — unusable for the model.
    n_before = len(matches)
    matches = matches[matches["hunter_player"].notna()].reset_index(drop=True)
    n_dropped = n_before - len(matches)
    if verbose and n_dropped:
        print(f"  dropped {n_dropped} row(s) with null hunter_player")

    matches.insert(0, "match_id", range(1, len(matches) + 1))

    if verbose:
        _print_match_qc(matches)
    return matches


def _print_match_qc(m: pd.DataFrame) -> None:
    n = len(m)
    decided = m["hunter_wins"].notna().sum()
    hwins   = (m["hunter_wins"] == 1).sum()
    draws   = m["hunter_wins"].isna().sum()
    esc_cov = m["n_escaped"].notna().sum()
    dates   = m["date"].dropna()
    print(f"\nMatches assembled: {n:,}")
    print(f"  decided (non-draw)  : {decided:,}")
    print(f"  draws (平)          : {draws:,}  ({draws/n*100:.1f}%)")
    if decided:
        print(f"  hunter win rate     : {hwins/decided*100:.1f}%  (of decided)")
    print(f"  n_escaped present   : {esc_cov:,}  ({esc_cov/n*100:.1f}%)")
    if len(dates):
        print(f"  date range          : {dates.min()} → {dates.max()}")
    print(f"  null date rows      : {m['date'].isna().sum():,}")


# ---------------------------------------------------------------------------
# Build players
# ---------------------------------------------------------------------------

def build_players_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Derive the players dimension (roles, first/last seen, n_games)."""
    frames = []
    h = matches[["match_id", "date", "hunter_player"]].rename(columns={"hunter_player": "pid"})
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

    # first/last seen from non-null dates only (ISO strings sort chronologically)
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
    players = players[["canonical_id", "known_roles", "first_seen", "last_seen", "n_games"]]
    return players.sort_values("canonical_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id          INTEGER PRIMARY KEY,
    date              TEXT,
    tournament        TEXT,
    hunter_player     TEXT,
    survivor1_player  TEXT,
    survivor2_player  TEXT,
    survivor3_player  TEXT,
    survivor4_player  TEXT,
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
    "CREATE INDEX IF NOT EXISTS idx_matches_date          ON matches(date);",
    "CREATE INDEX IF NOT EXISTS idx_matches_hunter_player ON matches(hunter_player);",
    "CREATE INDEX IF NOT EXISTS idx_matches_tournament    ON matches(tournament);",
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
    """Build matches + players and write to SQLite in one call."""
    print("Building matches …\n")
    matches = build_matches()
    print("\nBuilding players …")
    players = build_players_table(matches)
    print(f"  players: {len(players):,}")
    write_to_db(matches, players, db_path)


if __name__ == "__main__":
    build_database()
