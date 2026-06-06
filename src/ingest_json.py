"""
src/ingest_json.py
==================
JSON-format ingestion (the cleaner data source — supersedes ingest.py).

Source files in data/raw_json/:
    IVL_COA_2020-2024.json     IVL + COA matches 2020-2024
    IVL_COA_2025-2026.json     IVL + COA matches 2025-2026 (incl. COA9)
    oversea_2020-2024.json     Non-IVL/COA: IJL, IVS, IVT, IVC 2020-2024
    oversea_2025.json          Non-IVL/COA matches in 2025
    id.json                    Player canonical → alias mapping
    team.json                  Team aliases (not used by the model)

JSON structure
--------------
Each file is a list of MATCH objects (one team-vs-team series), each with:
    {
      "赛段": 2024,                    # calendar year (int)
      "阶段": "IVL夏季赛常规赛",            # tournament stage
      "日期": "2024-06-09",            # ISO date
      "主场": "TE", "客场": "GG",        # teams
      "games": [ {game_half_1}, {game_half_2}, ... ]
    }

Each game half within `games` has fields like:
    屠名 → hunter player ID
    求生者1ID..求生者4ID → 4 survivor player IDs
    胜利方 → 屠 / 人 / 平
    主分 / 客分 → home / away score
    地图 → map name
    出门1..出门4 → per-survivor outcome (出门 = escaped, 淘汰 = eliminated)

Missing values are encoded as -999 (or the string "-999" or "-").

Public API
----------
    read_json_file(path)            → DataFrame of game halves
    read_all()                      → DataFrame combining all 4 JSON files
    extract_tier(stage)             → 'IVL' | 'IJL' | 'COA' | 'IVS' | 'IVT' | 'IVC' | 'other'
    tournament_id(stage, year)      → clean string identifier (e.g. 'IVL_2024_summer_regular')

Output columns (matching downstream db.py expectations):
    date, year, tournament, tournament_tier, stage,
    home_team, away_team,
    hunter_player,
    survivor1_player, survivor2_player, survivor3_player, survivor4_player,
    map_name, winner_side,
    home_score, away_score,
    survivor1_result, survivor2_result, survivor3_result, survivor4_result,
    source_file, source_row
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT     = Path(__file__).parent.parent
DATA_DIR  = _ROOT / "data" / "raw_json"

JSON_FILES = [
    "IVL_COA_2020-2024.json",
    "IVL_COA_2025-2026.json",
    "oversea_2020-2024.json",
    "oversea_2025.json",
]

# COA edition ↔ calendar-year mapping (year - 2017 = edition).
# COA happens in spring; both small-group and finals are part of the same edition.
def _coa_edition(year: int) -> int:
    return year - 2017


# Values used as "missing" sentinels in the JSON
_MISSING_VALUES = {-999, "-999", "-", "", None}


def _clean(val: Any) -> Any:
    """Map sentinel values to None; pass real values through."""
    if val in _MISSING_VALUES:
        return None
    if isinstance(val, str):
        s = val.strip()
        if s in _MISSING_VALUES:
            return None
        return s
    return val


# ---------------------------------------------------------------------------
# Tournament tier + identifier
# ---------------------------------------------------------------------------

def extract_tier(stage: str) -> str:
    """Map 阶段 string → tier code."""
    if stage is None:
        return "other"
    s = str(stage)
    if s.startswith("IVL"):
        return "IVL"
    if s.startswith("IJL"):
        return "IJL"
    if s.startswith("COA"):
        return "COA"
    if s == "IVS":
        return "IVS"
    if "IVT" in s:
        return "IVT"
    if "IVC" in s:
        return "IVC"
    return "other"


def _season_str(s: str) -> str:
    return "summer" if "夏" in s else "fall" if "秋" in s else ""


def _stage_str(s: str) -> str:
    return "regular" if "常规" in s else "playoffs" if "季后" in s else ""


def tournament_id(stage: str, year: int) -> str:
    """
    Build a clean tournament identifier from (阶段, 赛段).

    Examples:
        ('IVL夏季赛常规赛', 2024)   → 'IVL_2024_summer_regular'
        ('IVL秋季赛季后赛', 2024)   → 'IVL_2024_fall_playoffs'
        ('IJL夏季赛常规赛', 2025)   → 'IJL_2025_summer_regular'
        ('COA小组赛', 2026)        → 'COA9'              (groups + finals merged)
        ('COA总决赛', 2025)        → 'COA8'
        ('COA预选赛', 2026)        → 'COA9_qualifier'
        ('IVS', 2025)             → 'IVS_2025'
        ('日本IVT夏季赛', 2023)     → 'IVT_Japan_2023_summer'
        ('欧美IVT', 2022)          → 'IVT_EU_NA_2022'
        ('韩国IVT', 2023)          → 'IVT_Korea_2023'
        ('港澳台IVT', 2024)        → 'IVT_HKMacauTW_2024'
        ('东南亚IVC', 2023)        → 'IVC_SEA_2023'
        ('日本IVC夏季赛', 2024)     → 'IVC_Japan_2024_summer'
    """
    s = str(stage) if stage else ""

    if s.startswith("IVL") or s.startswith("IJL"):
        tier   = s[:3]
        season = _season_str(s)
        stage_ = _stage_str(s)
        parts  = [tier, str(year)]
        if season: parts.append(season)
        if stage_: parts.append(stage_)
        return "_".join(parts)

    if s.startswith("COA"):
        ed = _coa_edition(year)
        if "预选" in s:
            return f"COA{ed}_qualifier"
        return f"COA{ed}"  # groups + finals merged

    if s == "IVS":
        return f"IVS_{year}"

    if "IVT" in s:
        region = ("Japan"      if "日本" in s
                  else "Korea"  if "韩国" in s
                  else "HKMacauTW" if "港澳台" in s
                  else "EU_NA"  if "欧美" in s
                  else "")
        season = _season_str(s)
        parts  = ["IVT"]
        if region: parts.append(region)
        parts.append(str(year))
        if season: parts.append(season)
        return "_".join(parts)

    if "IVC" in s:
        region = ("Japan" if "日本" in s
                  else "SEA" if "东南亚" in s
                  else "")
        season = _season_str(s)
        parts  = ["IVC"]
        if region: parts.append(region)
        parts.append(str(year))
        if season: parts.append(season)
        return "_".join(parts)

    return f"other_{year}"


# ---------------------------------------------------------------------------
# Flatten a single JSON file → DataFrame of game halves
# ---------------------------------------------------------------------------

_SURV_IDX = [1, 2, 3, 4]
# Mapping: output_col → JSON field name (with formatting for survivor index)
_FIELD_MAP = {
    "hunter_player":   ("屠名",      None),
    "winner_side":     ("胜利方",    None),
    "home_score":      ("主分",      None),
    "away_score":      ("客分",      None),
    "map_name":        ("地图",      None),
}


def _flatten_match(match: dict, source_file: str, source_row: int) -> list[dict]:
    """
    Flatten one match dict (with its `games` list) into one or more
    game-half row dicts.
    """
    year   = match.get("赛段")
    stage  = match.get("阶段")
    date   = match.get("日期")
    home   = match.get("主场")
    away   = match.get("客场")

    rows: list[dict] = []
    for game in match.get("games", []):
        row: dict[str, Any] = {
            "date":            _clean(date),
            "year":            _clean(year),
            "stage":           _clean(stage),
            "tournament_tier": extract_tier(stage),
            "tournament":      tournament_id(stage, year) if year is not None else None,
            "home_team":       _clean(home),
            "away_team":       _clean(away),
            "source_file":     source_file,
            "source_row":      source_row,
        }
        for out_col, (json_field, _) in _FIELD_MAP.items():
            row[out_col] = _clean(game.get(json_field))

        # Survivor IDs and per-survivor results
        for i in _SURV_IDX:
            row[f"survivor{i}_player"] = _clean(game.get(f"求生者{i}ID"))
            row[f"survivor{i}_result"] = _clean(game.get(f"出门{i}"))

        rows.append(row)
    return rows


def read_json_file(path: str | Path) -> pd.DataFrame:
    """Read one JSON file → DataFrame of game halves."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for i, match in enumerate(data):
        rows.extend(_flatten_match(match, source_file=path.name, source_row=i))
    return pd.DataFrame(rows)


def read_all() -> pd.DataFrame:
    """Read all 4 match-data JSON files → unified DataFrame, sorted by date."""
    frames = []
    for fn in JSON_FILES:
        p = DATA_DIR / fn
        if p.exists():
            frames.append(read_json_file(p))
        else:
            print(f"  WARN: missing {fn}")
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "source_file", "source_row"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Player alias map from id.json
# ---------------------------------------------------------------------------

def load_id_alias_map() -> dict[str, str]:
    """
    Load id.json and return a flat dict mapping every historical ID to its
    canonical ID.

    id.json structure: {canonical_id: [canonical_id, alias_1, alias_2, ...]}
    """
    with open(DATA_DIR / "id.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    flat: dict[str, str] = {}
    for canonical, ids in data.items():
        canonical_lower = str(canonical).lower().strip()
        for alias in ids:
            alias_lower = str(alias).lower().strip()
            if alias_lower != canonical_lower:
                flat[alias_lower] = canonical_lower
    return flat
