"""
src/ingest.py
=============
File reading and column normalization for IDV esports data.

Public API:
    Constants:
        LEGACY_FILE, LEGACY_RAW_TABS, LEGACY_COA_TABS
        IVL_FILES, IJL_FILES, COA_FILES, ALL_MODERN_FILES
        MODERN_RAW_SHEET, MODERN_PLAYER_SHEET, MODERN_GAME_SHEET
        COLUMN_ALIASES

    Functions:
        detect_header_row(path, sheet_name, markers, max_search=5)
        normalize_columns(df, alias_map=None, verbose=False)
        read_legacy_tab(tab_name, nrows=None, normalize=True)
        read_modern_sheet(filename, sheet_name, nrows=None, normalize=True)
"""

import os
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).parent        # .../src/data/
_ROOT    = _HERE.parent.parent          # project root
DATA_DIR = str(_ROOT / "data" / "raw")

# ---------------------------------------------------------------------------
# File registry
# ---------------------------------------------------------------------------
LEGACY_FILE     = "2020-2023.xlsx"
LEGACY_RAW_TABS = ["2020原始", "2021原始", "2022原始", "2023原始"]
LEGACY_COA_TABS = ["COA4", "COA5", "COA6"]

IVL_FILES = [
    "2024IVL夏季赛常规赛.xlsx",
    "2024IVL夏季赛季后赛.xlsx",
    "2024IVL秋季赛常规赛.xlsx",
    "2024IVL秋季赛季后赛.xlsx",
    "2025IVL夏季赛常规赛.xlsx",
    "2025IVL夏季赛季后赛.xlsx",
    "2025IVL秋季赛常规赛.xlsx",
    "2025IVL秋季赛季后赛.xlsx",
]

IJL_FILES = [
    "2024IJL夏季赛常规赛.xlsx",
    "2024IJL秋季赛季后赛.xlsx",
    "2025IJL夏季赛常规赛.xlsx",
    "2025IJL夏季赛季后赛.xlsx",
    "2025IJL秋季赛常规赛.xlsx",
    "2025IJL秋季赛季后赛.xlsx",
]

COA_FILES = [
    "COA8 全球总决赛小组赛.xlsx",
    "COA8 全球总决赛淘汰赛.xlsx",
    "COA9 全球总决赛小组赛.xlsx",
    "COA9 全球总决赛淘汰赛.xlsx",
]

EXCLUDED_FILES = [
    "2025IVS.xlsx",
    "COA8 日本赛区预选赛.xlsx",
    "COA9 日本赛区预选赛.xlsx",
]

ALL_MODERN_FILES    = IVL_FILES + IJL_FILES + COA_FILES
MODERN_RAW_SHEET    = "原始数据"
MODERN_PLAYER_SHEET = "赛后数据"
MODERN_GAME_SHEET   = "对局数据"

# ---------------------------------------------------------------------------
# Canonical column alias dictionary
# ---------------------------------------------------------------------------
COLUMN_ALIASES: dict[str, str] = {
    # --- Game-level metadata ---
    "本页出错":           "page_error",
    "阶段":               "stage",
    "日期":               "date",
    "月":                 "month",
    "日":                 "day",
    "时间":               "time",
    "场次":               "match_num",
    "场次.1":             "match_num_2",
    "半场":               "half",
    "主场":               "home_team",
    "客场":               "away_team",

    # --- Hunter side ---
    "屠队主客":           "hunter_side",
    "屠队":               "hunter_team",
    "屠名":               "hunter_player",
    "屠ID":               "hunter_player",      # modern alias
    "屠选":               "hunter_character",

    # --- Survivor side ---
    "人队":               "survivor_team",

    # --- Outcome / scoring ---
    "胜利方":             "winner_side",
    "胜利":               "winner_side",
    "局分":               "half_score",
    "局总分":             "game_score",
    "场分":               "match_score",
    "主分":               "home_score",
    "客分":               "away_score",
    "剩机台数":           "gens_remaining",
    "剩机":               "gens_remaining",

    # --- Map / draft ---
    "地图":               "map_name",
    "BAN图":              "banned_map",
    "BAN图.1":            "banned_map_2",
    "BAN图1":             "banned_map",
    "BAN图2":             "banned_map_2",
    "地图主客":           "map_picker_side",
    "地图选择方主客":     "map_picker_side",
    "选图队伍":           "map_picker_team",
    "地图选择方队名":     "map_picker_team",
    "选边队伍":           "side_picker_team",

    # --- Match meta ---
    "用时 (分)":          "duration_min",
    "用时 (秒)":          "duration_sec",
    "游戏用时":           "game_duration_sec",
    "MVP":                "mvp",
    "暂停或重赛":         "paused_or_rematch",
    "备注":               "notes",

    # --- Hunter-side aggregate stats (modern 赛后数据) ---
    "击倒":               "hunter_knockdowns",
    "命中":               "hunter_hits",
    "震慑":               "hunter_stuns",
    "破板":               "hunter_boards_broken",
    "角色.4":             "hunter_character",    # modern 赛后: 5th character col = hunter
    "角色.5":             "hunter_talent",

    # --- Match-level outcome aggregates ---
    "总逃脱":             "total_escapes",
    "逃生数":             "_qc_escape_count_check",   # duplicate QC column
    "淘汰":               "eliminations",
    "淘汰数":             "eliminations",
    "总破译进度":         "total_repair_progress",
    "总牵制时长":         "total_harassment",

    # --- 2020-only draft fields (pre-phased draft system) ---
    "人BAN":              "survivor_ban_2020",
    "人选":               "survivor_pick_2020",
    "屠BAN":              "hunter_ban_2020",

    # --- Phased draft fields (2021+) ---
    "第一阶段人BAN":      "phase1_survivor_ban",
    "第一阶段屠BAN":      "phase1_hunter_ban",
    "第一阶段人PICK":     "phase1_survivor_pick",
    "第二阶段人BAN":      "phase2_survivor_ban",
    "第二阶段人PICK":     "phase2_survivor_pick",
    "第三阶段人BAN":      "phase3_survivor_ban",
    "第三阶段人PICK":     "phase3_survivor_pick",

    # --- Auto-generated ban summaries (2023+) ---
    "求生者首抢禁用【自动生成】": "survivor_first_ban_auto",
    "求生者全局禁用【自动生成】": "survivor_global_ban_auto",
    "监管者全局禁用【自动生成】": "hunter_global_ban_auto",

    # --- Talent / supplementary picks ---
    "辅助特质":           "support_talent",
    "辅助特质1":          "support_talent",
    "底牌后":             "post_trump_talent",
    "光明之星":           "mvp",        # 2020-only MVP-like award

    # --- QC / system columns (prefixed _ → drop before modeling) ---
    "逃脱数据出错":       "_qc_escape_error_flag",
    "逃脱数据错误":       "_qc_escape_error_flag",
}

# Survivor slot columns (1–4), built programmatically.
# Suffix pattern: slot 1 → '', slot 2 → '.1', slot 3 → '.2', slot 4 → '.3'
_SURVIVOR_ALIASES: dict[str, str] = {}
for _i in range(1, 5):
    _sfx = "" if _i == 1 else f".{_i - 1}"
    _SURVIVOR_ALIASES[f"求生者{_i}ID"]       = f"survivor{_i}_player"
    _SURVIVOR_ALIASES[f"人ID{_i}"]           = f"survivor{_i}_player"
    _SURVIVOR_ALIASES[f"使用角色{_sfx}"]     = f"survivor{_i}_character"
    _SURVIVOR_ALIASES[f"角色{_sfx}"]         = f"survivor{_i}_character"
    _SURVIVOR_ALIASES[f"修机进度{_sfx}"]     = f"survivor{_i}_repairs"
    _SURVIVOR_ALIASES[f"修机{_sfx}"]         = f"survivor{_i}_repairs"
    _SURVIVOR_ALIASES[f"救人数{_sfx}"]       = f"survivor{_i}_rescues"
    _SURVIVOR_ALIASES[f"救人{_sfx}"]         = f"survivor{_i}_rescues"
    _SURVIVOR_ALIASES[f"治疗数{_sfx}"]       = f"survivor{_i}_heals"
    _SURVIVOR_ALIASES[f"治疗{_sfx}"]         = f"survivor{_i}_heals"
    _SURVIVOR_ALIASES[f"砸板命中{_sfx}"]     = f"survivor{_i}_boards"
    _SURVIVOR_ALIASES[f"砸板{_sfx}"]         = f"survivor{_i}_boards"
    _SURVIVOR_ALIASES[f"牵制时长{_sfx}"]     = f"survivor{_i}_harassment"
    _SURVIVOR_ALIASES[f"牵制{_sfx}"]         = f"survivor{_i}_harassment"
    _SURVIVOR_ALIASES[f"结果{_sfx}"]         = f"survivor{_i}_result"
    _SURVIVOR_ALIASES[f"淘汰{_sfx}"]         = f"survivor{_i}_eliminated"

COLUMN_ALIASES.update(_SURVIVOR_ALIASES)

# ---------------------------------------------------------------------------
# Header detection marker sets
# ---------------------------------------------------------------------------
MODERN_RAW_MARKERS: set[str] = {
    "本页出错", "主场", "客场", "胜利方", "屠选", "屠名",
    "人队", "屠队", "地图", "场次",
}
MODERN_PLAYER_MARKERS: set[str] = {
    "人ID1", "人ID2", "人ID3", "人ID4", "屠ID", "求生者1ID",
    "修机", "救人", "牵制", "结果", "角色",
    "主场", "客场", "场次",
}
LEGACY_MARKERS: set[str] = {
    "阶段", "主场", "客场", "屠队", "地图", "胜利方", "屠选", "屠名",
}

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def detect_header_row(
    path: str,
    sheet_name: str,
    markers: set[str],
    max_search: int = 5,
) -> tuple[int, int]:
    """
    Scan header rows 0..max_search-1 and return (row, score) for the row
    whose column set has the most overlap with *markers*.
    """
    best_row, best_score = 0, -1
    for h in range(max_search):
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, header=h, nrows=0)
            cols = {str(c).strip() for c in df.columns}
            score = len(cols & markers)
            if score > best_score:
                best_row, best_score = h, score
        except Exception:
            continue
    return best_row, best_score


def normalize_columns(
    df: pd.DataFrame,
    alias_map: dict[str, str] | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Rename df columns via *alias_map* (defaults to COLUMN_ALIASES).
    Columns starting with 'Unnamed' or '_' are left untouched.

    Returns:
        (renamed_df, unmapped_col_names)
    """
    if alias_map is None:
        alias_map = COLUMN_ALIASES
    unmapped: list[str] = []
    new_names: dict = {}
    for col in df.columns:
        col_str = str(col).strip()
        if col_str in alias_map:
            new_names[col] = alias_map[col_str]
        elif col_str.startswith("Unnamed") or col_str.startswith("_"):
            pass  # system / metadata columns — leave alone
        else:
            unmapped.append(col_str)
    renamed = df.rename(columns=new_names)
    if verbose and unmapped:
        print(f"  Unmapped columns: {unmapped}")
    return renamed, unmapped


def _drop_template_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Excel files have pre-formatted template rows filled with default zeros.
    Drop them by requiring a known 'anchor' column to be non-null.
    Tries anchor columns in priority order; uses first match found.
    """
    for anchor in ["hunter_team", "hunter_player", "survivor1_player", "home_team"]:
        if anchor in df.columns:
            return df[df[anchor].notna()].copy()
    return df  # no anchor found — return as-is


def read_legacy_tab(
    tab_name: str,
    nrows: int | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Read one tab from the legacy 2020-2023.xlsx file.

    Auto-detects the header row, applies the legacy 角色 pre-rename quirk
    (bare '角色' and '角色.4' are MVP-related in legacy, NOT
    survivor1_character / hunter_character as they are in modern sheets),
    normalizes column names, and drops template rows.
    """
    path = os.path.join(DATA_DIR, LEGACY_FILE)
    header_row, _ = detect_header_row(path, tab_name, LEGACY_MARKERS)
    # keep_default_na=False + na_values=[''] preserves the literal string 'nan'
    # as a real player ID (player 楠 uses 'nan' as their in-game ID) while
    # still treating genuinely empty cells as NaN.
    df = pd.read_excel(path, sheet_name=tab_name, header=header_row, nrows=nrows,
                       keep_default_na=False, na_values=[""])

    # Pre-rename BEFORE the canonical alias pass to avoid the context collision.
    df = df.rename(columns={
        "角色":   "_qc_mvp_character",
        "角色.4": "_qc_mvp_character_alt",
    })

    df["_source"]     = f"legacy:{tab_name}"
    df["_header_row"] = header_row
    if normalize:
        df, _ = normalize_columns(df)
        df = _drop_template_rows(df)
    return df


def read_modern_sheet(
    filename: str,
    sheet_name: str,
    nrows: int | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Read one sheet from a modern (2024+) single-season xlsx file.

    Auto-detects the header row, normalizes column names, and drops
    template rows. Warns if header detection confidence is low (score < 2).
    """
    path = os.path.join(DATA_DIR, filename)
    markers = (
        MODERN_RAW_MARKERS if sheet_name == MODERN_RAW_SHEET
        else MODERN_PLAYER_MARKERS
    )
    header_row, score = detect_header_row(path, sheet_name, markers)
    if score < 2:
        print(f"⚠️  Low header detection confidence: {filename}:{sheet_name} (score={score})")
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, nrows=nrows,
                       keep_default_na=False, na_values=[""])
    df["_source"]     = f"{filename}:{sheet_name}"
    df["_header_row"] = header_row
    if normalize:
        df, _ = normalize_columns(df)
        df = _drop_template_rows(df)
    return df
