"""
tests/test_players.py
Tests for src/players.py — alias map construction and ID normalization.

Run:
    pytest tests/test_players.py -v
"""

import sys
import os

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from players import (
    ALIAS_MAP,
    MANUAL_OVERRIDES,
    PLAYER_COLUMNS,
    normalize_player_id,
    normalize_player_ids,
    _parse_alias_tab,
    _build_alias_map,
    DATA_DIR,
    LEGACY_FILE,
)

# ===========================================================================
# UNIT TESTS — alias map structure
# ===========================================================================

class TestAliasMap:
    def test_has_expected_entries(self):
        # 15 multi-ID players × ≥1 alias each; jiaxin and yan have 2 each
        assert len(ALIAS_MAP) >= 15

    def test_known_hunter_aliases(self):
        """Legacy hunter IDs that should be mapped."""
        assert ALIAS_MAP.get("dx")       == "dongx"    # 东玄
        assert ALIAS_MAP.get("qy")       == "yan"      # 祈颜 (old)
        assert ALIAS_MAP.get("qyq")      == "yan"      # 祈颜 (middle)
        assert ALIAS_MAP.get("ssy")      == "sese"     # 瑟瑟
        assert ALIAS_MAP.get("taoxing")  == "tx"       # 逃行  (_CANONICAL_FIXES)

    def test_known_survivor_aliases(self):
        """Legacy survivor IDs that should be mapped."""
        assert ALIAS_MAP.get("huli")    == "fox"      # 狐狸
        assert ALIAS_MAP.get("mm")      == "drop"     # 神坠
        assert ALIAS_MAP.get("angeldd") == "add"      # 啊咚咚
        assert ALIAS_MAP.get("xiaoci")  == "chuc"     # 小慈
        assert ALIAS_MAP.get("df")      == "bm"       # 东风
        assert ALIAS_MAP.get("bbe")     == "jiaxin"   # 嘉新 (oldest)
        assert ALIAS_MAP.get("qingkk")  == "jiaxin"   # 嘉新 (middle)

    def test_canonical_ids_not_in_alias_keys(self):
        """A canonical ID should never point somewhere else (no chain)."""
        for old, canon in ALIAS_MAP.items():
            assert canon not in ALIAS_MAP, (
                f"Chain detected: {old!r} → {canon!r} → {ALIAS_MAP[canon]!r}"
            )

    def test_no_self_mappings(self):
        for old, canon in ALIAS_MAP.items():
            assert old != canon, f"Self-mapping: {old!r} → {canon!r}"

    def test_all_values_lowercase(self):
        for old, canon in ALIAS_MAP.items():
            assert canon == canon.lower(), f"Non-lowercase canonical: {canon!r}"
            assert old   == old.lower(),   f"Non-lowercase alias key: {old!r}"

    def test_manual_overrides_merged(self):
        """If MANUAL_OVERRIDES is non-empty, its entries must appear in ALIAS_MAP."""
        for old, canon in MANUAL_OVERRIDES.items():
            assert ALIAS_MAP.get(old.lower().strip()) == canon.lower().strip(), (
                f"MANUAL_OVERRIDE {old!r} → {canon!r} not found in ALIAS_MAP"
            )



# ===========================================================================
# UNIT TESTS — normalize_player_id
# ===========================================================================

class TestNormalizePlayerId:
    def test_alias_resolved(self):
        assert normalize_player_id("dx")      == "dongx"
        assert normalize_player_id("qy")      == "yan"
        assert normalize_player_id("taoxing") == "tx"
        assert normalize_player_id("huli")    == "fox"

    def test_uppercase_lowercased_and_resolved(self):
        assert normalize_player_id("DongX")  == "dongx"
        assert normalize_player_id("Fox")    == "fox"
        assert normalize_player_id("DX")     == "dongx"

    def test_canonical_id_unchanged(self):
        assert normalize_player_id("dongx")   == "dongx"
        assert normalize_player_id("yan")     == "yan"
        assert normalize_player_id("tx")      == "tx"      # override made tx canonical
        assert normalize_player_id("taoxing") == "tx"      # taoxing → tx via MANUAL_OVERRIDES

    def test_unknown_id_returned_as_lowercase(self):
        assert normalize_player_id("jin")     == "jin"
        assert normalize_player_id("choai")   == "choai"
        assert normalize_player_id("UnknOwn") == "unknown"

    def test_empty_string_returns_empty(self):
        assert normalize_player_id("") == ""

    def test_nan_string_is_real_player_id(self):
        # 'nan' is the actual in-game ID of player 楠 — must NOT be treated as null
        assert normalize_player_id("nan") == "nan"
        assert normalize_player_id("NaN") == "nan"   # lowercased, preserved

    def test_none_string_returns_empty(self):
        assert normalize_player_id("none") == ""

    def test_whitespace_stripped(self):
        assert normalize_player_id("  dx  ") == "dongx"
        assert normalize_player_id("  jin ") == "jin"

    def test_non_string_input(self):
        # float NaN
        assert normalize_player_id(float("nan")) == ""
        # None-like handled without crash
        assert normalize_player_id(None) == ""


# ===========================================================================
# UNIT TESTS — normalize_player_ids (DataFrame)
# ===========================================================================

class TestNormalizePlayerIds:
    def _df(self, **cols):
        return pd.DataFrame(cols)

    def test_aliases_resolved_in_hunter_column(self):
        df = self._df(hunter_player=["dongx", "dx", "DongX"])
        result = normalize_player_ids(df)
        assert list(result["hunter_player"]) == ["dongx", "dongx", "dongx"]

    def test_aliases_resolved_in_survivor_columns(self):
        df = self._df(
            survivor1_player=["fox", "huli"],
            survivor2_player=["qy", "yan"],
        )
        result = normalize_player_ids(df)
        assert list(result["survivor1_player"]) == ["fox", "fox"]
        assert list(result["survivor2_player"]) == ["yan", "yan"]

    def test_null_cells_preserved(self):
        df = self._df(hunter_player=["dongx", None, np.nan, "dx"])
        result = normalize_player_ids(df)
        assert pd.isna(result["hunter_player"].iloc[1])
        assert pd.isna(result["hunter_player"].iloc[2])
        assert result["hunter_player"].iloc[0] == "dongx"
        assert result["hunter_player"].iloc[3] == "dongx"

    def test_missing_columns_skipped_silently(self):
        """DataFrames without all 5 player columns should not raise."""
        df = self._df(hunter_player=["dx"])
        result = normalize_player_ids(df)
        assert result["hunter_player"].iloc[0] == "dongx"
        for col in ["survivor1_player", "survivor2_player",
                    "survivor3_player", "survivor4_player"]:
            assert col not in result.columns

    def test_returns_copy(self):
        df = self._df(hunter_player=["dx"])
        result = normalize_player_ids(df)
        assert df["hunter_player"].iloc[0] == "dx"   # original unchanged
        assert result["hunter_player"].iloc[0] == "dongx"

    def test_unrelated_columns_untouched(self):
        df = self._df(hunter_player=["dx"], map_name=["moon_river"])
        result = normalize_player_ids(df)
        assert result["map_name"].iloc[0] == "moon_river"


# ===========================================================================
# INTEGRATION TEST — parse real tab
# ===========================================================================

class TestParseAliasTab:
    def test_tab_loads_without_error(self):
        import os
        path = os.path.join(DATA_DIR, LEGACY_FILE)
        entries = _parse_alias_tab(path)
        assert len(entries) > 0

    def test_returns_expected_count(self):
        import os
        path = os.path.join(DATA_DIR, LEGACY_FILE)
        entries = _parse_alias_tab(path)
        # Tab has ~120 survivor + hunter entries combined
        assert len(entries) >= 100

    def test_all_ids_lowercase(self):
        import os
        path = os.path.join(DATA_DIR, LEGACY_FILE)
        entries = _parse_alias_tab(path)
        for role, nick, ids, counts in entries:
            for id_ in ids:
                assert id_ == id_.lower(), f"Non-lowercase ID from tab: {id_!r}"

    def test_known_multi_id_entries_present(self):
        import os
        path = os.path.join(DATA_DIR, LEGACY_FILE)
        entries = _parse_alias_tab(path)
        all_ids = {i for _, _, ids, _ in entries for i in ids}
        for expected in ["dongx", "dx", "yan", "qy", "qyq", "tx", "taoxing",
                         "fox", "huli", "drop", "mm"]:
            assert expected in all_ids, f"Expected ID {expected!r} not found in tab"


# ===========================================================================
# INTEGRATION TEST — apply to real data
# ===========================================================================

@pytest.mark.slow
def test_normalize_removes_all_aliases_from_legacy():
    """After normalization, no legacy row should contain a known alias ID."""
    from ingest import read_legacy_tab, LEGACY_RAW_TABS
    for tab in LEGACY_RAW_TABS:
        df = normalize_player_ids(read_legacy_tab(tab))
        for col in PLAYER_COLUMNS:
            if col not in df.columns:
                continue
            found = set(df[col].dropna().str.lower()) & set(ALIAS_MAP.keys())
            assert not found, (
                f"{tab}:{col} still contains alias IDs after normalization: {found}"
            )


@pytest.mark.slow
def test_normalize_removes_all_aliases_from_modern():
    """After normalization, no modern file should contain a known alias ID."""
    from ingest import read_modern_sheet, ALL_MODERN_FILES, MODERN_PLAYER_SHEET
    for f in ALL_MODERN_FILES:
        df = normalize_player_ids(read_modern_sheet(f, MODERN_PLAYER_SHEET))
        for col in PLAYER_COLUMNS:
            if col not in df.columns:
                continue
            found = set(df[col].dropna().str.lower()) & set(ALIAS_MAP.keys())
            assert not found, (
                f"{f}:{col} still contains alias IDs after normalization: {found}"
            )
