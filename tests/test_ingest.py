"""
tests/test_ingest.py
====================
Tests for src/ingest.py.

Structure
---------
Unit tests   (no file I/O, instant):
  - COLUMN_ALIASES contents
  - normalize_columns behaviour
  - _drop_template_rows behaviour
  - survivor alias coverage

Integration tests (read real xlsx files, ~seconds each):
  - read_legacy_tab   — schema, pre-rename quirk, no template rows
  - read_modern_sheet — schema, header detection, no template rows

Parametrized smoke tests (all files load without error):
  - every legacy tab
  - every modern file (原始数据 + 赛后数据)

Run:
    pytest tests/                  # everything
    pytest tests/ -m "not slow"    # unit + 2-file integration only
    pytest tests/ -v               # verbose
"""

import pandas as pd
import pytest

from data.ingest import (
    COLUMN_ALIASES,
    LEGACY_FILE, LEGACY_RAW_TABS, LEGACY_COA_TABS,
    IVL_FILES, IJL_FILES, COA_FILES, ALL_MODERN_FILES,
    MODERN_RAW_SHEET, MODERN_PLAYER_SHEET,
    DATA_DIR,
    detect_header_row,
    normalize_columns,
    _drop_template_rows,
    read_legacy_tab,
    read_modern_sheet,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_df(**cols):
    """Build a tiny DataFrame from keyword args: col_name=[values, ...]."""
    return pd.DataFrame(cols)


# ===========================================================================
# UNIT TESTS — no file I/O
# ===========================================================================

class TestColumnAliases:
    def test_has_enough_entries(self):
        assert len(COLUMN_ALIASES) >= 130, (
            f"Expected ≥130 entries, got {len(COLUMN_ALIASES)}"
        )

    def test_key_game_columns_present(self):
        required = {
            "主场": "home_team",
            "客场": "away_team",
            "屠队": "hunter_team",
            "屠名": "hunter_player",
            "屠ID": "hunter_player",
            "屠选": "hunter_character",
            "人队": "survivor_team",
            "胜利方": "winner_side",
            "地图": "map_name",
            "半场": "half",
            "剩机台数": "gens_remaining",
            "剩机": "gens_remaining",
        }
        for chinese, english in required.items():
            assert COLUMN_ALIASES.get(chinese) == english, (
                f"'{chinese}' should map to '{english}', "
                f"got '{COLUMN_ALIASES.get(chinese)}'"
            )

    def test_all_four_survivor_slots_have_player(self):
        for i in range(1, 5):
            assert f"survivor{i}_player" in COLUMN_ALIASES.values(), (
                f"survivor{i}_player missing from COLUMN_ALIASES values"
            )

    def test_all_four_survivor_slots_have_character(self):
        for i in range(1, 5):
            assert f"survivor{i}_character" in COLUMN_ALIASES.values()

    def test_all_four_survivor_slots_have_repairs(self):
        for i in range(1, 5):
            assert f"survivor{i}_repairs" in COLUMN_ALIASES.values()

    def test_all_four_survivor_slots_have_result(self):
        for i in range(1, 5):
            assert f"survivor{i}_result" in COLUMN_ALIASES.values()

    def test_qc_columns_prefixed_with_underscore(self):
        """Columns meant to be dropped pre-modeling must start with '_qc_'."""
        qc_values = [v for v in COLUMN_ALIASES.values() if "qc" in v]
        for v in qc_values:
            assert v.startswith("_qc_"), f"QC column '{v}' should start with '_qc_'"

    def test_modern_survivor_id_aliases(self):
        """2025/COA9 use 人ID1..4 instead of 求生者1ID..4."""
        for i in range(1, 5):
            assert COLUMN_ALIASES.get(f"人ID{i}") == f"survivor{i}_player"

    def test_legacy_survivor_id_aliases(self):
        for i in range(1, 5):
            assert COLUMN_ALIASES.get(f"求生者{i}ID") == f"survivor{i}_player"


class TestNormalizeColumns:
    def test_renames_known_columns(self):
        df = _synthetic_df(**{"主场": ["A"], "客场": ["B"], "屠队": ["C"]})
        renamed, _ = normalize_columns(df)
        assert "home_team"   in renamed.columns
        assert "away_team"   in renamed.columns
        assert "hunter_team" in renamed.columns

    def test_leaves_unmapped_columns_untouched(self):
        df = _synthetic_df(**{"主场": ["A"], "局": ["1"]})
        renamed, unmapped = normalize_columns(df)
        assert "home_team" in renamed.columns   # known → renamed
        assert "局" in renamed.columns          # unknown → kept
        assert "局" in unmapped

    def test_leaves_unnamed_columns_untouched(self):
        df = _synthetic_df(**{"主场": ["A"]})
        df["Unnamed: 3"] = None
        renamed, unmapped = normalize_columns(df)
        assert "Unnamed: 3" in renamed.columns
        assert "Unnamed: 3" not in unmapped     # not reported as unmapped

    def test_leaves_underscore_columns_untouched(self):
        df = _synthetic_df(**{"主场": ["A"]})
        df["_source"] = "test"
        renamed, unmapped = normalize_columns(df)
        assert "_source" in renamed.columns
        assert "_source" not in unmapped

    def test_returns_unmapped_list(self):
        df = _synthetic_df(**{"主场": ["A"], "whatever": ["x"]})
        _, unmapped = normalize_columns(df)
        assert "whatever" in unmapped
        assert "主场" not in unmapped           # known cols not in unmapped

    def test_empty_dataframe_ok(self):
        df = pd.DataFrame()
        renamed, unmapped = normalize_columns(df)
        assert list(renamed.columns) == []
        assert unmapped == []

    def test_custom_alias_map(self):
        df = _synthetic_df(**{"foo": [1], "bar": [2]})
        renamed, _ = normalize_columns(df, alias_map={"foo": "baz"})
        assert "baz" in renamed.columns
        assert "bar" in renamed.columns  # not in custom map → kept


class TestDropTemplateRows:
    def test_drops_null_hunter_team_rows(self):
        df = _synthetic_df(
            hunter_team=["TeamA", None, "TeamB", None],
            hunter_player=["p1", None, "p2", None],
        )
        result = _drop_template_rows(df)
        assert len(result) == 2
        assert result["hunter_team"].notna().all()

    def test_falls_back_to_hunter_player(self):
        """If hunter_team absent, should use hunter_player."""
        df = _synthetic_df(
            hunter_player=["p1", None, "p2"],
            home_team=["A", "B", "C"],
        )
        result = _drop_template_rows(df)
        assert len(result) == 2

    def test_falls_back_to_survivor1_player(self):
        df = _synthetic_df(
            survivor1_player=["s1", None, "s2"],
        )
        result = _drop_template_rows(df)
        assert len(result) == 2

    def test_falls_back_to_home_team(self):
        df = _synthetic_df(
            home_team=["A", None, "B"],
        )
        result = _drop_template_rows(df)
        assert len(result) == 2

    def test_no_anchor_returns_unchanged(self):
        df = _synthetic_df(score=[1, 2, 3])
        result = _drop_template_rows(df)
        assert len(result) == 3

    def test_prefers_hunter_team_over_others(self):
        """hunter_team takes priority; rows null in hunter_team but non-null
        in survivor1_player should still be dropped."""
        df = _synthetic_df(
            hunter_team=[None, "TeamA"],
            survivor1_player=["s1", "s2"],  # both non-null
        )
        result = _drop_template_rows(df)
        assert len(result) == 1             # only the row with hunter_team kept

    def test_returns_copy(self):
        df = _synthetic_df(hunter_team=["A", None])
        result = _drop_template_rows(df)
        result["hunter_team"] = "X"         # mutating result should not affect df
        assert df.loc[0, "hunter_team"] == "A"


# ===========================================================================
# INTEGRATION TESTS — reads real xlsx files
# ===========================================================================

class TestReadLegacyTab:
    def test_2023_loads_and_has_rows(self):
        df = read_legacy_tab("2023原始")
        assert len(df) > 0, "2023原始 loaded 0 rows"

    def test_2023_key_columns_present(self):
        df = read_legacy_tab("2023原始")
        for col in ["home_team", "away_team", "hunter_team", "hunter_player",
                    "hunter_character", "winner_side", "map_name"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_2023_survivor_columns_present(self):
        df = read_legacy_tab("2023原始")
        for i in range(1, 5):
            assert f"survivor{i}_player" in df.columns, f"survivor{i}_player missing"
            assert f"survivor{i}_repairs" in df.columns, f"survivor{i}_repairs missing"

        # Slots 2-4: '角色.1' / '角色.2' / '角色.3' normalise to survivorN_character.
        for i in range(2, 5):
            assert f"survivor{i}_character" in df.columns, (
                f"survivor{i}_character missing"
            )

        # Slot 1: bare '角色' is pre-renamed to _qc_mvp_character in legacy files
        # (it's the MVP character field, not survivor 1's pick), so
        # survivor1_character is intentionally absent in legacy data.
        assert "survivor1_character" not in df.columns, (
            "survivor1_character should NOT exist in legacy — "
            "bare '角色' is pre-renamed to _qc_mvp_character"
        )
        assert "_qc_mvp_character" in df.columns

    def test_2023_no_template_rows(self):
        df = read_legacy_tab("2023原始")
        assert df["hunter_team"].isna().sum() == 0, (
            "Template rows remain — hunter_team has nulls"
        )

    def test_legacy_pre_rename_quirk(self):
        """Bare '角色' must be pre-renamed; must NOT appear as survivor1_character."""
        df = read_legacy_tab("2023原始")
        # The pre-rename should have converted 角色 → _qc_mvp_character
        assert "_qc_mvp_character" in df.columns, (
            "Legacy pre-rename quirk: _qc_mvp_character column missing"
        )
        assert "_qc_mvp_character_alt" in df.columns, (
            "Legacy pre-rename quirk: _qc_mvp_character_alt column missing"
        )

    def test_2020_loads_without_survivor_team(self):
        """2020 has no survivor_team column — must not crash."""
        df = read_legacy_tab("2020原始")
        assert len(df) > 0
        assert "survivor_team" not in df.columns  # absent in 2020

    def test_source_column_set(self):
        df = read_legacy_tab("2021原始")
        assert "_source" in df.columns
        assert (df["_source"] == "legacy:2021原始").all()

    def test_normalize_false_returns_raw_columns(self):
        """normalize=False should leave Chinese column names intact."""
        df = read_legacy_tab("2023原始", normalize=False)
        assert "主场" in df.columns
        assert "home_team" not in df.columns


class TestReadModernSheet:
    SAMPLE_FILE = "2025IVL夏季赛常规赛.xlsx"

    def test_raw_sheet_loads_and_has_rows(self):
        df = read_modern_sheet(self.SAMPLE_FILE, MODERN_RAW_SHEET)
        assert len(df) > 0

    def test_raw_sheet_key_columns_present(self):
        df = read_modern_sheet(self.SAMPLE_FILE, MODERN_RAW_SHEET)
        for col in ["home_team", "away_team", "hunter_team", "hunter_player",
                    "hunter_character", "winner_side", "map_name", "half"]:
            assert col in df.columns, f"Missing column in 原始数据: {col}"

    def test_raw_sheet_no_template_rows(self):
        df = read_modern_sheet(self.SAMPLE_FILE, MODERN_RAW_SHEET)
        assert df["hunter_team"].isna().sum() == 0

    def test_player_sheet_loads_and_has_rows(self):
        df = read_modern_sheet(self.SAMPLE_FILE, MODERN_PLAYER_SHEET)
        assert len(df) > 0

    def test_player_sheet_survivor_columns_present(self):
        df = read_modern_sheet(self.SAMPLE_FILE, MODERN_PLAYER_SHEET)
        for i in range(1, 5):
            assert f"survivor{i}_player" in df.columns, (
                f"survivor{i}_player missing from 赛后数据"
            )

    def test_modern_bare_jiao_se_maps_to_survivor1_character(self):
        """In modern 赛后数据, bare '角色' (no suffix) = survivor 1's character."""
        df = read_modern_sheet(self.SAMPLE_FILE, MODERN_PLAYER_SHEET)
        assert "survivor1_character" in df.columns
        # The raw Chinese column '角色' must NOT survive normalization
        assert "角色" not in df.columns

    def test_source_column_set(self):
        df = read_modern_sheet(self.SAMPLE_FILE, MODERN_RAW_SHEET)
        assert "_source" in df.columns
        expected = f"{self.SAMPLE_FILE}:{MODERN_RAW_SHEET}"
        assert (df["_source"] == expected).all()

    def test_normalize_false_returns_raw_columns(self):
        df = read_modern_sheet(self.SAMPLE_FILE, MODERN_RAW_SHEET, normalize=False)
        assert "主场" in df.columns
        assert "home_team" not in df.columns

    def test_coa_file_loads(self):
        df = read_modern_sheet("COA9 全球总决赛小组赛.xlsx", MODERN_RAW_SHEET)
        assert len(df) > 0
        assert "hunter_character" in df.columns


# ===========================================================================
# PARAMETRIZED SMOKE TESTS — every file loads without error
# ===========================================================================

@pytest.mark.slow
@pytest.mark.parametrize("tab", LEGACY_RAW_TABS + LEGACY_COA_TABS)
def test_legacy_tab_smoke(tab):
    """Every legacy tab loads, normalizes, and has at least 1 row."""
    df = read_legacy_tab(tab)
    assert len(df) > 0, f"{tab} loaded 0 rows"
    assert "hunter_team" in df.columns or "home_team" in df.columns


@pytest.mark.slow
@pytest.mark.parametrize("filename", ALL_MODERN_FILES)
def test_modern_raw_smoke(filename):
    """Every modern file's 原始数据 loads without error and has rows."""
    df = read_modern_sheet(filename, MODERN_RAW_SHEET)
    assert len(df) > 0, f"{filename}:原始数据 loaded 0 rows"
    assert "hunter_team" in df.columns


@pytest.mark.slow
@pytest.mark.parametrize("filename", ALL_MODERN_FILES)
def test_modern_player_smoke(filename):
    """Every modern file's 赛后数据 loads without error and has rows."""
    df = read_modern_sheet(filename, MODERN_PLAYER_SHEET)
    assert len(df) > 0, f"{filename}:赛后数据 loaded 0 rows"
    assert "survivor1_player" in df.columns
