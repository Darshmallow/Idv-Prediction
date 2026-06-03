"""
tests/test_db.py
================
Tests for src/db.py — match assembly, outcome parsing, hunter backfill,
players table, and the SQLite schema.

Structure
---------
Unit tests (no file I/O, instant):
  - tournament_id / _source_year / _coa_year
  - _escaped_flag / _hunter_wins / _hw_from_escapes
  - _outcome_frame (escape-priority hunter_wins, 2020 fallback)
  - date parsers
  - build_players_table on synthetic matches

Integration tests (build the real DB once; marked slow):
  - matches: no null dates, value ranges, escape/winner consistency,
             draws retained, tournament IDs, COA years, alias normalization
  - players: roles, n_games, dual-role
  - SQLite: tables + indexes created, row counts match

Run:
    pytest tests/test_db.py -v
    pytest tests/test_db.py -m "not slow"      # unit only
"""

import sys
import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from db import (
    tournament_id,
    _source_year,
    _coa_year,
    COA_YEAR,
    _escaped_flag,
    _hunter_wins,
    _hw_from_escapes,
    _outcome_frame,
    _escapes_from_score,
    _parse_legacy_dates,
    _make_modern_dates,
    build_matches,
    build_players_table,
    write_to_db,
    PLAYER_ID_COLS,
)


# ===========================================================================
# UNIT — tournament identifiers
# ===========================================================================

class TestTournamentId:
    def test_modern_ivl(self):
        assert tournament_id("2024IVL夏季赛常规赛.xlsx") == "IVL_2024_summer_regular"
        assert tournament_id("2025IVL秋季赛季后赛.xlsx") == "IVL_2025_fall_playoffs"

    def test_modern_ijl(self):
        assert tournament_id("2024IJL夏季赛常规赛.xlsx") == "IJL_2024_summer_regular"
        assert tournament_id("2025IJL秋季赛季后赛.xlsx") == "IJL_2025_fall_playoffs"

    def test_modern_coa_merges_groups_and_knockout(self):
        assert tournament_id("COA9 全球总决赛小组赛.xlsx") == "COA9"
        assert tournament_id("COA9 全球总决赛淘汰赛.xlsx") == "COA9"
        assert tournament_id("COA8 全球总决赛小组赛.xlsx") == "COA8"

    def test_legacy_raw_tab_with_stage(self):
        assert tournament_id("2020原始", stage="夏季赛常规赛") == "IVL_2020_summer_regular"
        assert tournament_id("2023原始", stage="秋季赛季后赛") == "IVL_2023_fall_playoffs"

    def test_legacy_coa_tab(self):
        assert tournament_id("COA4") == "COA4"
        assert tournament_id("COA6") == "COA6"


class TestYearResolution:
    def test_coa9_is_2026(self):
        assert COA_YEAR[9] == 2026

    def test_coa_sequence_steps_by_one(self):
        assert COA_YEAR == {4: 2021, 5: 2022, 6: 2023, 7: 2024, 8: 2025, 9: 2026}

    def test_source_year_from_prefix(self):
        assert _source_year("2024IVL夏季赛常规赛.xlsx") == 2024
        assert _source_year("2020原始") == 2020

    def test_source_year_falls_back_to_coa(self):
        assert _source_year("COA9 全球总决赛小组赛.xlsx") == 2026
        assert _source_year("COA4") == 2021

    def test_coa_year_helper(self):
        assert _coa_year("COA8 ...") == 2025
        assert _coa_year("nothing") is None


# ===========================================================================
# UNIT — outcome parsing
# ===========================================================================

class TestEscapedFlag:
    def test_escaped(self):
        assert _escaped_flag("出门") == 1

    def test_eliminated(self):
        assert _escaped_flag("淘汰") == 0

    def test_eliminated_variant(self):
        assert _escaped_flag("淘汰·") == 0          # the dotted variant

    def test_junk_is_none(self):
        assert _escaped_flag("0") is None
        assert _escaped_flag("58") is None
        assert _escaped_flag("") is None

    def test_nan_is_none(self):
        assert _escaped_flag(np.nan) is None
        assert _escaped_flag(None) is None


class TestHunterWins:
    def test_basic(self):
        assert _hunter_wins("屠") == 1
        assert _hunter_wins("人") == 0

    def test_draw_and_unknown(self):
        assert _hunter_wins("平") is None
        assert _hunter_wins(np.nan) is None


class TestHwFromEscapes:
    def test_hunter_dominant(self):
        assert _hw_from_escapes(0) == 1
        assert _hw_from_escapes(1) == 1

    def test_survivor_dominant(self):
        assert _hw_from_escapes(3) == 0
        assert _hw_from_escapes(4) == 0

    def test_draw_is_none(self):
        assert _hw_from_escapes(2) is None

    def test_nan_is_none(self):
        assert _hw_from_escapes(pd.NA) is None
        assert _hw_from_escapes(np.nan) is None


class TestOutcomeFrame:
    def _df(self, r1, r2, r3, r4, winner):
        return pd.DataFrame({
            "survivor1_result": [r1], "survivor2_result": [r2],
            "survivor3_result": [r3], "survivor4_result": [r4],
            "winner_side": [winner],
        })

    def test_all_escaped(self):
        out = _outcome_frame(self._df("出门", "出门", "出门", "出门", "人"))
        assert out["n_escaped"].iloc[0] == 4
        assert out["hunter_wins"].iloc[0] == 0          # survivors win

    def test_all_eliminated(self):
        out = _outcome_frame(self._df("淘汰", "淘汰", "淘汰", "淘汰", "屠"))
        assert out["n_escaped"].iloc[0] == 0
        assert out["hunter_wins"].iloc[0] == 1

    def test_two_two_is_draw(self):
        out = _outcome_frame(self._df("出门", "出门", "淘汰", "淘汰", "平"))
        assert out["n_escaped"].iloc[0] == 2
        assert pd.isna(out["hunter_wins"].iloc[0])      # draw retained

    def test_escape_priority_over_winner(self):
        """Even if winner_side disagrees, hunter_wins follows the escape count."""
        out = _outcome_frame(self._df("淘汰", "淘汰", "淘汰", "淘汰", "人"))
        assert out["n_escaped"].iloc[0] == 0
        assert out["hunter_wins"].iloc[0] == 1          # escapes win, not winner_side

    def test_partial_results_give_null_n_escaped(self):
        out = _outcome_frame(self._df("出门", "出门", None, "淘汰", "人"))
        assert pd.isna(out["n_escaped"].iloc[0])        # incomplete → NULL

    def test_no_per_survivor_escaped_columns(self):
        """The per-survivor flags are internal only — not in the output."""
        out = _outcome_frame(self._df("出门", "淘汰", "出门", "淘汰", "平"))
        assert list(out.columns) == ["n_escaped", "hunter_wins"]


# ===========================================================================
# UNIT — _escapes_from_score (2020 path)
# ===========================================================================

class TestEscapesFromScore:
    def _df(self, winner, home, away):
        return pd.DataFrame({
            "winner_side": [winner],
            "home_score":  [home],
            "away_score":  [away],
        })

    def test_hunter_shutout(self):
        assert _escapes_from_score(self._df("屠", 0, 5)).iloc[0] == 0

    def test_hunter_one_escape(self):
        assert _escapes_from_score(self._df("屠", 1, 3)).iloc[0] == 1

    def test_survivor_shutout(self):
        assert _escapes_from_score(self._df("人", 5, 0)).iloc[0] == 4

    def test_survivor_one_eliminated(self):
        assert _escapes_from_score(self._df("人", 3, 1)).iloc[0] == 3

    def test_draw(self):
        assert _escapes_from_score(self._df("平", 2, 2)).iloc[0] == 2

    def test_home_away_symmetry(self):
        # score is symmetric — same result regardless of which team is home
        a = _escapes_from_score(self._df("屠", 0, 5)).iloc[0]
        b = _escapes_from_score(self._df("屠", 5, 0)).iloc[0]
        assert a == b == 0

    def test_missing_score_gives_null(self):
        assert pd.isna(_escapes_from_score(self._df("屠", None, None)).iloc[0])


class TestNEscapedPriority:
    """n_escaped uses total_escapes first, then per-survivor sum, then scores."""

    def test_total_escapes_column_used_directly(self):
        df = pd.DataFrame({
            "total_escapes":   [3],
            "survivor1_result": ["淘汰"],  # would give 0 if summed — ignored
            "survivor2_result": ["淘汰"],
            "survivor3_result": ["淘汰"],
            "survivor4_result": ["淘汰"],
            "winner_side":      ["人"],
            "home_score":       [3], "away_score": [1],
        })
        out = _outcome_frame(df)
        assert out["n_escaped"].iloc[0] == 3   # from total_escapes, not summed

    def test_per_survivor_sum_used_when_no_total(self):
        df = pd.DataFrame({
            "survivor1_result": ["出门"],
            "survivor2_result": ["出门"],
            "survivor3_result": ["淘汰"],
            "survivor4_result": ["淘汰"],
            "winner_side":      ["平"],
        })
        out = _outcome_frame(df)
        assert out["n_escaped"].iloc[0] == 2

    def test_score_fallback_for_2020_style(self):
        """No total_escapes, no results → score derivation."""
        df = pd.DataFrame({
            "winner_side": ["人"],
            "home_score":  [3], "away_score": [1],
        })
        out = _outcome_frame(df)
        assert out["n_escaped"].iloc[0] == 3


# ===========================================================================
# UNIT — date parsers
# ===========================================================================

class TestDateParsers:
    def test_legacy_dot_and_star(self):
        out = _parse_legacy_dates(pd.Series(["6.25", "6*9", "4/16"]), 2023)
        assert out == ["2023-06-25", "2023-06-09", "2023-04-16"]

    def test_legacy_no_year_returns_none(self):
        assert _parse_legacy_dates(pd.Series(["6.25"]), None) == [None]

    def test_legacy_nan(self):
        assert _parse_legacy_dates(pd.Series([np.nan]), 2023) == [None]

    def test_modern_month_day(self):
        out = _make_modern_dates(pd.Series([6, 6]), pd.Series([6, 25]), 2025)
        assert out == ["2025-06-06", "2025-06-25"]

    def test_modern_month_day_distinct_months(self):
        out = _make_modern_dates(pd.Series([4, 11]), pd.Series([1, 30]), 2026)
        assert out == ["2026-04-01", "2026-11-30"]

    def test_modern_missing(self):
        assert _make_modern_dates(pd.Series([np.nan]), pd.Series([5]), 2025) == [None]
        assert _make_modern_dates(pd.Series([5]), pd.Series([5]), None) == [None]


# ===========================================================================
# UNIT — players table from synthetic matches
# ===========================================================================

class TestBuildPlayersTable:
    def _matches(self):
        return pd.DataFrame({
            "match_id": [1, 2, 3],
            "date": ["2024-01-01", "2024-06-01", "2025-01-01"],
            "hunter_player":    ["alice", "bob",   "alice"],
            "survivor1_player": ["carol", "alice", "carol"],   # alice is both
            "survivor2_player": ["dave",  "carol", "dave"],
            "survivor3_player": [None,    "dave",  "erin"],
            "survivor4_player": [None,    None,    None],
        })

    def test_roles(self):
        p = build_players_table(self._matches()).set_index("canonical_id")
        assert p.loc["alice", "known_roles"] == "both"
        assert p.loc["bob",   "known_roles"] == "hunter"
        assert p.loc["carol", "known_roles"] == "survivor"

    def test_n_games(self):
        p = build_players_table(self._matches()).set_index("canonical_id")
        assert p.loc["alice", "n_games"] == 3      # hunter×2 + survivor×1, 3 distinct matches
        assert p.loc["carol", "n_games"] == 3
        assert p.loc["erin",  "n_games"] == 1

    def test_first_last_seen(self):
        p = build_players_table(self._matches()).set_index("canonical_id")
        assert p.loc["alice", "first_seen"] == "2024-01-01"
        assert p.loc["alice", "last_seen"]  == "2025-01-01"

    def test_no_null_or_empty_ids(self):
        p = build_players_table(self._matches())
        assert p["canonical_id"].notna().all()
        assert (p["canonical_id"] != "").all()


# ===========================================================================
# INTEGRATION — build the real database once
# ===========================================================================

@pytest.fixture(scope="module")
def matches():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_matches(verbose=False)


@pytest.fixture(scope="module")
def players(matches):
    return build_players_table(matches)


@pytest.fixture(scope="module")
def db_path(matches, players, tmp_path_factory):
    p = tmp_path_factory.mktemp("db") / "test_idv.db"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_to_db(matches, players, str(p))
    return str(p)


@pytest.mark.slow
class TestMatchesData:
    def test_nonempty(self, matches):
        assert len(matches) > 5000

    def test_no_null_dates(self, matches):
        assert matches["date"].isna().sum() == 0

    def test_match_id_unique_and_sequential(self, matches):
        assert matches["match_id"].is_unique
        assert matches["match_id"].min() == 1
        assert matches["match_id"].max() == len(matches)

    def test_n_escaped_range(self, matches):
        vals = set(matches["n_escaped"].dropna().unique())
        assert vals <= {0, 1, 2, 3, 4}

    def test_no_per_survivor_escaped_columns(self, matches):
        leftover = [c for c in matches.columns
                    if c.startswith("survivor") and c.endswith("_escaped")]
        assert leftover == [], f"per-survivor escaped flags should be gone: {leftover}"

    def test_expected_columns(self, matches):
        assert list(matches.columns) == [
            "match_id", "date", "tournament",
            "hunter_player", "survivor1_player", "survivor2_player",
            "survivor3_player", "survivor4_player",
            "n_escaped", "hunter_wins", "source_file", "source_row",
        ]

    def test_hunter_wins_consistent_with_escapes(self, matches):
        """By construction hunter_wins must agree with n_escaped everywhere."""
        m = matches.dropna(subset=["n_escaped"]).copy()
        expected = m["n_escaped"].map(_hw_from_escapes)
        got = m["hunter_wins"]
        # both NA where draw, equal where decided
        mism = (got.fillna(-1) != pd.array(expected, dtype="Int64").to_numpy(na_value=-1))
        assert mism.sum() == 0, f"{mism.sum()} rows where hunter_wins disagrees with n_escaped"

    def test_draws_retained(self, matches):
        n_draws = (matches["n_escaped"] == 2).sum()
        assert n_draws > 1000, "draws should be retained, not dropped"

    def test_hunter_win_rate_plausible(self, matches):
        decided = matches["hunter_wins"].dropna()
        rate = (decided == 1).mean()
        assert 0.45 <= rate <= 0.58, f"hunter win rate {rate:.3f} out of expected band"

    def test_tournament_ids_clean(self, matches):
        t = set(matches["tournament"].unique())
        # COA merged (no _groups/_knockout suffix)
        assert "COA9" in t and "COA8" in t
        assert not any("groups" in x or "knockout" in x for x in t)
        # modern leagues present with the expected format
        assert "IVL_2024_summer_regular" in t
        assert "IJL_2025_fall_regular" in t

    def test_coa9_dates_in_2026(self, matches):
        coa9 = matches[matches["tournament"] == "COA9"]["date"].dropna()
        assert coa9.str.startswith("2026").all()

    def test_coa8_dates_in_2025(self, matches):
        coa8 = matches[matches["tournament"] == "COA8"]["date"].dropna()
        assert coa8.str.startswith("2025").all()

    def test_no_alias_ids_leak(self, matches):
        """Normalized IDs only — canonical present, aliases absent."""
        ids = set()
        for col in PLAYER_ID_COLS:
            ids |= set(matches[col].dropna().unique())
        assert "dongx" in ids        # canonical
        assert "dx" not in ids       # alias of dongx
        assert "yan" in ids
        assert "qy" not in ids       # alias of yan

    def test_hunter_coverage_high(self, matches):
        null_h = matches["hunter_player"].isna().sum()
        assert null_h < 30, f"{null_h} matches missing hunter — backfill regressed?"


@pytest.mark.slow
class TestPlayersData:
    def test_nonempty(self, players):
        assert len(players) > 300

    def test_roles_valid(self, players):
        assert set(players["known_roles"].unique()) <= {"hunter", "survivor", "both"}

    def test_n_games_positive(self, players):
        assert (players["n_games"] > 0).all()

    def test_ppxia_is_dual_role(self, players):
        row = players[players["canonical_id"] == "ppxia"]
        assert len(row) == 1
        assert row["known_roles"].iloc[0] == "both"

    def test_canonical_ids_unique(self, players):
        assert players["canonical_id"].is_unique


@pytest.mark.slow
class TestSqlite:
    def test_tables_exist(self, db_path):
        conn = sqlite3.connect(db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        assert {"matches", "players"} <= tables

    def test_indexes_exist(self, db_path):
        conn = sqlite3.connect(db_path)
        idx = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
        conn.close()
        assert "idx_matches_date" in idx
        assert "idx_matches_hunter_player" in idx
        assert "idx_matches_tournament" in idx

    def test_row_counts_match(self, db_path, matches, players):
        conn = sqlite3.connect(db_path)
        nm = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        npl = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        conn.close()
        assert nm == len(matches)
        assert npl == len(players)

    def test_matches_pk_is_match_id(self, db_path):
        conn = sqlite3.connect(db_path)
        cols = {r[1]: r[5] for r in conn.execute("PRAGMA table_info(matches)").fetchall()}
        conn.close()
        assert cols["match_id"] == 1          # pk flag
