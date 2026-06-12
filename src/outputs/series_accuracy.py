"""
src/outputs/series_accuracy.py
===============================
Conditional series-win accuracy: given the *actual* per-round lineups, how
often does the ordinal BT model correctly predict which team wins the series?

"Conditional" means we know who played in every round retroactively —
this isolates whether the player ratings carry signal about series outcomes,
stripping out lineup-selection uncertainty.

Metrics
-------
  accuracy   — fraction of series where predicted winner (P > 0.5) == actual
  Brier      — mean squared error on P(home wins series), baseline = 0.25 (50/50)

Output
------
  outputs/series_accuracy.txt  — console-style report
  (prints to stdout as well)

Usage
-----
    python src/outputs/series_accuracy.py               # top tiers only
    python src/outputs/series_accuracy.py --all         # both configs
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_DB = str(_ROOT / "data" / "processed" / "idv.db")
OUTPUTS    = _ROOT / "outputs"
OPTUNA_DIR = _ROOT / "outputs" / "optuna"
JSON_DIR   = _ROOT / "data" / "raw_json"

from model.bradley_terry import filter_complete, compute_weights
from model.ordinal import fit_ordinal, predict_probs
from model.team_prior import identify_new_players, compute_team_mean_priors

# IDV scoring table: n_escaped -> (hunter_pts, survivor_pts)
HALF_SCORE = {0: (5, 0), 1: (3, 1), 2: (2, 2), 3: (1, 3), 4: (0, 5)}

# ---------------------------------------------------------------------------
# Alias map: raw_name -> canonical
# ---------------------------------------------------------------------------

def build_alias_map() -> dict[str, str]:
    id_path = JSON_DIR / "id.json"
    with open(id_path) as f:
        id_data = json.load(f)
    alias_to_canon: dict[str, str] = {}
    for canon, aliases in id_data.items():
        for alias in aliases:
            alias_to_canon[str(alias).lower()] = str(canon).lower()
    return alias_to_canon


def normalize(name: str, alias_map: dict[str, str]) -> str | None:
    if not name or str(name) in ("-999", "-", "nan", ""):
        return None
    n = str(name).lower().strip()
    return alias_map.get(n, n)


# ---------------------------------------------------------------------------
# Parse JSON files → series records
# ---------------------------------------------------------------------------

JSON_FILES = [
    "IVL_COA_2020-2024.json",
    "IVL_COA_2025-2026.json",
    "oversea_2020-2024.json",
    "oversea_2025.json",
]

TIER_MAP = {
    "IVL": "IVL", "IJL": "IJL", "COA": "COA",
    "IVC": "IVC", "IVT": "IVT", "IVS": "IVS",
}

def _infer_tier(stage: str) -> str:
    for prefix, tier in TIER_MAP.items():
        if prefix.lower() in stage.lower():
            return tier
    return "OTHER"

def _infer_tier_from_tournament(tournament: str) -> str:
    for prefix, tier in TIER_MAP.items():
        if tournament.upper().startswith(prefix):
            return prefix
    return "OTHER"


def parse_all_series(alias_map: dict[str, str]) -> list[dict]:
    """
    Returns one record per series with:
      source_file, source_row, date, tournament, tier,
      home_team, away_team,
      rounds: list of {round_no, halves: [{home_hunts, hunter, survs, n_esc}x2]}
      actual_winner: 'home' | 'away' | 'draw'
      format: 3 | 5  (Bo3 or Bo5)
    """
    series_list = []

    for fname in JSON_FILES:
        fpath = JSON_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)

        for row_idx, rec in enumerate(data):
            home_team = str(rec.get("主场", ""))
            away_team = str(rec.get("客场", ""))
            date_str  = str(rec.get("日期", ""))
            stage     = str(rec.get("阶段", ""))
            games     = rec.get("games", [])
            if not games:
                continue

            # Infer tournament name and tier
            tournament = stage
            tier = _infer_tier(stage)

            # Group games by round number
            rounds_dict: dict = defaultdict(list)
            for g in games:
                rno_raw = g.get("场次号", 0)
                try:
                    rno = int(rno_raw)
                except (ValueError, TypeError):
                    rno = 99  # extra/tiebreaker round

                hunt_team  = str(g.get("屠队", ""))
                home_hunts = (hunt_team == home_team)

                hunter_raw = g.get("屠名", "")
                hunter     = normalize(hunter_raw, alias_map)

                survs = []
                for k in range(1, 5):
                    raw = g.get(f"求生者{k}ID", "")
                    survs.append(normalize(raw, alias_map))

                # Determine n_escaped from scores
                # home_pts = points scored by home team in this half
                # If home_hunts: home=hunter_pts, away=surv_pts
                # If away_hunts: home=surv_pts, away=hunter_pts
                home_pts = g.get("主分", -999)
                away_pts = g.get("客分", -999)
                n_esc = None
                if home_pts != -999 and away_pts != -999:
                    try:
                        hp, ap = int(home_pts), int(away_pts)
                        if home_hunts:
                            hunt_pts, surv_pts = hp, ap
                        else:
                            surv_pts, hunt_pts = hp, ap
                        # reverse lookup HALF_SCORE
                        for n, (hs, ss) in HALF_SCORE.items():
                            if hs == hunt_pts and ss == surv_pts:
                                n_esc = n
                                break
                    except (ValueError, TypeError):
                        pass

                rounds_dict[rno].append({
                    "home_hunts": home_hunts,
                    "hunter":     hunter,
                    "survs":      survs,
                    "n_esc":      n_esc,
                    "home_pts":   home_pts,
                    "away_pts":   away_pts,
                })

            # Build ordered rounds list (exclude extra tiebreaker rounds for format detection)
            sorted_rnos = sorted(rounds_dict.keys())
            rounds = []
            for rno in sorted_rnos:
                halves = rounds_dict[rno]
                if len(halves) == 2:
                    rounds.append({"round_no": rno, "halves": halves})

            if not rounds:
                continue

            # Compute round wins to determine actual series winner and format
            home_wins = 0
            away_wins = 0
            home_cum  = 0
            away_cum  = 0
            for rnd in rounds:
                h_pts = sum(
                    h["home_pts"] for h in rnd["halves"]
                    if h["home_pts"] not in (-999, "-999")
                    and isinstance(h["home_pts"], (int, float))
                )
                a_pts = sum(
                    h["away_pts"] for h in rnd["halves"]
                    if h["away_pts"] not in (-999, "-999")
                    and isinstance(h["away_pts"], (int, float))
                )
                home_cum += h_pts
                away_cum += a_pts
                if h_pts > a_pts:
                    home_wins += 1
                elif a_pts > h_pts:
                    away_wins += 1

            # Infer format
            max_wins = max(home_wins, away_wins)
            series_format = 5 if max_wins >= 3 else 3

            # Actual winner
            if home_wins > away_wins:
                actual_winner = "home"
            elif away_wins > home_wins:
                actual_winner = "away"
            elif home_cum > away_cum:
                actual_winner = "home"
            elif away_cum > home_cum:
                actual_winner = "away"
            else:
                actual_winner = "draw"

            series_list.append({
                "source_file":   fname,
                "source_row":    row_idx,
                "date":          date_str,
                "tournament":    tournament,
                "tier":          tier,
                "home_team":     home_team,
                "away_team":     away_team,
                "rounds":        rounds,
                "actual_winner": actual_winner,
                "format":        series_format,
                "home_wins":     home_wins,
                "away_wins":     away_wins,
            })

    series_list.sort(key=lambda s: s["date"])
    return series_list


# ---------------------------------------------------------------------------
# Series win probability via DP
# ---------------------------------------------------------------------------

def _round_win_probs(
    halves: list[dict],
    res: dict,
    player_index: dict,
) -> tuple[float, float, float]:
    """
    Given 2 halves for a round and a fitted ordinal model, compute:
      (p_home_wins_round, p_away_wins_round, p_draw_round)
    """
    if len(halves) != 2:
        return 0.5, 0.5, 0.0

    # Build a 1-row dataframe for each half and get P(n=k)
    probs = []
    for half in halves:
        hunter  = half["hunter"]
        survs   = half["survs"]
        # Build a minimal test row matching the model's expected format
        row = {
            "hunter_player":   hunter or "__unk__",
            "survivor1_player": survs[0] if survs[0] else "__unk__",
            "survivor2_player": survs[1] if len(survs) > 1 and survs[1] else "__unk__",
            "survivor3_player": survs[2] if len(survs) > 2 and survs[2] else "__unk__",
            "survivor4_player": survs[3] if len(survs) > 3 and survs[3] else "__unk__",
            "n_escaped": 2,  # placeholder
        }
        df_half = pd.DataFrame([row])
        p = predict_probs(df_half, res)[0]  # shape (5,)
        probs.append(p)

    p1, p2 = probs[0], probs[1]  # half1, half2

    # half1: home_hunts=halves[0]["home_hunts"]
    # For each (n1, n2), compute home_score and away_score
    p_home = 0.0
    p_away = 0.0
    p_draw = 0.0

    home_hunts_h1 = halves[0]["home_hunts"]
    home_hunts_h2 = halves[1]["home_hunts"]

    for n1 in range(5):
        for n2 in range(5):
            prob = float(p1[n1]) * float(p2[n2])
            hs1, ss1 = HALF_SCORE[n1]
            hs2, ss2 = HALF_SCORE[n2]

            home_score = (hs1 if home_hunts_h1 else ss1) + (hs2 if home_hunts_h2 else ss2)
            away_score = (ss1 if home_hunts_h1 else hs1) + (ss2 if home_hunts_h2 else hs2)

            if home_score > away_score:
                p_home += prob
            elif away_score > home_score:
                p_away += prob
            else:
                p_draw += prob

    return p_home, p_away, p_draw


def series_win_probability(
    series: dict,
    res: dict,
) -> float:
    """
    P(home wins series) using exact DP.

    Draws in a round are split 50/50 (approximation for the tiebreaker chain).
    """
    fmt = series["format"]
    wins_needed = (fmt + 1) // 2  # 2 for Bo3, 3 for Bo5

    rounds = series["rounds"]
    # Compute per-round home win probability
    round_probs = []
    for rnd in rounds:
        ph, pa, pd_ = _round_win_probs(rnd["halves"], res, res["player_index"])
        # Redistribute draw probability 50/50
        ph_eff = ph + 0.5 * pd_
        pa_eff = pa + 0.5 * pd_
        round_probs.append((ph_eff, pa_eff))

    # For unplayed rounds (series ended early), repeat last round's probs
    n_full_rounds = wins_needed * 2 - 1  # max rounds possible
    while len(round_probs) < n_full_rounds:
        round_probs.append(round_probs[-1])

    # DP: state[wh][wa] = probability of being in that state
    # wh, wa ∈ {0, ..., wins_needed}; absorbing at wins_needed
    state = np.zeros((wins_needed + 1, wins_needed + 1))
    state[0][0] = 1.0

    for ph, pa in round_probs[:n_full_rounds]:
        new_state = np.zeros_like(state)
        for wh in range(wins_needed + 1):
            for wa in range(wins_needed + 1):
                if state[wh][wa] == 0.0:
                    continue
                if wh == wins_needed or wa == wins_needed:
                    # Absorbing — stay
                    new_state[wh][wa] += state[wh][wa]
                    continue
                new_state[min(wh + 1, wins_needed)][wa] += state[wh][wa] * ph
                new_state[wh][min(wa + 1, wins_needed)] += state[wh][wa] * pa
        state = new_state

    return float(state[wins_needed, :].sum())  # home reaches wins_needed


# ---------------------------------------------------------------------------
# Temporal CV evaluation
# ---------------------------------------------------------------------------

def _fit_with_prior(train, ref, tau, l2, threshold):
    from model.bradley_terry import filter_complete as fc
    m = fc(train)
    w = compute_weights(m["date"], ref, tau)
    res0 = fit_ordinal(m, l2_lambda=l2, weights=w)
    new_keys = identify_new_players(m, res0["player_index"], threshold)
    mu = compute_team_mean_priors(m, res0["player_index"], res0["beta"], new_keys)
    return fit_ordinal(m, l2_lambda=l2, weights=w,
                       init_theta=res0["theta"], prior_mean=mu)


def evaluate_series_cv(
    matches_df: pd.DataFrame,
    series_list: list[dict],
    tau: float,
    l2: float,
    threshold: int,
    tiers: list[str] | None = None,
    n_splits: int = 5,
) -> dict:
    """
    Temporal CV at the series level.

    For each fold, trains on matches from earlier series and evaluates
    P(home wins) on test series.
    """
    if tiers:
        series_list = [s for s in series_list if s["tier"] in tiers]
        matches_df  = matches_df[matches_df["tournament_tier"].isin(tiers)].copy()

    series_list = [
        s for s in series_list if s["actual_winner"] in ("home", "away")
    ]

    # Sort by date and split at series level
    series_arr = np.array(series_list, dtype=object)
    dates      = pd.to_datetime([s["date"] for s in series_list])
    order      = np.argsort(dates)
    series_arr = series_arr[order]
    dates      = dates[order]

    matches_df = filter_complete(matches_df).sort_values("date").reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(series_arr), start=1):
        train_series = series_arr[tr_idx]
        test_series  = series_arr[te_idx]

        # Training date cutoff = last date in training series
        cutoff = max(s["date"] for s in train_series)

        # Training matches = all matches on or before cutoff
        train_matches = matches_df[matches_df["date"] <= cutoff]
        if len(train_matches) < 50:
            continue

        ref = train_matches["date"].max()
        res = _fit_with_prior(train_matches, ref, tau, l2, threshold)

        print(f"  fold {fold}: train_series={len(train_series)}, "
              f"test_series={len(test_series)}, cutoff={cutoff}", flush=True)

        for series in test_series:
            if series["actual_winner"] not in ("home", "away"):
                continue
            p_home = series_win_probability(series, res)
            actual = 1.0 if series["actual_winner"] == "home" else 0.0
            rows.append({
                "fold":         fold,
                "date":         series["date"],
                "tier":         series["tier"],
                "p_home":       p_home,
                "actual_home":  actual,
                "predicted_home": int(p_home > 0.5),
                "correct":      int((p_home > 0.5) == (actual == 1.0)),
                "brier":        (p_home - actual) ** 2,
                "format":       series["format"],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"  No data for {label}")
        return

    acc       = df["correct"].mean()
    brier     = df["brier"].mean()
    n         = len(df)
    null_brier = 0.25  # always predict 50/50

    print(f"\n{'='*60}")
    print(f"SERIES WIN ACCURACY  {label}  (n = {n:,} series)")
    print(f"{'='*60}")
    print(f"  Accuracy (predicted winner correct): {acc:.1%}")
    print(f"  Brier score:                         {brier:.4f}")
    print(f"  Brier — null (50/50):                {null_brier:.4f}")
    print(f"  Brier reduction vs null:             {(null_brier - brier)/null_brier*100:.1f}%")

    print(f"\n── Per-fold breakdown ──")
    print(f"  {'fold':>5}  {'n':>5}  {'acc':>7}  {'brier':>7}")
    for fold, grp in df.groupby("fold"):
        print(f"  {fold:>5}  {len(grp):>5}  {grp['correct'].mean():>7.1%}  {grp['brier'].mean():>7.4f}")

    print(f"\n── By format ──")
    for fmt, grp in df.groupby("format"):
        print(f"  Bo{fmt}: n={len(grp)}, acc={grp['correct'].mean():.1%}, "
              f"brier={grp['brier'].mean():.4f}")

    # Calibration buckets
    print(f"\n── Calibration (P(home) bucket → actual home win rate) ──")
    bins = [0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0]
    labels = ["<40%", "40-45%", "45-50%", "50-55%", "55-60%", ">60%"]
    df["bucket"] = pd.cut(df["p_home"], bins=bins, labels=labels, include_lowest=True)
    for bucket, grp in df.groupby("bucket", observed=True):
        if len(grp) == 0:
            continue
        print(f"  pred {bucket:>7}: n={len(grp):>4}, actual win rate={grp['actual_home'].mean():.1%}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONFIGS = {
    "top_tiers": {
        "json":  OPTUNA_DIR / "top_tiers_best.json",
        "tiers": ["IVL", "IJL", "COA"],
    },
    "all_tiers": {
        "json":  OPTUNA_DIR / "ordinal_best_best.json",
        "tiers": None,
    },
}


def run_config(
    matches_df: pd.DataFrame,
    series_list: list[dict],
    cfg_name: str,
) -> pd.DataFrame:
    cfg    = CONFIGS[cfg_name]
    params = json.loads(cfg["json"].read_text())["best_params"]
    tau    = params["half_life_days"]
    l2     = params["l2_lambda"]
    thr    = int(params["threshold"])
    tiers  = cfg["tiers"]

    print(f"\nConfig: {cfg_name}")
    print(f"  τ={tau:.1f}d  l2={l2:.3f}  threshold={thr}  tiers={tiers or 'all'}")

    cache = OUTPUTS / f"series_accuracy_{cfg_name}.csv"
    if cache.exists():
        print(f"  Loading cached results from {cache}")
        df = pd.read_csv(cache)
    else:
        print("  Running temporal CV…")
        df = evaluate_series_cv(
            matches_df, series_list, tau, l2, thr, tiers=tiers
        )
        df.to_csv(cache, index=False)

    tier_str = "/".join(tiers) if tiers else "all tiers"
    print_report(df, label=f"({tier_str}, τ={tau:.0f}d)")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["top_tiers", "all_tiers"],
                        default="top_tiers")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    conn      = sqlite3.connect(DEFAULT_DB)
    matches   = pd.read_sql("SELECT * FROM matches ORDER BY date", conn)
    conn.close()
    print(f"Loaded {len(matches)} matches")

    alias_map   = build_alias_map()
    series_list = parse_all_series(alias_map)
    print(f"Parsed {len(series_list)} series")

    configs_to_run = list(CONFIGS.keys()) if args.all else [args.config]
    for cfg_name in configs_to_run:
        run_config(matches, series_list, cfg_name)
