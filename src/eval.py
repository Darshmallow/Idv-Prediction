"""
src/eval.py
===========
Temporal cross-validation and half-life sweep for the Bradley-Terry
margin model.

Core question
-------------
How quickly does competitive skill information decay?
We sweep the exponential half-life τ and pick the value that minimises
out-of-sample RMSE under strict temporal CV (no future data leaks into
training).

Evaluation metric
-----------------
RMSE on the margin  y = n_escaped − 2 ∈ {−2,−1,0,+1,+2}.

    RMSE = sqrt( mean( (y − ŷ)² ) )

Baseline: predict ŷ = 0 always (everyone is average) → RMSE ≈ 1.15.
A useful model must beat that.

Two split strategies are supported:

  "time_series"  — sklearn TimeSeriesSplit with N+1 equal chunks.
                   N splits; each test fold = N/(n_splits+1) consecutive
                   matches by date.  Default for backward compatibility.

  "seasons"      — one fold per competitive year, derived from tournament
                   names.  Each test fold = all matches of one season
                   (IVL summer + fall + IJL summer + fall + COA).  Train
                   = all matches from earlier seasons (expanding window).
                   Number of splits = (n_seasons − 1).

Public API
----------
    season_splits(matches)
        → list of (train_idx, test_idx) tuples for the season strategy

    get_splits(matches, strategy, n_splits)
        → unified dispatcher

    temporal_cv(matches, half_life_days, n_splits, l2_lambda, split_strategy)
        → per-fold results DataFrame

    null_model_rmse(matches, n_splits, split_strategy)
        → float — RMSE of the ŷ=0 baseline, same CV setup

    sweep_half_lives(matches, half_lives, n_splits, l2_lambda, verbose, split_strategy)
        → long-form results DataFrame (one row per fold × half-life)

    summarise(results)
        → aggregated DataFrame (one row per half-life)

    plot_sweep(results, save_path)
        → saves the sweep curve
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from bradley_terry import (
    build_design_matrix,
    build_player_index,
    compute_weights,
    filter_complete,
    fit,
    predict,
)

_ROOT      = Path(__file__).parent.parent
DEFAULT_DB = str(_ROOT / "data" / "processed" / "idv.db")
OUTPUTS    = _ROOT / "outputs"


# ---------------------------------------------------------------------------
# Default half-life grid
# ---------------------------------------------------------------------------

def default_half_lives() -> np.ndarray:
    """
    20 log-spaced candidates from 30 days (≈1 month) to 1825 days (5 years).
    Covers everything from aggressive recency-weighting to near-uniform.
    """
    return np.logspace(np.log10(30), np.log10(1825), 20)


# ---------------------------------------------------------------------------
# Cold-start detection
# ---------------------------------------------------------------------------

def _cold_start_mask(
    test: pd.DataFrame,
    player_index: dict[tuple[str, str], int],
) -> pd.Series:
    """
    Returns a boolean Series (aligned to test.index) that is True for any
    match where at least one player was unseen in training.
    These matches get ŷ = 0 for that player's contribution (the prior).
    """
    from bradley_terry import SURVIVOR_COLS

    unseen = pd.Series(False, index=test.index)
    for pid in test["hunter_player"]:
        pass  # vectorised below

    h_seen  = test["hunter_player"].map(lambda p: (str(p), "hunter") in player_index)
    unseen  = ~h_seen
    for col in SURVIVOR_COLS:
        s_seen = test[col].map(
            lambda p: (str(p), "survivor") in player_index if pd.notna(p) else True
        )
        unseen |= ~s_seen
    return unseen


# ---------------------------------------------------------------------------
# Split strategies
# ---------------------------------------------------------------------------

# Maps COA edition → competitive year it wraps up.
# COA spring N world finals belong with season (N−1).
_COA_TO_SEASON = {4: 2020, 5: 2021, 6: 2022, 7: 2023, 8: 2024, 9: 2025}


def _season_year(tournament: str) -> int | None:
    """
    Map a tournament-id string to its competitive-year fold.

    IVL_2024_summer_regular     → 2024
    IJL_2025_fall_playoffs      → 2025
    COA8 (April-May 2025)       → 2024  (wraps up the 2024 season)
    COA4 (April-May 2021)       → 2020
    """
    if tournament.startswith("COA"):
        try:
            return _COA_TO_SEASON[int(tournament[3:])]
        except (KeyError, ValueError):
            return None
    parts = tournament.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def season_splits(matches: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Season-aligned expanding-window splits.

    Each split's test set = all matches from one competitive-year fold;
    training set = all earlier folds.

    Number of splits = (number of seasons in the data) − 1.  The earliest
    season is never used as a test fold (only as training data).

    Parameters
    ----------
    matches : DataFrame with a `tournament` column.
              Should be the row-filtered, date-sorted version that
              `temporal_cv` operates on (so positional indices match).

    Returns
    -------
    List of (train_idx, test_idx) tuples — each idx is a 1-D numpy array
    of integer positions into `matches`.
    """
    seasons = matches["tournament"].map(_season_year)
    unique_seasons = sorted(s for s in seasons.dropna().unique())
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(len(unique_seasons) - 1):
        train_mask = seasons <= unique_seasons[k]
        test_mask  = seasons == unique_seasons[k + 1]
        splits.append((
            np.where(train_mask)[0],
            np.where(test_mask)[0],
        ))
    return splits


def get_splits(
    matches: pd.DataFrame,
    strategy: str = "time_series",
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Unified dispatcher.  Returns a list of (train_idx, test_idx) tuples.

    strategy:
        "time_series" - sklearn TimeSeriesSplit (expanding window)
        "seasons"     - one test fold per competitive season

    For "seasons", n_splits is ignored (determined by data).
    """
    if strategy == "time_series":
        return list(TimeSeriesSplit(n_splits=n_splits).split(matches))
    if strategy == "seasons":
        return season_splits(matches)
    raise ValueError(f"Unknown split_strategy {strategy!r}; "
                     f"use 'time_series' or 'seasons'.")


def _test_period_label(matches: pd.DataFrame, test_idx: np.ndarray,
                       strategy: str) -> str:
    """Human-readable label for a test fold."""
    if strategy == "seasons":
        seasons = matches.iloc[test_idx]["tournament"].map(_season_year).dropna()
        if len(seasons):
            return str(int(seasons.iloc[0]))
    sub = matches.iloc[test_idx]
    return f"{sub['date'].min()}→{sub['date'].max()}"


# ---------------------------------------------------------------------------
# Single CV run
# ---------------------------------------------------------------------------

def temporal_cv(
    matches: pd.DataFrame,
    half_life_days: float | None,
    n_splits: int = 5,
    l2_lambda: float = 1.0,
    split_strategy: str = "time_series",
    test_tiers: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Run temporal cross-validation for one half-life value.

    Matches are sorted by date; each fold's test set is strictly after its
    training set.  The reference date for decay weights is the last date
    in the training set (not a global constant) so weights correctly
    reflect the information available at that point in time.

    Parameters
    ----------
    matches        : Full matches DataFrame (will be filtered + sorted).
    half_life_days : τ in days.  None = uniform weights (no decay).
    n_splits       : Number of temporal folds (ignored for "seasons" strategy).
    l2_lambda      : Ridge penalty.
    split_strategy : "time_series" (default) or "seasons".
    test_tiers     : Optional list of tournament_tier values to restrict the
                     test set to (e.g. ['IVL', 'IJL']).  Training set is
                     unchanged.  Requires the matches DataFrame to have a
                     `tournament_tier` column.

    Returns
    -------
    DataFrame with columns:
        half_life_days, fold, test_period, train_size, test_size,
        rmse, null_rmse, n_cold_start
    """
    matches = filter_complete(matches).sort_values("date").reset_index(drop=True)
    splits  = get_splits(matches, split_strategy, n_splits)
    rows    = []

    # Filter test indices by tier if requested
    if test_tiers is not None:
        if "tournament_tier" not in matches.columns:
            raise ValueError(
                "test_tiers requires a 'tournament_tier' column in matches"
            )
        tier_set = set(test_tiers)
        tier_mask = matches["tournament_tier"].isin(tier_set).to_numpy()
        splits = [
            (tr, te[tier_mask[te]])
            for tr, te in splits
        ]

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train = matches.iloc[train_idx]
        test  = matches.iloc[test_idx]

        # Reference date = last training date
        ref_date     = train["date"].max()
        player_index = build_player_index(train)

        # Fit
        X_train, y_train = build_design_matrix(train, player_index)
        weights          = compute_weights(train["date"], ref_date, half_life_days)
        beta             = fit(X_train, y_train, weights, l2_lambda)

        # Predict on test (cold-start players contribute 0)
        X_test, y_test = build_design_matrix(test, player_index)
        y_hat          = predict(X_test, beta)

        rmse      = float(np.sqrt(np.mean((y_test - y_hat) ** 2)))
        null_rmse = float(np.sqrt(np.mean(y_test ** 2)))   # predict 0 always
        n_cold    = int(_cold_start_mask(test, player_index).sum())

        rows.append({
            "half_life_days": half_life_days,
            "fold":           fold,
            "test_period":    _test_period_label(matches, test_idx, split_strategy),
            "train_size":     len(train),
            "test_size":      len(test),
            "rmse":           rmse,
            "null_rmse":      null_rmse,
            "n_cold_start":   n_cold,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Null-model baseline
# ---------------------------------------------------------------------------

def null_model_rmse(
    matches: pd.DataFrame,
    n_splits: int = 5,
    split_strategy: str = "time_series",
) -> float:
    """
    RMSE of the ŷ = 0 baseline (everyone is average) using the same CV folds.
    Returns the mean across folds.
    """
    matches = filter_complete(matches).sort_values("date").reset_index(drop=True)
    splits  = get_splits(matches, split_strategy, n_splits)
    rmses   = []
    for _, test_idx in splits:
        y = (matches.iloc[test_idx]["n_escaped"] - 2).to_numpy(dtype=float)
        rmses.append(float(np.sqrt(np.mean(y ** 2))))
    return float(np.mean(rmses))


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_half_lives(
    matches: pd.DataFrame,
    half_lives: np.ndarray | None = None,
    n_splits: int = 5,
    l2_lambda: float = 1.0,
    verbose: bool = True,
    split_strategy: str = "time_series",
) -> pd.DataFrame:
    """
    Run temporal CV for each candidate half-life.

    Parameters
    ----------
    matches     : Full matches DataFrame.
    half_lives  : Array of τ values in days.  Defaults to 20 log-spaced
                  values from 30 to 1825.  Pass np.array([np.inf]) or None
                  to include the no-decay baseline.
    n_splits    : Temporal folds per half-life.
    l2_lambda   : Ridge penalty (same for all half-lives).
    verbose     : Print progress.

    Returns
    -------
    Long-form DataFrame with one row per (half_life, fold).
    Columns: half_life_days, fold, train_size, test_size,
             rmse, null_rmse, n_cold_start.
    """
    if half_lives is None:
        half_lives = default_half_lives()

    all_results = []
    t0 = time.time()

    for i, tau in enumerate(half_lives, start=1):
        tau_val = None if np.isinf(tau) else float(tau)
        label   = f"τ=∞ (no decay)" if tau_val is None else f"τ={tau:.0f}d"

        if verbose:
            print(f"  [{i:2d}/{len(half_lives)}]  {label} …", end="", flush=True)

        fold_t0 = time.time()
        cv      = temporal_cv(matches, tau_val, n_splits, l2_lambda, split_strategy)
        elapsed = time.time() - fold_t0

        # Store the original tau value (possibly inf) for plotting
        cv["half_life_days"] = tau
        all_results.append(cv)

        if verbose:
            mean_rmse = cv["rmse"].mean()
            print(f"  mean RMSE={mean_rmse:.4f}  ({elapsed:.1f}s)")

    total = time.time() - t0
    if verbose:
        print(f"\nSweep complete — {len(half_lives)} × {n_splits} folds in {total:.1f}s")

    return pd.concat(all_results, ignore_index=True)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-fold results to one row per half-life.

    Returns columns:
        half_life_days, half_life_months,
        mean_rmse, se_rmse,
        vs_null_mean,            # mean_rmse − mean(null_rmse): improvement
        n_cold_start_mean
    """
    agg = (
        results
        .groupby("half_life_days")
        .agg(
            mean_rmse      =("rmse",          "mean"),
            se_rmse        =("rmse",          lambda x: x.std() / np.sqrt(len(x))),
            mean_null_rmse =("null_rmse",      "mean"),
            n_cold_start   =("n_cold_start",   "mean"),
            n_folds        =("fold",           "count"),
        )
        .reset_index()
        .sort_values("half_life_days")
    )
    agg["vs_null"]          = agg["mean_rmse"] - agg["mean_null_rmse"]
    agg["half_life_months"] = agg["half_life_days"] / 30.44
    return agg


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_sweep(
    results: pd.DataFrame,
    save_path: str | None = None,
    title: str = "Skill decay half-life sweep — Identity V pro play",
) -> None:
    """
    Plot mean CV RMSE ± 1 SE vs half-life (log x-axis).
    Marks the optimal half-life and the null-model baseline.

    Parameters
    ----------
    results   : Output of sweep_half_lives().
    save_path : If given, save figure here.  Defaults to outputs/sweep.png.
    """
    agg     = summarise(results)
    finite  = agg[np.isfinite(agg["half_life_days"])].copy()
    inf_row = agg[~np.isfinite(agg["half_life_days"])]

    best    = finite.loc[finite["mean_rmse"].idxmin()]
    best_hl = best["half_life_days"]
    best_mo = best["half_life_months"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Main sweep curve
    ax.errorbar(
        finite["half_life_days"],
        finite["mean_rmse"],
        yerr=finite["se_rmse"],
        fmt="o-",
        capsize=4,
        linewidth=1.8,
        markersize=5,
        label="CV RMSE ± 1 SE",
        color="steelblue",
    )

    # Optimal point
    ax.scatter(
        [best_hl], [best["mean_rmse"]],
        color="crimson", zorder=6, s=80,
        label=f"Optimal: {best_hl:.0f} days ({best_mo:.1f} months)",
    )
    ax.axvline(best_hl, color="crimson", linestyle="--", alpha=0.4, linewidth=1.2)

    # Null model reference line
    null_level = agg["mean_null_rmse"].mean()
    ax.axhline(
        null_level,
        color="grey", linestyle=":", linewidth=1.5,
        label=f"Null model (ŷ=0): RMSE={null_level:.4f}",
    )

    # No-decay reference line (if included)
    if len(inf_row):
        nd_level = inf_row["mean_rmse"].iloc[0]
        ax.axhline(
            nd_level,
            color="darkorange", linestyle="-.", linewidth=1.2,
            label=f"No decay (τ=∞): RMSE={nd_level:.4f}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Half-life τ (days)", fontsize=12)
    ax.set_ylabel("Out-of-sample RMSE (margin)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    # Secondary x-axis in months
    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda d: d / 30.44, lambda m: m * 30.44),
    )
    secax.set_xlabel("Half-life (months)", fontsize=10)

    plt.tight_layout()

    if save_path is None:
        OUTPUTS.mkdir(exist_ok=True)
        save_path = str(OUTPUTS / "sweep.png")

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    db_path: str = DEFAULT_DB,
    n_splits: int = 5,
    l2_lambda: float = 1.0,
    save_csv: str | None = None,
) -> pd.DataFrame:
    """
    Full pipeline: load data → sweep → print summary → plot.

    Parameters
    ----------
    db_path   : Path to the SQLite database.
    n_splits  : Temporal CV folds.
    l2_lambda : Ridge penalty.
    save_csv  : Path to save raw fold-level results.  Defaults to
                outputs/sweep_results.csv.
    """
    conn    = sqlite3.connect(db_path)
    matches = pd.read_sql("SELECT * FROM matches ORDER BY date", conn)
    conn.close()

    # Include no-decay baseline at the end of the grid
    half_lives = np.append(default_half_lives(), np.inf)

    print(f"Running {len(half_lives)} half-life candidates × {n_splits} folds …\n")
    results = sweep_half_lives(matches, half_lives, n_splits, l2_lambda, verbose=True)

    # Save raw results
    if save_csv is None:
        OUTPUTS.mkdir(exist_ok=True)
        save_csv = str(OUTPUTS / "sweep_results.csv")
    results.to_csv(save_csv, index=False)
    print(f"\nRaw results saved → {save_csv}")

    # Print summary table
    print("\nSummary (mean RMSE ± SE across folds):\n")
    summary = summarise(results)
    # Format for display
    disp = summary[["half_life_days", "half_life_months",
                     "mean_rmse", "se_rmse", "vs_null", "n_cold_start"]].copy()
    disp["half_life_days"]   = disp["half_life_days"].apply(
        lambda x: "∞" if np.isinf(x) else f"{x:.0f}")
    disp["half_life_months"] = disp["half_life_months"].apply(
        lambda x: "∞" if np.isinf(x) else f"{x:.1f}")
    disp["mean_rmse"]        = disp["mean_rmse"].round(5)
    disp["se_rmse"]          = disp["se_rmse"].round(5)
    disp["vs_null"]          = disp["vs_null"].round(5)
    disp["n_cold_start"]     = disp["n_cold_start"].round(1)
    print(disp.to_string(index=False))

    # Best finite half-life
    finite  = summary[np.isfinite(summary["half_life_days"])]
    best    = finite.loc[finite["mean_rmse"].idxmin()]
    print(f"\n→ Optimal half-life: {best['half_life_days']:.0f} days "
          f"({best['half_life_months']:.1f} months)  "
          f"RMSE={best['mean_rmse']:.5f}")

    plot_sweep(results)
    return results


if __name__ == "__main__":
    run()
