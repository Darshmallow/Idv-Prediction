"""
src/outputs/comparison_sweep.py
================================
Half-life sweep that compares three models side-by-side:
  1. Ordinal BT (the main model)
  2. Naive rolling-average baseline (K=30)
  3. Null model (predict ŷ = 0 always)

Runs two configurations driven by the Optuna best-params JSON files:

  Config A — top_tiers_best.json
    tiers : IVL, IJL, COA
    l2_lambda, threshold fixed at Optuna best; tau swept

  Config B — ordinal_best_best.json  (all tiers)
    tiers : all
    l2_lambda, threshold fixed at Optuna best; tau swept

Usage
-----
    python src/outputs/comparison_sweep.py
    → outputs/comparison_sweep_top_tiers.png
    → outputs/comparison_sweep_all_tiers.png
    → outputs/comparison_sweep_top_tiers.csv
    → outputs/comparison_sweep_all_tiers.csv
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_DB = str(_ROOT / "data" / "processed" / "idv.db")
OUTPUTS    = _ROOT / "outputs"
OPTUNA_DIR = _ROOT / "outputs" / "optuna"

from model.bradley_terry import filter_complete, compute_weights
from model.ordinal import fit_ordinal, predict_expected_margin
from model.team_prior import identify_new_players, compute_team_mean_priors
from outputs.naive_rolling import cv_rmse as rolling_cv_rmse


# ---------------------------------------------------------------------------
# BT ordinal CV with team-mean prior (single tau)
# ---------------------------------------------------------------------------

def _fit_ordinal_with_prior(train, ref, tau, l2, threshold):
    m = filter_complete(train)
    w = compute_weights(m["date"], ref, tau)
    res0 = fit_ordinal(m, l2_lambda=l2, weights=w)
    new_keys = identify_new_players(m, res0["player_index"], threshold)
    mu = compute_team_mean_priors(m, res0["player_index"], res0["beta"], new_keys)
    return fit_ordinal(m, l2_lambda=l2, weights=w,
                       init_theta=res0["theta"], prior_mean=mu)


def bt_fold_rmses(
    matches: pd.DataFrame,
    tau: float,
    l2: float,
    threshold: int,
    n_splits: int = 5,
) -> list[float]:
    matches = filter_complete(matches).sort_values("date").reset_index(drop=True)
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    rmses   = []
    for tr, te in tscv.split(matches):
        train, test = matches.iloc[tr], matches.iloc[te]
        ref  = train["date"].max()
        res  = _fit_ordinal_with_prior(train, ref, tau, l2, threshold)
        yhat = predict_expected_margin(test, res)
        y    = (test["n_escaped"] - 2).to_numpy(dtype=float)
        rmses.append(float(np.sqrt(np.mean((y - yhat) ** 2))))
    return rmses


# ---------------------------------------------------------------------------
# Null baseline
# ---------------------------------------------------------------------------

def null_fold_rmses(matches: pd.DataFrame, n_splits: int = 5) -> list[float]:
    matches = filter_complete(matches).sort_values("date").reset_index(drop=True)
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    rmses   = []
    for _, te in tscv.split(matches):
        y = (matches.iloc[te]["n_escaped"] - 2).to_numpy(dtype=float)
        rmses.append(float(np.sqrt(np.mean(y ** 2))))
    return rmses


# ---------------------------------------------------------------------------
# Full sweep for one config
# ---------------------------------------------------------------------------

def run_sweep(
    matches: pd.DataFrame,
    tau_grid: np.ndarray,
    l2: float,
    threshold: int,
    tiers: list[str] | None = None,
    n_splits: int = 5,
    label: str = "",
    K: int = 30,
) -> pd.DataFrame:
    if tiers:
        matches = matches[matches["tournament_tier"].isin(tiers)].copy()
        print(f"  Filtered to {tiers}: {len(matches)} rows")

    matches = filter_complete(matches).sort_values("date").reset_index(drop=True)
    print(f"  Complete rows: {len(matches)}")

    # Null (same for all tau)
    print("  Computing null baseline…")
    null_rmses = null_fold_rmses(matches, n_splits)
    null_mean  = float(np.mean(null_rmses))
    null_se    = float(np.std(null_rmses, ddof=1) / np.sqrt(n_splits))
    print(f"  Null RMSE: {null_mean:.4f} ± {null_se:.4f}")

    # Rolling K=30 (same for all tau)
    print(f"  Computing rolling K={K} baseline…")
    roll_mean, roll_se, roll_folds = rolling_cv_rmse(matches, K=K, n_splits=n_splits)
    print(f"  Rolling RMSE: {roll_mean:.4f} ± {roll_se:.4f}")

    # BT sweep
    rows = []
    for i, tau in enumerate(tau_grid, 1):
        t0 = time.time()
        print(f"  [{i:2d}/{len(tau_grid)}]  τ={tau:.0f}d …", end="", flush=True)
        fold_rmses = bt_fold_rmses(matches, tau, l2, threshold, n_splits)
        mean_rmse  = float(np.mean(fold_rmses))
        se_rmse    = float(np.std(fold_rmses, ddof=1) / np.sqrt(n_splits))
        print(f"  RMSE={mean_rmse:.4f} ± {se_rmse:.4f}  ({time.time()-t0:.1f}s)")
        rows.append({
            "tau":            tau,
            "bt_mean":        mean_rmse,
            "bt_se":          se_rmse,
            "null_mean":      null_mean,
            "null_se":        null_se,
            "rolling_mean":   roll_mean,
            "rolling_se":     roll_se,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_sweep(
    df: pd.DataFrame,
    title: str,
    best_tau: float,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    # BT model
    ax.errorbar(
        df["tau"], df["bt_mean"], yerr=df["bt_se"],
        fmt="o-", color="steelblue", capsize=3, linewidth=1.8, markersize=5,
        label="Ordinal BT (ordinal + team-mean prior)",
    )

    # Rolling baseline (horizontal band)
    roll_mean = df["rolling_mean"].iloc[0]
    roll_se   = df["rolling_se"].iloc[0]
    ax.axhline(roll_mean, color="darkorange", linewidth=1.6, linestyle="--",
               label=f"Rolling avg K=30: {roll_mean:.4f}")
    ax.fill_between(df["tau"],
                    roll_mean - roll_se, roll_mean + roll_se,
                    color="darkorange", alpha=0.15)

    # Null baseline
    null_mean = df["null_mean"].iloc[0]
    null_se   = df["null_se"].iloc[0]
    ax.axhline(null_mean, color="grey", linewidth=1.4, linestyle=":",
               label=f"Null (ŷ=0): {null_mean:.4f}")
    ax.fill_between(df["tau"],
                    null_mean - null_se, null_mean + null_se,
                    color="grey", alpha=0.1)

    # Mark optimal tau
    best_idx = df["bt_mean"].idxmin()
    best_row = df.loc[best_idx]
    ax.axvline(best_row["tau"], color="steelblue", linewidth=1.0,
               linestyle=":", alpha=0.6)
    ax.annotate(
        f"τ* = {best_row['tau']:.0f}d\nRMSE = {best_row['bt_mean']:.4f}",
        xy=(best_row["tau"], best_row["bt_mean"]),
        xytext=(best_row["tau"] * 1.3, best_row["bt_mean"] + 0.003),
        fontsize=8.5, color="steelblue",
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8),
    )

    # Mark the Optuna best tau
    ax.axvline(best_tau, color="steelblue", linewidth=1.2,
               linestyle="-.", alpha=0.4,
               label=f"Optuna best τ = {best_tau:.0f}d")

    ax.set_xscale("log")
    ax.set_xlabel("Half-life τ (days)", fontsize=11)
    ax.set_ylabel("OOS RMSE on margin", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda d: d / 30.44, lambda m: m * 30.44),
    )
    secax.set_xlabel("Half-life (months)", fontsize=9)

    plt.tight_layout()
    OUTPUTS.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"→ {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    conn    = sqlite3.connect(DEFAULT_DB)
    matches = pd.read_sql("SELECT * FROM matches ORDER BY date", conn)
    conn.close()
    print(f"Loaded {len(matches)} matches\n")

    tau_grid = np.logspace(np.log10(30), np.log10(1800), 22)

    configs = [
        {
            "json":     OPTUNA_DIR / "top_tiers_best.json",
            "tiers":    ["IVL", "IJL", "COA"],
            "label":    "top_tiers",
            "title":    "Half-life sweep — top tiers (IVL / IJL / COA)\n"
                        "Ordinal BT vs rolling avg vs null  "
                        "[l2 & threshold fixed at Optuna best]",
            "out_png":  OUTPUTS / "comparison_sweep_top_tiers.png",
            "out_csv":  OUTPUTS / "comparison_sweep_top_tiers.csv",
        },
        {
            "json":     OPTUNA_DIR / "ordinal_best_best.json",
            "tiers":    None,
            "label":    "all_tiers",
            "title":    "Half-life sweep — all tiers\n"
                        "Ordinal BT vs rolling avg vs null  "
                        "[l2 & threshold fixed at Optuna best]",
            "out_png":  OUTPUTS / "comparison_sweep_all_tiers.png",
            "out_csv":  OUTPUTS / "comparison_sweep_all_tiers.csv",
        },
    ]

    for cfg in configs:
        params = json.loads(cfg["json"].read_text())["best_params"]
        l2        = params["l2_lambda"]
        threshold = int(params["threshold"])
        best_tau  = params["half_life_days"]

        print(f"\n{'='*60}")
        print(f"Config: {cfg['label']}")
        print(f"  l2={l2:.3f}  threshold={threshold}  best_tau={best_tau:.1f}d")
        print(f"  tiers: {cfg['tiers'] or 'all'}")
        print(f"{'='*60}")

        df = run_sweep(
            matches.copy(),
            tau_grid,
            l2=l2,
            threshold=threshold,
            tiers=cfg["tiers"],
            label=cfg["label"],
        )
        df.to_csv(cfg["out_csv"], index=False)
        print(f"→ {cfg['out_csv']}")

        plot_sweep(df, title=cfg["title"], best_tau=best_tau,
                   save_path=cfg["out_png"])
