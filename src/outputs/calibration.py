"""
src/outputs/calibration.py
==========================
Generate calibration (reliability) diagrams for the ordinal BT model.

For each test-set game-half, the ordinal model outputs P(n_escaped = k)
for k in {0,1,2,3,4}.  We derive:

    P(survivor win) = P(n_escaped > 2) = P(n=3) + P(n=4)

Then bucket predictions into 10 equal-width bins and compare the
predicted probability to the actual survivor win rate in each bucket.

A perfectly calibrated model lies on the diagonal y = x.

Usage
-----
    python src/outputs/calibration.py                         # all tiers (ordinal_best)
    python src/outputs/calibration.py --config top_tiers      # IVL/IJL/COA only
    python src/outputs/calibration.py --config all_tiers       # explicit all-tiers
    python src/outputs/calibration.py --all                    # run both configs
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_DB  = str(_ROOT / "data" / "processed" / "idv.db")
OUTPUTS     = _ROOT / "outputs"
OPTUNA_DIR  = _ROOT / "outputs" / "optuna"

from model.bradley_terry import filter_complete, compute_weights
from model.ordinal import fit_ordinal, predict_probs
from model.team_prior import identify_new_players, compute_team_mean_priors

CONFIGS = {
    "top_tiers": {
        "json":   OPTUNA_DIR / "top_tiers_best.json",
        "tiers":  ["IVL", "IJL", "COA"],
        "suffix": "top_tiers",
    },
    "all_tiers": {
        "json":   OPTUNA_DIR / "ordinal_best_best.json",
        "tiers":  None,
        "suffix": "all_tiers",
    },
}


def _fit_with_prior(train, ref, tau, l2, threshold):
    from model.bradley_terry import filter_complete as fc
    m = fc(train)
    w = compute_weights(m["date"], ref, tau)
    res0 = fit_ordinal(m, l2_lambda=l2, weights=w)
    new_keys = identify_new_players(m, res0["player_index"], threshold)
    mu = compute_team_mean_priors(m, res0["player_index"], res0["beta"], new_keys)
    return fit_ordinal(m, l2_lambda=l2, weights=w,
                       init_theta=res0["theta"], prior_mean=mu)


def collect_test_predictions(
    matches: pd.DataFrame,
    tau: float,
    l2: float,
    threshold: int,
    tiers: list[str] | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    if tiers:
        matches = matches[matches["tournament_tier"].isin(tiers)].copy()
    matches = filter_complete(matches).sort_values("date").reset_index(drop=True)
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    rows    = []

    for fold, (tr, te) in enumerate(tscv.split(matches), start=1):
        train, test = matches.iloc[tr], matches.iloc[te]
        ref = train["date"].max()
        print(f"  fold {fold}: train={len(train)}, test={len(test)}", flush=True)

        res        = _fit_with_prior(train, ref, tau, l2, threshold)
        probs      = predict_probs(test, res)
        p_surv_win = probs[:, 3] + probs[:, 4]
        actual_win = (test["n_escaped"].to_numpy() > 2).astype(int)

        for p, a in zip(p_surv_win, actual_win):
            rows.append({"p_surv_win": float(p), "actual_win": int(a), "fold": fold})

    return pd.DataFrame(rows)


def plot_calibration(
    preds: pd.DataFrame,
    title: str,
    n_bins: int = 10,
    save_path: Path | None = None,
) -> None:
    bins = np.linspace(0, 1, n_bins + 1)
    pred_means, actual_rates, counts = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (preds["p_surv_win"] >= lo) & (preds["p_surv_win"] < hi)
        if lo == bins[-2]:
            mask = (preds["p_surv_win"] >= lo) & (preds["p_surv_win"] <= hi)
        n = mask.sum()
        if n == 0:
            continue
        pred_means.append(preds.loc[mask, "p_surv_win"].mean())
        actual_rates.append(preds.loc[mask, "actual_win"].mean())
        counts.append(n)

    pred_means   = np.array(pred_means)
    actual_rates = np.array(actual_rates)
    counts       = np.array(counts)

    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax_main.plot([0, 1], [0, 1], "--", color="grey", linewidth=1.2,
                 label="Perfect calibration")
    ax_main.fill_between([0, 1], [0, 1], [1, 1], alpha=0.04, color="steelblue",
                         label="Underconfident region")
    ax_main.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color="darkorange",
                         label="Overconfident region")
    ax_main.scatter(pred_means, actual_rates,
                    s=counts / counts.max() * 300 + 40,
                    color="steelblue", edgecolors="white", linewidths=0.8,
                    zorder=5, label="Model")
    ax_main.plot(pred_means, actual_rates, "-", color="steelblue",
                 linewidth=1.6, alpha=0.7)
    for x, y, n in zip(pred_means, actual_rates, counts):
        ax_main.annotate(f"n={n}", (x, y), textcoords="offset points",
                         xytext=(6, 4), fontsize=7.5, color="#444")

    ax_main.set_ylabel("Actual survivor win rate", fontsize=11)
    ax_main.set_xlim(0, 1); ax_main.set_ylim(0, 1)
    ax_main.set_title(title, fontsize=11)
    ax_main.legend(fontsize=9, loc="upper left")
    ax_main.grid(True, alpha=0.3)

    ax_hist.bar(pred_means, counts, width=(bins[1] - bins[0]) * 0.8,
                color="steelblue", alpha=0.6, edgecolor="white")
    ax_hist.set_xlabel("Predicted P(survivor win)", fontsize=11)
    ax_hist.set_ylabel("# game-halves", fontsize=9)
    ax_hist.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = save_path or (OUTPUTS / "calibration.png")
    OUTPUTS.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"→ {out}")


def run_config(matches: pd.DataFrame, cfg_name: str) -> None:
    cfg    = CONFIGS[cfg_name]
    params = json.loads(cfg["json"].read_text())["best_params"]
    tau    = params["half_life_days"]
    l2     = params["l2_lambda"]
    thr    = int(params["threshold"])
    tiers  = cfg["tiers"]
    suffix = cfg["suffix"]

    print(f"\nConfig: {cfg_name}")
    print(f"  τ={tau:.1f}d  l2={l2:.3f}  threshold={thr}  tiers={tiers or 'all'}")

    cache = OUTPUTS / f"calibration_preds_{suffix}.csv"
    if cache.exists():
        print(f"  Loading cached predictions from {cache}")
        preds = pd.read_csv(cache)
    else:
        print("  Running temporal CV…")
        preds = collect_test_predictions(matches, tau, l2, thr, tiers)
        preds.to_csv(cache, index=False)

    print(f"  {len(preds)} predictions  "
          f"actual win rate={preds['actual_win'].mean():.3f}  "
          f"mean predicted={preds['p_surv_win'].mean():.3f}")

    tier_str = "/".join(tiers) if tiers else "all tiers"
    title = (
        f"Calibration — Ordinal BT ({tier_str})\n"
        f"τ={tau:.0f}d  l2={l2:.2f}  threshold={thr}  5-fold temporal CV"
    )
    plot_calibration(preds, title=title,
                     save_path=OUTPUTS / f"calibration_{suffix}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["top_tiers", "all_tiers"],
                        default="all_tiers")
    parser.add_argument("--all", action="store_true",
                        help="Run both configs")
    args = parser.parse_args()

    conn    = sqlite3.connect(DEFAULT_DB)
    matches = pd.read_sql("SELECT * FROM matches ORDER BY date", conn)
    conn.close()
    print(f"Loaded {len(matches)} matches")

    configs_to_run = list(CONFIGS.keys()) if args.all else [args.config]
    for cfg_name in configs_to_run:
        run_config(matches, cfg_name)
