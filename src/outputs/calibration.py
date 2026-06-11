"""
src/calibration.py
==================
Generate a calibration (reliability) diagram for the ordinal BT model.

For each test-set game-half, the ordinal model outputs P(n_escaped = k)
for k in {0,1,2,3,4}.  We derive:

    P(survivor win) = P(n_escaped > 2) = P(n=3) + P(n=4)

Then we bucket predictions into 10 equal-width bins and compare the
predicted probability to the actual survivor win rate in each bucket.

A perfectly calibrated model lies on the diagonal y = x.

Usage
-----
    python src/calibration.py
    → saves outputs/calibration.png
"""

from __future__ import annotations

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

DEFAULT_DB = str(_ROOT / "data" / "processed" / "idv.db")
OUTPUTS    = _ROOT / "outputs"

from model.bradley_terry import filter_complete, build_player_index, compute_weights
from model.ordinal import fit_ordinal, predict_probs
from model.team_prior import identify_new_players, compute_team_mean_priors


def _fit_with_prior(train: pd.DataFrame, ref: str, half_life_days: float,
                    l2_lambda: float, threshold: int) -> dict:
    """Two-pass ordinal fit with new-player team-mean prior."""
    from model.bradley_terry import filter_complete as fc
    m = fc(train)

    w = compute_weights(m["date"], ref, half_life_days)

    # Pass 1
    res0 = fit_ordinal(m, l2_lambda=l2_lambda, weights=w)
    beta0 = res0["beta"]
    theta0 = res0["theta"]
    player_index = res0["player_index"]

    # Pass 2: team-mean prior for new players
    new_keys = identify_new_players(m, player_index, threshold)
    mu = compute_team_mean_priors(m, player_index, beta0, new_keys)

    res1 = fit_ordinal(m, l2_lambda=l2_lambda, weights=w,
                       init_theta=theta0, prior_mean=mu)
    return res1


def collect_test_predictions(
    matches: pd.DataFrame,
    half_life_days: float = 136.0,
    l2_lambda: float = 1.0,
    threshold: int = 5,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Run temporal CV and collect (p_survivor_win, actual_survivor_win) for
    every test-set game-half across all folds.
    """
    matches = filter_complete(matches).sort_values("date").reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rows = []
    for fold, (tr, te) in enumerate(tscv.split(matches), start=1):
        train, test = matches.iloc[tr], matches.iloc[te]
        ref = train["date"].max()

        print(f"  fold {fold}: train={len(train)}, test={len(test)}", flush=True)
        res = _fit_with_prior(train, ref, half_life_days, l2_lambda, threshold)

        probs = predict_probs(test, res)             # (N, 5)
        p_surv_win = probs[:, 3] + probs[:, 4]      # P(n_escaped in {3, 4})
        actual_win = (test["n_escaped"].to_numpy() > 2).astype(int)

        for p, a in zip(p_surv_win, actual_win):
            rows.append({"p_surv_win": float(p), "actual_win": int(a), "fold": fold})

    return pd.DataFrame(rows)


def plot_calibration(preds: pd.DataFrame, n_bins: int = 10,
                     save_path: Path | None = None) -> None:
    """
    Reliability diagram: bin predictions by predicted P(survivor win),
    plot mean predicted vs actual win rate per bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_mids, actual_rates, pred_means, counts = [], [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (preds["p_surv_win"] >= lo) & (preds["p_surv_win"] < hi)
        if lo == bins[-2]:       # include right edge for last bin
            mask = (preds["p_surv_win"] >= lo) & (preds["p_surv_win"] <= hi)
        n = mask.sum()
        if n == 0:
            continue
        bin_mids.append(preds.loc[mask, "p_surv_win"].mean())
        pred_means.append(preds.loc[mask, "p_surv_win"].mean())
        actual_rates.append(preds.loc[mask, "actual_win"].mean())
        counts.append(n)

    bin_mids    = np.array(bin_mids)
    pred_means  = np.array(pred_means)
    actual_rates = np.array(actual_rates)
    counts      = np.array(counts)

    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    # ── Reliability diagram ──
    ax_main.plot([0, 1], [0, 1], "--", color="grey", linewidth=1.2,
                 label="Perfect calibration")

    # Shade over/under confidence regions
    ax_main.fill_between([0, 1], [0, 1], [1, 1], alpha=0.04, color="steelblue",
                         label="Underconfident region")
    ax_main.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color="darkorange",
                         label="Overconfident region")

    # Points sized by count
    sc = ax_main.scatter(pred_means, actual_rates, s=counts / counts.max() * 300 + 40,
                         color="steelblue", edgecolors="white", linewidths=0.8,
                         zorder=5, label="Model")
    ax_main.plot(pred_means, actual_rates, "-", color="steelblue",
                 linewidth=1.6, alpha=0.7)

    # Annotate counts
    for x, y, n in zip(pred_means, actual_rates, counts):
        ax_main.annotate(f"n={n}", (x, y), textcoords="offset points",
                         xytext=(6, 4), fontsize=7.5, color="#444")

    ax_main.set_ylabel("Actual survivor win rate", fontsize=11)
    ax_main.set_xlim(0, 1); ax_main.set_ylim(0, 1)
    ax_main.set_title(
        "Calibration — Ordinal BT model\n"
        "(τ=136d, ordinal + team-mean prior, 5-fold temporal CV)",
        fontsize=11
    )
    ax_main.legend(fontsize=9, loc="upper left")
    ax_main.grid(True, alpha=0.3)

    # ── Histogram of predicted probabilities ──
    ax_hist.bar(bin_mids, counts, width=(bins[1] - bins[0]) * 0.8,
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


if __name__ == "__main__":
    conn = sqlite3.connect(DEFAULT_DB)
    matches = pd.read_sql("SELECT * FROM matches ORDER BY date", conn)
    conn.close()
    print(f"Loaded {len(matches)} matches\n")

    print("Running temporal CV to collect predictions…")
    preds = collect_test_predictions(matches)
    preds.to_csv(OUTPUTS / "calibration_preds.csv", index=False)
    print(f"\nCollected {len(preds)} test predictions across all folds")

    # Summary stats
    overall_win_rate = preds["actual_win"].mean()
    mean_pred = preds["p_surv_win"].mean()
    print(f"Actual survivor win rate: {overall_win_rate:.3f}")
    print(f"Mean predicted P(win):    {mean_pred:.3f}")

    plot_calibration(preds)
