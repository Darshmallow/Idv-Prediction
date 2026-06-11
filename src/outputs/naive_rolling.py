"""
src/outputs/naive_rolling.py
============================
Stupid baseline — no Bradley-Terry, no ridge, no fitting at all.

For each match, predict:

    ŷ = hunter_avg + survivor_avg

where:
    hunter_avg   = mean margin over the hunter's K most recent prior matches
    survivor_avg = mean over the 4 survivors of each one's K-recent-prior avg

The intuition: margin is signed (n_escaped − 2, positive = survivor side
won).  Strong hunters → past matches had negative margin → hunter_avg < 0 →
pulls the prediction down.  Strong survivors → positive margin → pulls
the prediction up.  Two equally strong sides → averages cancel → predict 0.

This is the "if I just averaged the past 30 matches" model — useful as a
floor against which the linear / ordinal Bradley-Terry models earn their
credibility.

Public API
----------
    rolling_predict(matches, K=30)         → (preds ndarray, cleaned df)
    cv_rmse(matches, K=30, n_splits=5)     → (mean, se, fold_rmses)
    sweep_K(matches, K_values)             → DataFrame
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.bradley_terry import SURVIVOR_COLS, filter_complete


def rolling_predict(matches: pd.DataFrame, K: int = 30) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Walk through matches in chronological order.  For each match, predict
    using only past data, then update the per-player rolling state with
    this match's actual margin.

    Cold-start contributions (player with no prior history) are treated
    as 0 — the population prior.

    Returns (preds, cleaned_sorted_df).
    """
    df       = filter_complete(matches).sort_values("date").reset_index(drop=True)
    margins  = (df["n_escaped"] - 2).to_numpy(dtype=np.float64)
    hunter_h: dict[str, deque] = {}
    surv_h:   dict[str, deque] = {}

    preds      = np.zeros(len(df), dtype=np.float64)
    hunter_arr = df["hunter_player"].astype(str).to_numpy()
    surv_arrs  = [df[c].astype(str).to_numpy() for c in SURVIVOR_COLS]

    for i in range(len(df)):
        h     = hunter_arr[i]
        h_dq  = hunter_h.get(h)
        h_avg = float(np.mean(h_dq)) if h_dq else 0.0

        s_avgs = []
        for s_arr in surv_arrs:
            s  = s_arr[i]
            dq = surv_h.get(s)
            if dq:
                s_avgs.append(float(np.mean(dq)))
        s_avg = float(np.mean(s_avgs)) if s_avgs else 0.0

        preds[i] = h_avg + s_avg

        actual = margins[i]
        if h_dq is None:
            hunter_h[h] = deque([actual], maxlen=K)
        else:
            h_dq.append(actual)

        for s_arr in surv_arrs:
            s  = s_arr[i]
            dq = surv_h.get(s)
            if dq is None:
                surv_h[s] = deque([actual], maxlen=K)
            else:
                dq.append(actual)

    return preds, df


def cv_rmse(
    matches: pd.DataFrame,
    K: int = 30,
    n_splits: int = 5,
) -> tuple[float, float, list[float]]:
    """
    Compute per-fold RMSE on TimeSeriesSplit test folds.
    The rolling state is built from the FULL dataset (all folds),
    then evaluated only on each fold's test indices — matching the
    way the BT model uses its training data.
    """
    preds, df = rolling_predict(matches, K)
    y         = (df["n_escaped"] - 2).to_numpy(dtype=np.float64)
    tscv      = TimeSeriesSplit(n_splits=n_splits)

    fold_rmses = []
    for _, te in tscv.split(df):
        r = float(np.sqrt(np.mean((y[te] - preds[te]) ** 2)))
        fold_rmses.append(r)
    mean = float(np.mean(fold_rmses))
    se   = float(np.std(fold_rmses, ddof=1) / np.sqrt(n_splits))
    return mean, se, fold_rmses


def sweep_K(
    matches: pd.DataFrame,
    K_values: Iterable[int] = (5, 10, 20, 30, 50, 100, 200),
    n_splits: int = 5,
) -> pd.DataFrame:
    rows = []
    for K in K_values:
        mean, se, _ = cv_rmse(matches, K, n_splits)
        rows.append({"K": K, "rmse_mean": mean, "rmse_se": se})
    return pd.DataFrame(rows)
