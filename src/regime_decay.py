"""
src/regime_decay.py
===================
EXPERIMENT — piecewise exponential decay with a manual regime-shift drop.

Motivation
----------
The 2023 IVL season is the worst season-fold for the model: trained on
2020-2022, the model systematically over-predicts survivor margin
because the meta shifted toward hunters. The stretch-analysis previously
showed pre-2023 has optimal τ ≈ 110 d while post-2023 has τ ≈ 71 d, so
"the right τ" isn't a single number across eras.

This module implements a piecewise weighting that:

  1. Uses one half-life (τ_pre) for matches before the regime shift.
  2. Uses another half-life (τ_post) for matches after.
  3. Multiplies every pre-shift match by a fixed discount α ∈ (0, 1] to
     model an additional information loss caused by the discrete meta change.

    w_i = α · 0.5^( Δt_i / τ_pre )      if  t_i  <  t_shift
        =     0.5^( Δt_i / τ_post )    if  t_i  ≥  t_shift

When τ_pre = τ_post and α = 1, this reduces to the baseline single-rate decay.

Default shift date is 2023-01-01 (start of the 2023 competitive season,
the first negative-R² fold under season-aligned CV).

Public API
----------
    compute_weights_regime(dates, reference_date, tau_post, tau_pre,
                           shift_date='2023-01-01', alpha=1.0)
        → ndarray of per-match weights

    sweep_regime(matches, taus_post, alphas, tau_pre, shift_date,
                  n_splits=5, l2_lambda=1.0, split_strategy='seasons',
                  model='ordinal')
        → long-form DataFrame (one row per τ_post × α × fold)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from bradley_terry import filter_complete
from eval import get_splits


# ---------------------------------------------------------------------------
# Weight function
# ---------------------------------------------------------------------------

def compute_weights_regime(
    dates: pd.Series,
    reference_date: str | pd.Timestamp,
    tau_post: float,
    tau_pre: float | None = None,
    shift_date: str | pd.Timestamp = "2023-01-01",
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Piecewise exponential decay with an optional multiplicative drop at the
    regime boundary.

    Parameters
    ----------
    dates           : Series of ISO date strings (match dates).
    reference_date  : Anchor for decay (matches on this date get weight 1).
    tau_post        : Half-life applied to matches with date ≥ shift_date.
    tau_pre         : Half-life applied to matches with date < shift_date.
                      Defaults to tau_post (same rate both eras).
    shift_date      : The regime-change date.  Pre-shift matches are
                      multiplied by `alpha`.
    alpha           : Discount applied to pre-shift matches AFTER decay.
                      alpha = 1.0 → no extra drop.  alpha → 0 → ignore pre.

    Returns
    -------
    weights : ndarray(N,), all positive.
    """
    if tau_pre is None:
        tau_pre = tau_post

    ref     = pd.Timestamp(reference_date)
    shift   = pd.Timestamp(shift_date)
    dt      = (ref - pd.to_datetime(dates)).dt.days.to_numpy(dtype=np.float64)
    dt      = np.clip(dt, 0.0, None)
    pre     = (pd.to_datetime(dates) < shift).to_numpy()

    w = np.empty(len(dates), dtype=np.float64)
    w[~pre] = np.power(0.5, dt[~pre] / tau_post)
    w[pre]  = alpha * np.power(0.5, dt[pre] / tau_pre)
    return w


# ---------------------------------------------------------------------------
# Sweep over (τ_post, α)
# ---------------------------------------------------------------------------

def sweep_regime(
    matches: pd.DataFrame,
    taus_post: list[float] | np.ndarray,
    alphas: list[float] | np.ndarray,
    tau_pre: float = 110.0,
    shift_date: str = "2023-01-01",
    n_splits: int = 5,
    l2_lambda: float = 1.0,
    split_strategy: str = "seasons",
    model: str = "ordinal",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Grid sweep over (τ_post, α) at fixed τ_pre and shift_date.

    For each combination, run temporal CV with regime-shift weights.

    Parameters
    ----------
    matches         : Full matches DataFrame.
    taus_post       : Half-life candidates for post-shift matches.
    alphas          : Discount multipliers ∈ (0, 1] for pre-shift matches.
    tau_pre         : Fixed half-life for pre-shift matches.
    shift_date      : Boundary date.
    n_splits        : CV folds (ignored under "seasons").
    l2_lambda       : Ridge penalty.
    split_strategy  : "time_series" or "seasons".
    model           : "linear" or "ordinal".  Ordinal is the canonical
                      headline model; linear is faster for exploration.
    verbose         : Print progress.

    Returns
    -------
    Long-form DataFrame with columns:
        tau_post, alpha, tau_pre, shift_date, fold, test_period,
        rmse, null_rmse, r2
    """
    matches = filter_complete(matches).sort_values("date").reset_index(drop=True)
    splits  = get_splits(matches, split_strategy, n_splits)

    if model == "ordinal":
        from ordinal import fit_ordinal, predict_expected_margin
        from bradley_terry import build_design_matrix
        def _fit(train, test, weights, refdate):
            res  = fit_ordinal(train, half_life_days=None, l2_lambda=l2_lambda)
            # need to pass custom weights; ordinal.fit_ordinal accepts only τ.
            # Inject by re-fitting with a custom path:
            return None
        raise NotImplementedError("Use model='linear' for the regime sweep; "
                                  "ordinal would require an internal refactor "
                                  "to accept pre-computed weights.")

    # Linear path — closed-form ridge, supports arbitrary weight vectors.
    from bradley_terry import build_design_matrix, build_player_index, fit, predict

    all_rows = []
    t0 = time.time()
    n_combos = len(taus_post) * len(alphas)
    combo_idx = 0

    for tau_post in taus_post:
        for alpha in alphas:
            combo_idx += 1
            if verbose:
                print(f"  [{combo_idx:2d}/{n_combos}]  τ_post={tau_post:>5.0f}d  "
                      f"α={alpha:>4.2f} …", end="", flush=True)
            fold_t0 = time.time()

            for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
                train, test = matches.iloc[tr_idx], matches.iloc[te_idx]
                ref         = train["date"].max()
                idx         = build_player_index(train)
                X_tr, y_tr  = build_design_matrix(train, idx)
                w           = compute_weights_regime(
                    train["date"], ref,
                    tau_post=float(tau_post), tau_pre=float(tau_pre),
                    shift_date=shift_date, alpha=float(alpha),
                )
                beta        = fit(X_tr, y_tr, w, l2_lambda)
                X_te, y_te  = build_design_matrix(test, idx)
                y_hat       = predict(X_te, beta)
                rmse        = float(np.sqrt(np.mean((y_te - y_hat) ** 2)))
                null        = float(np.sqrt(np.mean(y_te ** 2)))
                # human-readable test period
                from eval import _test_period_label
                period      = _test_period_label(matches, te_idx, split_strategy)
                all_rows.append({
                    "tau_post":   float(tau_post),
                    "alpha":      float(alpha),
                    "tau_pre":    float(tau_pre),
                    "shift_date": shift_date,
                    "fold":       fold,
                    "test_period": period,
                    "rmse":       rmse,
                    "null_rmse":  null,
                    "r2":         1.0 - rmse ** 2 / null ** 2,
                })

            if verbose:
                mean_rmse = np.mean([r["rmse"] for r in all_rows[-len(splits):]])
                print(f"  mean RMSE={mean_rmse:.4f}  ({time.time()-fold_t0:.1f}s)")

    if verbose:
        print(f"\nRegime-decay sweep complete — {n_combos} (τ_post, α) × "
              f"{len(splits)} folds in {time.time()-t0:.0f}s")
    return pd.DataFrame(all_rows)
