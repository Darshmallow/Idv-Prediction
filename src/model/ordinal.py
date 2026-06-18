"""
src/ordinal.py
==============
Proportional-odds (ordinal logistic) Bradley-Terry margin model.

Motivation
----------
The linear margin model assumes equal skill-spacing between consecutive
escape counts.  Empirically the cumulative thresholds are very unequal:

    n=0→1:  Δθ ≈ 0.71
    n=1→2:  Δθ ≈ 1.60
    n=2→3:  Δθ ≈ 2.33

The proportional-odds model lets the data choose the spacing.

Model
-----
Latent score for match i:
    η_i = (1/4) Σ_k β^S_{s_ik} − β^H_{h_i}

Five ordered outcome categories n ∈ {0, 1, 2, 3, 4}, four thresholds
θ_1 < θ_2 < θ_3 < θ_4.  Cumulative-logit link:

    P(n_i ≤ j) = σ(θ_{j+1} − η_i)   for j = 0..3,  P(n_i ≤ 4) = 1

So:
    P(n = 0)        = σ(θ_1 − η)
    P(n = j)        = σ(θ_{j+1} − η) − σ(θ_j − η)   for j = 1,2,3
    P(n = 4)        = 1 − σ(θ_4 − η)

Threshold ordering is enforced via reparameterisation:
    θ_1 = a
    θ_{k+1} = θ_k + exp(b_k)   for k = 1,2,3.

So the unconstrained optimiser sees (a, b_1, b_2, b_3) ∈ ℝ^4 with
guaranteed θ_1 < θ_2 < θ_3 < θ_4.

Fit
---
Minimise weighted negative log-likelihood with L2 ridge on β only:

    L(β, θ) = −Σ_i w_i log P(n_i | η_i, θ) + (λ/2) ‖β‖²

L-BFGS-B with analytical gradient.  Convex in β and (linearly in) θ.

Prediction
----------
Expected margin (for comparing to the linear model's RMSE):

    E[margin_i] = E[n_i] − 2 = Σ_j j · P(n_i = j) − 2

Public API
----------
    init_thresholds(matches)                     -> ndarray(4,)
    fit_ordinal(matches, half_life_days, ...)    -> dict
    predict_expected_margin(matches, fit_result) -> ndarray(N,)
    predict_probs(matches, fit_result)           -> ndarray(N, 5)
    sanity_check(db_path)                        -> None
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.bradley_terry import (
    SURVIVOR_COLS,
    build_design_matrix,
    build_player_index,
    compute_weights,
    filter_complete,
)


_ROOT      = Path(__file__).parent.parent.parent
DEFAULT_DB = str(_ROOT / "data" / "processed" / "idv.db")
N_CATS     = 5      # n_escaped ∈ {0, 1, 2, 3, 4}
N_THRESH   = 4      # one threshold between each adjacent pair


# ---------------------------------------------------------------------------
# Threshold parameterisation: (a, b1, b2, b3) ↔ (θ_1, θ_2, θ_3, θ_4)
# ---------------------------------------------------------------------------

def _ab_to_theta(ab: np.ndarray) -> np.ndarray:
    """(a, b_1, b_2, b_3) → (θ_1, θ_2, θ_3, θ_4) with θ_1 < … < θ_4."""
    a, b = ab[0], ab[1:]
    return a + np.concatenate(([0.0], np.cumsum(np.exp(b))))


def _theta_to_ab(theta: np.ndarray) -> np.ndarray:
    """Inverse: only used for initialisation."""
    return np.concatenate(([theta[0]], np.log(np.diff(theta))))


def init_thresholds(matches: pd.DataFrame) -> np.ndarray:
    """
    Initial θ from the empirical cumulative distribution of n_escaped:
        θ_{j+1} = logit( P(n ≤ j) )   for j = 0, 1, 2, 3.
    """
    n = matches["n_escaped"].dropna().astype(int)
    return np.array([logit((n <= j).mean()) for j in range(N_THRESH)])


# ---------------------------------------------------------------------------
# Log-likelihood with analytical gradient
# ---------------------------------------------------------------------------

def _logp_and_grad(
    eta: np.ndarray,
    theta: np.ndarray,
    n_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-match log P(n_i = obs_i | η_i, θ) and its gradient w.r.t. η_i and θ.

    Returns
    -------
    log_p  : (N,)   log-likelihood per match
    d_eta  : (N,)   ∂ log_p / ∂ η_i
    d_th   : (N, 4) ∂ log_p / ∂ θ_k     (only ≤2 nonzero entries per row)
    """
    EPS  = 1e-12
    # u[:, k] = σ(θ_{k+1} − η),  k = 0..3   (matches θ_1..θ_4)
    u    = expit(theta[None, :] - eta[:, None])
    N    = len(eta)
    log_p = np.empty(N)
    d_eta = np.empty(N)
    d_th  = np.zeros((N, N_THRESH))

    # n = 0:  P = u_1
    m = n_obs == 0
    if m.any():
        p          = np.clip(u[m, 0], EPS, 1 - EPS)
        log_p[m]   = np.log(p)
        d_eta[m]   = -(1 - p)
        d_th[m, 0] =  (1 - p)

    # n = j ∈ {1, 2, 3}:  P = u_{j+1} − u_j
    for j in (1, 2, 3):
        m = n_obs == j
        if not m.any():
            continue
        u_lo = u[m, j-1]           # u_j
        u_hi = u[m, j]             # u_{j+1}
        p    = np.clip(u_hi - u_lo, EPS, 1.0)
        log_p[m]    = np.log(p)
        d_eta[m]    = (u_lo * (1 - u_lo) - u_hi * (1 - u_hi)) / p
        d_th[m, j-1] = -u_lo * (1 - u_lo) / p
        d_th[m, j]   =  u_hi * (1 - u_hi) / p

    # n = 4:  P = 1 − u_4
    m = n_obs == 4
    if m.any():
        u4         = u[m, 3]
        p          = np.clip(1 - u4, EPS, 1 - EPS)
        log_p[m]   = np.log(p)
        d_eta[m]   = u4
        d_th[m, 3] = -u4

    return log_p, d_eta, d_th


def _objective_and_grad(
    params: np.ndarray,
    X: sparse.csr_matrix,
    n_obs: np.ndarray,
    w: np.ndarray,
    P: int,
    l2_lambda: float,
    prior_mean: np.ndarray | None = None,
    fixed_theta: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """
    Compute (NLL + ridge) and its gradient.

    params layout (fixed_theta=None)
    ---------------------------------
    [β (P,),  a, b_1, b_2, b_3]    →  shape (P + 4,)

    params layout (fixed_theta given)
    ----------------------------------
    [β (P,)]    →  shape (P,)   — θ held constant, only β is optimised.

    The ridge term is  0.5 * λ * ‖β − μ‖²  where μ defaults to zeros.
    """
    if fixed_theta is not None:
        beta = params          # params is just β
        th   = fixed_theta
        eta  = X @ beta

        log_p, d_eta, _ = _logp_and_grad(eta, th, n_obs)

        beta_centred = beta if prior_mean is None else beta - prior_mean
        nll    = -np.sum(w * log_p) + 0.5 * l2_lambda * float(beta_centred @ beta_centred)
        g_beta = -np.asarray(X.T @ (w * d_eta)).ravel() + l2_lambda * beta_centred
        return float(nll), g_beta

    beta = params[:P]
    ab   = params[P:]
    th   = _ab_to_theta(ab)
    eta  = X @ beta

    log_p, d_eta, d_th = _logp_and_grad(eta, th, n_obs)

    if prior_mean is None:
        beta_centred = beta
    else:
        beta_centred = beta - prior_mean

    nll = -np.sum(w * log_p) + 0.5 * l2_lambda * float(beta_centred @ beta_centred)

    # ∂nll / ∂β = −Xᵀ(w · d_eta) + λ(β − μ)
    g_beta = -np.asarray(X.T @ (w * d_eta)).ravel() + l2_lambda * beta_centred

    # ∂nll / ∂θ_k = −Σ_i w_i · d_th[i, k]
    g_theta = -(w[:, None] * d_th).sum(axis=0)

    # Chain through a, b_1, b_2, b_3
    e        = np.exp(ab[1:])                 # (3,)
    g_a      = g_theta.sum()
    g_b      = np.array([
        e[0] * g_theta[1:].sum(),
        e[1] * g_theta[2:].sum(),
        e[2] * g_theta[3:].sum(),
    ])
    g_ab     = np.concatenate(([g_a], g_b))

    return float(nll), np.concatenate([g_beta, g_ab])


# ---------------------------------------------------------------------------
# Fit + Predict
# ---------------------------------------------------------------------------

def fit_ordinal(
    matches: pd.DataFrame,
    half_life_days: float | None = None,
    reference_date: str | None = None,
    l2_lambda: float = 1.0,
    init_beta: np.ndarray | None = None,
    init_theta: np.ndarray | None = None,
    fixed_theta: np.ndarray | None = None,
    maxiter: int = 1000,
    verbose: bool = False,
    weights: np.ndarray | None = None,
    prior_mean: np.ndarray | None = None,
) -> dict:
    """
    Fit the proportional-odds margin model on a single dataset.

    Parameters
    ----------
    weights : Optional pre-computed per-match weight vector.  If given,
              `half_life_days` and `reference_date` are ignored.  Must be
              the same length as `filter_complete(matches)`.  Useful for
              experimenting with non-standard weight schemes (e.g.
              piecewise / regime-shift decay — see src/regime_decay.py).
    prior_mean : Optional P-vector setting the centre of the L2 ridge for
                 each player.  Default (None) → zeros (standard L2 prior
                 N(0, σ²)).  Used to inject team-anchored priors for
                 cold-start players — see src/team_prior.py.

    All other parameters and return value unchanged.

    Returns
    -------
    dict with keys:
        beta           : (P,) player ratings
        theta          : (4,) cumulative thresholds
        player_index   : (player_id, role) → column index
        n_iter         : optimisation iterations
        converged      : bool
        final_nll      : objective value at solution
    """
    m = filter_complete(matches)
    if reference_date is None:
        reference_date = m["date"].max()

    player_index = build_player_index(m)
    X, _         = build_design_matrix(m, player_index)
    n_obs        = m["n_escaped"].to_numpy(dtype=int)
    if weights is None:
        w = compute_weights(m["date"], reference_date, half_life_days)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(m):
            raise ValueError(
                f"weights length {len(w)} doesn't match "
                f"filter_complete(matches) length {len(m)}"
            )
    P            = X.shape[1]

    # Initialisation
    beta0  = np.zeros(P) if init_beta is None else init_beta.copy()

    if prior_mean is not None:
        prior_mean = np.asarray(prior_mean, dtype=np.float64)
        if len(prior_mean) != P:
            raise ValueError(f"prior_mean length {len(prior_mean)} != P={P}")

    if fixed_theta is not None:
        # θ is held constant — only optimise β
        theta_hat = np.asarray(fixed_theta, dtype=np.float64)
        result = minimize(
            _objective_and_grad,
            beta0,
            args=(X, n_obs, w, P, l2_lambda, prior_mean, theta_hat),
            method="L-BFGS-B",
            jac=True,
            options=dict(maxiter=maxiter, ftol=1e-9, gtol=1e-7),
        )
        beta_hat = result.x
    else:
        theta0 = init_thresholds(m) if init_theta is None else init_theta.copy()
        ab0    = _theta_to_ab(theta0)
        x0     = np.concatenate([beta0, ab0])
        result = minimize(
            _objective_and_grad,
            x0,
            args=(X, n_obs, w, P, l2_lambda, prior_mean, None),
            method="L-BFGS-B",
            jac=True,
            options=dict(maxiter=maxiter, ftol=1e-9, gtol=1e-7),
        )
        beta_hat  = result.x[:P]
        theta_hat = _ab_to_theta(result.x[P:])

    return {
        "beta":         beta_hat,
        "theta":        theta_hat,
        "player_index": player_index,
        "n_iter":       result.nit,
        "converged":    result.success,
        "final_nll":    float(result.fun),
    }


def predict_probs(
    matches: pd.DataFrame,
    fit_result: dict,
) -> np.ndarray:
    """
    Per-match probability distribution over n ∈ {0, 1, 2, 3, 4}.

    Returns
    -------
    probs : (N, 5)
    """
    beta  = fit_result["beta"]
    theta = fit_result["theta"]
    idx   = fit_result["player_index"]
    X, _  = build_design_matrix(matches, idx)
    eta   = X @ beta

    u = expit(theta[None, :] - eta[:, None])     # (N, 4)
    N = len(eta)
    p = np.empty((N, N_CATS))
    p[:, 0] = u[:, 0]
    p[:, 1] = u[:, 1] - u[:, 0]
    p[:, 2] = u[:, 2] - u[:, 1]
    p[:, 3] = u[:, 3] - u[:, 2]
    p[:, 4] = 1.0 - u[:, 3]
    return np.clip(p, 0.0, 1.0)


def predict_expected_margin(
    matches: pd.DataFrame,
    fit_result: dict,
) -> np.ndarray:
    """
    Expected margin E[n] − 2 for each match.  Use this for RMSE
    comparison with the linear model on the same target.
    """
    probs = predict_probs(matches, fit_result)
    levels = np.arange(N_CATS)
    return probs @ levels - 2.0


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(db_path: str = DEFAULT_DB, l2_lambda: float = 1.0) -> None:
    """
    Fit on all data (no time decay) and print top-10 hunters and survivors,
    plus the learned threshold spacings.
    """
    conn = sqlite3.connect(db_path)
    matches = pd.read_sql("SELECT * FROM matches ORDER BY date", conn)
    conn.close()

    print("Fitting ordinal margin model (no time decay)…")
    res = fit_ordinal(matches, half_life_days=None, l2_lambda=l2_lambda, verbose=False)

    theta = res["theta"]
    print(f"\nConverged: {res['converged']}  iters: {res['n_iter']}  "
          f"NLL: {res['final_nll']:.1f}")

    print(f"\nLearned thresholds (vs initialisation):")
    init_th = init_thresholds(filter_complete(matches))
    print(f"  k   init θ_k    fit θ_k")
    for k in range(N_THRESH):
        print(f"  {k+1}   {init_th[k]:>+7.3f}    {theta[k]:>+7.3f}")
    spacings = np.diff(theta)
    print(f"\nFit spacings: {spacings.round(3)}")
    print(f"  → n=0→1 gap is {spacings[0]:.2f}, n=3→4 gap is {spacings[2]:.2f}  "
          f"(ratio {spacings[2]/spacings[0]:.2f}×)")

    # Top players (using same approach as bradley_terry.top_ratings)
    rows = [
        {"player_id": k[0], "role": k[1], "beta": float(res["beta"][i])}
        for k, i in res["player_index"].items()
    ]
    df = pd.DataFrame(rows).sort_values(["role", "beta"], ascending=[True, False])

    print(f"\n──── Top 10 HUNTERS ────")
    print(df[df["role"] == "hunter"].head(10).to_string(index=False))
    print(f"\n──── Top 10 SURVIVORS ────")
    print(df[df["role"] == "survivor"].head(10).to_string(index=False))


if __name__ == "__main__":
    sanity_check()
