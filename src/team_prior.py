"""
src/team_prior.py
=================
EXPERIMENT (branch: team-prior)

Replace the standard centred-at-zero L2 prior on player ratings with a
**team-anchored prior**:

    β_i ~ N(μ_team(i), σ²)         instead of    β_i ~ N(0, σ²)

where μ_team(i) is the (co-appearance-weighted) mean rating of survivor i's
teammates.  Implemented as a graph-Laplacian-style structured penalty:

    L(β) = ½ ‖y − Xβ‖²_W + λ_player ‖β‖²  +  λ_team ‖(I − A)β‖²

A is the row-normalised survivor co-appearance adjacency matrix:

    A[i, j] = (# matches with survivors i and j together)
              / (# matches involving survivor i)

so (Aβ)_i is the co-appearance-weighted mean of survivor i's teammates'
ratings, and (I − A)β is the deviation of each survivor's rating from
that mean.  Hunters are exempt from the team-anchored term — they don't
have teammates.

Intuition for "informed prior on new players"
---------------------------------------------
A player with N training matches has their β estimated jointly from
their own observations and the prior.  When N is small, the prior
dominates.  With standard L2 the prior is N(0, σ²), pulling new players
to "average".  With team-anchoring, the prior is centred at the
teammates' mean, so a player who only ever shows up alongside a strong
roster is presumed strong themselves — consistent with the empirical
observation that strong teams recruit talented players.

Public API
----------
    build_team_adjacency(matches, player_index)
        → ndarray (P, P)   survivor-co-appearance row-normalised adjacency

    fit_team_anchored(matches, half_life_days=None, ...,
                       l2_player=0.1, l2_team=1.0, weights=None)
        → (beta, player_index)

When λ_team = 0 this reduces exactly to the baseline linear BT model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg, sparse

from bradley_terry import (
    SURVIVOR_COLS,
    build_design_matrix,
    build_player_index,
    compute_weights,
    filter_complete,
)


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------

def build_team_adjacency(
    matches: pd.DataFrame,
    player_index: dict[tuple[str, str], int],
) -> np.ndarray:
    """
    Build the (P × P) survivor-co-appearance adjacency, row-normalised.

    Hunter rows (and the only entry on their row, A[h, h]) are left at 0.
    A survivor's row sums to 1 (or 0 if they appear in zero training
    matches alongside any other survivor, which can't happen if they're
    in `player_index` at all).
    """
    P = len(player_index)
    A = np.zeros((P, P), dtype=np.float64)

    # Vectorised pair counting over the four survivor slots
    surv_idx_arrays = []
    for col in SURVIVOR_COLS:
        col_idx = np.array([
            player_index.get((str(p), "survivor"), -1)
            for p in matches[col]
        ], dtype=np.int64)
        surv_idx_arrays.append(col_idx)

    # Accumulate co-appearance counts (symmetric)
    for i in range(4):
        for j in range(i + 1, 4):
            a, b   = surv_idx_arrays[i], surv_idx_arrays[j]
            valid  = (a >= 0) & (b >= 0)
            np.add.at(A, (a[valid], b[valid]), 1.0)
            np.add.at(A, (b[valid], a[valid]), 1.0)

    # Row-normalise: each survivor's row → distribution over teammates
    row_sums = A.sum(axis=1)
    mask     = row_sums > 0
    A[mask]  = A[mask] / row_sums[mask, None]
    return A


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_team_anchored(
    matches: pd.DataFrame,
    half_life_days: float | None = None,
    reference_date: str | None = None,
    l2_player: float = 0.1,
    l2_team: float = 1.0,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[tuple[str, str], int]]:
    """
    Closed-form weighted ridge with team-anchored penalty.

    Solves the augmented normal equations

        (XᵀWX  +  λ_player·I  +  λ_team·(I−A)ᵀ(I−A)) β  =  XᵀW y

    via Cholesky.  When λ_team = 0 this is exactly the baseline linear model.

    Parameters
    ----------
    matches        : DataFrame of matches (will be filter_complete'd).
    half_life_days : Single-rate decay if `weights` is None.  Ignored if
                     `weights` is provided.
    reference_date : Anchor date for decay.
    l2_player      : Standard ridge on β (small ε for identifiability).
    l2_team        : Strength of the team-anchored shrinkage.
    weights        : Optional pre-computed per-match weights (e.g. from
                     `regime_decay.compute_weights_regime`).

    Returns
    -------
    beta         : (P,) array of player ratings
    player_index : (player_id, role) → column index
    """
    m = filter_complete(matches)
    if reference_date is None:
        reference_date = m["date"].max()

    player_index = build_player_index(m)
    X, y         = build_design_matrix(m, player_index)
    if weights is None:
        w = compute_weights(m["date"], reference_date, half_life_days)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(m):
            raise ValueError(f"weights length {len(w)} != filter_complete length {len(m)}")
    P = X.shape[1]

    # XᵀW X term (dense P × P)
    sqrt_w = np.sqrt(w)
    Xw     = X.multiply(sqrt_w[:, np.newaxis])
    XtWX   = (Xw.T @ Xw).toarray()
    XtWy   = np.asarray(X.T @ (w * y)).ravel()

    # Penalty matrix
    A      = build_team_adjacency(m, player_index)
    # Zero out hunter rows of (I − A) so hunters carry only λ_player ridge
    L = np.eye(P) - A
    is_hunter = np.array([
        key[1] == "hunter"
        for key, _ in sorted(player_index.items(), key=lambda kv: kv[1])
    ])
    L[is_hunter, :] = 0.0    # hunters not subject to team-anchored term

    Pen = l2_player * np.eye(P) + l2_team * (L.T @ L)

    Amat = XtWX + Pen

    # Cholesky solve (Amat is SPD as long as l2_player > 0)
    c, low = linalg.cho_factor(Amat)
    beta   = linalg.cho_solve((c, low), XtWy)
    return beta, player_index
