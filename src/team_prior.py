"""
src/team_prior.py
=================
EXPERIMENT (branch: team-prior)

Initialize NEW players' ridge prior centred at the mean of their first-match
teammates' β values, rather than at zero.  Established players keep the
standard N(0, σ²) prior.

This is a *targeted* informed prior, NOT a universal team-anchored
penalty — established players have enough individual data; only the
sparse-history players benefit from the team-anchoring.

Definition of "new"
-------------------
A player is "new" iff they appear in fewer than `new_player_threshold`
matches in the training set.  Default threshold = 5.

Procedure (two-pass)
--------------------
1. Fit the model (linear or ordinal) with the standard prior (μ=0).
   → preliminary β estimates for all players.
2. For each new player i:
     a. Find their first match in training.
     b. team_mean_i = mean of teammates' preliminary β values from that
        match (their 3 co-survivors if i is a survivor; their 4 opposing
        survivors if i is a hunter).
     c. Set prior_mean[i] = team_mean_i.
3. Refit with the per-player prior:    penalty = 0.5 λ ‖β − μ‖²

For closed-form linear ridge this just shifts the RHS:
    (XᵀWX + λI) β = XᵀW y  +  λ μ

For ordinal it modifies the gradient analogously (already supported via
`ordinal.fit_ordinal`'s `prior_mean` parameter).

Public API
----------
    identify_new_players(matches, player_index, threshold)
        → set of (player_id, role) keys

    compute_team_mean_priors(matches, player_index, beta, new_keys)
        → ndarray (P,) — zeros for established players, team mean for new

    fit_with_new_player_prior(matches, model='linear', ...,
                              threshold=5, weights=None)
        → (beta, player_index)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg

from bradley_terry import (
    SURVIVOR_COLS,
    build_design_matrix,
    build_player_index,
    compute_weights,
    fit,
    filter_complete,
)


# ---------------------------------------------------------------------------
# Identifying new players
# ---------------------------------------------------------------------------

def identify_new_players(
    matches: pd.DataFrame,
    player_index: dict[tuple[str, str], int],
    threshold: int = 5,
) -> set[tuple[str, str]]:
    """
    Return the set of (player_id, role) keys for players appearing in fewer
    than `threshold` matches in the training set.
    """
    counts: dict[tuple[str, str], int] = {k: 0 for k in player_index}
    h = matches["hunter_player"].astype(str)
    for pid in h:
        key = (pid, "hunter")
        if key in counts:
            counts[key] += 1
    for col in SURVIVOR_COLS:
        for pid in matches[col].dropna().astype(str):
            key = (pid, "survivor")
            if key in counts:
                counts[key] += 1
    return {k for k, n in counts.items() if n < threshold}


# ---------------------------------------------------------------------------
# Computing team-mean priors at first appearance
# ---------------------------------------------------------------------------

def compute_team_mean_priors(
    matches: pd.DataFrame,
    player_index: dict[tuple[str, str], int],
    beta: np.ndarray,
    new_keys: set[tuple[str, str]],
) -> np.ndarray:
    """
    For each new player, find their first training match and compute the
    mean β of their teammates from that match (using the preliminary β
    passed in).  Return a P-vector of prior means (zeros for established
    players).

    "Teammates" of:
        a new SURVIVOR  = the other 3 survivors in their first match
        a new HUNTER    = the 4 opposing survivors in their first match

    Teammates who are themselves "new" (and therefore have unreliable
    preliminary β) are excluded from the mean — if all teammates are
    new, prior stays at 0.
    """
    P = len(player_index)
    mu = np.zeros(P, dtype=np.float64)

    df = matches.sort_values("date").reset_index(drop=True)
    surv_cols = SURVIVOR_COLS

    # For each new player, find first appearance
    found: dict[tuple[str, str], int] = {}
    for i, row in df.iterrows():
        if len(found) == len(new_keys):
            break
        # hunter
        hk = (str(row["hunter_player"]), "hunter")
        if hk in new_keys and hk not in found:
            found[hk] = i
        for col in surv_cols:
            pid = row[col]
            if pd.notna(pid):
                sk = (str(pid), "survivor")
                if sk in new_keys and sk not in found:
                    found[sk] = i

    # For each new player, compute teammates' β mean (excluding new teammates)
    for key, first_idx in found.items():
        row = df.iloc[first_idx]
        if key[1] == "survivor":
            # teammates = other survivors in the same row
            teammate_keys = []
            for col in surv_cols:
                pid = row[col]
                if pd.notna(pid):
                    tk = (str(pid), "survivor")
                    if tk != key and tk in player_index:
                        teammate_keys.append(tk)
        else:  # hunter
            # teammates = the 4 survivors in the same row (opposing side
            # but they're what defines the matchup; use them as anchor)
            teammate_keys = []
            for col in surv_cols:
                pid = row[col]
                if pd.notna(pid):
                    tk = (str(pid), "survivor")
                    if tk in player_index:
                        teammate_keys.append(tk)

        # Exclude teammates who are themselves new (unreliable estimates)
        established = [tk for tk in teammate_keys if tk not in new_keys]
        if established:
            tm_betas = np.array([beta[player_index[tk]] for tk in established])
            # For a HUNTER new player, the team-mean should be NEGATED:
            # if their opposing survivors are strong (high β), the hunter
            # facing them must be strong too (low/negative-margin contributor).
            # But our hunter parameter convention is β^H subtracted from
            # margin: stronger hunter -> more negative β^H means more
            # negative margin contribution (-β^H added → margin -= β^H).
            # Wait: design matrix has -1 for hunter column. So predicted
            # margin contribution = -β^H_h. Strong hunter (suppresses
            # escapes) has positive β^H. Conversely, weak hunters have
            # negative β^H.
            # If a new hunter consistently faces strong survivors and
            # somehow wins, we'd infer the hunter is strong too. As a
            # first-appearance prior, we don't have the outcome info yet,
            # so we just say "this hunter is at the level of strong
            # opponents" → use mean of opponents' β directly (positive
            # mean of strong survivors → hunter prior is also positive,
            # meaning strong).
            mu[player_index[key]] = float(tm_betas.mean())
        # else: leave mu at 0

    return mu


# ---------------------------------------------------------------------------
# Two-pass fit
# ---------------------------------------------------------------------------

def fit_with_new_player_prior(
    matches: pd.DataFrame,
    model: str = "linear",
    half_life_days: float | None = None,
    reference_date: str | None = None,
    l2_lambda: float = 1.0,
    threshold: int = 5,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[tuple[str, str], int]]:
    """
    Fit the BT margin model with a team-anchored prior applied ONLY to
    players appearing in fewer than `threshold` training matches.

    Procedure:
      1. Standard fit (prior μ = 0) → preliminary β
      2. Identify new players (training-set appearances < threshold)
      3. Compute their team-mean priors from preliminary β
      4. Refit with the per-player prior shifted to those means

    Parameters
    ----------
    model           : "linear" or "ordinal"
    half_life_days  : single-rate decay (ignored if `weights` is given)
    weights         : optional pre-computed weight vector
    threshold       : player is "new" if training appearances < threshold
    """
    m = filter_complete(matches)
    if reference_date is None:
        reference_date = m["date"].max()

    player_index = build_player_index(m)
    if weights is None:
        w = compute_weights(m["date"], reference_date, half_life_days)
    else:
        w = np.asarray(weights, dtype=np.float64)
    P = len(player_index)

    # --- Pass 1: standard fit ---
    if model == "linear":
        X, y = build_design_matrix(m, player_index)
        beta0 = fit(X, y, w, l2_lambda)
    elif model == "ordinal":
        from ordinal import fit_ordinal
        res0 = fit_ordinal(m, l2_lambda=l2_lambda, weights=w)
        beta0 = res0["beta"]
        theta0 = res0["theta"]
    else:
        raise ValueError(f"model must be 'linear' or 'ordinal', got {model!r}")

    # --- Identify new players + compute team-mean priors ---
    new_keys = identify_new_players(m, player_index, threshold)
    mu = compute_team_mean_priors(m, player_index, beta0, new_keys)

    # --- Pass 2: refit with prior_mean = mu ---
    if model == "linear":
        # Standard ridge with shifted RHS.  Solve (X'WX + λI) β = X'Wy + λμ.
        X, y   = build_design_matrix(m, player_index)
        sqrt_w = np.sqrt(w)
        Xw     = X.multiply(sqrt_w[:, np.newaxis])
        A      = (Xw.T @ Xw).toarray()
        A[np.diag_indices_from(A)] += l2_lambda
        b      = np.asarray(X.T @ (w * y)).ravel() + l2_lambda * mu
        c, low = linalg.cho_factor(A)
        beta1  = linalg.cho_solve((c, low), b)
        return beta1, player_index
    else:
        from ordinal import fit_ordinal
        res1 = fit_ordinal(m, l2_lambda=l2_lambda, weights=w,
                            init_theta=theta0, prior_mean=mu)
        return res1["beta"], player_index, res1  # also return res for theta access
