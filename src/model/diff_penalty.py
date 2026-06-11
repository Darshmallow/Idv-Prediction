"""
src/diff_penalty.py
===================
EXPERIMENT (branch: diff-penalty)

Option 4: a *targeted* difference penalty on detected collinear survivor
pairs, instead of the blunt team-intercept of Option 3.

Objective
---------
    min_β  Σ_i w_i (y_i − x_iβ)²  +  λ₁‖β‖²  +  λ_d Σ_{(j,k)∈P} (β_j − β_k)²

The last term says: "for pairs of players we cannot tell apart (they almost
always appear together), assume similar skill."  Σ (β_j−β_k)² = β'Lβ where L
is the graph Laplacian of the collinear-pair graph, so the solution stays
closed-form:

    β* = (X'WX + λ₁I + λ_d L)⁻¹ X'Wy

Unlike a team intercept, this only shrinks the *difference* within detected
pairs — players identified across many distinct rosters (e.g. huan) are
untouched, because they don't form high-overlap pairs.

Collinear-pair detection
-------------------------
A survivor pair (a, b) is collinear when the rarer player almost always
appears alongside the other:

    co_appearances(a,b) / min(games(a), games(b))  ≥  theta
    and min(games(a), games(b)) ≥ min_games

Pairs are detected on the *training* data inside each CV fold.

Public API
----------
    detect_collinear_pairs(matches, index, theta, min_games) → list[(a,b)]
    build_laplacian(pairs, index)                            → ndarray (P×P)
    fit_diff(X, y, weights, laplacian, l2, l2_diff)          → beta
    fit_ratings(matches, half_life, ref, l2, l2_diff, theta, min_games)
                                                              → (beta, index, pairs)
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.bradley_terry import (
    SURVIVOR_COLS,
    filter_complete,
    build_player_index,
    build_design_matrix,
    compute_weights,
)


# ---------------------------------------------------------------------------
# Collinear-pair detection
# ---------------------------------------------------------------------------

def detect_collinear_pairs(
    matches: pd.DataFrame,
    player_index: dict,
    theta: float = 0.9,
    min_games: float = 8,
) -> list[tuple[str, str]]:
    """
    Find survivor pairs that almost always appear together.

    Returns a list of (id_a, id_b) tuples (a < b), restricted to players
    present in player_index as survivors.
    """
    matches = matches.reset_index(drop=True)

    appears: dict[str, set] = defaultdict(set)
    for col in SURVIVOR_COLS:
        for i, pid in enumerate(matches[col]):
            if pd.notna(pid):
                appears[str(pid)].add(i)

    survivors = [p for p in appears if (p, "survivor") in player_index]
    pairs: list[tuple[str, str]] = []

    for ai in range(len(survivors)):
        a = survivors[ai]
        ga = appears[a]
        na = len(ga)
        if na < min_games:
            continue
        for bi in range(ai + 1, len(survivors)):
            b = survivors[bi]
            gb = appears[b]
            nb = len(gb)
            if nb < min_games:
                continue
            n_ab = len(ga & gb)
            if n_ab == 0:
                continue
            if n_ab / min(na, nb) >= theta:
                pairs.append((a, b))
    return pairs


# ---------------------------------------------------------------------------
# Graph Laplacian of the collinear-pair graph
# ---------------------------------------------------------------------------

def build_laplacian(pairs: list[tuple[str, str]], player_index: dict) -> np.ndarray:
    """
    L such that  β'Lβ = Σ_{(a,b)∈pairs} (β_a − β_b)².
    Only couples survivor columns of paired players.
    """
    P = len(player_index)
    L = np.zeros((P, P), dtype=np.float64)
    for a, b in pairs:
        ca = player_index[(a, "survivor")]
        cb = player_index[(b, "survivor")]
        L[ca, ca] += 1.0
        L[cb, cb] += 1.0
        L[ca, cb] -= 1.0
        L[cb, ca] -= 1.0
    return L


# ---------------------------------------------------------------------------
# Penalised solve
# ---------------------------------------------------------------------------

def fit_diff(
    X,
    y: np.ndarray,
    weights: np.ndarray,
    laplacian: np.ndarray,
    l2: float = 1.0,
    l2_diff: float = 0.0,
) -> np.ndarray:
    """
    Solve  (X'WX + λ₁I + λ_d L) β = X'Wy  via Cholesky.

    l2      = λ₁ ridge (same as the baseline model)
    l2_diff = λ_d strength of the difference penalty
    """
    sqrt_w = np.sqrt(weights)
    X_w    = X.multiply(sqrt_w[:, np.newaxis])

    A = (X_w.T @ X_w).toarray()
    A[np.diag_indices_from(A)] += l2
    if l2_diff:
        A += l2_diff * laplacian
    b = np.asarray(X.T @ (weights * y)).ravel()

    c, low = linalg.cho_factor(A)
    return linalg.cho_solve((c, low), b)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def fit_ratings(
    matches: pd.DataFrame,
    half_life_days: float | None = None,
    reference_date: str | None = None,
    l2: float = 1.0,
    l2_diff: float = 3.0,        # RMSE-optimal in 5-fold temporal CV
    theta: float = 0.9,
    min_games: float = 8,
) -> tuple[np.ndarray, dict, list]:
    """Fit the difference-penalised model on a single dataset."""
    matches = filter_complete(matches)
    if reference_date is None:
        reference_date = matches["date"].max()

    index = build_player_index(matches)
    X, y  = build_design_matrix(matches, index)
    w     = compute_weights(matches["date"], reference_date, half_life_days)
    pairs = detect_collinear_pairs(matches, index, theta, min_games)
    L     = build_laplacian(pairs, index)
    beta  = fit_diff(X, y, w, L, l2, l2_diff)
    return beta, index, pairs
