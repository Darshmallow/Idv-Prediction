"""
src/bradley_terry.py
====================
Time-weighted linear Bradley-Terry model for IDV match outcomes.

Model
-----
    E[margin_i | β] = (1/4) Σ_k β^S_{s_{ik}} − β^H_{h_i}

    margin_i = n_escaped_i − 2  ∈ {−2, −1, 0, +1, +2}

    Positive margin → survivor advantage.  Negative → hunter advantage.
    High β always means "more skilled at your role":
      high β^H → hunter wins more (negative margin)
      high β^S → survivor escapes more (positive margin)

Fit via weighted ridge (closed-form normal equations)
-----------------------------------------------------
    β* = (X'WX + λI)⁻¹ X'Wy

    W = diag(w_i),   w_i = 0.5^(Δt_i / τ)     exponential time decay
    λ                                            L2 regularisation (ridge)

    X is a sparse N×P design matrix:
        X[i, (h_i,   "hunter")]   = +1
        X[i, (s_{ik},"survivor")] = −0.25   for k = 1..4

    Players not seen in training have β = 0 (the regularisation prior).

Parameters are indexed by (player_id, role) tuples, so dual-role players
(e.g. ppxia) get independent hunter and survivor ratings.

Public API
----------
    filter_complete(matches)                → DataFrame
    build_player_index(matches)             → dict[(str,str), int]
    build_design_matrix(matches, index)     → (csr_matrix, ndarray)
    compute_weights(dates, ref, half_life)  → ndarray
    fit(X, y, weights, l2_lambda)           → ndarray   (β vector)
    predict(X, beta)                        → ndarray   (ŷ vector)
    fit_ratings(matches, half_life_days,
                reference_date, l2_lambda)  → (beta, player_index)
    ratings_df(beta, player_index)          → DataFrame
    top_ratings(beta, player_index, role)   → DataFrame
    sanity_check(db_path)                   → None  (prints top-10 each role)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, sparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOT        = Path(__file__).parent.parent.parent
DEFAULT_DB   = str(_ROOT / "data" / "processed" / "idv.db")

SURVIVOR_COLS   = [f"survivor{i}_player" for i in range(1, 5)]
ALL_PLAYER_COLS = ["hunter_player"] + SURVIVOR_COLS

# Design-matrix entry weights per column type.
# E[margin] = mean(β^S) − β^H, so margin grows when survivors are strong
# and shrinks when the hunter is strong.
_HUNTER_WEIGHT   = -1.0         # hunter skill opposes escapes
_SURVIVOR_WEIGHT =  0.25        # +1/4: each survivor contributes positively


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def filter_complete(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where all five player IDs and n_escaped are non-null.
    Drops the two known legacy rows with missing survivor data.
    """
    mask = matches["n_escaped"].notna() & matches["hunter_player"].notna()
    for col in SURVIVOR_COLS:
        mask &= matches[col].notna()
    out = matches[mask].copy()
    dropped = len(matches) - len(out)
    if dropped:
        print(f"  filter_complete: dropped {dropped} incomplete row(s)")
    return out


def build_player_index(matches: pd.DataFrame) -> dict[tuple[str, str], int]:
    """
    Build a deterministic (player_id, role) → column-index mapping from
    the players seen in *matches*.  Sorted so the mapping is reproducible.

    Only players actually appearing in this DataFrame get a column.
    Players unseen at fit time are handled as cold-start (β = 0) at
    prediction time.
    """
    players: set[tuple[str, str]] = set()

    for pid in matches["hunter_player"].dropna().unique():
        players.add((str(pid), "hunter"))
    for col in SURVIVOR_COLS:
        for pid in matches[col].dropna().unique():
            players.add((str(pid), "survivor"))

    return {player: idx for idx, player in enumerate(sorted(players))}


def build_design_matrix(
    matches: pd.DataFrame,
    player_index: dict[tuple[str, str], int],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Build the sparse N×P design matrix X and target vector y.

    For match i:
        X[i, (hunter, "hunter")]     = +1
        X[i, (survivor_k,"survivor")]= −0.25  for k = 1..4

    Players not in player_index (cold-start) contribute 0 to that match's
    predicted margin — equivalent to assigning them β = 0 (the prior).

    Parameters
    ----------
    matches      : DataFrame with player ID columns and n_escaped.
                   Should be pre-filtered with filter_complete().
    player_index : (player_id, role) → column index, from build_player_index().

    Returns
    -------
    X : csr_matrix of shape (N, P)
    y : ndarray of shape (N,), where y_i = n_escaped_i − 2
    """
    matches = matches.reset_index(drop=True)
    N = len(matches)
    P = len(player_index)

    row_idx, col_idx, values = [], [], []

    # Hunter entries: +1
    for i, pid in enumerate(matches["hunter_player"]):
        key = (str(pid), "hunter")
        if key in player_index:
            row_idx.append(i)
            col_idx.append(player_index[key])
            values.append(_HUNTER_WEIGHT)

    # Survivor entries: −0.25 each
    for col in SURVIVOR_COLS:
        for i, pid in enumerate(matches[col]):
            if pd.notna(pid):
                key = (str(pid), "survivor")
                if key in player_index:
                    row_idx.append(i)
                    col_idx.append(player_index[key])
                    values.append(_SURVIVOR_WEIGHT)

    X = sparse.csr_matrix(
        (values, (row_idx, col_idx)),
        shape=(N, P),
        dtype=np.float64,
    )
    y = (matches["n_escaped"] - 2).to_numpy(dtype=np.float64)
    return X, y


# ---------------------------------------------------------------------------
# Time-decay weights
# ---------------------------------------------------------------------------

def compute_weights(
    dates: pd.Series,
    reference_date: str | pd.Timestamp,
    half_life_days: float | None,
) -> np.ndarray:
    """
    Compute per-match exponential decay weights.

        w_i = 0.5 ^ (Δt_i / τ)

    where Δt_i = (reference_date − date_i) in days and τ = half_life_days.

    Parameters
    ----------
    dates          : Series of ISO date strings (match dates).
    reference_date : The anchor date — typically the last date in the
                     training set.  Matches on this date get weight 1.0.
    half_life_days : τ in days.  Pass None for uniform weights (no decay).

    Returns
    -------
    weights : ndarray of shape (N,), all positive.
    """
    if half_life_days is None:
        return np.ones(len(dates), dtype=np.float64)

    ref = pd.Timestamp(reference_date)
    delta = (ref - pd.to_datetime(dates)).dt.days.to_numpy(dtype=np.float64)
    delta = np.clip(delta, 0.0, None)   # future dates (shouldn't occur) → weight 1
    return np.power(0.5, delta / half_life_days)


# ---------------------------------------------------------------------------
# Core linear algebra
# ---------------------------------------------------------------------------

def fit(
    X: sparse.csr_matrix,
    y: np.ndarray,
    weights: np.ndarray,
    l2_lambda: float = 1.0,
) -> np.ndarray:
    """
    Solve the weighted ridge normal equations:

        β* = (X'WX + λI)⁻¹ X'Wy

    X'WX + λI is symmetric positive-definite, so we use a Cholesky
    decomposition — about 2× faster than general LU and more numerically
    stable.

    Parameters
    ----------
    X         : sparse design matrix (N×P).
    y         : target margins (N,).
    weights   : per-match weights (N,), all positive.
    l2_lambda : ridge penalty λ.  Larger → more shrinkage toward zero.

    Returns
    -------
    beta : ndarray of shape (P,).
    """
    P = X.shape[1]

    # Scale rows of X by sqrt(w_i) so X_w'X_w = X'WX
    sqrt_w  = np.sqrt(weights)
    X_w     = X.multiply(sqrt_w[:, np.newaxis])          # N×P sparse

    XtWX    = (X_w.T @ X_w).toarray()                    # P×P dense
    XtWy    = np.asarray(X.T @ (weights * y)).ravel()    # P,

    # Add ridge penalty to the diagonal
    XtWX   += l2_lambda * np.eye(P)

    # Cholesky solve: faster than linalg.solve for SPD matrices
    c, low  = linalg.cho_factor(XtWX)
    beta    = linalg.cho_solve((c, low), XtWy)
    return beta


def predict(X: sparse.csr_matrix, beta: np.ndarray) -> np.ndarray:
    """
    Compute predicted margins ŷ = Xβ.

    Parameters
    ----------
    X    : sparse design matrix (N×P).
    beta : coefficient vector (P,), from fit().

    Returns
    -------
    y_hat : ndarray of shape (N,).
    """
    return np.asarray(X @ beta).ravel()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def fit_ratings(
    matches: pd.DataFrame,
    half_life_days: float | None = None,
    reference_date: str | pd.Timestamp | None = None,
    l2_lambda: float = 1.0,
) -> tuple[np.ndarray, dict[tuple[str, str], int]]:
    """
    Fit ratings on a single dataset (no train/test split).

    Parameters
    ----------
    matches        : DataFrame from the matches table, sorted by date.
    half_life_days : τ for exponential decay.  None = uniform weights.
    reference_date : Anchor for decay weights.  Defaults to the latest
                     date in matches.
    l2_lambda      : Ridge penalty.

    Returns
    -------
    beta         : ndarray of shape (P,) — one value per (player, role).
    player_index : dict mapping (player_id, role) → position in beta.
    """
    matches = filter_complete(matches)

    if reference_date is None:
        reference_date = matches["date"].max()

    player_index = build_player_index(matches)
    X, y         = build_design_matrix(matches, player_index)
    weights      = compute_weights(matches["date"], reference_date, half_life_days)
    beta         = fit(X, y, weights, l2_lambda)

    return beta, player_index


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

def ratings_df(
    beta: np.ndarray,
    player_index: dict[tuple[str, str], int],
) -> pd.DataFrame:
    """
    Convert (beta, player_index) into a tidy DataFrame with columns:
        player_id, role, beta
    sorted by beta descending within each role.
    """
    rows = [
        {"player_id": pid, "role": role, "beta": float(beta[idx])}
        for (pid, role), idx in player_index.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(["role", "beta"], ascending=[True, False])
        .reset_index(drop=True)
    )


def top_ratings(
    beta: np.ndarray,
    player_index: dict[tuple[str, str], int],
    role: str,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return the top-n players by beta for a given role ('hunter' or 'survivor').
    """
    df = ratings_df(beta, player_index)
    return (
        df[df["role"] == role]
        .head(n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(
    db_path: str = DEFAULT_DB,
    l2_lambda: float = 1.0,
) -> None:
    """
    Fit with uniform weights (no decay) on all data and print the top-10
    hunters and top-10 survivors.  Use this to verify the model is working:
    the top players should be recognisable names.
    """
    conn = sqlite3.connect(db_path)
    matches = pd.read_sql("SELECT * FROM matches ORDER BY date", conn)
    conn.close()

    print("Fitting with no time decay (uniform weights)…")
    beta, player_index = fit_ratings(matches, half_life_days=None, l2_lambda=l2_lambda)

    df = ratings_df(beta, player_index)

    print(f"\nParameters fit: {len(beta)}")
    print(f"Beta range: [{beta.min():.3f}, {beta.max():.3f}]")
    print(f"Beta mean:  {beta.mean():.4f}  (close to 0 = regularisation working)")

    print(f"\n{'─'*40}")
    print("Top 10 HUNTERS  (higher β = stronger hunter)")
    print(f"{'─'*40}")
    print(top_ratings(beta, player_index, "hunter", 10).to_string(index=False))

    print(f"\n{'─'*40}")
    print("Top 10 SURVIVORS  (higher β = stronger survivor)")
    print(f"{'─'*40}")
    print(top_ratings(beta, player_index, "survivor", 10).to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sanity_check()
