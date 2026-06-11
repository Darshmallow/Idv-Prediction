"""
src/stretch.py
==============
Phase 4 stretch analyses for the IDV skill-decay project.

Three self-contained extensions of the baseline half-life sweep, each
answering a sharper sub-question than the overall headline:

1. **Role asymmetry** — separate sweeps for hunter and survivor skill.
   Hypothesis: hunter half-life is shorter because hunter skill is tied
   to character mechanics and characters get patched.

2. **Tier comparison** — sweep IVL, IJL, and COA tournaments separately.
   Question: does competitive intensity affect skill stability?

3. **Temporal stability** — sweep pre-2023 vs post-2023 matches.
   Question: has the meta sped up over time?

For role asymmetry we use a role-restricted design matrix (only the
hunter columns, or only the survivor columns) so the half-life captures
"information persistence about *that role's* skill."  For the tier and
temporal analyses the baseline sweep machinery in `eval.py` is reused on
filtered data.

Public API
----------
    role_asymmetry_sweep(matches, half_lives, n_splits, l2_lambda)
        → long-form DataFrame (one row per role × half-life × fold)

    tier_comparison_sweep(matches, half_lives, n_splits, l2_lambda)
        → long-form DataFrame (one row per tier × half-life × fold)

    temporal_stability_sweep(matches, half_lives, n_splits, l2_lambda)
        → long-form DataFrame (one row per era × half-life × fold)

    plot_multi_curve(results, group_col, title, save_path)
        → dict[label → best-row]

    run_all(db_path)
        → run all three, save CSVs + plots to outputs/
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
from scipy import sparse
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.bradley_terry import (
    SURVIVOR_COLS,
    compute_weights,
    filter_complete,
    fit,
    predict,
)
from outputs.eval import default_half_lives, sweep_half_lives

_ROOT      = Path(__file__).parent.parent.parent
DEFAULT_DB = str(_ROOT / "data" / "processed" / "idv.db")
OUTPUTS    = _ROOT / "outputs"


# ===========================================================================
# Role-restricted building blocks
# ===========================================================================

def _build_role_index(matches: pd.DataFrame, role: str) -> dict[str, int]:
    """Player → column-index map, restricted to one role."""
    players: set[str] = set()
    if role == "hunter":
        for p in matches["hunter_player"].dropna().unique():
            players.add(str(p))
    elif role == "survivor":
        for col in SURVIVOR_COLS:
            for p in matches[col].dropna().unique():
                players.add(str(p))
    else:
        raise ValueError(f"role must be 'hunter' or 'survivor', got {role!r}")
    return {p: i for i, p in enumerate(sorted(players))}


def _build_role_design_matrix(
    matches: pd.DataFrame,
    role: str,
    index: dict[str, int],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Role-restricted sparse design matrix.

        hunter-only:    X[i, h_i]     = -1
        survivor-only:  X[i, s_{i,k}] = +0.25  for k = 1..4

    Players not in `index` (cold-start, including all of the OTHER role
    by construction) contribute 0 — the prediction collapses to the
    chosen role's contribution only.
    """
    matches = matches.reset_index(drop=True)
    N, P    = len(matches), len(index)
    rows, cols, vals = [], [], []

    if role == "hunter":
        for i, p in enumerate(matches["hunter_player"]):
            if pd.notna(p):
                pid = str(p)
                if pid in index:
                    rows.append(i); cols.append(index[pid]); vals.append(-1.0)
    else:  # survivor
        for col in SURVIVOR_COLS:
            for i, p in enumerate(matches[col]):
                if pd.notna(p):
                    pid = str(p)
                    if pid in index:
                        rows.append(i); cols.append(index[pid]); vals.append(0.25)

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(N, P), dtype=np.float64)
    y = (matches["n_escaped"] - 2).to_numpy(dtype=np.float64)
    return X, y


def _temporal_cv_role(
    matches: pd.DataFrame,
    role: str,
    half_life_days: float | None,
    n_splits: int = 5,
    l2_lambda: float = 1.0,
) -> pd.DataFrame:
    """Role-restricted temporal CV — single half-life value."""
    matches = filter_complete(matches).sort_values("date").reset_index(drop=True)
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    rows    = []

    for fold, (tr, te) in enumerate(tscv.split(matches), start=1):
        train, test = matches.iloc[tr], matches.iloc[te]
        ref         = train["date"].max()
        index       = _build_role_index(train, role)

        X_tr, y_tr = _build_role_design_matrix(train, role, index)
        w          = compute_weights(train["date"], ref, half_life_days)
        beta       = fit(X_tr, y_tr, w, l2_lambda)

        X_te, y_te = _build_role_design_matrix(test, role, index)
        y_hat      = predict(X_te, beta)

        rmse = float(np.sqrt(np.mean((y_te - y_hat) ** 2)))
        rows.append({
            "role":           role,
            "half_life_days": half_life_days if half_life_days is not None else np.inf,
            "fold":           fold,
            "train_size":     len(train),
            "test_size":      len(test),
            "rmse":           rmse,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Three analyses
# ===========================================================================

def role_asymmetry_sweep(
    matches: pd.DataFrame,
    half_lives: np.ndarray | None = None,
    n_splits: int = 5,
    l2_lambda: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Sweep half-lives for hunter-only and survivor-only models.

    Each model is misspecified relative to the joint baseline (the other
    role's variance becomes noise), but the *shape* of RMSE-vs-τ isolates
    how quickly that role's information becomes stale.
    """
    if half_lives is None:
        half_lives = default_half_lives()

    all_rows: list[pd.DataFrame] = []
    for role in ("hunter", "survivor"):
        if verbose:
            print(f"\n  [{role}-only sweep]")
        for tau in half_lives:
            tau_val = None if np.isinf(tau) else float(tau)
            cv = _temporal_cv_role(matches, role, tau_val, n_splits, l2_lambda)
            cv["half_life_days"] = tau
            all_rows.append(cv)
            if verbose:
                label = "∞" if np.isinf(tau) else f"{tau:>4.0f}d"
                print(f"    τ={label}:  mean RMSE = {cv['rmse'].mean():.4f}")
    return pd.concat(all_rows, ignore_index=True)


def tier_comparison_sweep(
    matches: pd.DataFrame,
    half_lives: np.ndarray | None = None,
    n_splits: int = 5,
    l2_lambda: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the standard baseline sweep separately on each competitive tier.
    Uses eval.sweep_half_lives on filtered subsets.
    """
    if half_lives is None:
        half_lives = default_half_lives()

    tiers = {
        "IVL": matches[matches["tournament"].str.startswith("IVL")].copy(),
        "IJL": matches[matches["tournament"].str.startswith("IJL")].copy(),
        "COA": matches[matches["tournament"].str.startswith("COA")].copy(),
    }

    all_rows: list[pd.DataFrame] = []
    for tier, sub in tiers.items():
        if verbose:
            print(f"\n  [{tier} sweep, {len(sub)} matches]")
        if len(sub) < 100:
            print(f"    skipping — only {len(sub)} matches")
            continue
        res = sweep_half_lives(sub, half_lives, n_splits, l2_lambda, verbose=False)
        res["tier"] = tier
        all_rows.append(res)
        if verbose:
            agg  = res.groupby("half_life_days")["rmse"].mean()
            best = agg.idxmin()
            print(f"    optimal τ ≈ {best:.0f}d  (RMSE={agg.min():.4f})")
    return pd.concat(all_rows, ignore_index=True)


def temporal_stability_sweep(
    matches: pd.DataFrame,
    half_lives: np.ndarray | None = None,
    n_splits: int = 5,
    l2_lambda: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Sweep half-lives separately on pre-2023 and post-2023 matches.
    A shorter post-2023 optimum would suggest the meta has sped up.
    """
    if half_lives is None:
        half_lives = default_half_lives()

    eras = {
        "early (2020-2022)": matches[matches["date"] < "2023-01-01"].copy(),
        "late (2023+)":       matches[matches["date"] >= "2023-01-01"].copy(),
    }

    all_rows: list[pd.DataFrame] = []
    for era, sub in eras.items():
        if verbose:
            print(f"\n  [{era}, {len(sub)} matches]")
        res = sweep_half_lives(sub, half_lives, n_splits, l2_lambda, verbose=False)
        res["era"] = era
        all_rows.append(res)
        if verbose:
            agg  = res.groupby("half_life_days")["rmse"].mean()
            best = agg.idxmin()
            print(f"    optimal τ ≈ {best:.0f}d  (RMSE={agg.min():.4f})")
    return pd.concat(all_rows, ignore_index=True)


# ===========================================================================
# Plotting
# ===========================================================================

def plot_multi_curve(
    results: pd.DataFrame,
    group_col: str,
    title: str,
    save_path: str,
    palette: list | None = None,
) -> dict[str, pd.Series]:
    """
    Plot one mean-RMSE-vs-τ curve per group with ±1 SE error bars and a
    dashed line at each group's optimal half-life.  Returns the optimum
    row per group.
    """
    if palette is None:
        palette = list(plt.cm.tab10.colors)

    fig, ax = plt.subplots(figsize=(10, 6))
    summaries: dict[str, pd.Series] = {}

    groups = list(results.groupby(group_col))
    for i, (label, sub) in enumerate(groups):
        agg = (
            sub.groupby("half_life_days")
               .agg(mean_rmse=("rmse", "mean"),
                    se_rmse=("rmse", lambda x: x.std() / np.sqrt(len(x))))
               .reset_index()
               .sort_values("half_life_days")
        )
        finite = agg[np.isfinite(agg["half_life_days"])]
        if len(finite) == 0:
            continue

        color = palette[i % len(palette)]
        ax.errorbar(
            finite["half_life_days"], finite["mean_rmse"],
            yerr=finite["se_rmse"], fmt="o-",
            capsize=3, color=color, label=str(label),
            linewidth=1.7, markersize=5,
        )
        best = finite.loc[finite["mean_rmse"].idxmin()]
        ax.axvline(best["half_life_days"], color=color, linestyle=":", alpha=0.4)
        summaries[str(label)] = best

    ax.set_xscale("log")
    ax.set_xlabel("Half-life τ (days)", fontsize=11)
    ax.set_ylabel("OOS RMSE (margin)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    secax = ax.secondary_xaxis(
        "top", functions=(lambda d: d / 30.44, lambda m: m * 30.44),
    )
    secax.set_xlabel("Half-life (months)", fontsize=9)

    plt.tight_layout()
    OUTPUTS.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return summaries


# ===========================================================================
# Entry point — run all three
# ===========================================================================

def run_all(db_path: str = DEFAULT_DB) -> None:
    """Run all three stretch analyses end-to-end."""
    conn    = sqlite3.connect(db_path)
    matches = pd.read_sql("SELECT * FROM matches ORDER BY date", conn)
    conn.close()
    print(f"Loaded {len(matches)} matches\n")

    # --- 1. Role asymmetry ---
    print("=" * 60)
    print("1. ROLE ASYMMETRY (hunter vs survivor information decay)")
    print("=" * 60)
    role_results = role_asymmetry_sweep(matches)
    role_results.to_csv(OUTPUTS / "stretch_role_asymmetry.csv", index=False)
    role_summary = plot_multi_curve(
        role_results, "role",
        "Skill decay half-life — hunter vs survivor",
        str(OUTPUTS / "stretch_role_asymmetry.png"),
    )
    print(f"\n  → outputs/stretch_role_asymmetry.png")
    for role, row in role_summary.items():
        print(f"    {role}: optimal τ ≈ {row['half_life_days']:.0f} days  "
              f"({row['half_life_days']/30.44:.1f} months)")

    # --- 2. Tier comparison ---
    print("\n" + "=" * 60)
    print("2. TIER COMPARISON (IVL vs IJL vs COA)")
    print("=" * 60)
    tier_results = tier_comparison_sweep(matches)
    tier_results.to_csv(OUTPUTS / "stretch_tier_comparison.csv", index=False)
    tier_summary = plot_multi_curve(
        tier_results, "tier",
        "Skill decay half-life by competitive tier",
        str(OUTPUTS / "stretch_tier_comparison.png"),
    )
    print(f"\n  → outputs/stretch_tier_comparison.png")
    for tier, row in tier_summary.items():
        print(f"    {tier}: optimal τ ≈ {row['half_life_days']:.0f} days  "
              f"({row['half_life_days']/30.44:.1f} months)")

    # --- 3. Temporal stability ---
    print("\n" + "=" * 60)
    print("3. TEMPORAL STABILITY (pre-2023 vs post-2023)")
    print("=" * 60)
    temp_results = temporal_stability_sweep(matches)
    temp_results.to_csv(OUTPUTS / "stretch_temporal_stability.csv", index=False)
    temp_summary = plot_multi_curve(
        temp_results, "era",
        "Skill decay half-life by era",
        str(OUTPUTS / "stretch_temporal_stability.png"),
    )
    print(f"\n  → outputs/stretch_temporal_stability.png")
    for era, row in temp_summary.items():
        print(f"    {era}: optimal τ ≈ {row['half_life_days']:.0f} days  "
              f"({row['half_life_days']/30.44:.1f} months)")


if __name__ == "__main__":
    run_all()
