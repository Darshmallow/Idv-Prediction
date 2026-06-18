"""
src/optimize.py
===============
Hyperparameter optimisation for the BT margin model via Optuna.

Uses single-rate exponential decay (`compute_weights`).  By default all
tiers are used in both training and test sets, and the full joint model
(hunter + 4 survivors) is fit.

Optimised hyperparameters (any can be FIXED instead — see --fix below):
    half_life_days (τ) : continuous, log-spaced 30 → 1000 days
    l2_lambda          : continuous, log-spaced 0.1 → 10
    threshold          : integer, 0 → 20 (new-player team-mean prior)
    model              : categorical, 'linear' | 'ordinal'

Filters:
    --train-tiers IVL,IJL     restrict training set to these tiers
    --test-tiers  IVL         restrict test set to these tiers
    --tiers       IVL,IJL     shorthand: apply same filter to both
    --role        hunter      use only hunter columns in the design matrix
                              (or 'survivor', or 'both' for the default joint)
    --start-date  2022-01-01  drop training matches before this date
    --end-date    2025-12-31  drop test matches after this date

Fixing hyperparameters:
    --fix half_life_days=120         pin τ; Optuna won't tune it
    --fix l2_lambda=1.0,threshold=5  pin multiple

Run from the project root:
    python src/optimize.py
    python src/optimize.py --trials 200 --model ordinal
    python src/optimize.py --tiers IVL,IJL --role hunter --trials 60
    python src/optimize.py --train-tiers IVL,IJL,COA --test-tiers IVL
    python src/optimize.py --fix half_life_days=120,l2_lambda=1.0

Output:
    outputs/optuna_study.csv      — every trial's params + score
    outputs/optuna_best.json      — the winning configuration
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy import linalg, sparse
from sklearn.model_selection import TimeSeriesSplit

# Make `src/` importable when run as a script
_HERE = Path(__file__).parent
_SRC  = _HERE.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.bradley_terry import (
    SURVIVOR_COLS,
    build_design_matrix,
    build_player_index,
    compute_weights,
    filter_complete,
    fit,
    predict,
)
from model.ordinal import fit_ordinal, predict_expected_margin
from model.team_prior import fit_with_new_player_prior

_ROOT      = _HERE.parent.parent
DB_PATH    = _ROOT / "data" / "processed" / "idv.db"
OUTPUTS    = _ROOT / "outputs"

VALID_TIERS = {"IVL", "IJL", "COA", "IVS", "IVT", "IVC"}


# ---------------------------------------------------------------------------
# Data loading + filtering
# ---------------------------------------------------------------------------

def load_matches(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """Load matches table from SQLite and filter to complete rows."""
    conn = sqlite3.connect(str(db_path))
    matches = pd.read_sql("SELECT * FROM matches ORDER BY date", conn)
    conn.close()
    return filter_complete(matches).sort_values("date").reset_index(drop=True)


def apply_date_filter(
    d: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Drop rows outside [start_date, end_date]; ISO strings, inclusive."""
    mask = pd.Series(True, index=d.index)
    if start_date is not None:
        mask &= d["date"] >= start_date
    if end_date is not None:
        mask &= d["date"] <= end_date
    return d[mask].reset_index(drop=True) if not mask.all() else d


def tier_mask(d: pd.DataFrame, tiers: list[str] | None) -> np.ndarray:
    """Boolean mask True where tournament_tier ∈ tiers.  None ⇒ all-True."""
    if tiers is None:
        return np.ones(len(d), dtype=bool)
    if "tournament_tier" not in d.columns:
        raise ValueError(
            "tier filter requested but matches DataFrame has no "
            "'tournament_tier' column. Did you build the db from JSON sources?"
        )
    return d["tournament_tier"].isin(set(tiers)).to_numpy()


def build_splits(
    d: pd.DataFrame,
    strategy: str = "time_series",
    n_splits: int = 5,
    train_tiers: list[str] | None = None,
    test_tiers: list[str] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build CV splits, then apply train/test tier filters to the indices.

    `d` stays as the single source of truth — both masks are applied to
    indices, not to the DataFrame itself.
    """
    if strategy == "time_series":
        raw = list(TimeSeriesSplit(n_splits=n_splits).split(d))
    elif strategy == "seasons":
        from eval import get_splits
        raw = get_splits(d, strategy="seasons")
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    tr_mask = tier_mask(d, train_tiers)
    te_mask = tier_mask(d, test_tiers)
    return [(tr[tr_mask[tr]], te[te_mask[te]]) for tr, te in raw]


# ---------------------------------------------------------------------------
# Role-restricted design matrices
# ---------------------------------------------------------------------------

def build_role_index(matches: pd.DataFrame, role: str) -> dict[str, int]:
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


def build_role_design(
    matches: pd.DataFrame,
    role: str,
    index: dict[str, int],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Role-restricted sparse design matrix."""
    matches = matches.reset_index(drop=True)
    N, P    = len(matches), len(index)
    rows, cols, vals = [], [], []
    if role == "hunter":
        for i, p in enumerate(matches["hunter_player"]):
            if pd.notna(p) and str(p) in index:
                rows.append(i); cols.append(index[str(p)]); vals.append(-1.0)
    else:
        for col in SURVIVOR_COLS:
            for i, p in enumerate(matches[col]):
                if pd.notna(p) and str(p) in index:
                    rows.append(i); cols.append(index[str(p)]); vals.append(0.25)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(N, P), dtype=np.float64)
    y = (matches["n_escaped"] - 2).to_numpy(dtype=np.float64)
    return X, y


# ---------------------------------------------------------------------------
# CV scoring
# ---------------------------------------------------------------------------

def _eval_fold_full(
    train: pd.DataFrame,
    test: pd.DataFrame,
    weights: np.ndarray,
    l2_lambda: float,
    threshold: int,
    model: str,
) -> tuple[float, float]:
    """Standard full joint model fit + predict; returns (rmse, null_rmse)."""
    result = fit_with_new_player_prior(
        train, model=model, l2_lambda=l2_lambda,
        threshold=threshold, weights=weights,
    )
    if model == "ordinal":
        _, _, res = result
        yhat = predict_expected_margin(test, res)
    else:
        beta, idx = result
        X_te, _   = build_design_matrix(test, idx)
        yhat      = predict(X_te, beta)
    y    = (test["n_escaped"] - 2).to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    null = float(np.sqrt(np.mean(y ** 2)))
    return rmse, null


def _eval_fold_role(
    train: pd.DataFrame,
    test: pd.DataFrame,
    role: str,
    weights: np.ndarray,
    l2_lambda: float,
    threshold: int = 0,
) -> tuple[float, float]:
    """
    Role-restricted ordinal fit with θ fixed to the combined model's thresholds.

    Fits the combined ordinal model first (cheaply, same training data) to
    obtain θ, then optimises only the role-restricted β with θ held constant.
    This lets role-only analysis use the ordinal link function without
    learning separate thresholds per role.
    """
    from model.ordinal import fit_ordinal, predict_expected_margin
    from scipy.special import expit

    # Step 1: fit combined model to get θ
    combined = fit_with_new_player_prior(
        train, model="ordinal", l2_lambda=l2_lambda,
        threshold=threshold, weights=weights,
    )
    _, _, combined_res = combined
    theta = combined_res["theta"]

    # Step 2: role-only β with fixed θ
    idx        = build_role_index(train, role)
    X_tr, y_tr = build_role_design(train, role, idx)
    n_obs_tr   = (train.reset_index(drop=True)["n_escaped"]).to_numpy(dtype=int)

    res = fit_ordinal(
        train,
        weights=weights,
        l2_lambda=l2_lambda,
        fixed_theta=theta,
    )
    # Override player_index with the role-restricted one and refit β only
    # via the design matrix (fit_ordinal uses the full joint index — we need
    # role-restricted instead)
    from model.ordinal import _objective_and_grad
    from scipy.optimize import minimize as _minimize

    n_obs_tr = train.reset_index(drop=True)["n_escaped"].to_numpy(dtype=int)
    beta0    = np.zeros(X_tr.shape[1])
    result   = _minimize(
        _objective_and_grad,
        beta0,
        args=(X_tr, n_obs_tr, weights, X_tr.shape[1], l2_lambda, None, theta),
        method="L-BFGS-B",
        jac=True,
        options=dict(maxiter=1000, ftol=1e-9, gtol=1e-7),
    )
    beta_hat = result.x

    # Predict on test using role-restricted β + fixed θ
    X_te, y_te = build_role_design(test, role, idx)
    eta_te     = np.asarray(X_te @ beta_hat).ravel()
    u          = expit(theta[None, :] - eta_te[:, None])
    probs      = np.column_stack([
        u[:, 0],
        u[:, 1] - u[:, 0],
        u[:, 2] - u[:, 1],
        u[:, 3] - u[:, 2],
        1.0 - u[:, 3],
    ])
    probs  = np.clip(probs, 0.0, 1.0)
    yhat   = probs @ np.arange(5) - 2.0

    rmse = float(np.sqrt(np.mean((y_te - yhat) ** 2)))
    null = float(np.sqrt(np.mean(y_te ** 2)))
    return rmse, null


def evaluate(
    d: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    half_life_days: float,
    l2_lambda: float,
    threshold: int,
    model: str = "ordinal",
    role: str = "both",
) -> float:
    """Run temporal CV and return pooled R²."""
    rmses, nulls = [], []
    for tr, te in splits:
        train, test = d.iloc[tr], d.iloc[te]
        if len(test) == 0 or len(train) == 0:
            continue

        w = compute_weights(
            train["date"], train["date"].max(),
            half_life_days=half_life_days,
        )

        if role == "both":
            rmse, null = _eval_fold_full(train, test, w, l2_lambda, threshold, model)
        else:
            rmse, null = _eval_fold_role(train, test, role, w, l2_lambda, threshold)

        rmses.append(rmse)
        nulls.append(null)

    if not rmses:
        return float("nan")
    return 1.0 - float(np.mean([r ** 2 for r in rmses])) / float(np.mean([n ** 2 for n in nulls]))


# ---------------------------------------------------------------------------
# Optuna objective with fixed-parameter support
# ---------------------------------------------------------------------------

def make_objective(
    d: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    fixed_model: str | None = None,
    role: str = "both",
    fixed_params: dict | None = None,
):
    """
    Build the Optuna objective.  Any hyperparameter in `fixed_params` is
    not searched — Optuna uses the fixed value instead.
    """
    fixed_params = fixed_params or {}

    def _get(trial, name, suggest_fn):
        """Use fixed value if provided, otherwise let Optuna suggest."""
        if name in fixed_params:
            trial.set_user_attr(f"fixed_{name}", fixed_params[name])
            return fixed_params[name]
        return suggest_fn(trial)

    def objective(trial: optuna.trial.Trial) -> float:
        tau       = _get(trial, "half_life_days",
                         lambda t: t.suggest_float("half_life_days", 30.0, 1000.0, log=True))
        l2        = _get(trial, "l2_lambda",
                         lambda t: t.suggest_float("l2_lambda", 0.1, 10.0, log=True))
        threshold = _get(trial, "threshold",
                         lambda t: t.suggest_int("threshold", 0, 20))

        if role != "both":
            # Role-only path uses ordinal with fixed θ from combined model.
            model = "ordinal"
        elif fixed_model is not None:
            model = fixed_model
        elif "model" in fixed_params:
            model = fixed_params["model"]
        else:
            model = trial.suggest_categorical("model", ["linear", "ordinal"])

        r2 = evaluate(
            d, splits,
            half_life_days=tau, l2_lambda=l2,
            threshold=int(threshold), model=model, role=role,
        )
        return -r2

    return objective


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    n_trials: int = 80,
    fixed_model: str | None = None,
    fixed_params: dict | None = None,
    split_strategy: str = "time_series",
    n_splits: int = 5,
    train_tiers: list[str] | None = None,
    test_tiers: list[str] | None = None,
    role: str = "both",
    start_date: str | None = None,
    end_date: str | None = None,
    seed: int = 42,
    db_path: str | Path = DB_PATH,
    save_dir: str | Path = OUTPUTS,
    out_prefix: str = "optuna",
) -> optuna.study.Study:
    """
    Run the Optuna search end-to-end.

    See module docstring for filter / fix semantics.
    """
    fixed_params = fixed_params or {}

    print(f"Loading matches from {db_path} …")
    d = load_matches(db_path)
    print(f"  {len(d):,} complete matches")
    if start_date or end_date:
        d = apply_date_filter(d, start_date, end_date)
        print(f"  after date filter [{start_date or '...'} → {end_date or '...'}]: {len(d):,}")

    splits = build_splits(d, strategy=split_strategy, n_splits=n_splits,
                          train_tiers=train_tiers, test_tiers=test_tiers)
    n_train_total = sum(len(tr) for tr, _ in splits)
    n_test_total  = sum(len(te) for _, te in splits)
    print(f"  splits: {len(splits)} ({split_strategy})  "
          f"train rows total={n_train_total:,}  test rows total={n_test_total:,}")
    print(f"  train tiers: {train_tiers or 'all'}    test tiers: {test_tiers or 'all'}")
    print(f"  role:        {role}")
    if fixed_params:
        print(f"  fixed: {fixed_params}")

    objective = make_objective(d, splits, fixed_model=fixed_model,
                               role=role, fixed_params=fixed_params)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction="minimize", sampler=sampler)

    print(f"\nStarting search — {n_trials} trials, "
          f"fixed_model={fixed_model or 'auto'}\n")

    best_so_far = float("inf")
    t0 = time.time()

    def callback(study_, trial):
        nonlocal best_so_far
        if trial.value is not None and trial.value < best_so_far:
            best_so_far = trial.value
            elapsed = time.time() - t0
            print(f"  [{trial.number:3d}/{n_trials}]  R²={-trial.value*100:+.2f}%  "
                  f"params={trial.params}  ({elapsed:.0f}s)")

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    elapsed = time.time() - t0
    print(f"\nFinished — {len(study.trials)} trials in {elapsed/60:.1f} min")
    print(f"\nBest R²:    {-study.best_value*100:.3f}%")
    print(f"Best params: {study.best_params}")
    if fixed_params:
        print(f"Fixed:       {fixed_params}")

    # Save
    save_dir = Path(save_dir); save_dir.mkdir(exist_ok=True)
    df = study.trials_dataframe()
    df["r2"] = -df["value"]
    df.to_csv(save_dir / f"{out_prefix}_study.csv", index=False)
    with open(save_dir / f"{out_prefix}_best.json", "w") as f:
        json.dump({
            "best_r2":      -study.best_value,
            "best_params":  study.best_params,
            "fixed_params": fixed_params,
            "n_trials":     len(study.trials),
            "elapsed_s":    elapsed,
            "split_strategy": split_strategy,
            "fixed_model":    fixed_model,
            "train_tiers":    train_tiers,
            "test_tiers":     test_tiers,
            "role":           role,
            "start_date":     start_date,
            "end_date":       end_date,
        }, f, indent=2)
    print(f"\n→ {save_dir / f'{out_prefix}_study.csv'}")
    print(f"→ {save_dir / f'{out_prefix}_best.json'}")
    return study


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_csv(s: str | None) -> list[str] | None:
    """Parse 'IVL,IJL' → ['IVL', 'IJL'].  Empty / None returns None."""
    if not s:
        return None
    out = [x.strip() for x in s.split(",") if x.strip()]
    unknown = [t for t in out if t not in VALID_TIERS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown tier(s): {unknown}. Valid: {sorted(VALID_TIERS)}")
    return out


def _parse_fix(s: str | None) -> dict:
    """Parse 'half_life_days=120,l2_lambda=1.0' → {'half_life_days': 120.0, ...}."""
    if not s:
        return {}
    out: dict = {}
    for kv in s.split(","):
        if "=" not in kv:
            raise argparse.ArgumentTypeError(
                f"--fix entry must be NAME=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        k = k.strip(); v = v.strip()
        if k in {"half_life_days", "l2_lambda"}:
            out[k] = float(v)
        elif k == "threshold":
            out[k] = int(v)
        elif k == "model":
            if v not in {"linear", "ordinal"}:
                raise argparse.ArgumentTypeError(
                    f"model must be 'linear' or 'ordinal', got {v!r}")
            out[k] = v
        else:
            raise argparse.ArgumentTypeError(
                f"unknown fix-able parameter {k!r}. "
                f"Allowed: half_life_days, l2_lambda, threshold, model")
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--trials",   type=int, default=80,
                   help="Number of Optuna trials (default 80)")
    p.add_argument("--model",    choices=["linear", "ordinal", "auto"],
                   default="ordinal",
                   help="Fix model class, or 'auto' to let Optuna pick (default ordinal)")
    p.add_argument("--splits",   choices=["time_series", "seasons"],
                   default="time_series",
                   help="CV split strategy (default time_series)")
    p.add_argument("--n-splits", type=int, default=5,
                   help="Folds for time_series (default 5)")

    p.add_argument("--train-tiers", type=_parse_csv, default=None,
                   help=f"Comma-separated training-set tiers, e.g. 'IVL,IJL'. "
                        f"Valid: {sorted(VALID_TIERS)}")
    p.add_argument("--test-tiers", type=_parse_csv, default=None,
                   help="Comma-separated test-set tiers (default: all)")
    p.add_argument("--tiers", type=_parse_csv, default=None,
                   help="Shorthand: apply same tier filter to both train and test")

    p.add_argument("--role", choices=["both", "hunter", "survivor"], default="both",
                   help="Use only one role's columns in the design matrix. "
                        "Forces model=linear when not 'both'. (default both)")

    p.add_argument("--start-date", default=None,
                   help="Drop matches with date < this (ISO 'YYYY-MM-DD')")
    p.add_argument("--end-date",   default=None,
                   help="Drop matches with date > this (ISO 'YYYY-MM-DD')")

    p.add_argument("--fix", type=_parse_fix, default={},
                   help="Pin hyperparameters Optuna won't tune. "
                        "Format: NAME=VALUE,NAME=VALUE. Allowed names: "
                        "half_life_days, l2_lambda, threshold, model. "
                        "Example: --fix half_life_days=120,l2_lambda=1.0")

    p.add_argument("--out-prefix", default="optuna",
                   help="Prefix for output filenames (default 'optuna'). "
                        "Useful when running multiple configs side-by-side.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for the TPE sampler (default 42)")
    args = p.parse_args()

    # Resolve tiers — --tiers is shorthand for both
    if args.tiers is not None:
        if args.train_tiers or args.test_tiers:
            p.error("--tiers cannot be combined with --train-tiers or --test-tiers")
        train_tiers = test_tiers = args.tiers
    else:
        train_tiers = args.train_tiers
        test_tiers  = args.test_tiers

    fixed_model = None if args.model == "auto" else args.model
    # If --fix specifies model, CLI --model becomes redundant — we honour --fix
    if "model" in args.fix:
        fixed_model = args.fix["model"]

    run(
        n_trials=args.trials,
        fixed_model=fixed_model,
        fixed_params={k: v for k, v in args.fix.items() if k != "model"},
        split_strategy=args.splits,
        n_splits=args.n_splits,
        train_tiers=train_tiers,
        test_tiers=test_tiers,
        role=args.role,
        start_date=args.start_date,
        end_date=args.end_date,
        seed=args.seed,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()
