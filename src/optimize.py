"""
src/optimize.py
===============
Hyperparameter optimisation for the BT margin model via Optuna.

Uses single-rate exponential decay (`compute_weights`) — regime decay
has been retired in favour of simpler training.  All tiers are used for
both training and testing (no tier filtering applied here).

Optimised hyperparameters (configurable):
    half_life_days (τ) : continuous, log-spaced 30 → 1000 days
    l2_lambda          : continuous, log-spaced 0.1 → 10
    threshold          : integer, 0 → 20 (new-player team-mean prior)
    model              : categorical, 'linear' | 'ordinal'

Run from the project root:
    python src/optimize.py                              # default: 80 trials
    python src/optimize.py --trials 200 --model ordinal # ordinal only, 200 trials
    python src/optimize.py --splits seasons             # use season-aligned CV

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
from sklearn.model_selection import TimeSeriesSplit

# Make `src/` importable when run as a script
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from bradley_terry import (
    build_design_matrix,
    compute_weights,
    filter_complete,
    predict,
)
from ordinal import predict_expected_margin
from team_prior import fit_with_new_player_prior

_ROOT      = _HERE.parent
DB_PATH    = _ROOT / "data" / "processed" / "idv.db"
OUTPUTS    = _ROOT / "outputs"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_matches(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    """Load matches table from SQLite and filter to complete rows."""
    conn = sqlite3.connect(str(db_path))
    matches = pd.read_sql("SELECT * FROM matches ORDER BY date", conn)
    conn.close()
    return filter_complete(matches).sort_values("date").reset_index(drop=True)


def build_splits(
    d: pd.DataFrame,
    strategy: str = "time_series",
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build CV splits.  All tiers in both train and test (no filtering)."""
    if strategy == "time_series":
        return list(TimeSeriesSplit(n_splits=n_splits).split(d))
    if strategy == "seasons":
        from eval import get_splits
        return get_splits(d, strategy="seasons")
    raise ValueError(f"Unknown strategy {strategy!r}")


# ---------------------------------------------------------------------------
# CV scoring function (the objective)
# ---------------------------------------------------------------------------

def evaluate(
    d: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    half_life_days: float,
    l2_lambda: float,
    threshold: int,
    model: str = "ordinal",
) -> float:
    """
    Run k-fold temporal CV with single-rate decay.  Returns pooled R².

    For each fold:
      1. weights = 0.5^(Δt / half_life_days)
      2. fit with new-player team-mean prior at the given threshold
      3. predict on the test set
      4. accumulate RMSE and null RMSE
    """
    rmses, nulls = [], []
    for tr, te in splits:
        train, test = d.iloc[tr], d.iloc[te]
        if len(test) == 0 or len(train) == 0:
            continue

        w = compute_weights(
            train["date"], train["date"].max(),
            half_life_days=half_life_days,
        )

        result = fit_with_new_player_prior(
            train, model=model, l2_lambda=l2_lambda,
            threshold=threshold, weights=w,
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
        rmses.append(rmse)
        nulls.append(null)

    if not rmses:
        return float("nan")
    return 1.0 - float(np.mean([r ** 2 for r in rmses])) / float(np.mean([n ** 2 for n in nulls]))


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(
    d: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    fixed_model: str | None = None,
):
    """Return an Optuna objective closure with the data baked in."""
    def objective(trial: optuna.trial.Trial) -> float:
        tau       = trial.suggest_float("half_life_days", 30.0, 1000.0, log=True)
        l2        = trial.suggest_float("l2_lambda",       0.1,   10.0, log=True)
        threshold = trial.suggest_int  ("threshold", 0, 20)
        model     = fixed_model or trial.suggest_categorical("model", ["linear", "ordinal"])

        r2 = evaluate(
            d, splits,
            half_life_days=tau, l2_lambda=l2,
            threshold=threshold, model=model,
        )
        # Optuna minimises by default — return negative R² so it MAXIMISES R²
        return -r2

    return objective


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    n_trials: int = 80,
    fixed_model: str | None = None,
    split_strategy: str = "time_series",
    n_splits: int = 5,
    seed: int = 42,
    db_path: str | Path = DB_PATH,
    save_dir: str | Path = OUTPUTS,
) -> optuna.study.Study:
    """
    Run the Optuna search end-to-end.

    Parameters
    ----------
    n_trials       : number of Optuna trials (50-100 is usually enough)
    fixed_model    : 'linear' or 'ordinal' to fix the model class.
                     None lets Optuna pick (mixes both — but linear is fast,
                     ordinal is slow, so the search budget can be dominated
                     by ordinal trials if you let it choose freely)
    split_strategy : 'time_series' (default) or 'seasons'
    n_splits       : only used for 'time_series'
    seed           : RNG seed for the TPE sampler
    """
    print(f"Loading matches from {db_path} …")
    d = load_matches(db_path)
    print(f"  {len(d):,} complete matches  ({d['date'].min()} → {d['date'].max()})")

    splits = build_splits(d, strategy=split_strategy, n_splits=n_splits)
    print(f"  {len(splits)} CV splits ({split_strategy})")

    objective = make_objective(d, splits, fixed_model=fixed_model)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(direction="minimize", sampler=sampler)

    print(f"\nStarting search — {n_trials} trials, "
          f"fixed_model={fixed_model or 'auto'}, "
          f"split_strategy={split_strategy}\n")

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

    # Save
    save_dir = Path(save_dir); save_dir.mkdir(exist_ok=True)
    df = study.trials_dataframe()
    df["r2"] = -df["value"]
    df.to_csv(save_dir / "optuna_study.csv", index=False)
    with open(save_dir / "optuna_best.json", "w") as f:
        json.dump({
            "best_r2":    -study.best_value,
            "best_params": study.best_params,
            "n_trials":   len(study.trials),
            "elapsed_s":  elapsed,
            "split_strategy": split_strategy,
            "fixed_model":    fixed_model,
        }, f, indent=2)
    print(f"\n→ {save_dir / 'optuna_study.csv'}")
    print(f"→ {save_dir / 'optuna_best.json'}")
    return study


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument("--trials",   type=int, default=80,
                   help="Number of Optuna trials (default 80)")
    p.add_argument("--model",    choices=["linear", "ordinal", "auto"],
                   default="ordinal",
                   help="Fix model class, or 'auto' to let Optuna pick (default ordinal)")
    p.add_argument("--splits",   choices=["time_series", "seasons"],
                   default="time_series",
                   help="CV split strategy (default time_series)")
    p.add_argument("--n-splits", type=int, default=5,
                   help="Number of folds for time_series (default 5)")
    p.add_argument("--seed",     type=int, default=42,
                   help="RNG seed for the TPE sampler (default 42)")
    args = p.parse_args()

    fixed_model = None if args.model == "auto" else args.model
    run(
        n_trials=args.trials,
        fixed_model=fixed_model,
        split_strategy=args.splits,
        n_splits=args.n_splits,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
