"""
src/outputs/metrics.py
======================
Accuracy metrics for the ordinal BT model under temporal CV.

Two metrics:
  1. Exact accuracy — round predicted expected margin to nearest integer,
     compare to actual n_escaped.  5-class problem: {0,1,2,3,4}.
  2. Win/draw/loss accuracy — collapse outcomes to 3 categories:
       hunter win  : n_escaped ∈ {0, 1}  (margin < 0)
       draw        : n_escaped = 2        (margin = 0)
       survivor win: n_escaped ∈ {3, 4}  (margin > 0)
     Predict the most probable category from the model's P(n=k).

For each metric, null baselines are computed (predict most-frequent class).

Usage
-----
    python src/outputs/metrics.py                         # all tiers
    python src/outputs/metrics.py --config top_tiers      # IVL/IJL/COA only
    python src/outputs/metrics.py --all                   # both configs
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_DB = str(_ROOT / "data" / "processed" / "idv.db")
OUTPUTS    = _ROOT / "outputs"
OPTUNA_DIR = _ROOT / "outputs" / "optuna"

from model.bradley_terry import filter_complete, compute_weights
from model.ordinal import fit_ordinal, predict_probs, predict_expected_margin
from model.team_prior import identify_new_players, compute_team_mean_priors

CONFIGS = {
    "top_tiers": {
        "json":   OPTUNA_DIR / "top_tiers_best.json",
        "tiers":  ["IVL", "IJL", "COA"],
        "suffix": "top_tiers",
    },
    "all_tiers": {
        "json":   OPTUNA_DIR / "ordinal_best_best.json",
        "tiers":  None,
        "suffix": "all_tiers",
    },
}


# ---------------------------------------------------------------------------
# CV prediction collector
# ---------------------------------------------------------------------------

def _fit_with_prior(train, ref, tau, l2, threshold):
    from model.bradley_terry import filter_complete as fc
    m = fc(train)
    w = compute_weights(m["date"], ref, tau)
    res0 = fit_ordinal(m, l2_lambda=l2, weights=w)
    new_keys = identify_new_players(m, res0["player_index"], threshold)
    mu = compute_team_mean_priors(m, res0["player_index"], res0["beta"], new_keys)
    return fit_ordinal(m, l2_lambda=l2, weights=w,
                       init_theta=res0["theta"], prior_mean=mu)


def collect_predictions(
    matches: pd.DataFrame,
    tau: float,
    l2: float,
    threshold: int,
    tiers: list[str] | None = None,
    n_splits: int = 5,
) -> pd.DataFrame:
    if tiers:
        matches = matches[matches["tournament_tier"].isin(tiers)].copy()
    matches = filter_complete(matches).sort_values("date").reset_index(drop=True)
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    rows    = []

    for fold, (tr, te) in enumerate(tscv.split(matches), start=1):
        train, test = matches.iloc[tr], matches.iloc[te]
        ref = train["date"].max()
        print(f"  fold {fold}: train={len(train)}, test={len(test)}", flush=True)

        res        = _fit_with_prior(train, ref, tau, l2, threshold)
        probs      = predict_probs(test, res)
        exp_margin = predict_expected_margin(test, res)
        actual_n   = test["n_escaped"].to_numpy(dtype=int)

        for i in range(len(test)):
            rows.append({
                "fold":       fold,
                "actual_n":   actual_n[i],
                "exp_margin": float(exp_margin[i]),
                "p0": float(probs[i, 0]),
                "p1": float(probs[i, 1]),
                "p2": float(probs[i, 2]),
                "p3": float(probs[i, 3]),
                "p4": float(probs[i, 4]),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    actual_n   = df["actual_n"].to_numpy()
    probs      = df[["p0", "p1", "p2", "p3", "p4"]].to_numpy()
    exp_margin = df["exp_margin"].to_numpy()

    pred_argmax   = np.argmax(probs, axis=1)
    pred_round    = np.clip(np.round(exp_margin + 2).astype(int), 0, 4)

    p_hunter      = probs[:, 0] + probs[:, 1]
    p_draw        = probs[:, 2]
    p_surv        = probs[:, 3] + probs[:, 4]
    p3            = np.stack([p_hunter, p_draw, p_surv], axis=1)
    pred_3class   = np.argmax(p3, axis=1)
    actual_3class = np.where(actual_n < 2, 0, np.where(actual_n == 2, 1, 2))

    null5_class = np.bincount(actual_n, minlength=5).argmax()
    null3_class = np.bincount(actual_3class, minlength=3).argmax()

    def confusion(actual, pred, n_classes):
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for a, p in zip(actual, pred):
            cm[a, p] += 1
        return cm

    return {
        "acc_argmax":    (pred_argmax == actual_n).mean(),
        "acc_round":     (pred_round  == actual_n).mean(),
        "acc_3class":    (pred_3class == actual_3class).mean(),
        "null5_acc":     (actual_n == null5_class).mean(),
        "null3_acc":     (actual_3class == null3_class).mean(),
        "null5_class":   null5_class,
        "null3_class":   null3_class,
        "cm5":           confusion(actual_n, pred_argmax, 5),
        "cm3":           confusion(actual_3class, pred_3class, 3),
        "actual_n":      actual_n,
        "pred_argmax":   pred_argmax,
        "actual_3class": actual_3class,
        "pred_3class":   pred_3class,
        "n_total":       len(df),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_report(m: dict, label: str = "") -> None:
    n = m["n_total"]
    print(f"\n{'='*60}")
    print(f"ACCURACY METRICS  {label}  (n = {n:,} test game-halves)")
    print("=" * 60)

    print("\n── 5-class exact accuracy (n_escaped ∈ {0,1,2,3,4}) ──")
    print(f"  Model (argmax of P(n=k)):      {m['acc_argmax']:.1%}")
    print(f"  Model (round expected margin): {m['acc_round']:.1%}")
    print(f"  Null (always predict {m['null5_class']}):         {m['null5_acc']:.1%}")
    print(f"  Lift over null:                +{m['acc_argmax'] - m['null5_acc']:.1%}")

    class_names3 = ["hunter win", "draw", "survivor win"]
    print("\n── 3-class accuracy (hunter win / draw / survivor win) ──")
    print(f"  Model:                         {m['acc_3class']:.1%}")
    print(f"  Null (always predict {class_names3[m['null3_class']]!r}):  {m['null3_acc']:.1%}")
    print(f"  Lift over null:                +{m['acc_3class'] - m['null3_acc']:.1%}")

    print("\n── 5-class confusion matrix (rows=actual, cols=predicted) ──")
    cm5 = m["cm5"]
    short = ["n=0", "n=1", "n=2", "n=3", "n=4"]
    print("  actual\\pred  " + "  ".join(f"{s:>5}" for s in short))
    for i, row in enumerate(cm5):
        print(f"  {short[i]:>11}  " + "  ".join(f"{v:>5}" for v in row))

    print("\n── 3-class confusion matrix ──")
    cm3 = m["cm3"]
    short3 = ["hunter", "draw", "surv"]
    print("  actual\\pred  " + "  ".join(f"{s:>7}" for s in short3))
    for i, row in enumerate(cm3):
        print(f"  {short3[i]:>11}  " + "  ".join(f"{v:>7}" for v in row))

    print("\n── Per-class recall (3-class) ──")
    for i, name in enumerate(["Hunter win", "Draw", "Survivor win"]):
        total   = cm3[i].sum()
        correct = cm3[i, i]
        print(f"  {name:<14}: {correct/total:.1%}  ({correct}/{total})")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrices(m: dict, title_suffix: str = "",
                             save_path: Path | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    labels5 = ["0\n(hunter\ncrush)", "1\n(hunter\nwin)", "2\n(draw)",
               "3\n(surv\nwin)", "4\n(surv\ncrush)"]
    labels3 = ["Hunter\nwin", "Draw", "Survivor\nwin"]

    for ax, cm, labels, title in [
        (axes[0], m["cm5"], labels5,
         f"5-class: exact n_escaped\n(acc={m['acc_argmax']:.1%}, null={m['null5_acc']:.1%})"),
        (axes[1], m["cm3"], labels3,
         f"3-class: hunter win / draw / survivor win\n(acc={m['acc_3class']:.1%}, null={m['null3_acc']:.1%})"),
    ]:
        n_cls    = len(labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = cm / np.where(row_sums == 0, 1, row_sums)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Recall (row-normalised)")
        ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(title, fontsize=10)

        for i in range(n_cls):
            for j in range(n_cls):
                pct   = cm_norm[i, j]
                color = "white" if pct > 0.55 else "black"
                ax.text(j, i, f"{pct:.0%}\n({cm[i,j]:,})",
                        ha="center", va="center", fontsize=8,
                        color=color, fontweight="bold" if i == j else "normal")

    plt.suptitle(f"Model accuracy — Ordinal BT  {title_suffix}",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = save_path or (OUTPUTS / "confusion_matrix.png")
    OUTPUTS.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"→ {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_config(matches: pd.DataFrame, cfg_name: str) -> None:
    cfg    = CONFIGS[cfg_name]
    params = json.loads(cfg["json"].read_text())["best_params"]
    tau    = params["half_life_days"]
    l2     = params["l2_lambda"]
    thr    = int(params["threshold"])
    tiers  = cfg["tiers"]
    suffix = cfg["suffix"]

    print(f"\nConfig: {cfg_name}")
    print(f"  τ={tau:.1f}d  l2={l2:.3f}  threshold={thr}  tiers={tiers or 'all'}")

    cache = OUTPUTS / f"metrics_preds_{suffix}.csv"
    if cache.exists():
        print(f"  Loading cached predictions from {cache}")
        df = pd.read_csv(cache)
    else:
        print("  Running temporal CV…")
        df = collect_predictions(matches, tau, l2, thr, tiers)
        df.to_csv(cache, index=False)

    tier_str    = "/".join(tiers) if tiers else "all tiers"
    title_sfx   = f"({tier_str}, τ={tau:.0f}d, l2={l2:.2f}, thr={thr})"
    m           = compute_metrics(df)
    print_report(m, label=title_sfx)
    plot_confusion_matrices(
        m,
        title_suffix=title_sfx,
        save_path=OUTPUTS / f"confusion_matrix_{suffix}.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["top_tiers", "all_tiers"],
                        default="all_tiers")
    parser.add_argument("--all", action="store_true",
                        help="Run both configs")
    args = parser.parse_args()

    conn    = sqlite3.connect(DEFAULT_DB)
    matches = pd.read_sql("SELECT * FROM matches ORDER BY date", conn)
    conn.close()
    print(f"Loaded {len(matches)} matches")

    configs_to_run = list(CONFIGS.keys()) if args.all else [args.config]
    for cfg_name in configs_to_run:
        run_config(matches, cfg_name)
