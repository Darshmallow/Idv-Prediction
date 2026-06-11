"""
src/metrics.py
==============
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
    python src/metrics.py
    → prints report, saves outputs/confusion_matrix.png
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
from sklearn.model_selection import TimeSeriesSplit

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_DB = str(_ROOT / "data" / "processed" / "idv.db")
OUTPUTS    = _ROOT / "outputs"
CACHE      = OUTPUTS / "metrics_preds.csv"

from model.bradley_terry import filter_complete, compute_weights
from model.ordinal import fit_ordinal, predict_probs, predict_expected_margin
from model.team_prior import identify_new_players, compute_team_mean_priors


# ---------------------------------------------------------------------------
# CV prediction collector
# ---------------------------------------------------------------------------

def _fit_with_prior(train, ref, half_life_days, l2_lambda, threshold):
    from bradley_terry import filter_complete as fc
    m = fc(train)
    w = compute_weights(m["date"], ref, half_life_days)
    res0 = fit_ordinal(m, l2_lambda=l2_lambda, weights=w)
    new_keys = identify_new_players(m, res0["player_index"], threshold)
    mu = compute_team_mean_priors(m, res0["player_index"], res0["beta"], new_keys)
    res1 = fit_ordinal(m, l2_lambda=l2_lambda, weights=w,
                       init_theta=res0["theta"], prior_mean=mu)
    return res1


def collect_predictions(
    matches: pd.DataFrame,
    half_life_days: float = 136.0,
    l2_lambda: float = 1.0,
    threshold: int = 5,
    n_splits: int = 5,
) -> pd.DataFrame:
    matches = filter_complete(matches).sort_values("date").reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for fold, (tr, te) in enumerate(tscv.split(matches), start=1):
        train, test = matches.iloc[tr], matches.iloc[te]
        ref = train["date"].max()
        print(f"  fold {fold}: train={len(train)}, test={len(test)}", flush=True)

        res = _fit_with_prior(train, ref, half_life_days, l2_lambda, threshold)
        probs = predict_probs(test, res)             # (N, 5)
        exp_margin = predict_expected_margin(test, res)  # (N,)
        actual_n = test["n_escaped"].to_numpy(dtype=int)

        for i in range(len(test)):
            rows.append({
                "fold":         fold,
                "actual_n":     actual_n[i],
                "exp_margin":   float(exp_margin[i]),
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
    actual_n = df["actual_n"].to_numpy()
    probs = df[["p0", "p1", "p2", "p3", "p4"]].to_numpy()
    exp_margin = df["exp_margin"].to_numpy()

    # ── 5-class predictions ──
    # Method 1: argmax of P(n=k) — minimises expected 0/1 loss
    pred_argmax = np.argmax(probs, axis=1)
    # Method 2: round expected margin + 2 — more natural interpretation
    pred_round = np.clip(np.round(exp_margin + 2).astype(int), 0, 4)

    # ── 3-class collapse ──
    p_hunter = probs[:, 0] + probs[:, 1]
    p_draw   = probs[:, 2]
    p_surv   = probs[:, 3] + probs[:, 4]
    p3 = np.stack([p_hunter, p_draw, p_surv], axis=1)
    pred_3class = np.argmax(p3, axis=1)   # 0=hunter, 1=draw, 2=survivor
    actual_3class = np.where(actual_n < 2, 0, np.where(actual_n == 2, 1, 2))

    # ── Null baselines ──
    null5_class = np.bincount(actual_n, minlength=5).argmax()
    null3_class = np.bincount(actual_3class, minlength=3).argmax()

    null5_acc = (actual_n == null5_class).mean()
    null3_acc = (actual_3class == null3_class).mean()

    # ── Accuracies ──
    acc_argmax = (pred_argmax == actual_n).mean()
    acc_round  = (pred_round  == actual_n).mean()
    acc_3class = (pred_3class == actual_3class).mean()

    # ── Confusion matrices ──
    def confusion(actual, pred, n_classes):
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for a, p in zip(actual, pred):
            cm[a, p] += 1
        return cm

    cm5 = confusion(actual_n, pred_argmax, 5)
    cm3 = confusion(actual_3class, pred_3class, 3)

    return {
        "acc_argmax":    acc_argmax,
        "acc_round":     acc_round,
        "acc_3class":    acc_3class,
        "null5_acc":     null5_acc,
        "null3_acc":     null3_acc,
        "null5_class":   null5_class,
        "null3_class":   null3_class,
        "cm5":           cm5,
        "cm3":           cm3,
        "actual_n":      actual_n,
        "pred_argmax":   pred_argmax,
        "actual_3class": actual_3class,
        "pred_3class":   pred_3class,
        "n_total":       len(df),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_report(m: dict) -> None:
    n = m["n_total"]
    labels5 = ["0 esc\n(hunter\ncrush)", "1 esc\n(hunter\nwin)", "2 esc\n(draw)",
               "3 esc\n(surv\nwin)", "4 esc\n(surv\ncrush)"]
    labels3 = ["Hunter\nwin", "Draw", "Survivor\nwin"]

    print("\n" + "=" * 60)
    print("ACCURACY METRICS  (n = {:,} test game-halves)".format(n))
    print("=" * 60)

    print("\n── 5-class exact accuracy (n_escaped ∈ {0,1,2,3,4}) ──")
    print(f"  Model (argmax of P(n=k)):   {m['acc_argmax']:.1%}")
    print(f"  Model (round expected margin): {m['acc_round']:.1%}")
    print(f"  Null (always predict {m['null5_class']}):      {m['null5_acc']:.1%}")
    print(f"  Lift over null:             "
          f"+{m['acc_argmax'] - m['null5_acc']:.1%}")

    print("\n── 3-class accuracy (hunter win / draw / survivor win) ──")
    class_names3 = ["hunter win", "draw", "survivor win"]
    print(f"  Model:                      {m['acc_3class']:.1%}")
    print(f"  Null (always predict {class_names3[m['null3_class']]!r}):  {m['null3_acc']:.1%}")
    print(f"  Lift over null:             "
          f"+{m['acc_3class'] - m['null3_acc']:.1%}")

    print("\n── 5-class confusion matrix (rows=actual, cols=predicted) ──")
    cm5 = m["cm5"]
    short = ["n=0", "n=1", "n=2", "n=3", "n=4"]
    header = "actual\\pred  " + "  ".join(f"{s:>5}" for s in short)
    print("  " + header)
    for i, row in enumerate(cm5):
        print(f"  {short[i]:>11}  " + "  ".join(f"{v:>5}" for v in row))

    print("\n── 3-class confusion matrix ──")
    cm3 = m["cm3"]
    short3 = ["hunter", "draw", "surv"]
    header3 = "actual\\pred  " + "  ".join(f"{s:>7}" for s in short3)
    print("  " + header3)
    for i, row in enumerate(cm3):
        print(f"  {short3[i]:>11}  " + "  ".join(f"{v:>7}" for v in row))

    print("\n── Per-class recall (3-class) ──")
    for i, name in enumerate(["Hunter win", "Draw", "Survivor win"]):
        total = cm3[i].sum()
        correct = cm3[i, i]
        print(f"  {name:<14}: {correct/total:.1%}  ({correct}/{total})")

    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrices(m: dict, save_path: Path | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    labels5 = ["0\n(hunter\ncrush)", "1\n(hunter\nwin)", "2\n(draw)",
               "3\n(surv\nwin)", "4\n(surv\ncrush)"]
    labels3 = ["Hunter\nwin", "Draw", "Survivor\nwin"]

    for ax, cm, labels, title, acc in [
        (axes[0], m["cm5"], labels5,
         f"5-class: exact n_escaped\n(accuracy = {m['acc_argmax']:.1%}, null = {m['null5_acc']:.1%})",
         m["acc_argmax"]),
        (axes[1], m["cm3"], labels3,
         f"3-class: hunter win / draw / survivor win\n(accuracy = {m['acc_3class']:.1%}, null = {m['null3_acc']:.1%})",
         m["acc_3class"]),
    ]:
        n_cls = len(labels)
        # Normalize by row (= recall per class)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm / np.where(row_sums == 0, 1, row_sums)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (row-normalised)")

        ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(title, fontsize=10)

        for i in range(n_cls):
            for j in range(n_cls):
                raw = cm[i, j]
                pct = cm_norm[i, j]
                color = "white" if pct > 0.55 else "black"
                ax.text(j, i, f"{pct:.0%}\n({raw:,})",
                        ha="center", va="center", fontsize=8,
                        color=color, fontweight="bold" if i == j else "normal")

    plt.suptitle("Model accuracy — Ordinal BT (τ=136d, team-mean prior, 5-fold temporal CV)",
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

if __name__ == "__main__":
    conn = sqlite3.connect(DEFAULT_DB)
    matches = pd.read_sql("SELECT * FROM matches ORDER BY date", conn)
    conn.close()
    print(f"Loaded {len(matches)} matches")

    if CACHE.exists():
        print(f"Loading cached predictions from {CACHE}")
        df = pd.read_csv(CACHE)
    else:
        print("\nRunning temporal CV…")
        df = collect_predictions(matches)
        df.to_csv(CACHE, index=False)
        print(f"Saved predictions to {CACHE}")

    m = compute_metrics(df)
    print_report(m)
    plot_confusion_matrices(m)
