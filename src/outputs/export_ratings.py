"""
src/outputs/export_ratings.py
==============================
Export final player ratings and model thresholds to website/js/data.js.

Reads outputs/db_export/ratings_top.csv (top-tiers model: IVL/IJL/COA)
and the learned ordinal thresholds, and writes a self-contained JS file
the website loads as static data.

Usage
-----
    python src/outputs/export_ratings.py
    → website/js/data.js
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.bradley_terry import filter_complete, compute_weights
from model.ordinal import fit_ordinal
from model.team_prior import identify_new_players, compute_team_mean_priors


def get_thresholds() -> list[float]:
    params = json.loads((_ROOT / "outputs" / "optuna" / "top_tiers_best.json").read_text())["best_params"]
    tau, l2, thr = params["half_life_days"], params["l2_lambda"], int(params["threshold"])
    tiers = ["IVL", "IJL", "COA"]

    conn = sqlite3.connect(str(_ROOT / "data" / "processed" / "idv.db"))
    matches = pd.read_sql("SELECT * FROM matches ORDER BY date", conn)
    conn.close()

    m = filter_complete(matches[matches["tournament_tier"].isin(tiers)]).sort_values("date").reset_index(drop=True)
    ref = m["date"].max()
    w   = compute_weights(m["date"], ref, tau)

    res0     = fit_ordinal(m, l2_lambda=l2, weights=w)
    new_keys = identify_new_players(m, res0["player_index"], thr)
    mu       = compute_team_mean_priors(m, res0["player_index"], res0["beta"], new_keys)
    res      = fit_ordinal(m, l2_lambda=l2, weights=w, init_theta=res0["theta"], prior_mean=mu)

    return res["theta"].tolist()


if __name__ == "__main__":
    ratings_path = _ROOT / "outputs" / "db_export" / "ratings_top.csv"
    df = pd.read_csv(ratings_path, keep_default_na=False)
    df = df.sort_values("beta", ascending=False)

    print("Computing model thresholds…")
    thresholds = get_thresholds()
    print(f"Thresholds: {[round(t, 4) for t in thresholds]}")

    def make_player(row):
        return {
            "id":       row["player_id"],
            "beta":     round(float(row["beta"]), 4),
            "nGames":   int(row["n_games"]),
            "lastSeen": str(row["last_seen"]),
        }

    hunters   = [make_player(r) for _, r in df[df["role"] == "hunter"].iterrows()]
    survivors = [make_player(r) for _, r in df[df["role"] == "survivor"].iterrows()]

    data = {
        "thresholds": [round(t, 6) for t in thresholds],
        "hunters":    hunters,
        "survivors":  survivors,
    }

    out = _ROOT / "website" / "js" / "data.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"const MODEL = {json.dumps(data, indent=2)};\n")

    print(f"Exported {len(hunters)} hunters, {len(survivors)} survivors")
    print(f"→ {out}")
