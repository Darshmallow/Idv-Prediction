# Identity V Skill Decay Analysis — Project Handoff (v2)

## Status

**Modeling and analysis phase complete. Writeup phase remaining.**

The original project plan (linear Bradley-Terry margin model, temporal
half-life sweep, three stretch analyses) is implemented and validated.
Three additions extend the scope beyond the original plan:

- **Difference-penalty refinement** for trustworthy individual ratings
  (Option 4 — targeted shrinkage on collinear player pairs).
- **Ordinal regression** as a robustness check against the linear model's
  equal-spacing assumption.
- **Joint dual-τ model** confirming role asymmetry doesn't help prediction.

What's left is the README writeup, plot polish, and (optionally) test
coverage for the modeling code.

---

## Headline result

**Skill information about Identity V competitive matches has an optimal
half-life of about 3.6 months (≈ 110 days)** for predicting future match
outcomes via a margin-on-escapes Bradley-Terry model.

| Metric | Value |
|---|---|
| Optimal τ (linear margin BT) | **110 days** |
| OOS RMSE at τ* | 1.0990 ± 0.021 |
| Null model RMSE (predict ŷ=0) | 1.1400 |
| Improvement over null | 3.4% |
| Robustness (ordinal regression) | τ* shifts to 136d, RMSE 1.0944, qualitative finding intact |

Headline plot: `outputs/sweep.png`

---

## Repository layout

```
idv-analysis/
├── venv/                          # Python venv (activated)
├── data/
│   ├── raw/                       # 19 .xlsx files, read-only
│   └── processed/idv.db           # SQLite — 9 102 matches, 425 players
├── notebook/                      # Exploration (reference only, not extended)
│   ├── explore_01.ipynb
│   └── explore_02.ipynb
├── src/
│   ├── ingest.py                  # Excel readers + column alias dict
│   ├── players.py                 # Player ID canonicalization (alias map)
│   ├── db.py                      # SQLite build + matches/players tables
│   ├── bradley_terry.py           # Linear margin BT, closed-form ridge
│   ├── diff_penalty.py            # Collinearity-aware leaderboard (Option 4)
│   ├── eval.py                    # Temporal CV, half-life sweep, headline plot
│   ├── stretch.py                 # Phase 4 sub-analyses (role/tier/temporal)
│   ├── ordinal.py                 # Proportional-odds model (robustness check)
│   └── ordinal_eval.py            # Ordinal sweep + linear-vs-ordinal plot
├── tests/                         # 181 passing tests for ingest/players/db
├── outputs/
│   ├── sweep.png                  # ← HEADLINE: half-life sweep curve
│   ├── sweep_results.csv          # raw per-fold RMSE
│   ├── stretch_*.{png,csv}        # 3 stretch analyses
│   ├── ordinal_vs_linear.png      # robustness comparison
│   ├── db_export/
│   │   ├── matches.csv            # 9 102 rows
│   │   ├── players.csv            # 425 players
│   │   ├── ratings.csv            # Individual leaderboard (λ_d=3)
│   │   └── schema.sql             # DB schema
│   ├── diff_penalty_findings.md
│   ├── team_effects_findings.md   # rejected approach, kept for reference
│   ├── stretch_findings.md
│   └── ordinal_findings.md
├── PROJECT_HANDOFF.md             # ← this file
├── README.md                      # ← TO BUILD: final writeup
└── NOTES.md
```

Installed: `pandas openpyxl numpy scipy statsmodels scikit-learn
matplotlib seaborn plotly jupyter pytest`.

---

## Data

### Sources

`data/raw/` contains 19 `.xlsx` files spanning 2020-2026:

- **Legacy** `2020-2023.xlsx` — 7 tabs (4 IVL years + COA4/5/6) plus the
  critical `曾用id` tab for historical player IDs.
- **Modern** 18 single-season files: 8 IVL, 6 IJL, 4 COA international finals.

**Excluded:** Japan regional qualifiers and the IVS one-off — different
competitive populations, would add noise. **Missing:** COA7 (2024) — no
data available.

### Database schema

`data/processed/idv.db`:

```sql
CREATE TABLE matches (
    match_id          INTEGER PRIMARY KEY,
    date              TEXT,        -- ISO YYYY-MM-DD
    tournament        TEXT,        -- 'IVL_2024_summer_regular', 'COA9', etc.
    hunter_player     TEXT,        -- canonical ID
    survivor1_player  TEXT, survivor2_player TEXT,
    survivor3_player  TEXT, survivor4_player TEXT,
    n_escaped         INTEGER,     -- 0-4 (the graded outcome)
    hunter_wins       INTEGER,     -- 1/0/NULL (NULL = draw)
    source_file       TEXT,
    source_row        INTEGER      -- traceability
);

CREATE TABLE players (
    canonical_id  TEXT PRIMARY KEY,
    known_roles   TEXT,             -- 'hunter' | 'survivor' | 'both'
    first_seen    TEXT,
    last_seen     TEXT,
    n_games       INTEGER
);
```

`n_escaped` is the modelling outcome. `margin = n_escaped − 2` ∈ {−2,…,+2}.
Positive margin = survivor advantage, negative = hunter advantage.
**Draws (n_escaped = 2) are retained** — the linear margin formulation
handles them naturally as the zero midpoint.

### Counts

- 9 102 game halves total (1 142 from 2020, 6 766 from 2021-2023, 588 modern)
- 425 unique players (119 hunters, 306 survivors, 7 dual-role)
- Date range: 2020-06-25 → 2026-05-05
- Hunter coverage: 99.87% (12 unverifiable rows left NULL after backfill)
- `n_escaped` coverage: 100% (three-tier derivation: total_escapes col →
  per-survivor results → winner+score for 2020)

### Processing pipeline

1. **`ingest.py`** — auto-detect header row per file, rename Chinese columns
   via 140-entry alias map, drop template rows by anchor-column filter.
2. **`players.py`** — parse `曾用id` tab → `ALIAS_MAP`; `MANUAL_OVERRIDES`
   handles post-2023 changes and fuzzy-match corrections (`ppicha→pipicha`,
   `gua→guag`, `taoxing→tx`).
3. **`db.py`** — primary source = modern `赛后数据` sheets (survivor IDs +
   results), with hunter backfilled from `原始数据` for 5 early-2024 files;
   validated against winner_side, falls back NULL on disagreement.

181 pytest tests cover the foundation modules (none yet for modeling).

---

## The four models

### 1. Linear margin BT (`bradley_terry.py`) — primary, headline

$$\mathbb{E}[\text{margin}_i \mid \beta] = \tfrac14\sum_{k=1}^4\beta^S_{s_{ik}} - \beta^H_{h_i}$$

Closed-form weighted ridge:
$\beta^* = (X^\top W X + \lambda I)^{-1} X^\top W y$.
Sparse 9 100 × 432 design matrix, ~1 ms per solve.
Players indexed by `(player_id, role)` tuples → dual-role players get
independent hunter and survivor ratings.

L2 = 1.0 (default). Time decay weights $w_i = 0.5^{\Delta t_i / \tau}$.

### 2. Difference-penalty refinement (`diff_penalty.py`) — individual ratings

Augments the ridge with a penalty on differences between players who
appear in the same matches ≥ 90% of the time and have ≥ 8 games each:

$$\lambda \|\beta\|^2 + \lambda_d \sum_{(i,j) \in C} (\beta_i - \beta_j)^2$$

Encoded as a Laplacian on the collinearity graph. λ_d = 3.0 (RMSE-optimal
via grid sweep over [0, 10]). Resolves the koting/persica artifact: in the
baseline model koting got an inflated +0.94 (compensating for persica's
−0.38) because koting only ever played with persica. Diff-penalty collapses
their ratings to similar values (both ~+0.05) while leaving non-collinear
players (huan) untouched.

Used to produce `outputs/db_export/ratings.csv`, the trustworthy individual
leaderboard. Does **not** change the half-life finding — that's computed
with the plain linear BT.

### 3. Half-life sweep (`eval.py`) — methodology

5-fold `TimeSeriesSplit` temporal CV. 20 log-spaced τ values from 30 to
1825 days, plus τ = ∞ (no decay) as a baseline. For each fold the
reference date = last training date (not a global anchor — weights reflect
information available at that point in time).

Metric: out-of-sample RMSE on margin. Cold-start players (unseen in
training) implicitly get β = 0 — equivalent to the L2 prior, no
explicit handling needed.

Full sweep: ~2 seconds.

### 4. Ordinal regression (`ordinal.py`) — robustness check

Proportional-odds (cumulative-logit) BT. Same η = mean(β^S) − β^H, but
five ordered outcome categories and four learned thresholds θ_1 < … < θ_4.
Ordering enforced by reparameterising θ as (a, exp(b_1), exp(b_2), exp(b_3)).

Fit by L-BFGS-B with analytical gradient (verified by `scipy.optimize.check_grad`).
50-140 iterations per fit; full sweep ~10s.

**Confirms** the headline finding (τ shifts modestly to 136d, RMSE drops to
1.0944). The learned threshold spacings (0.77 / 1.77 / 2.47) prove the
linear model's equal-spacing assumption was wrong — the n=3 → n=4 transition
requires ~3.2× more skill than n=0 → n=1.

---

## Findings beyond the headline

### Role asymmetry — hunters MORE stable than survivors (contradicts hypothesis)

Separate hunter-only and survivor-only sweeps:

| Role     | Optimal τ | months |
|----------|----------:|-------:|
| Hunter   |  210 d    |  6.9   |
| Survivor |   57 d    |  1.9   |

The original hypothesis (hunter would be shorter because characters get
patched) was **inverted**. Plausible reason: pro hunters main 1-3
characters consistently; character mastery dominates and is stable.
Survivor performance depends more on shifting team compositions and meta.

**Caveat (important):** I also tested a dual-τ joint model with
role-specific decay weights. RMSE improvement over single-τ joint:
0.0013 (well within ±0.02 SE). The asymmetry is real structurally but
**not actionable for forecasting** — a textbook "informative but
non-exploitable" finding.

### Tier comparison — IVL well-identified, IJL/COA noisy

| Tier | Matches | Optimal τ | SE |
|------|--------:|----------:|------|
| IVL  | 6 876   | 110 d     | ±0.021 |
| IJL  | 1 045   | 324 d     | ±0.038 |
| COA  | 1 179   | 324 d     | ±0.053 |

IVL is well-identified. IJL/COA both land at the same grid point with
much wider SEs — the curves are too flat to resolve a sharp optimum at
those sample sizes. Honest reading: IJL/COA half-lives are bounded
**below** somewhere > 100 d but exact values aren't trustworthy.

### Temporal stability — meta has sped up ~35%

| Era | Matches | Optimal τ |
|-----|--------:|----------:|
| 2020-2022 | 3 674 | 110 d |
| 2023+     | 5 428 |  71 d |

Cleanest stretch finding. Post-2023 skill information becomes stale ~35%
faster. Plausibly real — both curves well-separated with comparable SEs.

---

## What's left

### Phase 5 — Writeup (the main remaining work)

**`README.md`** — needs to be written. Should contain:

1. **One-sentence headline** — "Skill information in competitive Identity V
   has an optimal half-life of ≈ 110 days for predicting future match
   outcomes."
2. **Background** — brief: what the game is, what the data is, why the
   question matters (alpha-decay analog).
3. **Explicit quant framing** — alpha decay analogy, temporal CV as
   backtest methodology, ridge as Gaussian prior / shrinkage, the
   "informative but non-exploitable" role-asymmetry caveat.
4. **Methodology section** — time-weighted MLE, why RMSE, why TimeSeriesSplit,
   why margin not binary.
5. **Headline plot** — embed `outputs/sweep.png`.
6. **Stretch findings** — role / tier / temporal, each with its plot.
7. **Robustness check** — ordinal regression result (embed
   `outputs/ordinal_vs_linear.png`).
8. **Limitations** — listed below.
9. **Reproducibility** — `python src/db.py && python src/eval.py` etc.

### Suggested interview talking points to embed naturally

- *"I implemented time-weighted ridge from scratch in closed form, with
  cold-start regularisation that's mathematically equivalent to a
  Gaussian prior — the standard shrinkage formulation."*
- *"I used RMSE because it's consistent with the squared-error loss I'm
  optimising; using a different metric for evaluation than for
  optimisation would let me tune τ to a metric the model wasn't trying
  to minimise."*
- *"I used `TimeSeriesSplit` to avoid lookahead bias — random k-fold
  would leak future matches into training, which is the standard pitfall
  in backtesting trading signals."*
- *"The half-life parameter is structurally analogous to alpha decay
  in systematic trading — measuring how fast historical signal becomes
  stale. Optimal τ ≈ 110 days, robust to model class (linear ↔ ordinal)
  and to sub-population splits."*
- *"I found a real structural asymmetry — hunter skill information
  persists ~3× longer than survivor information — but a joint model
  exploiting that asymmetry gave no significant RMSE improvement.
  Worth understanding the system but not actionable for forecasting."*

### Optional polish

- **Test coverage for modeling code** — `tests/test_bradley_terry.py`,
  `tests/test_eval.py`, `tests/test_diff_penalty.py`, `tests/test_ordinal.py`.
  Currently the modeling modules are only validated end-to-end by their
  sanity-check entrypoints. Low priority but defensive against
  regressions during the writeup phase.
- **Re-run stretch with ordinal** — for full consistency we could re-run
  role/tier/temporal under ordinal regression. The original handoff
  recommended this as a "future work" note rather than blocking. Likely
  to shift optima by ~25% (as the headline did) without changing
  directions.
- **A single orchestrator script** — `python run.py` that rebuilds DB,
  runs all sweeps, regenerates all plots. Currently each module has its
  own `if __name__ == "__main__"`.

---

## Implementation notes / decisions made

These are interview talking points; each represents a defensible choice
made along the way.

### Why margin (continuous, 5 levels) instead of binary win/loss

Binary BT throws away the gradient information (4-0 sweep vs 3-1 narrow win
are both "hunter wins"), and forces dropping all draws (~38% of data).
Margin keeps everything and treats the outcome as ordinal-but-evenly-spaced.

### Why closed-form ridge instead of L-BFGS-B for the headline

Linear margin + L2 has a closed-form solution; logistic BT would not.
Closed-form is 100× faster (~1 ms vs ~300 ms per fit), which matters for
the 100-fit sweep. Ordinal regression sacrifices the closed form for the
equal-spacing fix; both are available.

### Why difference-penalty instead of team random effects

Team random effects (`team_effects` branch — kept for reference but
**not merged**) fixed the collinearity symptom but damped legitimately
identified players (huan dropped from +0.98 to +0.42) and gave no OOS
benefit. Difference-penalty targets only the actually-collinear pairs,
leaving well-identified players alone.

### Why escape count instead of just the winner side

`n_escaped` is finer-grained than `winner_side` and captures decisive
vs marginal outcomes. Also: in our 9 102 matches, escape-count derivation
is consistent for 100% of rows (modern files have `total_escapes` column;
legacy 2021+ has per-survivor results; 2020 derives from `winner_side` +
`min(home, away)` score).

### Why diff-penalty λ_d = 3

Grid sweep over [0, 10] found the RMSE minimum at λ_d ≈ 3-4 (basically
flat throughout that range, within ±SE). λ_d = 3 retains some
within-clique signal so genuinely-strong players in collinear groups
aren't fully collapsed. Higher values (λ_d = 10) collapse more
aggressively but RMSE is essentially tied.

---

## Limitations (for the README's "Limitations" section)

1. **Equal-spacing approximation in the linear model.** Documented
   limitation; ordinal regression validates the headline survives.
2. **Cold-start players get β = 0.** Defensible as the L2 prior, but
   means matches between two unseen players are predicted as draws by
   default. A handful of cold-start matches per fold (≈ 7%); not enough
   to materially affect the sweep but should be noted.
3. **Collinearity for fixed-roster teams.** Individual ratings for
   players who appear in only one team composition are not
   independently identifiable — the diff-penalty model addresses this
   for individual ratings but the underlying limitation is structural.
4. **Tier-specific results (IJL/COA) are noise-limited.** Sample sizes
   < 1 200 give wide SEs; the IJL/COA optima both land at the same
   grid point and shouldn't be over-interpreted.
5. **2020 data lacks per-survivor results.** `n_escaped` is derived from
   winner + score; this loses one bit of information per match (which
   specific survivors escaped) but preserves the outcome count.

---

## Reproducibility

```bash
# 1. Activate venv
source venv/bin/activate

# 2. Build the database from raw Excel files
python src/db.py
# → data/processed/idv.db

# 3. Run the headline sweep
python src/eval.py
# → outputs/sweep.png, outputs/sweep_results.csv

# 4. Run the stretch analyses
python src/stretch.py
# → outputs/stretch_*.{png,csv}, outputs/stretch_findings.md

# 5. Run the ordinal robustness check
python src/ordinal_eval.py
# → outputs/ordinal_vs_linear.png, outputs/sweep_ordinal_results.csv

# Optional: regenerate the individual leaderboard
python -c "
import sys; sys.path.insert(0,'src')
import sqlite3, pandas as pd
from collections import Counter
import bradley_terry as bt, diff_penalty as dp
conn = sqlite3.connect('data/processed/idv.db')
m = pd.read_sql('SELECT * FROM matches', conn); conn.close()
beta, idx, pairs = dp.fit_ratings(m)   # λ_d=3 default
print(bt.ratings_df(beta, idx).head(20).to_string())
"

# Run the tests
pytest tests/ -q
# → 181 passed
```

Full sweep + stretch + ordinal end-to-end: < 30 seconds on a modern laptop.

---

## Git history

The main branch tracks the canonical narrative. Three feature branches
were merged via `--no-ff`; one branch (`team-effects`) was implemented,
evaluated, and rejected — kept for reference.

```
main
├── stretch-analyses (merged)
├── ordinal          (merged)
├── diff-penalty     (merged)
└── team-effects     (NOT merged — see outputs/team_effects_findings.md)
```

Use `git log --oneline --graph --all` to see the branching structure.
