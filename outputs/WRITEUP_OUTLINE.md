# Writeup Outline — everything you can talk about

Bullet-point inventory of every decision, finding, and dead-end from this
project. Use this as a menu — keep what serves the narrative, drop what
clutters it.

Format recommendation: **README.md** (Markdown) as the primary artifact.
A LaTeX version is only worth it if you want a long-form technical paper
to link from the README — most readers won't open a PDF.

---

## 1. Framing / abstract

- The research question — *"How fast does competitive skill information
  become obsolete in a high-frequency esports environment?"*
- One-sentence headline finding (your choice of phrasing around the
  ≈ 110-day half-life from the simpler model, or ≈ 136-200 days from
  the ordinal-with-corrections model)
- Quant-trading analog: **alpha decay** as the structurally identical
  problem
- Concrete subject: Identity V pro esports, 9 102 game halves,
  2020-06-25 → 2026-05-05
- Why this dataset: 6 years, three competitive tiers, a documented meta
  shift inside the window — natural test bed for time-decay modelling

---

## 2. Data

### 2.1 Sources

- 19 `.xlsx` files in `data/raw/`: 1 legacy multi-tab file
  (`2020-2023.xlsx`), 18 modern single-season files (IVL/IJL/COA)
- Span: 2020-06-25 → 2026-05-05, three competitive tiers
- Excluded: 2025 IVS one-off, both Japan regional qualifier files
  (different competitive populations)
- Missing: COA7 (2024) — no data exists; documented as a gap

### 2.2 Schema evolution / data tidiness story

- Three generations of column naming (2020-2022, 2023, 2024+) requiring
  a 140-entry alias dictionary
- Header row detection — files use header rows 0 or 1 unpredictably;
  auto-detected via known-marker overlap
- Template rows — Excel files have 700-1632 rows per sheet with
  pre-formatted zero defaults; filtered out using anchor columns
- Three context-dependent column meanings resolved (`角色`, `角色.4`,
  `总逃脱` vs `逃生数`)
- Modern format split across two sheets (`原始数据` and `赛后数据`) —
  joined via sequential-group join validated against `winner_side`
- Five 2024 files missing hunter ID in `赛后数据` — backfilled from
  `原始数据`, validated against `winner_side`, 8 rows left NULL because
  they couldn't be verified
- Date parsing for legacy files supports three formats: decimal
  (`6.25`), asterisk (`6*9`), and bare integers like `1104` for Nov 4

### 2.3 Player ID canonicalization

- `曾用id` tab in legacy file tracks historical ID changes for
  ~120 players; parsed into a 17-entry alias map
- `MANUAL_OVERRIDES` for post-2023 changes and fuzzy-match corrections
  (`ppicha→pipicha`, `gua→guag`, `taoxing→tx`)
- Subtle bug: `pandas` interprets the string `"nan"` as NaN by default,
  silently dropping player 楠 (in-game ID literally `nan`); fixed with
  `keep_default_na=False, na_values=[""]`
- ~425 unique canonical players (119 hunters, 306 survivors, 7 dual-role)

### 2.4 Schema (final SQL)

- Two tables: `matches` (~9 100 rows) and `players` (~425 rows)
- Outcome stored at escape-count resolution (0-4) rather than collapsed
  to binary win/loss — preserves draws and information about decisiveness
- `n_escaped` derivation has three tiers: `total_escapes` column → sum
  of per-survivor results → derive from winner + half-score (for 2020,
  which has no per-survivor results)
- 100% n_escaped coverage, 99.87% hunter coverage, 0 null dates

### 2.5 Testing the data layer

- 181 pytest unit + integration tests across `ingest.py`, `players.py`,
  `db.py`

---

## 3. Modelling decisions (in order made)

### 3.1 Outcome representation: continuous margin

- Margin = n_escaped − 2 ∈ {−2,−1,0,+1,+2}; positive = survivor advantage
- Why margin not binary win/loss:
  - Binary throws away gradient information (4-0 sweep ≠ 3-1 narrow win)
  - Forces dropping ~38% of data (draws)
- Defensible approximation: ordinal-but-treated-as-evenly-spaced
- Linearity assumption to be revisited later (the ordinal regression
  robustness check)

### 3.2 Asymmetric 1v4 structure

- $\mathbb{E}[\text{margin}] = \tfrac14 \sum_k \beta^S_{s_k} − \beta^H_h$
- Why `mean of survivors / 4` and not `sum`:
  - Sum scales prediction with team size; mean keeps "1 skill unit = 1
    extra escape" interpretation clean
- Why subtract (mean of survivors − hunter):
  - Keeps "high β = more skilled at your role" intuitive
  - Critical sign convention — needed to be fixed at one point after a
    sign error

### 3.3 L2 regularization

- Why L2 not L1: smooth shrinkage (sparse players stay in the model
  with small β) vs Lasso's hard zeros
- Bayesian interpretation: equivalent to Gaussian prior β ~ N(0, σ²) —
  the "covariance shrinkage" analog in portfolio construction

### 3.4 Closed-form weighted ridge

- $\beta^* = (X^\top W X + \lambda I)^{-1} X^\top W y$
- Cholesky solve on the SPD normal-equations matrix
- 9100 × 432 sparse design matrix → ~1 ms per fit
- Why this matters: enables fast half-life sweep

### 3.5 Time-decay weighting

- $w_i = (½)^{Δt_i / τ}$, exponential decay
- Memoryless property — unique function where relative weight depends
  only on time between matches, not absolute time
- Direct quant analog: factor-decay weighting in trading signal research

---

## 4. Evaluation methodology

### 4.1 Temporal cross-validation

- 5-fold `TimeSeriesSplit`; train always strictly before test
- Why **not** k-fold: would leak future into training (lookahead bias —
  the standard pitfall in backtesting)
- Expanding-window, not sliding (time decay is the model's job)
- Trade-off: earliest matches are never used as test (price of pure
  temporal CV)

### 4.2 Alternative split strategy: season-aligned folds

- 6 folds = 6 competitive years (2020-2025); 5 expanding-window splits
- Each test fold = exactly one season's worth of matches
- Motivated by: TimeSeriesSplit's mid-season cuts cause cold-start
  artifacts in some folds
- Result: doesn't beat TimeSeriesSplit on mean R² but gives more
  interpretable per-fold diagnostics
- Available as `split_strategy="seasons"` in `eval.py`

### 4.3 Metric: RMSE on margin

- RMSE consistent with the squared-error loss being minimised
- Null baseline: predict ŷ = 0 always → RMSE ≈ 1.14
- Defensibility: "I used the same metric for evaluation that I'm
  optimising" — avoids the tuning-to-the-wrong-metric trap

### 4.4 Cold-start handling

- Players unseen in training implicitly get β = 0 = the L2 prior
- Not dropped from evaluation — kept in to honestly include them

---

## 5. The headline finding

- Optimal half-life **τ ≈ 110 days** (linear margin BT, single rate,
  TimeSeriesSplit) — basic version of the finding
- Improvement over null: 3.36% RMSE, R² ≈ 6.50%
- Headline plot: `outputs/sweep.png`
- Robust across stretch analyses (sub-population sweeps consistently
  land in same order of magnitude)
- Reframed if using ordinal + regime + new-player prior:
  **R² = 8.04%** under TimeSeriesSplit CV

### 5.1 Alternative phrasings to pick from

- "≈ 3.6 months" / "≈ 110 days"
- "1 skill unit = 1 extra escape" interpretation
- Shorter than typical Elo / chess-rating half-lives (which are
  effectively years) — explainable by character meta evolution

---

## 6. Stretch analyses (Phase 4)

### 6.1 Role asymmetry — finding contradicted hypothesis

- Hunter optimal τ ≈ 210 days; survivor optimal τ ≈ 57 days
- Hypothesis (from handoff): hunter τ would be **shorter** because
  characters get patched
- Result: hunter τ is ~3.7× **longer** than survivor
- Plausible reason: pro hunters main 1-3 characters; character mastery
  dominates and is stable. Survivor performance depends more on shifting
  team composition and meta
- Caveat — extra finding: dual-τ joint model with role-specific decay
  gives ΔRMSE = 0.0013 over single-τ — well within ±0.02 SE. The
  asymmetry is **informative but not exploitable**, classic
  "I learned something about the system but it doesn't beat the null
  baseline by enough to forecast better"

### 6.2 Tier comparison

- IVL well-identified at τ ≈ 110 d (largest sample, 6,876 matches)
- IJL τ ≈ 324 d (1,045 matches), COA τ ≈ 324 d (1,179 matches)
- IJL/COA results are noise-limited — wide SEs, flat curves, the
  324d optimum is a grid-point artifact
- Honest framing: IVL is the real finding; IJL/COA bounded below ~100d
  but not trustworthy beyond that

### 6.3 Temporal stability — meta sped up

- Pre-2023 optimal τ ≈ 110 d; post-2023 optimal τ ≈ 71 d
- Post-2023 skill information goes stale **~35% faster**
- Cleanest stretch finding — well-separated curves, comparable SEs

---

## 7. Robustness checks

### 7.1 Ordinal regression

- Proportional-odds Bradley-Terry, fit by L-BFGS-B with analytical
  gradient (verified by `scipy.optimize.check_grad`)
- Confirms headline τ shifts modestly (110 → 136 days) but qualitative
  finding survives
- Learned threshold spacings: 0.77 / 1.77 / 2.47 — confirms the
  linear model's equal-spacing assumption was wrong; n=3→n=4 transition
  requires 3.2× more skill than n=0→n=1
- RMSE improvement: 1.1017 → 1.0944, R² 6.50% → 7.70%
- Per-fold: ordinal beats linear in 4/5 folds; paired t-test
  p = 0.17 (n=5, directional but not significant at α=0.05)

### 7.2 Cross-population sub-analyses

- See stretch analyses (Section 6); the headline finding is structurally
  the same in every subset where the data supports a confident answer

---

## 8. Things that worked (kept in the final model)

### 8.1 Regime-shift decay (under TimeSeriesSplit only)

- Piecewise weighting with separate (τ_pre, τ_post) and a multiplicative
  discount α on pre-shift matches
- Best for linear: τ_pre=365d, τ_post=30d, α=0.20 → R² = 7.53%
  (+1.03pp over linear baseline)
- Best for ordinal: τ_pre=200d, τ_post=60d, α=0.60 → R² = 7.93%
  (+0.23pp over ordinal baseline)
- Why it doesn't help under season splits: the bad fold's training is
  entirely pre-shift in that protocol, so α has nothing to discount
- Why it does help under TimeSeriesSplit: the bad fold's training spans
  the shift date, so α can discount pre-shift matches in favour of the
  9 months of post-shift training data
- Direct interview hook: this is exactly how systematic traders handle
  known structural breaks — apply a discrete information discount at the
  regime boundary alongside continuous time-decay

### 8.2 Difference penalty on collinear pairs (for the individual leaderboard)

- Targeted L2 difference penalty between players who share ≥ 90% of
  matches and have ≥ 8 games each
- λ_d = 3 chosen via grid sweep (RMSE-optimal)
- Doesn't change the headline finding (RMSE improvement is within noise)
  but produces a defensible individual leaderboard
- Fixes the **koting/persica artifact** — koting only ever played with
  persica, leading to inflated +0.94 / deflated −0.38 ratings under the
  baseline; diff penalty pulls them to similar values (+0.05 each) while
  leaving non-collinear players (huan) untouched
- Used to produce `outputs/db_export/ratings.csv`

### 8.3 New-player team-mean prior (the current best)

- For players with fewer than 5 training matches, set their L2 prior
  centre at the mean β of their teammates from their first match
- Two-pass procedure: fit standard model first, identify new players,
  compute team-mean priors, refit
- R² 7.93% → 8.04% on top of ordinal+regime
- The improvement is **concentrated in fold 3** (the meta-shift fold):
  R² −0.51% → +0.05% — finally positive
- Tiny gain in absolute terms (+0.11 pp) but the fold-3 sign flip is
  the structurally meaningful narrative

### 8.4 Score-derived n_escaped for 2020

- 2020 data has no per-survivor results, only winner side and half score
- Recovered n_escaped via: winner + min(home_score, away_score)
- 0:5 hunter win → 0 escapes, 1:3 hunter win → 1, 2:2 → draw, etc.
- 100% recovery on 2020 data; cross-validated against the per-survivor
  derivation on 2021-2023 (99.9% agreement)
- Why this matters in the writeup: shows attention to data
  completeness, recovery of information that would otherwise be lost

---

## 9. Things tried that did NOT work — keep or drop per taste

(These are all defensible to mention as "alternatives explored". Mention
some — too many makes the writeup feel padded.)

### 9.1 Team random-effects intercepts

- Added one γ_T intercept per unique survivor team composition
- Fixed the koting/persica collinearity symptom but **damped legitimate
  individual signal** (huan dropped from +0.98 to +0.42)
- Zero OOS improvement (cold-start team intercepts dominate)
- Branch: `team-effects` (not merged) — `outputs/team_effects_findings.md`
- Compact mention worth keeping; demonstrates being able to evaluate and
  reject an idea

### 9.2 Map fixed effects

- Pulled `map_name` into the matches table; added per-map γ_m intercepts
- Map effects are **real** (Spearman ρ = 0.97 between learned γ and raw
  per-map margins): 圣心医院 favours survivors, 不归林 favours hunters
- **BUT doesn't help OOS prediction** — every τ in the sweep is 0.002–
  0.003 RMSE worse than baseline
- Reasons: cold-start maps in test folds (新 maps appear over time);
  confounding with team identity (map choice is part of draft);
  L2 rebalancing of overall bias
- Branch: `map-effects` (not merged) — `outputs/map_effects_findings.md`
- Compelling negative result for the "data has lots of features, only
  some are predictively useful" framing

### 9.3 Team-composition model (ratings per unique team, not individual)

- One β^T per unique 4-survivor team composition; hunters individual
- Severe cold-start: 55.6% of CV test rows feature a never-seen team
- Loses to baseline by 0.0070 RMSE at each model's optimum
- Best τ shifts dramatically (110d → 1825d)
- Structural finding: "skill information half-life depends on how skill
  is measured — at the individual level ~4 months, at the team level
  much longer because we can't reliably see new team identities"
- Branch: `team-comp` (not merged) — `outputs/team_comp_findings.md`

### 9.4 Naive rolling-30 baseline

- Predict each match by summing player's K=30-match rolling-average margin
- Under strict CV: K=30 → RMSE 1.149 (**worse than null model 1.140**)
- Under leaky streaming: K=500 → ties/beats Bradley-Terry, but that
  uses intra-fold data the BT model doesn't get
- Strong sanity-check finding: shows BT actually earns its predictive
  accuracy via opponent adjustment + shrinkage, not by trivially
  capturing "who's been winning lately"
- Worth a sentence in the methodology section as the floor BT is
  benchmarked against
- Branch: `naive-rolling` (not merged)

### 9.5 Universal team-anchored prior (Laplacian over teammate graph)

- Penalty: λ_team ‖(I − A)β‖² on every survivor
- Standalone gain +0.61 pp R² over baseline
- Heavy overlap with regime decay — combined gain only +0.06 pp on top
  of regime alone
- Replaced by the targeted new-player-only prior (Section 8.3) which
  has the same intent but doesn't damp established players' signal
- Worth mentioning briefly as "alternative parameterisation"

### 9.6 Joint dual-τ joint model (separate τ for hunter and survivor)

- Asymmetric normal equations; closed-form solve via custom assembly
- Best: τ_H = 169d, τ_S = 88d — directionally matches the role-only
  sweep finding
- ΔRMSE over single-τ joint = 0.0013 — well within ±0.02 SE
- Already mentioned in 6.1 as the "informative but not exploitable"
  caveat to the role-asymmetry finding

### 9.7 Map data extraction (kept the column, not the model)

- Added `map_name` to the matches table; the data extraction itself
  was a real piece of work (sequential-group join, header detection
  on yet another column)
- Currently 99.5% coverage of map_name across 9 active maps
- Even though map effects don't help prediction, the column is in the
  database for future descriptive analysis

---

## 10. Limitations

- **Linear margin model assumes equal escape-spacing** — documented;
  ordinal regression validates the headline survives
- **Cold-start matches**: a small fraction (~5-7%) of test matches each
  fold have at least one player unseen in training. Defensible default
  (β = 0 = L2 prior), but worth mentioning
- **Fold-by-fold R² ranges from −1% to +18%** — write up the per-fold
  diagnostic honestly. Fold 3 (the 2023 meta-shift fold) is the worst
  even with all corrections applied
- **IJL/COA half-lives are wide-SE estimates** — only IVL is precisely
  identified
- **2020 data is shallow**: no per-survivor results, n_escaped derived
  from winner + score
- **The regime-shift is a hand-set hyperparameter** (2023-01-01),
  validated by sensitivity sweep but not learned via changepoint
  detection
- **R² of 8% sounds modest** — but is good for forecasting an
  individual game-half outcome under temporal CV. Published xG models
  for individual football match outcomes typically achieve 5-15% under
  similar protocols
- **Sample size of 5 folds limits statistical power** — paired t-tests
  between model variants often have p > 0.1 even when point estimates
  improve meaningfully

---

## 11. Quant framing — interview talking points to weave in naturally

Each of these is a one-liner you can say in an interview without it
sounding rehearsed:

- *"I implemented time-weighted ridge regression from scratch in closed
  form, with cold-start regularization that's mathematically equivalent
  to a Gaussian prior — the standard shrinkage formulation."*
- *"I used `TimeSeriesSplit` cross-validation to avoid lookahead bias —
  random k-fold would leak future matches into training, which is the
  standard pitfall in backtesting trading signals."*
- *"The half-life parameter is structurally analogous to alpha decay
  in systematic trading — measuring how fast historical information
  becomes stale."*
- *"L2 regularization on player ratings is equivalent to a Gaussian prior;
  for sparse players it provides shrinkage analogous to the covariance
  shrinkage used in portfolio construction."*
- *"I found a real structural asymmetry — hunter information persists
  3× longer than survivor information — but a joint model exploiting
  that asymmetry gave no significant RMSE improvement. Worth
  understanding the system, but not actionable for forecasting."*
- *"I tested a piecewise time-decay specification with a multiplicative
  discount at the regime boundary — the same technique systematic
  traders use to handle known structural breaks in alpha decay."*
- *"On the headline test set, the model's confident predictions are
  accurate (43% recall on predicted +1 vs 25% empirical base rate) but
  the L2 prior correctly recognizes the per-match signal-to-noise ratio
  doesn't support extreme predictions — so the model rarely guesses ±2."*

---

## 12. Methodological choices to defend in the writeup (deeper)

- Why margin not binary (and what ordinal regression contributes as a
  robustness check)
- Why closed-form ridge not L-BFGS-B (compute budget for the sweep)
- Why difference penalty not team random effects (collateral damage on
  identifiable players)
- Why escape count over winner_side as the modelling outcome (finer
  signal, recovers draws)
- Why diff-penalty λ_d = 3 (RMSE-optimal grid sweep, retains some
  within-clique signal)
- Why expanding-window CV not sliding (model τ handles forgetting, not
  the CV protocol)

---

## 13. Reproducibility

- `python src/db.py` → builds `data/processed/idv.db` from 19 .xlsx files
- `python src/eval.py` → runs the headline half-life sweep
- `python src/stretch.py` → role / tier / temporal sub-analyses
- `python src/ordinal_eval.py` → linear-vs-ordinal robustness comparison
- `pytest tests/` → 181 tests
- Full pipeline end-to-end: < 30 seconds on a modern laptop

---

## 14. Suggested README structure (your call)

A possible top-level structure (skip / reorder as you prefer):

1. Headline finding (one-sentence + headline plot)
2. Why it matters (alpha-decay analog)
3. Data overview (1-2 paragraphs + counts table)
4. Methodology (model, loss, CV, decay)
5. Headline result (table + plot)
6. Robustness checks (ordinal, sub-population)
7. Stretch findings (role / tier / temporal)
8. What didn't work (1 paragraph; mention 1-2 of the negatives)
9. Limitations
10. Reproducibility
11. (Optional) Individual leaderboard with caveat

---

## 15. Outputs already on disk you can drop into the README

- `outputs/sweep.png` — headline half-life sweep
- `outputs/stretch_role_asymmetry.png` — hunter vs survivor
- `outputs/stretch_tier_comparison.png` — IVL/IJL/COA
- `outputs/stretch_temporal_stability.png` — pre/post 2023
- `outputs/ordinal_vs_linear.png` — robustness check
- `outputs/r2_heatmap.png` — per-fold R² (TimeSeriesSplit)
- `outputs/r2_heatmap_seasons.png` — per-fold R² (season splits)
- `outputs/regime_decay_tss_heatmap.png` — (τ_post, α) grid
- Various `findings.md` files for the negative results

---

## 16. Things to deliberately NOT include unless asked

- Detailed parsing-bug discoveries (the `"nan"` issue, the `角色`
  pre-rename quirk) — fascinating to us, irrelevant noise to a reader
- All the alternative branches (team-effects, map-effects, etc.) — pick
  1-2 max
- Per-class confusion matrix — interesting but adds a lot of bulk
- The `曾用id` tab gory details — say "I parsed the player-alias history
  table from the legacy file" and move on
- Multiple competing accuracy metrics — pick 1-2 that support your
  framing
- The full architecture diagram of files in `src/` — list paths in a
  reproducibility section
