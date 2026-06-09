# Writeup Outline — final version

Bullet-point inventory of every decision, finding, and dead-end, with
priority markers so you can decide what to cut.

Legend:
- ✅ **Definitely include** — core to the project story; the writeup is
  weaker without these
- 🟡 **Consider** — interesting, defensible, but the writeup can stand
  without them; include if you have room
- ⚪ **Skip** — implementation noise, parsing trivia, or dead ends that
  add clutter

Format recommendation: **README.md** as the primary artifact. LaTeX
is only worth doing as a secondary technical paper *if* you have time.

---

## 1. Framing / abstract

- ✅ Research question: *"How quickly does competitive skill information
  become obsolete in a high-frequency esports environment?"*
- ✅ One-sentence headline finding (R² ≈ 8 % under temporal CV, optimal
  half-life ≈ 5 months — pick your phrasing once you settle the Optuna
  number)
- ✅ Quant-trading analog: **alpha decay** — the structurally identical
  problem of measuring how fast historical information becomes stale
- ✅ Concrete subject: Identity V pro esports, ~15,700 game halves,
  2020-06-25 → 2026-05-05, six competitive tiers
- 🟡 Why this dataset specifically: 6 years, three tiers, a documented
  meta shift inside the window — natural test bed for time-decay
  modelling

---

## 2. Data

### 2.1 Sources and tiers

- ✅ Data source: 4 JSON files in `data/raw_json/` covering all
  competitive events 2020-06 → 2026-05
- ✅ Six tournament tiers, in rough order of competitive intensity:
  IVL, IJL, COA, IVS, IVT, IVC. Sample counts per tier.
- ✅ All 6 tiers included in BOTH training AND test sets — overseas
  data calibrates IVL/IJL players whom they meet at international
  events (COA)
- 🟡 Missing data: COA7 (2024 international finals) is not in any source;
  documented as a gap rather than imputed

### 2.2 Schema and processing pipeline

- 🟡 JSON → DataFrame pipeline: 4 files, one row per match (= series),
  flattened to one row per game half (~15,700 halves)
- 🟡 Tournament identifiers: derived from the 阶段 field
  (`IVL_2024_summer_regular`, `COA9`, `IVS_2025`, etc.) — tier extracted
  via prefix
- 🟡 Two-table SQLite schema: `matches` (one row per game half) and
  `players` (one row per canonical player)
- ⚪ Detailed JSON field mapping (屠名 → hunter_player, 求生者iID → survivor
  player IDs, 出门 → escape outcome, etc.)
- ⚪ Sentinel value handling (-999 / "-999" / "-" mapped to NULL)
- ⚪ Header-row auto-detection and template-row filtering for the
  legacy xlsx (no longer the data source, but earlier project work)

### 2.3 Outcome representation

- ✅ Outcome: **margin = n_escaped − 2** ∈ {−2, −1, 0, +1, +2}
- ✅ Positive = survivor advantage; negative = hunter advantage; 0 = draw
- ✅ Why margin not binary win/loss:
  - Binary throws away gradient information (4-0 sweep ≠ 3-1 narrow win)
  - Forces dropping ~38 % of data (draws)
  - Margin keeps all observations and preserves information about
    decisiveness
- 🟡 Three-tier derivation of `n_escaped`:
  1. Sum per-survivor escape results (出门 = escaped, 淘汰 = eliminated)
  2. Fallback for 2020 (which lacks per-survivor results): derive from
     winner + half-score: 0:5 → 0 escapes, 1:3 → 1 escape, 2:2 → 2 (draw)
- 🟡 Result: 100 % `n_escaped` coverage across all 15,691 game halves

### 2.4 Player ID canonicalization

- 🟡 Source: `id.json` provides 96 canonical player ↔ alias mappings,
  plus 2 manual corrections (`ppicha→pipicha`, `gua→guag`) for
  suspected fuzzy duplicates
- ⚪ Subtle bug encountered: pandas' default `na_values` includes the
  literal string `"nan"`, but a player's actual in-game ID is `nan`
  (the character 楠). Fixed by passing `keep_default_na=False`.
- ⚪ Detailed format of `id.json` (`{canonical_id: [canonical, alias_1, …]}`)

### 2.5 Testing the data layer

- 🟡 181 pytest unit + integration tests covering ingestion, alias
  normalization, and schema construction

---

## 3. Modelling decisions (in order made)

### 3.1 Outcome and structure

- ✅ Continuous margin as the modelling target (per §2.3 above)
- ✅ Asymmetric 1v4 structure:
    $$\mathbb{E}[\text{margin}_i] = \tfrac{1}{4} \sum_{k=1}^4 \beta^S_{s_{ik}} - \beta^H_{h_i}$$
- ✅ Two choices baked into this formula:
  - **Mean over 4 survivors (not sum)** — keeps the interpretation
    "1 skill unit = 1 extra escape on average" clean
  - **Survivors minus hunter** — high β always means "more skilled at
    your role"; positive on the survivor side, positive on the hunter side
- 🟡 Players indexed by `(player_id, role)` tuples → dual-role players
  (e.g. `ppxia`) get independent hunter and survivor ratings

### 3.2 Regularization choice

- ✅ L2 ridge penalty `λ ‖β‖²`
- 🟡 Why L2 over L1 / lasso:
  - Smooth shrinkage; sparse players stay in the model with small β
  - Bayesian interpretation: Gaussian prior, equivalent to portfolio
    "covariance shrinkage"
  - L1 would set sparse players to exactly 0, throwing them out entirely

### 3.3 Closed-form vs iterative optimization

- ✅ Linear model: closed-form weighted ridge via Cholesky solve on the
  symmetric positive-definite normal-equations matrix
- ✅ Ordinal model: L-BFGS-B with analytical gradient (verified by
  `scipy.optimize.check_grad`)
- 🟡 Why this matters: linear ~1 ms per fit enables wide hyperparameter
  sweeps; ordinal ~30 ms per fit is still fast enough for Optuna

### 3.4 Time-decay weighting

- ✅ Exponential decay: $w_i = (1/2)^{\Delta t_i / \tau}$
- ✅ Half-life $\tau$ has the memoryless property — relative weight
  between two matches depends only on the time between them
- ✅ Direct quant-trading analog: this is the same form as exponential
  signal-decay weighting used in factor research
- 🟡 Single-rate decay (one τ for all training matches) — earlier work
  explored piecewise decay (see §8.3) but it was retired in favour of
  the simpler model

---

## 4. Evaluation methodology

### 4.1 Temporal cross-validation

- ✅ 5-fold `TimeSeriesSplit`; train is always strictly before test
- ✅ Why **not** random k-fold: would leak future into training —
  the standard pitfall in backtesting trading signals
- ✅ Expanding window (not sliding) — time decay is the **model's** job,
  not the CV's
- 🟡 First ~1/6 of data never used as test (price of pure temporal CV)

### 4.2 Alternative split strategy: season-aligned folds

- 🟡 6 folds = 6 competitive years; 5 expanding-window splits
- 🟡 Each test fold = exactly one season (IVL summer + fall + IJL +
  COA finals for that year)
- 🟡 Motivation: TimeSeriesSplit mid-season cuts cause cold-start
  artifacts in some folds; season splits give more interpretable
  per-period numbers
- 🟡 Result: mean R² is slightly lower than TimeSeriesSplit because
  season splits don't smooth over mid-season-cut wins. Honest read of
  per-period model quality.
- ⚪ Available as `split_strategy="seasons"` in `eval.py`

### 4.3 Loss / metric: RMSE on margin

- ✅ Why RMSE: consistent with the squared-error loss the model minimises
- ✅ Defensibility: using a different metric for evaluation than for
  optimization invites tuning to a metric the model isn't trying to
  hit. RMSE avoids this.
- ✅ Null baseline: predict $\hat{y} = 0$ for everyone → RMSE ≈ 1.15.
  A useful model must beat that.
- ✅ R² = 1 − MSE_model / MSE_null is the headline scalar

### 4.4 Cold-start handling

- 🟡 Players unseen in training implicitly get β = 0 (the L2 prior)
- 🟡 Not dropped from evaluation — honest reporting includes them
- 🟡 Cold-start rates: ~10-30 % per fold, lower than the earlier xlsx
  dataset because the JSON has more training matches per player

### 4.5 Hyperparameter optimization via Optuna

- ✅ Optuna's TPE (Tree-structured Parzen Estimator) sampler used to
  jointly optimise τ, λ, threshold, and model class
- ✅ Why Optuna rather than grid search: 4-5 hyperparameters with
  mixed continuous + integer + categorical types; TPE converges in
  ~50-100 trials vs ~thousands for a comparable grid
- 🟡 Search ranges: τ log-spaced 30-1000d, λ log-spaced 0.1-10, threshold
  integer 0-20, model ∈ {linear, ordinal}
- 🟡 Direct quant-trading analog: hyperparameter optimization is what
  you'd do over signal decay rates and shrinkage strengths in a real
  factor model — same problem, same tools (Bayesian optimisation,
  random search, evolutionary algorithms)

---

## 5. The headline finding

- ✅ Model: **ordinal proportional-odds Bradley-Terry with single-rate
  exponential decay and an informed new-player team-mean prior**
- ✅ Headline R² (fill in the Optuna-optimised number; the previous
  non-Optuna best was 8.04 %)
- ✅ Optimal τ ≈ 110-140 days (~4-5 months) under the optimiser
- ✅ Same order of magnitude across every sub-population sweep
  (role / tier / temporal stretch analyses, §6)
- ✅ Headline plot: `outputs/sweep.png` — half-life vs RMSE curve
- 🟡 Quant framing: "skill information half-life of about 4-5 months"
  in the same way alpha decay is reported in trading

---

## 6. Stretch analyses (cross-population sweeps)

### 6.1 Role asymmetry — finding contradicted the hypothesis

- ✅ Separate hunter-only and survivor-only sweeps:
  - Hunter optimal τ ≈ 210 days
  - Survivor optimal τ ≈ 57 days
- ✅ Hypothesis (from project handoff): hunter τ would be **shorter**
  because characters get patched
- ✅ Result: hunter τ is ~3.7× **longer** than survivor — the
  hypothesis was **inverted**
- ✅ Plausible reason: pro hunters main 1-3 characters consistently
  (character mastery dominates and is stable); survivor performance
  depends more on shifting team composition and meta
- ✅ Crucial caveat: I also tested a dual-τ joint model with
  role-specific decay rates. RMSE improvement over single-τ = 0.001
  (well within ±0.02 SE). **The asymmetry is informative but not
  exploitable for forecasting** — classic "I learned about the system
  but it doesn't beat the null baseline."

### 6.2 Tier comparison

- 🟡 IVL well-identified at τ ≈ 110 d (6,877 matches — largest sample)
- 🟡 IJL τ ≈ 324 d (1,045 matches)
- 🟡 COA τ ≈ 324 d (1,180 matches)
- 🟡 IJL/COA results are noise-limited — wide SEs, flat curves; the
  identical 324d optimum is a grid-point artifact, not a real finding
- 🟡 Honest framing: IVL is the real finding; IJL/COA are bounded
  below ~100 d but not trustworthy beyond that

### 6.3 Temporal stability — meta sped up after 2023

- ✅ Pre-2023 optimal τ ≈ 110 d; post-2023 optimal τ ≈ 71 d
- ✅ Post-2023 skill information goes stale **~35 % faster**
- ✅ Cleanest of the three stretch analyses — well-separated curves,
  comparable SEs

---

## 7. The 2023 IVL meta shift (the diagnostic story)

A standalone section worth its own slot:

- ✅ Fold 3 of the temporal CV (test ≈ 2023) had R² of ~0 % or
  negative — every model variant struggled
- ✅ Diagnosis: it's not a model bug, it's a real structural break in
  the data:
  | Year | IVL draw rate | IVL hunter win rate (decisive) |
  |---|---:|---:|
  | 2020-2022 | ~35 % | 38-44 % |
  | 2023 | **42 %** (+6 pp) | 48 % |
  | 2024 | 41 % | 61 % |
  | 2025 | 42 % | **67 %** |
- ✅ Two simultaneous things happened in IVL starting 2023:
  - **Draw rate jumped +6 pp** — more games end 2-2
  - **Hunters got much stronger** — win rate climbed from 38 % to 67 %
- ✅ Why the model struggles on this fold specifically:
  1. **Higher draw rate → harder null baseline.** Predicting 0 every
     time becomes more accurate, so the bar to beat the null is higher.
  2. **Distributional shift the model can't anticipate.** Trained on
     2020-2022 where outcomes were balanced; the 2023+ distribution is
     systematically more hunter-favoured.
- ✅ IVL-specific: COA stayed at 30-35 % draws throughout; IJL only
  started in 2024 (so its training already reflects the new meta)
- ✅ Direct quant analog: this is exactly a **regime change** in
  trading — a structural break that no amount of time-weighting can
  fully recover from, because the information about the new regime
  simply isn't in the pre-2023 training data

---

## 8. Robustness checks and refinements

### 8.1 Ordinal regression as robustness check

- ✅ Proportional-odds Bradley-Terry with cumulative-logit thresholds
- ✅ Confirms the headline finding: τ shifts modestly (~110 → ~136 d)
  but the qualitative answer survives
- ✅ Learned threshold spacings: 0.77 / 1.77 / 2.47 — the linear
  model's equal-spacing assumption was wrong (n=3→n=4 transition
  requires ~3.2× more skill than n=0→n=1)
- ✅ R² improvement: +1.2 pp over linear baseline
- 🟡 Ordinal is the chosen final model class because of this
  robustness check

### 8.2 New-player team-mean prior

- ✅ **Targeted** informed prior on cold-start players — players with
  fewer than `threshold` (default 5) training matches get their L2
  prior centre set at the **team mean** β of their first-match
  teammates, rather than at 0
- ✅ Rationale: strong teams recruit talented players, so a new
  player's expected skill is closer to their team's average than to
  the population average
- ✅ Two-pass procedure: fit standard model → identify new players →
  compute team-mean priors → refit with per-player priors
- ✅ Improvement: +0.1 pp pooled R², but **concentrated in fold 3**
  (the meta-shift fold) where R² flips from negative to positive
- 🟡 Hyperparameter `threshold = 5` is the sweet spot; threshold = 0
  reduces to standard L2

### 8.3 Regime decay (explored and retired)

- 🟡 Piecewise decay with separate (τ_pre, τ_post) and a multiplicative
  pre-shift discount α, motivated by the 2023 meta shift
- 🟡 Gave +1.0 pp R² in earlier work, but largely overlaps with the
  new-player prior
- 🟡 **Retired** in favour of the simpler single-rate decay — Optuna
  with single-rate plus the new-player prior matches it without the
  extra parameters
- 🟡 Worth mentioning as: *"I explored piecewise time-decay with a
  multiplicative information discount at the regime boundary —
  mirroring how systematic traders handle known structural breaks in
  alpha decay. The gain overlapped with simpler informed-prior
  approaches, so the final model uses single-rate decay."*

---

## 9. Things tried that didn't help — keep or drop per taste

(Each defensible as "I tested it and it didn't pan out." Mention 1-3
max, otherwise the writeup feels like a catalogue of negatives.)

### 9.1 Team random-effects intercepts

- 🟡 One γ_T intercept per unique survivor team composition
- 🟡 Fixed the collinearity symptom (koting / persica artifact) but
  damped legitimate individual signal (e.g. huan dropped from +0.98
  to +0.42) and gave zero OOS improvement
- 🟡 Branch: `team-effects` (not merged) — `outputs/team_effects_findings.md`

### 9.2 Map fixed effects

- 🟡 Pulled `map_name` into the matches table; added per-map γ_m
  intercepts
- 🟡 Map effects are **real** (Spearman ρ = 0.97 between learned γ
  and raw per-map margins) but **don't help OOS prediction** — every
  τ in the sweep was 0.002-0.003 RMSE worse than baseline
- 🟡 Reasons: cold-start maps in test folds; confounding with team
  identity (map choice is part of the draft); L2 rebalancing of
  overall bias
- 🟡 Branch: `map-effects` (not merged)

### 9.3 Team-composition model (per-team rating, not per-individual)

- 🟡 One β^T per unique 4-survivor team composition; hunters individual
- 🟡 Severe cold-start: 55.6 % of CV test rows feature a never-seen team
- 🟡 Loses to baseline by 0.007 RMSE
- 🟡 Branch: `team-comp` (not merged)

### 9.4 Naive rolling-30 baseline

- 🟡 "Predict each match by summing player's K=30-match rolling-average
  margin"
- 🟡 Under strict CV: **K=30 → worse than null model**, K=500 ≈ matches
  BT but with intra-fold leakage
- 🟡 Strong sanity check: shows BT actually earns its accuracy through
  opponent adjustment + shrinkage, not by trivially capturing recent
  results
- 🟡 Branch: `naive-rolling` (not merged)

### 9.5 Universal team-anchored prior (Laplacian)

- 🟡 Laplacian penalty: pulls every survivor toward their teammate mean
- 🟡 +0.6 pp R² standalone, but heavy overlap with simpler informed-prior
  approaches
- 🟡 Replaced by the targeted new-player-only prior (§8.2)

### 9.6 Joint dual-τ model (separate τ for hunter and survivor)

- 🟡 Asymmetric normal equations; closed-form solve
- 🟡 ΔRMSE = 0.0013 over single-τ — well within ±0.02 SE
- 🟡 Already mentioned in §6.1 as the "informative but not exploitable"
  caveat to role asymmetry

### 9.7 Difference-penalty model for collinear pairs

- 🟡 Targeted L2 difference penalty between players who share ≥ 90 %
  of matches and have ≥ 8 games each
- 🟡 Doesn't change the headline R² but cleans up the **individual
  leaderboard** (resolves the koting / persica artifact specifically)
- 🟡 Used to produce `outputs/db_export/ratings.csv` — separate from
  the main model

---

## 10. Limitations

- ✅ **Fold-3 R² remains weak** even with regime corrections — the
  2023 IVL meta shift is genuinely hard to predict from pre-2023 data
- ✅ **R² of ~8 % sounds modest in absolute terms** — but is good for
  forecasting individual game-half outcomes under temporal CV. xG
  models for football match outcomes typically achieve 5-15 % under
  similar protocols.
- 🟡 **Equal-spacing approximation in the linear model** — relaxed by
  ordinal regression as a robustness check; final model uses ordinal
- 🟡 **Cold-start matches** (~10-30 % per fold) defaulted to the L2 prior
- 🟡 **IJL/COA half-lives are wide-SE estimates** — only IVL is
  precisely identified
- 🟡 **5 folds gives limited statistical power** — paired tests
  between model variants often have p > 0.1 even when point estimates
  improve meaningfully
- ⚪ Detail-level limitations (2020 data lacks per-survivor results,
  shift_date is hand-set, etc.)

---

## 11. Quant framing — interview talking points to weave in naturally

Each is a one-liner you can say in an interview without sounding rehearsed:

- ✅ *"I implemented time-weighted ridge regression from scratch in
  closed form, with cold-start regularisation that's mathematically
  equivalent to a Gaussian prior — the standard shrinkage formulation."*
- ✅ *"I used `TimeSeriesSplit` cross-validation to avoid lookahead
  bias — random k-fold would leak future matches into training, which
  is the standard pitfall in backtesting trading signals."*
- ✅ *"The half-life parameter is structurally analogous to alpha decay
  in systematic trading — measuring how fast historical information
  becomes stale."*
- ✅ *"I used Bayesian hyperparameter optimisation (Optuna's TPE
  sampler) to jointly tune the decay rate, regularisation strength,
  and the new-player prior threshold — much more efficient than grid
  search for 4+ mixed-type hyperparameters."*
- 🟡 *"L2 regularization on player ratings is equivalent to a Gaussian
  prior; for sparse players it provides shrinkage analogous to the
  covariance shrinkage used in portfolio construction."*
- 🟡 *"I found a real structural asymmetry — hunter information persists
  3× longer than survivor information — but a joint model exploiting
  that asymmetry gave no significant RMSE improvement. Worth
  understanding the system, but not actionable for forecasting."*
- 🟡 *"Fold 3 (the 2023 IVL test season) is the worst fold across every
  variant I tried. Diagnosed as a regime change — draw rate jumped
  +6 pp and hunter win rate climbed from 38 % to 67 % over 3 years.
  Pre-shift training data doesn't contain the information needed to
  predict post-shift outcomes — analogous to a trading signal whose
  alpha decays during a structural break."*
- 🟡 *"The model's confident predictions are accurate (43 % recall on
  predicted survivor wins vs 25 % empirical base rate), but L2 ridge
  correctly recognizes the per-match signal-to-noise ratio doesn't
  support extreme predictions — so the model rarely guesses ±2."*

---

## 12. Methodological choices worth being ready to defend

- 🟡 Why margin not binary (and what ordinal regression contributes
  as a robustness check)
- 🟡 Why closed-form ridge for linear + L-BFGS-B for ordinal (compute
  budget for Optuna)
- 🟡 Why difference penalty for the individual leaderboard but
  single-rate decay + new-player prior for the headline model
- 🟡 Why escape count over winner_side (finer signal, recovers draws)
- 🟡 Why expanding-window CV not sliding (model τ handles forgetting)
- 🟡 Why Optuna over grid search (smarter sampling, handles mixed
  parameter types)
- 🟡 Why include overseas tiers in training (calibration of IVL/IJL
  stars via cross-tier matches at COA)

---

## 13. Reproducibility

- ✅ `python src/db.py` → builds `data/processed/idv.db` from JSON files
- ✅ `python src/eval.py` → runs the headline half-life sweep
- ✅ `python src/optimize.py` → runs Optuna hyperparameter search
- 🟡 `python src/stretch.py` → role / tier / temporal sub-analyses
- 🟡 `python src/ordinal_eval.py` → linear-vs-ordinal robustness
- 🟡 `pytest tests/` → 181 tests
- 🟡 Full pipeline end-to-end: < 30 seconds on a modern laptop (linear);
  ~10-15 min for the Optuna ordinal sweep

---

## 14. Suggested README structure

A possible top-level structure (reorder as you prefer):

1. ✅ Headline finding (one sentence + headline plot)
2. ✅ Why it matters (alpha-decay analog, structural break framing)
3. ✅ Data overview (1-2 paragraphs + counts table)
4. ✅ Methodology (model class, loss, CV, time decay, Optuna)
5. ✅ Headline result (table + plot)
6. ✅ Stretch findings (role / tier / temporal — each with its plot)
7. ✅ The 2023 meta shift (its own section — strong narrative)
8. 🟡 Robustness checks (ordinal, naive baseline floor)
9. 🟡 What didn't work (1 short paragraph; pick 1-2)
10. ✅ Limitations
11. ✅ Reproducibility
12. 🟡 (Optional) Individual leaderboard with the diff-penalty caveat

---

## 15. Outputs already on disk you can embed

- ✅ `outputs/sweep.png` — headline half-life sweep
- 🟡 `outputs/stretch_role_asymmetry.png` — hunter vs survivor
- 🟡 `outputs/stretch_tier_comparison.png` — IVL / IJL / COA
- 🟡 `outputs/stretch_temporal_stability.png` — pre- vs post-2023
- 🟡 `outputs/ordinal_vs_linear.png` — robustness check
- 🟡 `outputs/r2_heatmap.png` — per-fold R² (TimeSeriesSplit)
- 🟡 `outputs/r2_heatmap_seasons.png` — per-fold R² (season splits)
- ⚪ `outputs/regime_decay_tss_heatmap.png` — old regime-decay sweep
  (only include if you discuss regime decay)
- ✅ The findings docs in `outputs/*_findings.md` for citation depth

---

## 16. Things to deliberately NOT include unless asked

- ⚪ Parsing-bug discoveries (the `"nan"` issue, the `角色` pre-rename
  quirk, header-row detection) — fascinating to us, irrelevant noise
  to a reader
- ⚪ All seven alternative branches at once — pick 1-2 maximum
- ⚪ Per-class confusion matrix — interesting but bulky; replace with
  one accuracy number + one calibration claim
- ⚪ The `曾用id` Excel-tab parsing — superseded by `id.json`; mention
  the alias map exists and move on
- ⚪ Multiple competing accuracy metrics — pick 1-2 that support your
  framing
- ⚪ Full architecture diagram of `src/` files — list paths in §13
- ⚪ Detailed COA edition / year mapping logic
- ⚪ Database schema DDL in the README (link to `src/db.py` instead)

---

## 17. Sanity check on the "headline" number

Once you've run Optuna on the final config (ordinal + single-rate
decay + new-player prior, all tiers, TimeSeriesSplit), make sure
your headline matches what you actually got. The previous-best on
xlsx data was R² = 8.04 %; with JSON data and the all-tiers test
set, expect 7.5-8.5 % depending on Optuna's exact landing spot.

- ✅ Always report the **pooled R²** (computed from pooled MSE
  across folds), not the mean of per-fold R²s, in the headline.
  They differ slightly when folds have different null variances.
- ✅ Always mention the **null baseline** for context — "8 % may
  sound modest, but it's measured against a null model that already
  gets ~38 % of matches right by always predicting a draw"
- 🟡 Per-fold R²s reveal fold-3 weakness honestly; either include the
  per-fold table or summarise as a range ("R² ranges from −1 % to
  +18 % across folds")
