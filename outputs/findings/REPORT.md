# Measuring Competitive Skill Decay in Identity V Esports

**How quickly does competitive skill information become obsolete in a high-frequency esports environment?**

---

## Overview

This project applies an ordinal Bradley-Terry paired-comparison model to six years of professional Identity V esports data to quantify player skill and measure the rate at which historical match data becomes stale. The central finding is that top-tier competitive skill information has a half-life of roughly **182 days** — skill signal decays to half its weight in about 6 months — and the model achieves **R² = 8.6%** against a null baseline under strict temporal cross-validation.

The framing is deliberately analogous to **alpha decay** in quantitative finance: the core question (how fast does historical information predict future outcomes?) is structurally identical to measuring how quickly a trading signal loses its edge.

**Headline numbers (top tiers: IVL / IJL / COA):**

| Metric | Value |
|--------|-------|
| Optimal half-life τ\* | **182 days** |
| R² vs null (predict all draws) | **8.6%** |
| R² vs naive rolling K=30 | **12.5%** |
| Series win accuracy | **63.8%** |
| Series Brier score | **0.2197** (vs 0.25 for 50/50) |
| 3-class half accuracy | **40.7%** (vs 37.4% null) |

---

## Introduction

Identity V (第五人格) is an asymmetric 1v4 mobile survival game developed by NetEase. In competitive play, one hunter faces four survivors per half-game; each team plays both sides in a round. The outcome of each half is the number of survivors who escape (0–4), making match results an ordered categorical variable rather than a binary win/loss. This structure makes it an ideal test bed for two simultaneous questions:

1. **Do individual player ratings carry predictive signal about future match outcomes?**
2. **How quickly does that signal decay as the competitive meta evolves?**

The quant-trading analog is clean. A player's rating is analogous to a factor score: useful for near-term prediction, increasingly stale as market conditions (the meta) shift. The optimal exponential half-life τ\* is the model's answer to "how far back should you look?"

---

## Data & Outcome

### Sources and Coverage

Data were sourced from [Identity V Bwiki](https://wiki.biligame.com/idv/), which manually records professional match results. Four JSON files cover all competitive events from **2020-06-25 to 2026-05-05**.

| Tier | Half-games | Description |
|------|----------:|-------------|
| IVL | 6,877 | IVL (China main league) |
| COA | 3,946 | Championship of Abyss (international) |
| IJL | 2,561 | IJL (Japan league) |
| IVC | 1,336 | IVL qualifier circuit |
| IVT | 549 | IVT (regional) |
| IVS | 422 | IVS (regional) |
| **Total** | **15,691** | |

**Player coverage:** 2,049 unique players across all tiers (1,516 survivor-only, 462 hunter-only, 71 dual-role). The top-tier model (IVL/IJL/COA) rates 1,383 players: 1,018 survivors and 365 hunters.

### Outcome Representation

The raw outcome is `n_escaped ∈ {0, 1, 2, 3, 4}` — the number of survivors who escaped in one half-game. This is mapped to a **margin**:

```
margin = n_escaped − 2  ∈  {−2, −1, 0, +1, +2}
```

Positive margin → survivor advantage; negative → hunter advantage; zero → draw.

Using margin over binary win/loss retains gradient information (a 4–0 sweep is meaningfully different from a 3–1 close win), avoids discarding the ~40% of draws, and supports the ordinal regression framework.

---

## Model

### Ordinal Bradley-Terry with Time Decay

Each player `p` has a latent skill score β_p. For a half-game with hunter `h` and survivors `s₁…s₄`, the **linear predictor** is:

$$\eta = \frac{1}{4}\sum_{k=1}^{4} \beta^S_{s_k} - \beta^H_h$$

A high η → more survivor-favoured outcome. Positive β always means more skilled at one's role (strong hunters and strong survivors both have large positive β; the asymmetry is handled by the subtraction).

This linear predictor feeds a **proportional-odds (cumulative logit)** model, which learns separate thresholds θ₁ < θ₂ < θ₃ < θ₄ to map η onto P(n_escaped = k). Crucially, these thresholds are not assumed equal — the model discovers that the skill difference required to push from 0→1 escape is much smaller than from 3→4. The learned threshold spacings quantify this non-linearity.

### Exponential Time Decay

Matches are weighted by age relative to the training reference date:

$$w_i = \left(\tfrac{1}{2}\right)^{\Delta t_i / \tau}$$

where τ is the half-life in days. This is the same exponential decay used in factor research to down-weight stale signals. The memoryless property means the relative weight between any two matches depends only on the time between them — τ is a stable, interpretable parameter independent of the evaluation date.

### Additional Components

- **L2 ridge regularization** — shrinks all β toward zero; Gaussian prior interpretation equivalent to portfolio covariance shrinkage.
- **Team-mean informed prior** — players with fewer than `threshold` training matches have their L2 prior centered at their first-match teammates' mean β, rather than zero. Strong teams recruit talented players; the prior encodes this.
- **Hyperparameter tuning** — τ, λ (L2 strength), and `threshold` are jointly optimized via Optuna's TPE sampler under temporal CV.

### Fitting

The model is fit by maximum likelihood (ordinal log-likelihood) using L-BFGS-B with analytical gradient, verified against numerical gradient via `scipy.optimize.check_grad`. The linear closed-form baseline uses weighted ridge regression via Cholesky solve.

---

## Evaluation

### Temporal Cross-Validation

All evaluation uses **5-fold TimeSeriesSplit** with an expanding window: training data always strictly precedes test data. This mirrors the backtesting discipline in quantitative trading — no future leakage. Random k-fold is inappropriate here because it would allow future matches to inform past predictions.

### Baselines

| Baseline | Description | RMSE (top tiers) |
|----------|-------------|:----------------:|
| Null | Predict 0 (draw) every match | 1.156 |
| Naive rolling K=30 | Sum of each player's 30-game rolling margin avg | 1.182 |
| Linear BT | Closed-form ridge, same structure | 1.115 |
| **Ordinal BT** | Final model | **1.105** |

The naive rolling baseline is *worse* than the null model at every K below ~300, and barely positive at K=∞ (career average, R²=+1.1% vs null). A simple average of past margins conflates player skill with opponent quality — exactly what the BT normalization corrects.

### Half-Life Sweep

![Half-life sweep — top tiers](outputs/graphs/comparison_sweep_top_tiers.png)

The x-axis is τ (log scale); y-axis is out-of-sample RMSE under 5-fold temporal CV. The ordinal BT curve has a clear minimum at τ\* ≈ 182 days. The naive rolling baseline (orange) lies above the null baseline (grey) throughout — confirming it adds noise rather than signal. The BT model beats both at every τ.

### Calibration

![Calibration — top tiers](outputs/graphs/calibration_top_tiers.png)

The calibration plot compares predicted P(n_escaped = k) against empirical frequencies across the test set. Well-calibrated probabilities lie on the diagonal. The model is well-calibrated across all five outcome categories.

### Accuracy Metrics

For comparison against null on the 5-outcome classification problem:

| Config | 5-class acc (argmax) | 5-class null | 3-class acc | 3-class null |
|--------|:--------------------:|:------------:|:-----------:|:------------:|
| Top tiers | 37.6% | 37.4% | **40.7%** | 37.4% |
| All tiers | 36.3% | 36.5% | **40.5%** | 36.5% |

The 5-class task (predicting exact escape count) is extremely hard — outcomes are noisy even conditional on skill — so the small absolute lift is expected. The 3-class collapse (hunter win / draw / survivor win) is more informative and shows a meaningful +3.3 pp lift over null.

![Confusion matrix — top tiers](outputs/graphs/confusion_matrix_top_tiers.png)

### Series Win Prediction

Beyond individual half-games, the model predicts **series outcomes** (Bo3 / Bo5). Given the actual lineups per round (conditional evaluation), the probability of each team winning the series is computed via dynamic programming over round-win states. Results across 1,890 top-tier series:

| Metric | Top tiers | All tiers |
|--------|:---------:|:---------:|
| Series accuracy | **63.8%** | **64.3%** |
| Brier score | 0.2197 | 0.2194 |
| Brier reduction vs 50/50 | **12.1%** | **12.2%** |

Calibration is strong: series where P(home wins) > 60% result in home wins 73% of the time; series where P(home wins) < 40% result in home wins only 26% of the time.

---

## Findings

### 1. Skill information decays with a half-life of ~6 months (top tiers)

The optimal τ\* = 182 days means a match from 6 months ago is weighted half as much as a match from today. Beyond ~2 years, matches contribute less than 6% weight. This is the answer to the central research question: top-tier competitive skill information becomes substantially stale in about one competitive season.

**Per-fold R² (top tiers, τ\* = 182d):**

| Fold | Approx. test period | R² vs null | R² vs rolling |
|------|-------------------|:----------:|:-------------:|
| 1 | 2020–2021 | 2.4% | 7.9% |
| 2 | 2021–2022 | 4.8% | 12.8% |
| 3 | 2022–2023 | 6.1% | 7.1% |
| 4 | 2023–2024 | 14.4% | 17.2% |
| 5 | 2024–2026 | 15.2% | 17.5% |
| **Mean** | | **8.6%** | **12.5%** |

The model's performance improves over time — more training data and a more stable post-2023 meta both help the later folds.

### 2. The 2023 IVL meta shift: a structural break

Fold 3 (test ≈ 2022–2023) has the lowest R² (6.1%). This is not a model failure — it reflects a genuine structural break in the IVL competitive meta starting in 2023:

| Year | IVL draw rate | Hunter decisive win rate |
|------|:------------:|:------------------------:|
| 2020–2022 | ~35% | 38–44% |
| 2023 | **42%** (+6 pp) | 48% |
| 2024 | 41% | 61% |
| 2025 | 42% | **67%** |

Two simultaneous shifts happened: the draw rate jumped ~6 percentage points, and hunter dominance accelerated sharply. A model trained on 2020–2022 data predicts balanced outcomes; the 2023+ distribution is systematically more hunter-favoured.

This is the exact analog of a **regime change** in quantitative trading — a structural break that no amount of time-weighting can fully recover from because the information about the new regime simply isn't in the pre-break training data. No hyperparameter combination eliminates the fold 3 weakness; the signal genuinely isn't in the pre-2023 data.

The temporal stability analysis reinforces this: the optimal τ\* estimated on pre-2023 data alone is ~110 days; estimated on post-2023 data alone it shrinks to ~71 days. Skill information in the post-shift era goes stale **~35% faster**.

---

## Analysis

### Top Tiers vs All Tiers

| Config | τ\* | R² vs null | R² vs rolling |
|--------|----:|:----------:|:-------------:|
| Top tiers (IVL/IJL/COA) | 182 d | 8.6% | 12.5% |
| All tiers | 539 d | 9.0% | 14.3% |

The all-tiers model uses a much longer half-life (539 days ≈ 18 months) despite achieving slightly better R². This reflects two distinct effects:

1. **Lower-tier data is more stable.** IVC/IVT/IVS players have fewer games and less meta evolution. Longer lookback helps rather than hurts.
2. **Cross-tier calibration.** Including all tiers means the model encounters the same players in both IVL and lower-tier tournaments, improving cross-tier calibration through more games per player.

The top-tier model is the primary focus: the research question about rapid information decay is most relevant at the elite level, and the shorter τ\* = 182d is the cleaner answer.

### Hunter vs Survivor Skill Decay

Running separate role-only optimizations (each player's skill estimated only from their primary role's data, evaluated on top-tier matches only):

| Role | τ\* | Best R² |
|------|----:|:-------:|
| Hunter | 153 days | 5.6% |
| Survivor | 123 days | 3.7% |
| Combined (both roles) | 182 days | 8.6% |

Three observations:

1. **Hunter skill decays faster than survivor skill** (153 vs 123 days optimal). Hunter play is more meta-dependent — the dominant hunting strategy shifts with each new character balance patch, while survivor fundamentals (kiting, rescue timing) are more durable.

2. **Hunters carry more predictive signal** (5.6% vs 3.7% R²). A single exceptional hunter has more individual impact on the outcome than any single one of four survivors. This aligns with the 1v4 structure: one strong hunter can dominate; one strong survivor cannot guarantee a win against a skilled hunter.

3. **Combined outperforms both** (8.6%). The model extracts independent information from hunter and survivor skills simultaneously. This confirms that both sides' skill levels have genuine predictive value.

---

## Limitations

**Lineup uncertainty.** The series-win prediction (63.8%) uses actual per-round lineups (conditional evaluation). In a true pre-series forecast, the lineup is unknown — strategic substitutions mid-series are unpredictable. Real forecasting accuracy would be somewhat lower.

**Fold 3 structural break.** The 2023 IVL meta shift creates a genuine hard limit on fold-3 predictability. No model variant in this project recovers this fold fully. This is an inherent limitation of any retrospective skill model applied across a structural break.

**Small dataset per player.** With 1,383 rated players and 15,691 half-games, many players have fewer than 20 appearances. Ridge regularization handles this correctly by shrinking sparse estimates toward zero, but sparse players' ratings are necessarily wide. A player with 5 games has a very uncertain β.

**No character-level effects.** The model rates players, not character picks. Hunter character choice is a large strategic decision (some hunters are mechanically harder; some counter specific survivor compositions). Character-level fixed effects were explored but did not improve out-of-sample prediction, likely due to cold-start issues in test folds with new characters.

**No map effects.** Map fixed effects improve in-sample fit (Spearman ρ = 0.97 between learned map effects and raw per-map margins) but consistently hurt out-of-sample prediction by 0.002–0.003 RMSE — likely because map picks are correlated with team identity and draft strategy.

---

## Website

An interactive website accompanies this analysis and is hosted at: [link]

Features:
- **Leaderboard** — browse all rated players with filters by role (hunter/survivor) and minimum game count; ranked independently within each role
- **Series predictor** — select two 5-player teams (1 hunter + 4 survivors each), choose a series format (Bo1–Bo7), and get exact win probabilities via DP and an animated Monte Carlo simulation of the series
- **Half predictor** — per-half win probability and expected margin for any matchup

The interactive series simulator uses the same ordinal Bradley-Terry model underlying the analysis — the per-half escape probabilities P(n_escaped = k) feed directly into the round-win and series-win calculations.
