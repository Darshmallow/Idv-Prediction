# Ordinal Regression Findings

Proportional-odds (ordinal logistic) Bradley-Terry margin model, replacing
the linear margin model's equal-spacing assumption.

## The case for ordinal

Under the linear model, the skill gap to go from 3 → 4 escapes is assumed
to be the same as 0 → 1.  Fit on the actual data, the proportional-odds
thresholds tell a different story:

| Transition | Spacing Δθ |
|------------|------------|
| n=0 → n=1  | 0.77 |
| n=1 → n=2  | 1.77 |
| n=2 → n=3  | 2.47 |

Getting that 4th escape requires roughly **3.2× more skill advantage**
than getting the 1st. The linear model can't represent this; the ordinal
model learns it from data.

## Headline result

5-fold temporal CV on the full 9 100-match dataset (same protocol as the
linear sweep):

| Model    | τ* (days) | RMSE at τ* | improvement over null |
|----------|----------:|-----------:|----------------------:|
| Linear   |  110      | 1.10170 ± 0.021 | 3.36 % |
| **Ordinal** |  **136**  | **1.09441 ± 0.023** | **4.00 %** |

**Ordinal gain over linear: 0.0073 RMSE**, which is roughly **19 % of the
improvement linear already achieved over the null model.**  The optimal
half-life shifts slightly longer (136 d ≈ 4.5 months vs linear's 110 d ≈
3.6 months) but stays in the same order of magnitude.

## Paired test (per fold)

| Fold | Linear (τ*=110d) | Ordinal (τ*=136d) | Δ |
|-----:|----------------:|------------------:|----:|
| 1 | 1.13606 | 1.12356 | −0.013 |
| 2 | 1.07063 | 1.06698 | −0.004 |
| 3 | 1.15341 | 1.16100 | +0.008 |
| 4 | 1.11108 | 1.09279 | −0.018 |
| 5 | 1.03732 | 1.02773 | −0.010 |

Ordinal beats linear in 4 of 5 folds.  Paired t-test: t = −1.65,
p = 0.17 (two-sided).  Not significant at p < 0.05 with n = 5, but the
direction is consistent and the magnitude is meaningful.

## Stronger evidence: dominance across the τ grid

| | Linear | Ordinal |
|---|---|---|
| Best RMSE (across grid) | 1.1017 | 1.0944 |
| Worst RMSE (across grid) | 1.1120 | 1.0997 |

**Ordinal's *worst* point (1.0997) is still better than linear's *best*
point (1.1017).** The improvement is structural — driven by the model
class, not by the τ choice. The equal-spacing assumption was costing
genuine predictive accuracy at every half-life.

## Computational cost

|  | Linear | Ordinal |
|---|---|---|
| Solve type | closed-form ridge | L-BFGS-B (50–140 iters) |
| One fit | < 0.1 s | 0.3–0.8 s |
| Full sweep | 2 s | 10 s |

Five times slower but still fast — entire sweep finishes in 10 s.

## Interview framing

> "The headline linear margin model assumes the skill gap from 3 to 4
> escapes equals the gap from 0 to 1.  Fit on the data, ordinal regression
> learns that the 3→4 gap is 3× larger.  Switching to ordinal recovers
> about 19 % of the predictive improvement that the linear model already
> made over the no-skill baseline — a structural gain that holds across
> every half-life value, not a τ-tuning artifact.  The optimal half-life
> shifts only slightly (110 → 136 days), so the qualitative headline is
> robust to the modelling choice."

## Caveats

- p = 0.17 with n = 5 folds is **directionally consistent but not
  formally significant**. A larger CV (10 folds) might tighten this; the
  paired SE is 0.0044 and the effect is 0.0073, so n = 10 would put us
  near p ≈ 0.05.
- The optimal τ has shifted modestly (110 → 136 days).  Sub-analyses
  (role / tier / temporal) might also shift; not re-run here to keep this
  branch focused on the model-class comparison.
- L-BFGS-B occasionally takes 100+ iterations to converge — initial
  thresholds via empirical quantiles + warm-starting θ across folds
  keeps this manageable but a smarter optimiser (Newton-CG with explicit
  Hessian) could speed it further if needed.
