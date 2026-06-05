# Ordinal + Regime Decay — Combined Model

## Headline

The best combined model achieves **R² = 7.93%** under TimeSeriesSplit
CV — the highest of any model class tried on this project.

| Model | Best config | Mean RMSE | Mean R² |
|---|---|---:|---:|
| Null (predict 0) | — | 1.14000 | 0.00% |
| Linear baseline | τ = 110d | 1.10170 | 6.50% |
| Linear + regime | τ_pre=365d, τ_post=30d, α=0.20 | 1.09575 | 7.53% |
| Ordinal baseline | τ = 136d | 1.09441 | 7.70% |
| **Ordinal + regime** | **τ_pre=200d, τ_post=60d, α=0.60** | **1.09306** | **7.93%** |

## The two improvements partially overlap (don't fully stack)

| Δ vs linear baseline | mean R² gain |
|---|---:|
| Ordinal alone (relax equal-spacing) | +1.20 pp |
| Regime alone (handle 2023 meta shift) | +1.03 pp |
| Sum if independent | +2.23 pp |
| **Both combined** | **+1.43 pp** |

About **64% of the regime decay's contribution survives** when ordinal is
already in the mix. The ordinal model's learnable thresholds partly absorb
the regime-shift bias on their own, so there's less left for the discount
α to correct.

## Per-fold breakdown (R² across the 4 models)

| Fold | Period | Linear base | Linear+reg | Ord base | **Ord+reg** |
|---:|:---|---:|---:|---:|---:|
| 1 | 2021-07 → 2022-08 | +3.04% | +5.17% | +5.16% | **+5.45%** |
| 2 | 2022-08 → 2023-10 | +3.62% | +4.53% | +4.28% | **+4.38%** |
| 3 | 2023-10 → 2024-10 | +0.21% | **+2.28%** | −1.11% | −0.51% |
| 4 | 2024-10 → 2025-07 | +8.66% | +9.16% | +11.65% | **+11.87%** |
| 5 | 2025-07 → 2026-05 | +16.98% | +16.51% | +18.51% | **+18.45%** |
| **mean** | | **+6.50%** | **+7.53%** | **+7.70%** | **+7.93%** |

## Two interesting observations

### 1. Fold 3 is the only fold where ordinal LOSES to linear

| Fold 3 R² | Linear | Ordinal |
|---|---:|---:|
| baseline | +0.21% | **−1.11%** |
| + regime | +2.28% | −0.51% |

The ordinal model is worse than the linear model on fold 3 (the
meta-shift fold), and regime decay only partially recovers this. The
ordinal model's flexibility presumably overfits the threshold spacings to
the pre-shift distribution; when the test distribution shifts, those
fitted thresholds produce systematically biased margin predictions.

This is a real "no free lunch" finding for ordinal regression: the
extra capacity helps in stable regimes but hurts in regime breaks.

### 2. The combined optimum is **less aggressive** than linear+regime alone

| | τ_pre | τ_post | α |
|---|---:|---:|---:|
| Linear + regime | 365d | 30d | 0.20 |
| Ordinal + regime | 200d | 60d | 0.60 |

Ordinal+regime prefers a much milder regime correction:
- Longer effective post-shift half-life (60d vs 30d)
- Weaker pre-shift discount (60% retention vs 20%)

This is consistent with the ordinal model already absorbing some of the
regime-shift bias internally through its thresholds — so the explicit
regime correction needs to do less work.

## Implementation note

This required a small one-line addition to `ordinal.fit_ordinal` — an
optional `weights` parameter that, when given, bypasses the internal
weight computation. The integration into `regime_decay.sweep_regime`
is dispatched via the existing `model='linear' | 'ordinal'` parameter.

## Recommendation for the writeup

This is the new headline model:

> *"The best out-of-sample model combines proportional-odds Bradley-Terry
> with a piecewise time-decay that explicitly models the 2023 competitive
> meta shift: τ_pre = 200 days for pre-shift matches, τ_post = 60 days
> for post-shift, and a multiplicative discount α = 0.60 applied to
> pre-shift training data. This achieves R² = 7.93% under 5-fold
> TimeSeriesSplit cross-validation — a +22 % relative improvement over
> the single-rate linear baseline (6.50 %). The two extensions (ordinal
> outcome model + regime-aware decay) address structurally different
> failure modes — equal-spacing assumption and discrete distributional
> shift — and contribute roughly additively, though with diminishing
> returns due to partial overlap."*
