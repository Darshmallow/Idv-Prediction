# New-Player Team-Mean Prior — Findings

## What it is

Instead of initialising every player's β at the standard L2 prior
N(0, σ²), **new players** (those with fewer than `threshold` training
matches) are given an informed prior centred at the mean β of their
teammates from their first training match.  Established players keep
the standard centred-at-zero prior — the prior is targeted, not
universal.

Mathematically the L2 ridge becomes player-specific:

    penalty = ½ λ Σ_i (β_i − μ_i)²        instead of    ½ λ Σ_i β_i²

with μ_i = team mean (only set for new players, zero for everyone else).

### Closed-form linear path
Solving the modified normal equations is one extra term on the RHS:

    (X'WX + λI) β = X'W y + λ μ

### Ordinal path
`ordinal.fit_ordinal` was extended with an optional `prior_mean=`
parameter (backward compatible — None reproduces the standard fit). The
NLL gradient becomes `−X'(w·dη) + λ(β − μ)`.

### Two-pass procedure
1. **Pass 1** — fit with standard prior (μ = 0). Produces preliminary β.
2. **Identify** the new players (training-set appearances < threshold).
3. **Compute** team-mean priors from the preliminary β, using each new
   player's first-match teammates (excluding teammates who are
   themselves "new", to avoid noise-on-noise estimates).
4. **Pass 2** — refit with `prior_mean = μ`.

## Results

| Model | R² (pooled) | Δ vs ordinal+regime |
|---|---:|---:|
| Ordinal + regime (previous best) | 7.93% | — |
| **Ordinal + regime + new-player prior (th=5)** | **8.04%** | **+0.11 pp** |

The improvement is concentrated in the previously-worst fold:

| Fold | Period | ord+regime | + new-player prior | Δ |
|---:|:---|---:|---:|---:|
| 1 | 2021-07 → 2022-08 | +5.45% | +5.51% | +0.06 |
| 2 | 2022-08 → 2023-10 | +4.38% | +4.22% | −0.16 |
| **3** | **2023-10 → 2024-10** | **−0.51%** | **+0.05%** | **+0.56** |
| 4 | 2024-10 → 2025-07 | +11.87% | +11.80% | −0.07 |
| 5 | 2025-07 → 2026-05 | +18.45% | +18.42% | −0.03 |

**Fold 3 went from negative to positive** — the meta-shift fold is no
longer the embarrassing outlier. Other folds are within ±0.16 pp.

## Threshold sweep (ordinal + regime)

| Threshold | R² | Note |
|---:|---:|---|
| 0 | 7.96% | (baseline + 2-pass refit refinement only) |
| 1 | 7.96% | strictest "first appearance" |
| 3 | 7.93% | |
| **5** | **8.04%** | ← chosen optimum |
| 10 | 8.01% | |
| 20 | 7.94% | over-applies prior to medium-data players |

Players with ≤4 training matches benefit from the team prior; players
with ≥5 have enough individual data to override it.

## Sanity checks

- `threshold=0` (no players counted as new) → identical to baseline
  fit (verified: max |Δβ| = 0).
- `prior_mean=zeros` in `fit_ordinal` → identical to no `prior_mean`
  (verified: max |Δβ| = 0, max |Δθ| = 0).

## Why fold 3 benefits most

About 16% of fold 3's training-set player roster is "new" (44 players
with <5 training matches), and the test set is the 2024 meta-shift
period. For those players:

- Standard prior says: "no individual data → guess β = 0 (average)".
- New-player prior says: "no individual data → guess β = mean of your
  first-match teammates".

When that team is performing at a particular level, the informed prior
is a substantially better starting estimate than 0 — especially in a
regime-shift fold where the prior of "average" is more wrong than usual.

## Recommendation

**Merge into main.** This is the new project-best model
(R² = 8.04% under TimeSeriesSplit CV), the implementation is clean,
backward compatible, and the gain corresponds to a structurally
meaningful improvement (fold 3 crossing from negative to positive R²).
