# Difference-Penalty Experiment (branch: `diff-penalty`)

Option 4: a targeted penalty on detected collinear survivor pairs:

    min_β  Σ w_i (y_i − x_iβ)²  +  λ₁‖β‖²  +  λ_d Σ_{(a,b)∈P} (β_a − β_b)²

P = survivor pairs that almost always appear together
    (co_appearances / min(games) ≥ θ, both ≥ min_games games).
Σ(β_a−β_b)² = β'Lβ (graph Laplacian), so the solve stays closed-form:

    β* = (X'WX + λ₁I + λ_d L)⁻¹ X'Wy

Pairs are detected on the training fold; with θ=0.9, min_games=8 we find
~186 pairs per fold (341 on the full data).

## Result 1 — fixes collinearity WITHOUT damping good players

Survivor β (full-data fit, λ₁=1):

| player  | baseline | λ_d=10 | λ_d=100 | note |
|---------|---------:|-------:|--------:|------|
| koting  |   0.937  |  0.211 |  0.098  | collapses to ~0 ✓ |
| persica |  −0.384  |  0.011 |  0.073  | rises toward the clique ✓ |
| huiyi   |   0.623  |  0.061 |  0.087  | joins clique ✓ |
| **huan**|   0.977  |**1.330**|**1.368**| **preserved / strengthened ✓** |

This is the key win over Option 3 (team effects), where huan was wrongly
damped 0.98 → 0.42.  huan plays across many distinct rosters, so it forms no
high-overlap pair and the penalty never touches it.  Only the koting/persica/
huiyi/guoker clique — who genuinely can't be told apart — gets collapsed.

## Result 2 — RMSE is unchanged-to-slightly-better

5-fold temporal CV, half-life = 110 days:

| model | RMSE | Δ vs baseline |
|-------|-----:|--------------:|
| baseline                | 1.1017 ± 0.019 | — |
| diff penalty λ_d = 1    | 1.0992 | −0.0025 |
| diff penalty λ_d = 10   | 1.0991 | −0.0026 |
| diff penalty λ_d = 100  | 1.0994 | −0.0023 |
| diff penalty λ_d = 1000 | 1.0994 | −0.0023 |

A small, consistent improvement (within ±1 SE, so not significant — but
notably *not worse*, unlike team effects which were +0.0017 at λ=1).  The
gain is robust across four orders of magnitude of λ_d, so it comes from the
structure (linking unidentifiable pairs) rather than tuning.  Intuition:
preventing koting's β from inflating to +0.94 reduces overfit, helping
slightly when koting appears in the test fold.

## Conclusion

Option 4 dominates Option 3:

| | collinearity fix | damages good players? | OOS RMSE | closed-form? |
|--|--|--|--|--|
| Team effects (Opt 3)  | yes | **yes (huan 0.98→0.42)** | no change | yes |
| Diff penalty (Opt 4)  | yes | **no (huan preserved)**  | −0.003 (tiny gain) | yes |

The difference penalty is the right tool: it shrinks only the
unidentifiable directions, leaves everything else alone, keeps the fast
closed-form solve, and is insensitive to its hyperparameter.

**Recommendation:** worth merging into the main model as an optional
`l2_diff` argument (default off to keep the headline reproducible; on for
the trustworthy individual-rating leaderboard). λ_d ≈ 10 and θ ≈ 0.9 are
reasonable defaults.
