# Phase 4 Stretch Analyses

Three sub-questions decomposing the headline finding (overall τ ≈ 110 days).
All sweeps use the same baseline model (linear margin, L2 ridge, 5-fold
temporal CV) on subsets of the data.

## 1. Role asymmetry — hunter vs survivor skill decay

Fit two role-restricted models: one with only hunter parameters (`y_hat = −β^H_h`)
and one with only survivor parameters (`y_hat = ¼ Σ β^S_k`). Each model is
misspecified (the other role's variance becomes noise) but the τ that
minimises RMSE captures how fast *that role's* information goes stale.

| Role     | Optimal τ | months | RMSE at τ* | range over sweep |
|----------|----------:|-------:|-----------:|-----------------:|
| hunter   |   210 d   |   6.9  |   1.1038   |   1.1038–1.1086  |
| survivor |    57 d   |   1.9  |   1.1193   |   1.1193–1.1283  |

**Hunter half-life is ≈ 3.7× LONGER than survivor half-life.** This *contradicts*
the original hypothesis (that hunter skill would be shorter-lived because
characters get patched). Plausible explanations for the observed direction:

- Pro hunters typically main 1-3 characters; character mastery dominates,
  so the underlying signal is stable.
- Survivor meta shifts on a per-tournament basis (which 4 characters get
  picked, which maps favour which rotations) — so individual survivor
  ratings are more situationally dependent.
- Survivor performance has more team-composition contamination
  (collinearity), which the time-decay weighting effectively trades off
  against by preferring recent — i.e. more roster-similar — matches.

Caveat: the hunter sweep curve is very flat (RMSE range only 0.005 across
the whole grid), so the *exact* hunter optimum is loosely identified. The
qualitative direction — hunter τ > survivor τ — is robust.

## 2. Tier comparison — IVL vs IJL vs COA

Sweep applied independently to each tournament family.

| Tier | Matches | Optimal τ | months | RMSE at τ* |
|------|--------:|----------:|-------:|-----------:|
| IVL (Chinese league)        | 6 876 | 110 d | 3.6  | 1.1033 ± 0.021 |
| IJL (Japanese league)       | 1 045 | 324 d | 10.6 | 1.1014 ± 0.038 |
| COA (international finals)  | 1 179 | 324 d | 10.6 | 1.1484 ± 0.053 |

**IVL's optimal half-life is ~3× shorter than IJL/COA.** Two plausible reasons:

1. **Real signal — IVL is the most competitive scene.** Higher density of
   pro play means meta evolves faster, so older matches stop being predictive
   of the present sooner.
2. **Statistical artefact — fewer matches → wider SE → flatter curve.**
   IJL/COA each have ~6-7× fewer matches than IVL; their SEs are 2-2.5× larger.
   At those noise levels the curve cannot resolve a sharp short-τ optimum
   even if one exists. Note that both IJL and COA optima land at the *same*
   grid point (324 d), which is suspicious — it suggests the curve is so flat
   that the gridded optimum is essentially arbitrary.

Honest verdict: the IVL result (110 d) is well-identified; the IJL/COA
results are bounded *below* (likely > 100 d) but their exact values aren't
trustworthy without more data.

## 3. Temporal stability — has the meta sped up?

Sweep applied independently to pre-2023 matches and 2023+ matches.

| Era | Matches | Optimal τ | months | RMSE at τ* |
|-----|--------:|----------:|-------:|-----------:|
| 2020 - 2022 | 3 674 | 110 d | 3.6 | 1.1331 ± 0.020 |
| 2023+       | 5 428 |  71 d | 2.3 | 1.0837 ± 0.022 |

**Skill information now becomes stale ~35 % faster than it did pre-2023.**
The post-2023 optimal half-life (71 d ≈ 10 weeks) suggests that for current
predictions, only the last ~10 weeks of matches carry full predictive weight.

This is the clearest direction-positive finding of the stretch analyses
— roughly a one-grid-point shift but the curves are well-separated and SEs
are comparable. Plausibly real, not a noise artefact.

---

## Summary table for the README

| Analysis | Sub-population | Optimal τ | Note |
|----------|----------------|----------:|------|
| **headline** | all matches | 110 d (3.6 mo) | baseline finding |
| role: hunter | all matches | 210 d (6.9 mo) | longer than expected |
| role: survivor | all matches |  57 d (1.9 mo) | shorter than hunter |
| tier: IVL  | 6.9k matches | 110 d (3.6 mo) | matches headline |
| tier: IJL  | 1.0k matches | 324 d (10.6 mo) | wide SE, weak signal |
| tier: COA  | 1.2k matches | 324 d (10.6 mo) | wide SE, weak signal |
| era: pre-2023 | 3.7k matches | 110 d (3.6 mo) | older era |
| era: 2023+    | 5.4k matches |  71 d (2.3 mo) | meta has sped up |
