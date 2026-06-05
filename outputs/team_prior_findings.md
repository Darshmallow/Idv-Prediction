# Team-Anchored Prior — Findings

**Branch:** `team-prior`

## What was tried

Replace the centred-at-zero L2 prior on player ratings with a **team-anchored
prior** for survivors:

$$\beta_i \sim \mathcal{N}(\bar\beta_{\text{teammates}(i)},\, \sigma^2)
\quad\text{instead of}\quad \beta_i \sim \mathcal{N}(0,\, \sigma^2)$$

Implemented as a Laplacian-style structured penalty:

$$L(\beta) = \tfrac12 \|y - X\beta\|^2_W \;+\; \lambda_\text{player} \|\beta\|^2
                                       \;+\; \lambda_\text{team} \|(I - A)\beta\|^2$$

where A is the row-normalised survivor co-appearance adjacency
(A[i,j] = fraction of i's matches that also include survivor j).
Hunters are exempt from the team-anchored term — they have no teammates.

When λ_team = 0 this reduces exactly to the baseline linear model
(verified: max |Δβ| = 0).

## Standalone effect

At fixed τ = 110d, sweep over (λ_player, λ_team):

|           | λ_team=0 | λ_team=0.5 | λ_team=1 | λ_team=2 | λ_team=5 | λ_team=10 |
|-----------|---------:|-----------:|---------:|---------:|---------:|----------:|
| λ_p=0.05  |   0.01%  |   5.29%    |   5.76%  |   5.97%  |   6.04%  |   6.05%   |
| λ_p=0.10  |   1.87%  |   5.51%    |   5.93%  |   6.13%  |   6.21%  |   6.22%   |
| λ_p=0.30  |   4.50%  |   6.02%    |   6.32%  |   6.49%  |   6.56%  |   6.57%   |
| λ_p=1.00  |  **6.51%** |   6.85%  |   6.97%  |   7.06%  |   **7.11%**  |   7.11%   |

**Best standalone**: λ_player=1.0, λ_team=5.0 → R² = 7.11%
(vs baseline 6.50%, **Δ = +0.61pp**)

## Combined with regime decay

At (τ_pre=365d, τ_post=30d, α=0.20):

|           | λ_team=0 | λ_team=0.5 | λ_team=1 | λ_team=2 | λ_team=5 |
|-----------|---------:|-----------:|---------:|---------:|---------:|
| λ_p=0.30  |   6.71%  |   7.25%    |   7.30%  |   7.31%  |   7.28%  |
| λ_p=0.50  |   7.20%  |   7.44%    |   7.47%  |   7.46%  |   7.42%  |
| λ_p=1.00  |   7.53%  |   7.59%    | **7.59%**|   7.57%  |   7.53%  |

**Best combined**: λ_player=1.0, λ_team=1.0 → R² = 7.59%

This is only **+0.06pp over regime alone (7.53%)** — the two corrections
substantially overlap.

## Why the overlap?

Both mechanisms address the same underlying problem (sparse individual
data for less-active players) but from different angles:

- **Team-anchored**: pulls sparse players toward team mean
- **Regime decay**: discounts pre-shift data, making post-shift dominant

For a player whose individual data is sparse, both mechanisms produce a
non-zero prior. Once one is applied, the other has less work to do.

## Per-fold pattern

The team-anchored prior helps the "easy" folds (1, 2, 4, 5) but doesn't
help the meta-shift fold (3):

| Fold | Linear+regime | Linear+regime+team-anchored | Δ |
|---:|---:|---:|---:|
| 1 | +5.17% | +5.39% | +0.22 |
| 2 | +4.53% | +4.63% | +0.10 |
| **3** | **+2.28%** | **+2.16%** | **−0.12** |
| 4 | +9.16% | +9.17% | +0.01 |
| 5 | +16.51% | +16.60% | +0.09 |

Fold 3's R² actually *decreases* slightly with team-anchoring added — because
fold 3 isn't about sparse data, it's about distributional shift. Pulling
2023 players toward their (pre-2023 training) teammates' mean doesn't help
when the meta has changed.

## Full model comparison

| Model | R² | Δ vs linear baseline |
|---|---:|---:|
| Linear baseline | 6.50% | — |
| Linear + team-anchored | 7.11% | +0.61pp |
| Linear + regime | 7.53% | +1.03pp |
| **Linear + regime + team-anchored** | **7.59%** | **+1.09pp** |
| Ordinal baseline | 7.70% | +1.20pp |
| **Ordinal + regime** | **7.93%** | **+1.43pp** ⟵ current best |

## What the model says about new players (sanity check)

The koting/persica clique (always-together COA roster — used as the test
case for collinearity earlier):

| Player | n_games | baseline β | team-anchored β (λ=5) |
|---|---:|---:|---:|
| koting | 135 | +0.94 | +0.49 |
| persica | 273 | −0.38 | −0.02 |
| guoker | 204 | −0.08 | +0.24 |
| huiyi | 766 | +0.62 | +0.15 |

The four members are pulled toward each other (clique mean ≈ +0.22),
consistent with the prior's intent. Notably this happens **automatically**
without the explicit collinearity detection of the `diff_penalty` module.

## Recommendation

**Don't merge.** The team-anchored prior is a legitimate idea with a
modest standalone benefit (+0.61pp R²) but it overlaps heavily with
regime decay (combined gain is only +0.06pp over regime alone) and
doesn't address the meta-shift fold (3).

The current best model (ordinal + regime at 7.93%) likely sees similar
diminishing returns — we'd expect ordinal + regime + team-anchored to
land somewhere in the 7.95-8.05% range, not a material improvement
over 7.93%.

Keep `src/team_prior.py` on the branch for reference. The
implementation is clean, backward compatible (λ_team=0 reproduces the
baseline exactly), and supports custom weights so it can be combined
with `regime_decay.compute_weights_regime` if desired.

For the writeup, this is worth a one-paragraph mention as **"an
informed-prior alternative explored and found redundant with the
chosen regime-decay correction"**.
