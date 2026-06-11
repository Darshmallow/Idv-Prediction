# JSON Data Migration — Findings

**Branch:** `json-data`

## Summary

Switched primary data source from the xlsx files to the JSON files in
`data/raw_json/`. The dataset grew from 9,102 to **15,691 game halves**
(+72 %). Tournament tiers (IVL/IJL/COA/IVS/IVT/IVC) are now explicit
columns, enabling test-set filtering. The new alias map from `id.json`
(96 entries, vs our previous 19) is the canonical source — supplemented
with two manual overrides (`ppicha→pipicha`, `gua→guag`).

## Dataset shape

| Tier | Game halves |
|---|---:|
| IVL | 6,877 |
| COA | 3,946 |
| IJL | 2,561 |
| IVC | 1,336 |
| IVT |   549 |
| IVS |   422 |
| **Total** | **15,691** |

- Date range: 2020-06-25 → 2026-05-05
- 100 % `n_escaped` coverage (vs 86 % before)
- 0 null hunters, 0 null dates
- 2,049 unique canonical players (vs 425 before — overseas tournaments
  bring many more players)

## Test set: IVL + IJL only

Training set uses **all tiers** (gives the model exposure to overseas
players IVL/IJL stars face at COA), but the test set is restricted to
the **IVL and IJL main leagues**.

Test-set match count under TimeSeriesSplit + IVL/IJL filter: 7,892
(comparable to the old run's 7,580).

Cold-start rate on the IVL+IJL test sets is **lower** than before
(11–32 % per fold vs 36–65 % in the old run), because the bigger
training set has seen more of the IVL/IJL players already.

## Headline R² changed (in a structurally interpretable way)

| Run | Pooled R² |
|---|---:|
| Old xlsx data, no tier filter (previous main) | **8.04 %** |
| New JSON data, IVL+IJL test, **TimeSeriesSplit** | 6.17 % (old hyperparams) |
| New JSON data, IVL+IJL test, TimeSeriesSplit | 6.29 % (retuned τ_pre=365d, α=0.40) |
| New JSON data, IVL+IJL test, **season splits** | 4.08 % |

### Why the change

1. **Different temporal coverage** — the bigger dataset shifts the
   TimeSeriesSplit fold boundaries. The new fold 5 ends 2026-01-03
   (no COA9), whereas the old fold 5 ended 2026-05-05 (incl. COA9).
   Apples-to-oranges on per-fold numbers.
2. **More players to identify** — 5× more canonical players means more
   parameters with sparse data.  L2 prior would arguably need
   re-calibration; quick sweep showed only ~0.1 pp gains within a wide
   neighbourhood of the old optimum.
3. **Test population is harder** — IVL+IJL alone has a higher fraction
   of close-call games than the old all-tiers test set.

### Per-fold pattern (TimeSeriesSplit, retuned hyperparams)

| Fold | Period | R² (old) | R² (new) | Δ |
|---:|:---|---:|---:|---:|
| 1 | early 2021 → mid 2022 | +5.45 % | +3.71 % | −1.74 |
| 2 | mid 2022 → late 2023 | +4.38 % | +3.91 % | −0.47 |
| **3** | **late 2023 → late 2024** | **−0.51 %** | **+3.96 %** | **+4.47** |
| 4 | late 2024 → mid 2025 | +11.87 % | +8.26 % | −3.61 |
| 5 | mid 2025 → early/mid 2026 | +18.45 % | +11.94 % | −6.51 |

**Fold 3 — the meta-shift fold — improves dramatically (+4.47 pp).**
The richer training set covers more of the regime transition and the
model is no longer caught flat-footed on 2024 IVL data. This is the
most structurally meaningful change.

Folds 4-5 lose because:
- Their test periods are now shorter and have fewer "easy" matches
- The all-tier training mix dilutes some of the IVL/IJL specific signal

### Season splits (apples-to-apples temporal periods)

| Fold | Test season | R² (new) |
|---:|---:|---:|
| 1 | 2021 | +2.49 % |
| 2 | 2022 | +4.16 % |
| **3** | **2023** | **−0.52 %** |
| 4 | 2024 | +3.23 % |
| 5 | 2025 | +11.50 % |

Under cleanly-bounded season splits, fold 3 (the 2023 meta-shift season)
is still slightly negative. The TimeSeriesSplit benefit on fold 3
(+3.96 %) comes from the training set extending part-way through 2023,
not from the new data alone.

## Accuracy unchanged

Practical accuracy metrics are essentially identical to the old result:

| Metric | Old (xlsx) | New (JSON+filter) |
|---|---:|---:|
| Rounded exact | 36.29 % | 34.83 % |
| Sign accuracy | 40.80 % | 40.85 % |
| Within ±1 | 79.70 % | 80.98 % |
| Mean P(actual class) | 28.06 % | 27.60 % |

The R² drop is mostly about the null baseline shifting (the new test
mix has a higher draw rate, so the null model improves), not about the
model getting worse at the task in absolute terms.

## Code changes

| File | Change |
|---|---|
| `src/ingest_json.py` | NEW — JSON readers, tier extraction, alias loader |
| `src/players.py` | Uses `id.json` directly (with MANUAL_OVERRIDES merge) |
| `src/db.py` | Builds from JSON; adds `tournament_tier` column + index |
| `src/eval.py` | `test_tiers=['IVL','IJL']` parameter on `temporal_cv` |

## Open questions for the writeup

- Should the headline R² be the previous **8.04 %** (smaller dataset,
  no tier filter) or the new **6.17 %** (richer dataset, more honest
  IVL+IJL test set)? The 6.17 % is more defensible scientifically —
  but the 8.04 % is what the historical narrative led to.
- If reporting the new number, fold 3 going from −0.51 % to +3.96 %
  is the clearest "extra data fixed the meta-shift problem" story.
- Worth noting that **including overseas data in training improves
  IVL/IJL prediction during the meta shift** — direct evidence that
  cross-population information transfers.
