# Tuning & Testing Guide

How to train on different data, test on different data, and change every
hyperparameter — copy-paste templates only.

---

## 0. The 30-second mental model

Every experiment is the same six choices:

| Choice | Lever |
|---|---|
| **Which matches train the model?** | Filter the `train` DataFrame inside the CV loop |
| **Which matches evaluate the model?** | Filter the `test` DataFrame inside the CV loop |
| **How to split train vs test?** | `split_strategy = "time_series" \| "seasons"` |
| **How much past data to weight?** | `tau_pre`, `tau_post`, `alpha`, `shift_date` |
| **How to handle cold-start players?** | `threshold` (new-player team-mean prior) |
| **Which model class?** | `model = "linear" \| "ordinal"` |

Every test script is a small variation of the same template below.

---

## 1. The template script

Save this as a `.py` file or paste into a Jupyter cell. Everything else
in this guide is just changing two or three lines of this.

```python
# ── 0. Path setup — works from any cwd (notebook, project root, etc.) ─
import sys, os
_p = os.path.abspath('.')
while not os.path.exists(os.path.join(_p, 'src', 'bradley_terry.py')):
    parent = os.path.dirname(_p)
    if parent == _p:
        raise RuntimeError("Couldn't find idv-analysis project root")
    _p = parent
sys.path.insert(0, os.path.join(_p, 'src'))
DB_PATH = os.path.join(_p, 'data', 'processed', 'idv.db')

import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from bradley_terry import filter_complete, build_design_matrix, predict
from ordinal import predict_expected_margin
from regime_decay import compute_weights_regime
from team_prior import fit_with_new_player_prior

# ── 1. Load the database ───────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
matches = pd.read_sql("SELECT * FROM matches ORDER BY date", conn)
conn.close()
d = filter_complete(matches).sort_values('date').reset_index(drop=True)

# ── 2. Choose your hyperparameters ─────────────────────────────────────
TAU_PRE    = 200.0       # half-life (days) for matches BEFORE shift_date
TAU_POST   = 60.0        # half-life (days) for matches AFTER shift_date
ALPHA      = 0.60        # 0–1; multiplicative discount on pre-shift matches
SHIFT_DATE = '2023-01-01'
L2_LAMBDA  = 1.0         # ridge strength
THRESHOLD  = 5           # players with <THRESHOLD training matches get team-mean prior
MODEL      = 'ordinal'   # 'ordinal' or 'linear'

# ── 3. Choose what to test on ──────────────────────────────────────────
TEST_TIERS = ['IVL', 'IJL']            # restrict TEST set to these tiers
N_SPLITS   = 5

# ── 4. Run 5-fold temporal CV ──────────────────────────────────────────
test_mask = d['tournament_tier'].isin(TEST_TIERS).to_numpy()
splits = [(tr, te[test_mask[te]])
          for tr, te in TimeSeriesSplit(n_splits=N_SPLITS).split(d)]

rmses, nulls = [], []
for fold, (tr, te) in enumerate(splits, 1):
    train, test = d.iloc[tr], d.iloc[te]
    if len(test) == 0:
        continue

    weights = compute_weights_regime(
        train['date'], train['date'].max(),
        tau_post=TAU_POST, tau_pre=TAU_PRE,
        shift_date=SHIFT_DATE, alpha=ALPHA,
    )

    result = fit_with_new_player_prior(
        train, model=MODEL, l2_lambda=L2_LAMBDA,
        threshold=THRESHOLD, weights=weights,
    )
    if MODEL == 'ordinal':
        beta, idx, res = result
        yhat = predict_expected_margin(test, res)
    else:
        beta, idx = result
        X_te, _   = build_design_matrix(test, idx)
        yhat      = predict(X_te, beta)

    y      = (test['n_escaped'] - 2).to_numpy(float)
    rmse   = float(np.sqrt(np.mean((y - yhat) ** 2)))
    null   = float(np.sqrt(np.mean(y ** 2)))
    r2     = 1 - rmse ** 2 / null ** 2
    print(f"  fold {fold}: n={len(test):>5}  RMSE={rmse:.5f}  R²={r2*100:+.2f}%")
    rmses.append(rmse); nulls.append(null)

# Pooled R²
pooled_r2 = 1 - np.mean([r**2 for r in rmses]) / np.mean([n**2 for n in nulls])
print(f"\nPooled R²: {pooled_r2*100:.2f}%")
```

---

## 2. How each knob behaves

### 2.1 Half-life parameters (τ_pre, τ_post, α)

The weight applied to a training match dated `t`:

```
if t < shift_date:    w = alpha * 0.5^( (ref_date − t) / tau_pre )
else:                 w =         0.5^( (ref_date − t) / tau_post )
```

| Lever | Effect of increasing |
|---|---|
| `tau_pre`  | Old (pre-2023) matches stay influential longer |
| `tau_post` | Recent matches stay influential longer |
| `alpha`    | Pre-shift matches get more weight (α=1 disables the regime discount) |

**Defaults that work:** `tau_pre=200, tau_post=60, alpha=0.60` for ordinal.
For linear, the optimum is more aggressive: `tau_pre=365, tau_post=30, alpha=0.20`.

**To disable regime decay entirely** (single-rate decay): set
`tau_pre = tau_post = tau` and `alpha = 1.0`. Or replace the
`compute_weights_regime` call with:

```python
from bradley_terry import compute_weights
weights = compute_weights(train['date'], train['date'].max(),
                           half_life_days=110.0)
```

### 2.2 `THRESHOLD` (new-player prior)

Players with **fewer than `THRESHOLD` training matches** get their prior
mean set to the average β of their first-match teammates (rather than 0).

- `THRESHOLD = 0` disables the prior entirely (standard L2)
- `THRESHOLD = 5` is the chosen optimum
- `THRESHOLD = 50` would treat almost everyone as "new" — over-applies

### 2.3 `L2_LAMBDA`

Stronger ridge → ratings shrink harder toward zero (or toward the team
prior, for new players). Typically `1.0` is good. Try `0.3, 0.5, 1.0, 3.0`
if you're sweeping.

### 2.4 `MODEL`

- `"linear"` — closed-form weighted ridge. Fast (~ms per fit).
- `"ordinal"` — proportional-odds Bradley-Terry, L-BFGS-B. ~50× slower
  but typically +1.5 pp R². Use this when you care about the best
  number; use linear when you're sweeping a wide grid.

### 2.5 `N_SPLITS`

- `5` is standard.
- Larger N → smaller test folds, more noise per fold, but higher mean
  variance in R² estimates. Stick to 5.

---

## 3. Filtering the **training** set

You filter the DataFrame `d` (or `train` inside the loop) using normal
pandas operations. Three common patterns:

### 3.1 Train only on certain tiers

**Don't** reassign `d` — apply the filter as a mask on indices instead.
This keeps `d` as the canonical row set so `TimeSeriesSplit.split(d)` and
`test_mask` stay aligned.

```python
# Build train and test masks side-by-side
train_mask = d['tournament_tier'].isin(['IVL', 'COA']).to_numpy()
test_mask  = d['tournament_tier'].isin(['IVL', 'IJL']).to_numpy()

# Apply BOTH masks inside the splits
splits = [
    (tr[train_mask[tr]], te[test_mask[te]])
    for tr, te in TimeSeriesSplit(n_splits=N_SPLITS).split(d)
]
# Loop body unchanged — `d.iloc[tr]` and `d.iloc[te]` work as normal
```

**Common gotcha**: do NOT do `d = d[d['tournament_tier'].isin(...)].reset_index(drop=True)`
and then call `.split(d)` afterwards on the smaller frame, and then index
into something else — the indices won't line up across the filtered and
unfiltered frames and you'll get `IndexError: positional indexers are
out-of-bounds` on later folds.

### 3.2 Train only on a date window

```python
# Only train on matches from 2022 onwards
d = d[d['date'] >= '2022-01-01'].reset_index(drop=True)
```

### 3.3 Drop a specific tournament

```python
# Exclude COA9 from training
d = d[d['tournament'] != 'COA9'].reset_index(drop=True)
```

### 3.4 Per-fold training filter (inside the CV loop)

If you want the training-set filter to depend on the fold:

```python
for fold, (tr, te) in enumerate(splits, 1):
    train, test = d.iloc[tr], d.iloc[te]
    # Drop overseas tournaments from this fold's training set
    train = train[train['tournament_tier'].isin(['IVL', 'IJL', 'COA'])]
    # ... rest of the loop unchanged
```

---

## 4. Filtering the **test** set

### 4.1 By tier — simplest (covered in template)

```python
TEST_TIERS = ['IVL', 'IJL']                       # default
TEST_TIERS = ['IVL']                              # IVL only
TEST_TIERS = ['IVL', 'IJL', 'COA']                # competitive tiers
test_mask = d['tournament_tier'].isin(TEST_TIERS).to_numpy()
splits = [(tr, te[test_mask[te]]) for tr, te in raw_splits]
```

### 4.2 By specific tournament

```python
# Test only on COA9 matches
test_mask = (d['tournament'] == 'COA9').to_numpy()
```

### 4.3 By date window

```python
# Test only on matches in 2025
test_mask = ((d['date'] >= '2025-01-01') & (d['date'] < '2026-01-01')).to_numpy()
```

### 4.4 Multi-condition

```python
# Test only on IVL playoffs (any year)
test_mask = (
    d['tournament_tier'].eq('IVL') &
    d['tournament'].str.contains('playoffs')
).to_numpy()
```

---

## 5. Changing the split strategy

```python
from eval import get_splits
splits_raw = get_splits(d, strategy='seasons')        # one fold per season year
# or
splits_raw = get_splits(d, strategy='time_series', n_splits=5)

# Apply test-tier filter
splits = [(tr, te[test_mask[te]]) for tr, te in splits_raw]
```

- `"time_series"` — `sklearn.TimeSeriesSplit`, equal-size test folds
- `"seasons"` — one fold per calendar year (5 splits over 6 seasons)

Use seasons for an **apples-to-apples** comparison across years.
Use time_series for the **most statistical power** (more matches per fold).

---

## 6. Reading the output

### 6.1 Per-fold output

For each fold, you'll see:

```
fold 1: n=1282  RMSE=1.13089  R²=+3.50%
```

- `n` = number of test matches in this fold
- `RMSE` = root mean squared error on margin (target is `n_escaped − 2`)
- `R²` = 1 − RMSE² / null_RMSE² where null = predict 0 for everyone
  - Positive = beats the null model
  - Negative = does worse than predicting 0 for every match

### 6.2 Pooled R² vs mean of fold R²

```python
# Pooled (recommended for headline)
pooled_r2 = 1 - np.mean([r**2 for r in rmses]) / np.mean([n**2 for n in nulls])

# Mean of fold R²s (equal-weighted)
mean_r2 = np.mean([1 - r**2/n**2 for r, n in zip(rmses, nulls)])
```

Both are valid. **Pooled** weights by per-fold variance; **mean** treats
each fold equally. They diverge when folds have very different variances.
For the headline, prefer **pooled**.

### 6.3 Accuracy metrics (in addition to R²)

```python
# Collect predictions across folds
all_y, all_yhat = [], []
for tr, te in splits:
    # ... fit code ...
    all_y.append(y); all_yhat.append(yhat)
y    = np.concatenate(all_y)
yhat = np.concatenate(all_yhat)

# Sign accuracy: hunter/draw/survivor
y_int      = y.astype(int)
yhat_round = np.clip(np.round(yhat), -2, 2).astype(int)
sign_acc = ((np.sign(yhat_round) == np.sign(y_int)) |
             ((yhat_round == 0) & (y_int == 0))).mean()

# Within ±1
within_1 = (np.abs(yhat_round - y_int) <= 1).mean()

print(f"Sign accuracy: {sign_acc*100:.2f}%")
print(f"Within ±1:     {within_1*100:.2f}%")
```

---

## 7. Common gotchas

- **Don't shuffle.** Never use `KFold` instead of `TimeSeriesSplit`. The
  template uses `TimeSeriesSplit` correctly.
- **Test mask must be aligned to `d` after `filter_complete`**, not to
  the raw `matches` DataFrame. The template does this correctly; just
  don't reorder rows after building `test_mask`.
- **Cold start ≠ bug.** If a player is unseen in training, their β = 0
  (the L2 prior). The model gracefully degrades.
- **Empty test folds.** When you filter test sets aggressively, some
  folds can end up with 0 test matches. The template's `if len(test) == 0:
  continue` handles this — but it skews the pooled R² because some
  folds contribute nothing.
- **Different folds, different periods.** When the dataset grows
  (e.g., switching from xlsx to JSON added 6,000 matches), the
  TimeSeriesSplit fold boundaries shift. Per-fold R² numbers from
  different dataset sizes aren't directly comparable. Use season splits
  for stable per-period comparisons.

---

## 8. Cheat sheet — what each knob does to R²

Rough effect on R² compared to baseline (linear, τ=110, on the JSON
data with IVL+IJL test filter):

| Change | Approx Δ R² |
|---|---:|
| Linear → Ordinal | +1.0–1.5 pp |
| Add regime decay (correct τ, α) | +0.5–1.0 pp |
| Add new-player prior (threshold=5) | +0.1–0.2 pp |
| Halve `THRESHOLD` (1 instead of 5) | −0.1 pp |
| Double `L2_LAMBDA` (2.0 instead of 1.0) | ±0.2 pp |
| Switch `TEST_TIERS` from `['IVL','IJL']` → `['IVL']` | depends on data |
| Switch split from `time_series` → `seasons` | −1 to −2 pp |

The biggest gains are from switching model class (ordinal) and getting
the regime-decay timing right. Everything else is fine-tuning in the
±0.1–0.5 pp range.

---

## 9. Quick sweep template

If you want to sweep one parameter:

```python
results = []
for tau_post in [30, 45, 60, 90, 110, 169]:
    # ... run the template with TAU_POST = tau_post ...
    results.append({'tau_post': tau_post, 'r2': pooled_r2})

import pandas as pd
print(pd.DataFrame(results).round(5).to_string(index=False))
```

For a 2D sweep, just nest two loops. With the linear model, you can
sweep 100 combinations in seconds; with ordinal expect ~1 minute per 100.

---

## 10. Where to look in the code

| Want to know | Look at |
|---|---|
| How decay weights are computed | `src/regime_decay.py:compute_weights_regime` |
| How the team prior is constructed | `src/team_prior.py:fit_with_new_player_prior` |
| How ordinal NLL is minimized | `src/ordinal.py:_objective_and_grad` |
| How splits work | `src/eval.py:get_splits` |
| Tournament-tier classification | `src/ingest_json.py:extract_tier` |
| Player alias resolution | `src/players.py:ALIAS_MAP` |
