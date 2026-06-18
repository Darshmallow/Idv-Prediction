# How fast does competitive skill information become obsolete in a high-frequency esports environment?

In this project, we applied an ordinal Bradley-Terry model with a few adaptations to ~15,700 games of pro Identity V esports to quantify player skill and measure the rate at which historical match data becomes stale. 

Website: [Identity V Bwiki](https://wiki.biligame.com/idv/)

## Introduction
Identity V is an asymmetric 1v4 mobile game launched by NetEase in 2018. In each game, 1 hunter faces 4 survivors, and the outcome is the number of survivors who escape (0–4), making match results an ordered categorical variable rather than a binary win/loss.

The competitive esports of Identity V started with the Championship of Abyss I (COA I) in 2018, and a professional league based in China (IVL) was formalized in 2020, with another league based in Japan (IJL) following in 2022. As a former co-founder of the Identity V Bwiki, I have been watching and recording esports data for Identity V for many years, which sparked me to dig deeper than the surface level statistics. The project is inspired and based on 2 key questions:

1. **Do individual player ratings carry predictive signal about future match outcomes?**
2. **How quickly does that signal decay as the competitive meta evolves?**

The central finding is that top-tier competitive skill information has a half-life of roughly **182 days**, and the model achieves **$R^2$ = 8.6%** against the null baseline (which predicts a draw for every game) under strict temporal cross-validation. The last fold achieves $R^2 =  15.2\%$ against the null model.

| Metric | Value |
|--------|-------|
| Optimal half-life τ\* | **182 days** |
| $R^2$ vs null (predict all draws) | **8.6%** |
| Series win accuracy | **63.8%** |


## Data & Outcome
Data were sourced from [Identity V Bwiki](https://wiki.biligame.com/idv/), which manually records professional match results. Data cover all official competitive events from **2020-06-25 to 2026-05-05** excluding the 2023 and 2024 Japan IVT. There are 2049 unique players across all tiers (1,516 survivor-only, 462 hunter-only, 71 dual-role). The top-tier model (IVL/IJL/COA) rates 1,383 players: 1,018 survivors and 365 hunters.

| Tier | Half-games | Description |
|------|----------:|-------------|
| IVL | 6,877 | Chinese professional league |
| COA | 3,946 | Annual international tournament|
| IJL | 2,561 | Japanese professional league  |
| IVC | 1,336 | Top amateur tournament |
| IVT | 549 | Regional |
| IVS | 422 | Regional |
| **Total** | **15,691** | |

The outcome for each game is determined by how many survivors escaped, where n_escaped $\in \{0,1,2,3,4\}$. We defined the margin as n_escaped $- 2$ for this project. That is, a positive score indicates a survivor advantage, while a negative score indicates a hunter advantage. 

## Methodology

### Ordinal Bradley-Terry
The main model is an ordinal Bradley-Terry model with a few adaptations. We model each player $p$ with a latent skill score $\beta_p$ estimated jointly via maximum likelihood. For a half-game with hunter $h$ and survivors $s_1 \ldots s_4$, the **linear predictor** is:
$$\eta = \frac{1}{4}\sum_{k=1}^{4} \beta^S_{s_k} - \beta^H_h$$
Intuitively, a high $\beta$ value means more skilled at the respective role.

We used L-BFGS-B with analytical gradient to fit the model ordinal MLE, with each player's skill as a parameter. Since the difficulty of going from 0 escape to 1 escape is not the same as 3 escape to 4 escapes, this linear predictor feeds a proportional-odds (cumulative logit) model to learn the threshold for each outcome. After ordinal regression, the fitted thresholds are -1.44 / -0.46 / 1.44 / 4.13 . The ordinal model achieves 15% lower RMSE than the linear model.

### Time Decay
To account for the fast-paced competitive landscape, every match in the training set has a weight $w_i$ associated with it, determined solely by the match's age relative to the newest match. That is,
$$ w_i = \left(\frac 12\right)^{\Delta t_i/\tau}$$
where $\Delta t_i$ is the age of the match and $\tau$ is the optimal half life. This is the key to answering our question of "how quickly does the signal of player's skill decay in competitive esports".

### Additional Components
- **L2 ridge regularization** — shrinks all $\beta$ toward zero and penalizes players with fewer recorded matches. Gaussian prior interpretation equivalent to portfolio covariance shrinkage.
- **Team-mean informed prior** — players with fewer than `threshold` training matches have their L2 prior centered at their first-match teammates' mean $\beta$, rather than zero. This is to encode the fact that strong teams tend to recruit strong players.
- **Hyperparameter tuning** — $\tau$, $\lambda$ (L2 strength), and `threshold` are jointly optimized via Optuna's TPE sampler under temporal CV.


## Evaluation
The model is evaluated by a 5 fold temporal cross-validation to prevent look-ahead bias: training data always strictly precedes test data, mirroring the backtesting discipline. The model is evaluated by both the top tiers only data and all tiers data, as the meta and player skills tend to shift faster in the more competitive environment compared to amateur tournaments.

### Baselines

| Baseline | Description | RMSE (top tiers) |
|----------|-------------|:----------------:|
| Null | Predict 0 (draw) every match | 1.156 |
| Average ($K = \infty$) | Take the mean of players' margin from the past $K$ games | 1.150 |
| Linear BT | Closed-form ridge, same structure | 1.115 |
| **Ordinal BT** | Final model | **1.105** |

Note that the average baseline is *worse* than the null model at every K below ~300, and $R^2$ is barely positive at $K = \infty$. 

### Half-Life Sweep
To directly observe the effect of half-life, we fix all other hyper-parameters, and sweep the half life from 1 to 1800, on a log scale.

![Half-life sweep — top tiers](outputs/graphs/comparison_sweep_top_tiers.png)
The x-axis is $\tau$, y-axis is out-of-sample RMSE under 5-fold temporal CV. The graph gives $\tau$ = 173d, consistent with the optuna best $\tau$ = 182d.

### Calibration
Next, the calibration plot compares predicted win rate of survivors against empirical frequencies across the test set.
![Calibration — top tiers](outputs/graphs/calibration_top_tiers.png)

Well-calibrated probabilities lie on the diagonal. The model is well-calibrated across all five outcome categories.

### Series Accuracy
The outcome prediction for a single game is noisy due to the huge variance and outside factors that were not included in the model, and as such are much less stable than series prediction. Given the actual lineups per round, the probability of each team winning the series is computed via dynamic programming over round-win states. Aside from the raw series accuracy, we also look at the brier score, a metric evaluating accuracy from probabalistic predictions.

| Model | Series Accuracy | Brier Score | Brier Reduction |
|-------|:--------------:|:-----------:|:---------------:|
| Null | ~50% | 0.2500 | -|
| Always predict home win | 51.4% | 0.4860 | -94.4% |
| Average ($K = \infty$) | 51.0% | 0.2570 | −2.8% |
| **Ordinal BT** | **63.8%** | **0.2197** | **+12.1%** |


## Findings

### Half-Life
The optimal $τ^* = 182$ days means a match from 6 months ago is weighted half as much as a match from today. Matches more than 2 years old contribute less than 6% weight. This is the answer to the central research question: top-tier competitive skill information becomes substantially stale in about one competitive season. 


### 2023 IVL Meta-Shift
The per-fold breakdown is shown below, where $R^2$ increased as more training data becomes available.

| Fold | Approx. test period | R² vs null 
|------|-------------------|:----------:|
| 1 | 2020–2021 | 2.4% |
| 2 | 2021–2022 | 4.8% | 
| 3 | 2022–2023 | 6.1% | 
| 4 | 2023–2024 | 14.4% | 
| 5 | 2024–2026 | 15.2% | 
| **Mean** | | **8.6%** | 

Interestingly, in 2023-2024 season, IVL alone suffered a severe loss in $R^2$. When computing the $R^2$ for 2023-2024 season for IVL only, the $R^2$ sits at -1.1%. That is, it is even worse than the null model during the time period.

This is not a model failure, but rather a structural break in the competitive atmosphere in IVL. In 2023, the meta started as extremely survivor favoring, and since then it has became more and more hunter favoring. The specific regime change point is around October 15th, 2023, when 2023 IVL Fall started. As the table shows, the hunter win rate jumped from 26.0% to 36.3%, while the survivor win rate dropped from 37.2% to 21.8%.

| Period | Survivor win | Draw | Hunter win |
|--------|:-----------:|:----:|:----------:|
| Pre 2023-10-15 | 37.2% | 36.8% | 26.0% |
| Post 2023-10-15 | 21.8% | 41.9% | 36.3% |


## Analysis
In this section, we will examine how the half-life changes across different data configurations.

### Top Tiers vs. All Tiers
We have focused on the top tier tournaments so far, as players in professional leagues have more information for analysis. The following table documents the results from fitting the same model to data from all tiers. 

| Config | $\tau^*$ | $R^2$ vs null |
|--------|----:|:----------:|
| Top tiers (IVL/IJL/COA) | 182 d | 8.6% | 
| All tiers | 539 d | 9.0% | 

The all-tiers model uses a much longer half-life (539 days ≈ 18 months) despite achieving slightly better $R^2$. This reflects two distinct effects:

1. **Data scarcity in lower tiers.** IVC/IVT/IVS players have far fewer games, with many appear only once or twice a year. Aggressive time decay would effectively erase most of a player's history, leaving the model with too little signal to rate them. The longer half-life retains older matches to keep lower-tier players identifiable.
2. **Cross-tier calibration.** Including all tiers means the model encounters the same players in both IVL and lower-tier tournaments, improving cross-tier calibration through more games per player.

### Pre-2023 vs. Post-2023
As a result of the 2023 meta shift, we looked into the half-life both before and after the regime change in October, 2023. In addition, we applied another weight $\alpha$ to all training data before 2023-10-15 when we are fitting the model for post 2023-10-15.

| Config | Single $\tau^*$ | Single $\tau$ $R^2$ | $\tau_{pre}$ | $\tau_{post}$ | $\alpha$ | Regime $R^2$ |
|--------|:----------:|:-----------:|:------:|:-------:|:-:|:---------:|
| Top tiers (IVL/IJL/COA) | 182d | 8.6% | 137d | 204d | 1.00 | 8.57% |
| IVL only | 203d | 6.1% | 164d | 144d | 0.70 | 6.59% |

As the table shows, simply adjusting the weights of the training data does not recover the lost signal. It is rather a data scarcity problem, which explains why fold 4 and 5 experiences significant increase in $R^2$ as training data accumulates. 

### Hunter vs. Survivors
Running separate role-only optimizations (each player's skill estimated only from their primary role's data, evaluated on top-tier matches only), we obtained the following results.

| Role | $\tau^*$ | Best $R^2$ |
|------|----:|:-------:|
| Hunter | 165 days | 6.53% |
| Survivor | 125 days | 5.61% |
| Combined (both roles) | 182 days | 8.6% |

There are 2 observations we can make:
1. **Hunter skill decay slower.** Although the original hypothesis was that hunter players are affected by the meta more, the optimal half-life says otherwise. This is likely due to the fact that survivors are heavily dependent on the team, and the same player within two different teams can have drastically different performances.
2. **Hunters carry more predictive signal**  A single exceptional hunter has more individual impact on the outcome than any single one of four survivors. This aligns with the 1v4 structure: one strong hunter can dominate; one strong survivor cannot guarantee a win against a skilled hunter.

## Limitations
The model can only assess player skills, but an esport game is far more complex than player skills alone. There are many other impactful factors that were not included in the model, and some of the important ones include:

1. **2023 Meta Shift.** Meta-shift in general is unpredictable. An additional regime decay score was added as an attempt to lighten the reliance on pre-2023 data after the meta shift, but no model variant in this project recovers this fold fully. This is an inherent limitation of any retrospective skill model applied across a structural break.


2. **Survivor collinearity.** It is impossible to distinguish between two survivors who have always been playing together, because their combined $\beta only appears in the average, their individual contributions are unidentifiable. 
   
3. **No in-game factors.** The model rates players without any consideration of in-game factors, including map choices and character choices. A map and hunter character fixed effects were added in an attempt, but neither improved the model. 

4. **No team synergy factors,** There are over 50 survivor characters in Identity V, and most players are only familiar with a fraction of them. A team with the four top survivor players is not necessarily the best team composition, as a good team needs players who can play all types of characters.


## Interactive Website
An interactive website accompanies this analysis and is hosted at [IDV Match Predictor](https://darshmallow.github.io/Idv-Prediction/), with both Chinese and English version available. 

Features:
- **Leaderboard** — browse all rated players with filters by role (hunter/survivor) and minimum game count. Players are ranked independently within each role
- **Series predictor** — select two 5-player teams (1 hunter + 4 survivors each), choose a series format (Bo3, Bo5, Bo7), and get exact win probabilities of the series with 
- **Match predictor** — predicts the win probability and the expected margin of a single match.


## Reproducibility
The expected runtime of each step is in the following table.

| Step | Approx. runtime |
|------|:---------------:|
| Build database | < 1 min |
| Main hyperparameter search (80 trials, top tiers) | ~3 min |
| Role-only analysis (60 trials × 2 roles) | ~3 min per role |
| Regime split (60 trials × 4 configs) | ~2 min per run |
| Half-life sweep | ~5–10 min |
| Series accuracy | ~1 min |

### Environment

```bash
git clone https://github.com/darshmallow/Idv-Prediction.git
cd Idv-Prediction
pip install -r requirements.txt
```

Tested on Python 3.11+. Key dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `optuna`, `matplotlib`.

### Data

Raw match data is stored as JSON files in `data/raw_json/`. Build the SQLite database:

```bash
python src/data/db.py
# → data/processed/idv.db  (~15,700 matches)
```

### Reproduce the main results

**1. Hyperparameter search (ordinal BT, top tiers)**

```bash
python src/outputs/optimize.py \
    --model ordinal \
    --tiers IVL,IJL,COA \
    --trials 80
# → outputs/optuna/top_tiers_best.json
```

**2. Role-only analysis (hunter / survivor, ordinal with fixed θ)**

```bash
python src/outputs/optimize.py --model ordinal --tiers IVL,IJL,COA --role hunter --trials 60 --out-prefix hunter_ordinal
python src/outputs/optimize.py --model ordinal --tiers IVL,IJL,COA --role survivor --trials 60 --out-prefix survivor_ordinal
# → outputs/optuna/hunter_ordinal_best.json
# → outputs/optuna/survivor_ordinal_best.json
```

**3. Regime split (pre / post 2023-10-15)**

```bash
# Top tiers
python src/outputs/optimize.py --model ordinal --tiers IVL,IJL,COA --end-date 2023-10-14 --trials 60
python src/outputs/optimize.py --model ordinal --tiers IVL,IJL,COA --start-date 2023-10-15 --trials 60

# IVL only
python src/outputs/optimize.py --model ordinal --tiers IVL --end-date 2023-10-14 --trials 60
python src/outputs/optimize.py --model ordinal --tiers IVL --start-date 2023-10-15 --trials 60
```

**4. Half-life sweep and calibration plots**

```bash
python src/outputs/ordinal_eval.py      # half-life sweep → outputs/graphs/
python src/outputs/calibration.py       # calibration plot
```

**5. Series win accuracy**

```bash
python src/outputs/series_accuracy.py
```

### Saved hyperparameter results

All Optuna results are saved in `outputs/optuna/`. The key files and their best parameters:

| File | Config | τ\* | R² |
|------|--------|:---:|:--:|
| `top_tiers_best.json` | Top tiers (IVL/IJL/COA), combined | 182d | 8.6% |
| `hunter_ordinal_best.json` | Hunter only, ordinal fixed θ | 165d | 6.5% |
| `survivor_ordinal_best.json` | Survivor only, ordinal fixed θ | 125d | 5.6% |
| `top_pre2023_best.json` | Top tiers, pre 2023-10-15 | 382d | 4.9% |
| `top_post2023_best.json` | Top tiers, post 2023-10-15 | 830d | 14.4% |
| `ivl_pre2023_best.json` | IVL only, pre 2023-10-15 | 277d | 4.9% |
| `ivl_post2023_best.json` | IVL only, post 2023-10-15 | 59d | 11.6% |



