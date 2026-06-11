# How fast does competitive skill information become obsolete in a high-frequency esports environment?

This project
Thank you Identity V Bwiki.
Website:
Technical write-up:

## Introduction
In this project, we analyzed ~15,700 games of pro Identity V esports to assess the impact of player skills on the game result, and used this information to compute the half-life decay for skill information. Essentially, we are measuring how fast historical information becomes stale.

In the top tiers competitions, evaluated with temporal CV, our model achieves 12.4% $R^2$ against the naive model (which takes the average performance of the past 30 games for each player), with the best fold reaching 17.5% $R^2$. The optimal half-life is 182 days, around half of a year. For all tiers competitions, our model achieves 14.2% $R^2$ , with the best fold reaching 18.9% $R^2$. The optimal half-life for all tiers is 540 days.

## Data & Outcome
Identity V is an asymmetry game consisting of 1 hunter and 4 survivors in each game, where the survivors try to escape and hunter try to prevent survivors from escaping.
Data span from May 2020 to May 2026, including 6877 games from IVL, 3946 games from COA, 2561 games from IJL, 1336 games from IVC, 549 games from IVT, and 422 games from IVS. The data were mostly recorded manually by Identity V Bwiki. There are 1516 survivor players, 462 hunter players, and 71 players who played both. For analysis on the top tiers, only games from IVL, IJL, and COA are included.

The outcome per game is determined by a margin = n_escaped - 2 ∈ {−2, −1, 0, +1, +2}. That is, a positive score indicates a survivor advantage, while a negative score indicates a hunter advantage. 

## Model
The main model is Ordinal Bradley-Terry model with a few adaptations. Each player P is assigned a score, $\beta_P$, which is a representation of their skill level. We compute a continuous margin by
$$\mathbb{E}[\text{margin}_i] = \tfrac{1}{4} \sum_{k=1}^4 \beta^S_{s_{ik}} - \beta^H_{h_i}.$$
Intuitively, this can be understood as 1 skill unit is roughly equivalent to 1 escape on average. A high $\beta$ value means more skilled at the respective role.

We used L-BFGS-B with analytical gradient to fit the model ordinal MLE, with each player's skill as a parameter. Since the difficulty of going from 0 escape to 1 escape is not the same as 3 escape to 4 escapes, the model optimizes for the threshold of each escape count as a parameter too. After ordinal regression, we discovered that the gap is 0.75 / 1.69 / 2.40. The ordinal model performs 15% better than the linear model.

To account for the fast-paced competitive landscape, we added a time-decay.

Lastly, some players L2 regularization (ridge regression).

Hyper parameters are optimized jointly via optuna's TPE (Tree-structured Parzen Estimator) sampler, chosen because of the mixed continuous + discrete + categorical parameters.

### Time Decay

## Evaluation
The model is evaluated by a 5 fold temporal cross-validation to prevent look-ahead bias. The model is evaluated by both the top tiers only data and all tiers data, as pro players in the top tiers tend to be fast pace in the more competitive environment compared to amateur players.

The baseline naive model is the rolling model that 

## Findings

## Analysis

## Limitations

## Website



