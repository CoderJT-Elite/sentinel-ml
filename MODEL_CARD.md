# Model card

This follows the outline from Mitchell et al., *Model Cards for Model Reporting* (2019). It covers the two champions Sentinel picked in the recorded runs behind the hosted demo. Every number below is read from those runs (`docs/demo/data/run.json` and `run_ai4i.json`), and a fresh run reproduces both run hashes.

Both models are prototypes trained on public benchmark data. Neither has been tested on a real plant.

## Model details

| | C-MAPSS turbofan | AI4I milling machine |
|---|---|---|
| Champion | XGBoost, 350 trees, max depth 3 | LightGBM, 400 trees, 48 leaves |
| Task | fails within 30 cycles (yes or no) | machine failure (yes or no) |
| Ranked by | ROC-AUC | average precision (failures are rare) |
| Inputs | 86 features built from 17 sensors | 9 features |
| Run hash | `f206c5c62729d1464cce14aa15775ea3e5593d5ec7529184c1411ce9ffc01c2c` | `619a8a6c0147ffcfa9bca83e255d35255678aa1e95e47afe9e9d55186597aaf5` |

Sentinel picked each family and its settings with a seeded Optuna search scored by grouped cross-validation. The full parameters are in the leaderboard of each run record. Developer: John Tewolde. License: MIT.

## Intended use

Ranking a fleet by failure risk and explaining each alert, as a screening aid for a reliability engineer who makes the call. It's built to be reproduced and audited, not to act on its own.

Not for safety-instrumented functions, automatic shutdowns, or any closed loop that acts without a person checking. It hasn't been validated for that.

## Training and evaluation data

- **C-MAPSS FD001.** 100 training engines (20,631 rows). The official test set (100 engines, 13,096 rows) is the holdout, and it never influences which model wins.
- **AI4I 2020.** 8,000 training rows and 2,000 held out, with about 3.4% failures. The Data Steward drops the identifier columns and the four failure-mode flags that leak the target.

Both datasets are simulated or synthetic. See `DATASET_DATASHEET.md`.

## Results

**C-MAPSS.** Five-fold cross-validation grouped by engine gives ROC-AUC 0.9930 (spread 0.0018). On the holdout, over every row: ROC-AUC 0.9934, average precision 0.8246, F1 0.7191 at the alert threshold of 0.7495. On each test engine's last reading alone: ROC-AUC 0.9808, average precision 0.9492, F1 0.8085.

The linear baseline lands within 0.001 of the champion on cross-validated ROC-AUC, so most of the signal in this benchmark is simple.

**AI4I.** Cross-validated average precision is 0.8163 with a wide spread of 0.0627, so the ranking of LightGBM ahead of XGBoost (0.7966) is not a clear win. On the 2,000 held-out rows: ROC-AUC 0.9753, average precision 0.7887, F1 0.7719 at the alert threshold of 0.4309. XGBoost's holdout average precision (0.7946) is slightly higher than the champion's. The holdout doesn't pick the winner, so I report it as it is.

## Explanations and checks

- Tree-SHAP comes from each library's native contribution output. It matches `shap.TreeExplainer` with a maximum difference of 0.0 on 400 sampled rows in both runs.
- The generated serving package reproduces the trained model's scores with a maximum difference of 0.0.
- The confidence in an alert sentence is the out-of-fold hit rate of the score band, with its sample size.

## Limits

- Simulated data has smooth degradation. Real machines also fail from sudden shocks that give no gradual warning.
- C-MAPSS FD001 has one operating condition and one fault mode.
- The AI4I result is on synthetic rows, with a small number of failures in each fold, so the numbers move a lot between folds.
- The alert threshold maximizes F1 on out-of-fold scores. It doesn't account for what a miss or a false alarm costs. The Fleet tab's cost calculator lets you put in your own costs.
- Drift checks use the population stability index on a healthy early-life window. A flagged shift needs a person to look at it.
