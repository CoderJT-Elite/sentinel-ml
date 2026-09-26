# Model Card: Sentinel Champion Models

This model card follows the framework proposed by Mitchell et al. (2019). It covers the two production models trained by Sentinel: the C-MAPSS FD001 turbofan degradation model and the UCI AI4I 2020 machine failure model.

## 1. Model details

- **Developer:** John Tewolde
- **Model date:** September 2026
- **Model type:** Gradient boosted decision trees (XGBoost 3.1.2 and LightGBM 4.6.0)
- **Task:** Binary classification for early failure warning
- **License:** MIT
- **Contact:** Repository issues or submission kit details

### Model architectures

1. **Turbofan champion (C-MAPSS FD001):**
   - Family: XGBoost (`n_estimators=350`, `max_depth=3`, `learning_rate=0.040331`, `subsample=0.776061`, `colsample_bytree=0.561019`)
   - Input shape: 86 engineered features (causal 5-step rolling means, standard deviations, 10-step slopes, 5-step lags, and operational cycle)
   - Cryptographic run hash: `f206c5c62729d1464cce14aa15775ea3e5593d5ec7529184c1411ce9ffc01c2c`

2. **Milling machine champion (UCI AI4I 2020):**
   - Family: LightGBM (`n_estimators=350`, `num_leaves=13`, `learning_rate=0.040331`, `min_child_samples=72`, `subsample=0.776061`, `colsample_bytree=0.561019`)
   - Input shape: 7 numeric sensors and one-hot tool quality types
   - Cryptographic run hash: `619a8a6c0147ffcfa9bca83e255d35255678aa1e95e47afe9e9d55186597aaf5`

## 2. Intended use

- **Primary intended uses:** Fleet-level health screening, condition monitoring triage, and remaining useful life warning alerts for rotating equipment.
- **Intended operators:** Plant reliability engineers, maintenance planners, and field service technicians.
- **Out-of-scope uses:** Automated emergency shutdown loops, safety-instrumented systems (SIS), or unmonitored closed-loop actuation without technician verification.

## 3. Training data

- **C-MAPSS FD001:** 100 simulated turbofan run-to-failure trajectories (20,631 records). Training used 5-fold grouped cross-validation grouped by engine ID. No engine appears in both train and validation splits.
- **UCI AI4I 2020:** 10,000 synthetic machine observations reflecting actual industrial milling parameters with 339 failure events (3.39% class imbalance).

## 4. Evaluation data and metrics

### C-MAPSS FD001 holdout (100 test engines)

- Metric: ROC-AUC (ranking capability on balanced temporal trajectories)
- 5-fold cross-validation ROC-AUC: 0.9930 +/- 0.0018
- Holdout ROC-AUC (all 13,096 test rows): 0.9934
- Holdout ROC-AUC (final observation per engine): 0.9808
- Holdout Average Precision: 0.8246
- Holdout F1 score at alert threshold (0.7495): 0.7191

### UCI AI4I 2020 holdout (2,000 test records)

- Metric: Average Precision (PR-AUC, selected by Sentinel due to 3.39% minority prevalence)
- 5-fold cross-validation Average Precision: 0.8163 +/- 0.0383
- Holdout ROC-AUC: 0.9753
- Holdout Average Precision: 0.8037
- Holdout F1 score at alert threshold (0.4284): 0.7473

## 5. Explainability and verification

- **Tree-SHAP parity:** Native C-based Tree-SHAP calculations are verified against the reference `shap.TreeExplainer` library. The maximum absolute difference across all features and samples is 0.000000.
- **Serving parity:** The exported standalone FastAPI service is tested against in-process Python predictions. Maximum score difference is 0.000000.
- **In-browser parity:** The 350 exported JSON decision trees evaluated in JavaScript match Python XGBoost score outputs to within floating-point epsilon (1e-6).

## 6. Uncertainty quantification

- **Conformal intervals:** Nonconformity scores are computed out-of-fold to provide distribution-free marginal coverage guarantees at 90% confidence ($1 - \alpha = 0.90$).
- **Reliability calibration:** Predictions are grouped into 10 historical probability bins. Bins report empirical out-of-fold hit rates to prevent overconfidence.

## 7. Limitations and ethical considerations

- Simulated and synthetic benchmarks reflect stationary degradation curves. Real-world machinery often exhibits sudden mechanical shocks (bearing spalls, debris strikes) that do not present gradual sensor drift.
- Drift detection relies on Population Stability Index (PSI). High PSI values flag changes in sensor distributions, requiring operator inspection rather than blind retraining.
