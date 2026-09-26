# Dataset Datasheet: Sentinel Reference Benchmarks

This datasheet follows the structure proposed by Gebru et al. (2021). It documents the two datasets packaged with Sentinel: NASA C-MAPSS FD001 and the UCI AI4I 2020 Predictive Maintenance Dataset.

## 1. Motivation

- **Purpose:** Provide reproducible, verifiable industrial benchmarks for predictive maintenance, condition monitoring, and remaining useful life estimation.
- **Creator:** NASA Ames Prognostics Center of Excellence (C-MAPSS) and Stephan Matzka, HTW Berlin (UCI AI4I 2020).
- **Funding:** Public research grants and institutional university funding.

## 2. Composition

### C-MAPSS FD001 (Turbofan degradation)

- **Instances:** 100 training engines (20,631 cycles) and 100 test engines (13,096 cycles).
- **Fields:** 26 columns total. Engine ID, operational cycle counter, 3 operational settings, and 21 sensor channels measuring temperatures, pressures, spool speeds, and coolant flows.
- **Target:** Remaining useful life (RUL), converted by Sentinel into a binary alarm horizon indicator (failure within 30 cycles) using a closed-form rule (15% of median unit life).
- **Sensors dropped by quality gate:** Seven columns with constant zero variance (`op3`, `s1`, `s5`, `s10`, `s16`, `s18`, `s19`).

### UCI AI4I 2020 (Milling machine failure)

- **Instances:** 10,000 synthetic rows reflecting real industrial tool wear parameters.
- **Fields:** Product type (L, M, H quality variants), air temperature (K), process temperature (K), rotational speed (rpm), torque (Nm), and tool wear (minutes).
- **Target:** Binary machine failure (`Machine failure`), consisting of 339 positive failure events across tool wear failure, heat dissipation, power failure, and overstrain failure modes.
- **Class imbalance:** 3.39% positive class prevalence. Handled by Sentinel via automated metric switching from ROC-AUC to Average Precision (PR-AUC).

## 3. Collection process

- **C-MAPSS:** Generated using the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) software environment. Injected synthetic high-pressure compressor degradation profiles with added sensor noise.
- **UCI AI4I 2020:** Generated via a physical process model parameterized by industrial milling telemetry to mirror actual tool wear dynamics without exposing proprietary manufacturer data.

## 4. Preprocessing and cleaning

- **Causal rolling statistics:** Sentinel applies forward-looking-free rolling windows (5-step means, 5-step standard deviations, 10-step linear slopes, and 5-step difference lags).
- **Leakage prevention:** Grouped cross-validation ensures that all records belonging to a specific engine or asset remain within the same split. No time steps from test engines leak into training folds.

## 5. Uses

- **Approved uses:** Benchmarking condition-monitoring algorithms, AutoML pipeline evaluations, and explainable AI verification.
- **Prohibited uses:** Operational flight control or direct safety-critical actuation without physical inspection.

## 6. Distribution and licensing

- **C-MAPSS FD001:** Public domain (NASA Open Data Agreement).
- **UCI AI4I 2020:** Creative Commons Attribution 4.0 International (CC BY 4.0).
