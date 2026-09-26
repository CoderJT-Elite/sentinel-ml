# Dataset datasheet

This follows the outline from Gebru et al., *Datasheets for Datasets* (2021), for the two public datasets Sentinel ships examples for. Neither is redistributed in this repository. `scripts/fetch_data.py` downloads them from their sources.

## NASA C-MAPSS FD001

- **What it is.** A simulation of turbofan engines running to failure, made with NASA's C-MAPSS software at the Prognostics Center of Excellence, Ames Research Center. Creators: Saxena, Goebel, Simon and Eklund (2008).
- **Contents.** 100 training engines with 20,631 cycles and 100 test engines with 13,096 cycles. Each row has the engine number, the cycle, three operating settings and 21 sensor channels. The test file comes with each engine's true remaining life.
- **Condition.** FD001 has one operating condition and one fault mode (high-pressure compressor degradation).
- **What Sentinel does with it.** It sees that every engine's series ends at failure, so it derives a target: fails within 30 cycles, where 30 is 0.15 times the median engine life of 199 cycles. Seven columns carry no information and are dropped (op3, s1, s5, s10, s16, s18, s19).
- **Splits.** Cross-validation folds are grouped by engine, so no engine is in both a training and a validation fold. The official test set is the holdout.
- **Limits.** It's simulated. Real sensors have gaps, recalibrations and more than one fault at a time.
- **Terms.** Public domain under NASA's open data policy.

## UCI AI4I 2020 Predictive Maintenance Dataset

- **What it is.** A synthetic dataset built to resemble milling-machine process data. Creator: Stephan Matzka, HTW Berlin (2020). UCI repository dataset #601.
- **Contents.** 10,000 rows with a product quality type, air and process temperature, rotational speed, torque and tool wear, plus a machine-failure label and five failure-mode flags.
- **Balance.** About 3.4% of rows are failures.
- **What Sentinel does with it.** The Data Steward drops the two identifier columns, and the leakage guard drops the four failure-mode flags that predict the label almost perfectly (`TWF`, `HDF`, `PWF`, `OSF`). It switches the ranking metric to average precision because failures are rare. It splits 8,000 rows for training and holds out 2,000.
- **Limits.** The rows are independent, with no time series and no machine history. The failure count in each fold is small.
- **Terms.** CC BY 4.0.

## Citations

1. Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08)*.
2. Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications. *2020 Third International Conference on Artificial Intelligence for Industries (AI4I)*, pp. 69-74. IEEE.
