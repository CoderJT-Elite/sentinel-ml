# Third-Party Notices & Licenses

Sentinel incorporates open-source software, publicly accessible benchmarks, and open typography. This document details the respective licenses and citations.

---

## 1. Primary Datasets

### A. NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
- **Source**: NASA Prognostics Center of Excellence (PCoE), Ames Research Center.
- **Dataset**: Turbofan Engine Degradation Simulation (FD001).
- **Citation**: Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation.* In Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO.
- **License / Terms**: Public Domain / NASA Open Data Policy.

### B. UCI AI4I 2020 Predictive Maintenance Dataset
- **Source**: UCI Machine Learning Repository (Dataset #601).
- **Citation**: Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications.* In Third International Conference on Artificial Intelligence for Industries (AI4I 2020), pp. 69-74. IEEE.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## 2. Typography

### IBM Plex Sans, IBM Plex Sans Condensed, IBM Plex Mono
- **Author**: IBM Corp.
- **License**: SIL Open Font License, Version 1.1 (OFL-1.1).
- **Copyright**: Copyright (c) 2017 IBM Corp. with Reserved Font Name "Plex".
- **Bundled Location**: `web/fonts/` (Self-hosted for offline air-gapped industrial deployment).

---

## 3. Core Software Libraries

| Library | License | Usage |
|:---|:---:|:---|
| **Python** | PSF License | Runtime environment |
| **XGBoost** | Apache 2.0 | Gradient boosting decision trees |
| **LightGBM** | MIT License | High-efficiency gradient boosting |
| **scikit-learn** | BSD-3-Clause | Preprocessing, metrics, Random Forest, linear baselines |
| **SHAP** | MIT License | Tree-SHAP explainer verification |
| **Optuna** | MIT License | Deterministic Bayesian hyperparameter optimization |
| **MLflow** | Apache 2.0 | Model registry, experiment tracking, artifact storage |
| **FastAPI** | MIT License | REST API & microservice serving runtime |
| **Uvicorn** | BSD-3-Clause | ASGI web server |
| **Pydantic** | MIT License | Data validation and schema enforcement |
| **SQLAlchemy** | MIT License | ORM & database abstraction |
| **psycopg2-binary** | LGPL-3.0 with OpenSSL exception | PostgreSQL database adapter |
| **Pandas / NumPy / SciPy** | BSD-3-Clause | Tabular data manipulation & numerical computations |
| **Pytest** | MIT License | Automated testing suite |
