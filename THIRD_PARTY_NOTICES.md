# Third-party notices

Sentinel builds on open-source software, public datasets and an open typeface. This file lists what each one is and under which license.

## Datasets

**NASA C-MAPSS (FD001).** Turbofan engine degradation simulation from the NASA Prognostics Center of Excellence, Ames Research Center. Terms: public domain under NASA's open data policy.
Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation.* In Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO.

**UCI AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository, dataset #601). License: CC BY 4.0.
Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications.* In Third International Conference on Artificial Intelligence for Industries (AI4I 2020), pp. 69-74. IEEE.

## Typography

**IBM Plex Sans, IBM Plex Sans Condensed and IBM Plex Mono.** Copyright (c) 2017 IBM Corp., with Reserved Font Name "Plex". License: SIL Open Font License 1.1. The font files are bundled in `web/fonts/` so the app makes no request to a font CDN. The license text is in `web/fonts/OFL.txt`.

## Software libraries

| Library | License | What Sentinel uses it for |
|:---|:---:|:---|
| Python | PSF | runtime |
| XGBoost | Apache 2.0 | gradient-boosted trees |
| LightGBM | MIT | gradient-boosted trees |
| scikit-learn | BSD-3-Clause | preprocessing, metrics, Random Forest, linear baseline |
| SHAP | MIT | checking the native Tree-SHAP output |
| Optuna | MIT | seeded hyperparameter search |
| MLflow | Apache 2.0 | tracking and the model registry |
| FastAPI | MIT | the REST API and the generated model service |
| Uvicorn | BSD-3-Clause | ASGI server |
| Pydantic | MIT | request validation |
| SQLAlchemy | MIT | database access |
| psycopg2-binary | LGPL-3.0 with OpenSSL exception | PostgreSQL driver |
| pandas, NumPy, SciPy | BSD-3-Clause | data handling and statistics |
| pytest | MIT | tests |

Sentinel itself is released under the MIT License (see `LICENSE`).
