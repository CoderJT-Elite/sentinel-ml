# Sentinel

**Deterministic AutoML + MLOps for predictive maintenance. No language model anywhere in the decision path.**

Give Sentinel a sensor log. It profiles the data against fixed thresholds, frames the problem with rules, engineers features, trains and ranks four model families by cross-validated score, explains every alert with Tree-SHAP, packages the winner as a versioned FastAPI/Docker service, and watches it for drift. Every decision is written to an auditable ledger, and the whole run reduces to one hash that anyone can reproduce.

> Built for the ABB Accelerator hackathon, Theme 1 (Agentic Predictive Maintenance Studio), Prototype Phase.
> **Live demo (a recorded run of the real pipeline, always on): https://coderjt-elite.github.io/sentinel-ml/demo/**
> **Full live app:** `docker compose up --build` (below).

## Why deterministic

A maintenance tool earns trust only when an engineer can check it. Most "agentic" tools wrap the workflow in a generative model, so the reasoning cannot be reproduced, audited, or defended in a safety review. Sentinel splits the job differently:

| Where machine learning is the right tool | Where Sentinel uses rules and statistics |
|---|---|
| The failure-prediction models themselves (XGBoost, LightGBM, Random Forest, linear baseline) | Data profiling and the quality gate |
| | Task framing, ranking metric, alert horizon |
| | Model selection (cross-validated score, nothing else) |
| | Feature engineering |
| | Explanations (Tree-SHAP plus fixed sentence templates) |
| | Drift detection and the retrain decision (PSI threshold) |

`tests/test_pipeline.py::test_no_llm_imports_anywhere_in_the_package` fails the build if any language-model client is imported.

## Results on NASA C-MAPSS FD001 (turbofan engines)

The pipeline is given only the raw training file: no target column, no hints. It detects the `unit`/`cycle` structure, derives remaining useful life, sets an alarm horizon of 30 cycles (0.15 x median life of 199), and predicts "fails within 30 cycles". The official NASA test set is the holdout and is never used to choose a model.

| Model | CV ROC-AUC (5 folds grouped by engine) | Holdout ROC-AUC (all rows) |
|---|---|---|
| **XGBoost (champion)** | **0.9930 +/- 0.0018** | 0.9934 |
| LightGBM | 0.9930 +/- 0.0017 | 0.9932 |
| Linear baseline | 0.9921 +/- 0.0008 | 0.9918 |
| Random Forest | 0.9909 +/- 0.0020 | 0.9924 |

On the 100 test engines' final observations the champion reaches ROC-AUC 0.981, average precision 0.949 and F1 0.809. The linear baseline is only a few thousandths behind: on this benchmark most of the signal is simple, and Sentinel says so rather than hiding it.

Same data + same seed + same thresholds reproduce the same **run hash**. Two independent full runs (separate processes and registries) both produced
`f206c5c62729d1464cce14aa15775ea3e5593d5ec7529184c1411ce9ffc01c2c`, and CI checks determinism on synthetic data on every push.

## Quick start

**Full stack** (app + PostgreSQL + MLflow server):

```bash
git clone https://github.com/CoderJT-Elite/sentinel-ml && cd sentinel-ml
python scripts/fetch_data.py        # NASA C-MAPSS + UCI AI4I (not committed)
docker compose up --build           # then open http://localhost:8000
```

Pick a dataset, press **Run pipeline** (about 2 minutes on a laptop). Watch the six nodes and the decision ledger fire, then open the Leaderboard, Explain, Fleet, Deploy and Monitor tabs.

**Single container, no compose:**

```bash
docker build -t sentinel . && docker run -p 8000:8000 -v "$PWD/data:/app/data" sentinel
```

**Command line:**

```bash
pip install -r requirements.txt
python scripts/fetch_data.py
python -m sentinel.cli run cmapss_fd001 --budget fast
pytest -q            # 19 tests, synthetic data, no download needed
```

## The pipeline

A LangGraph `StateGraph` used purely as a workflow engine (typed state, a conditional edge for the quality gate). No node calls a model.

1. **Data Steward** (`sentinel/steward.py`): row count, missingness, duplicates, constant columns, identifier columns, multicollinearity (pairwise r and VIF), distribution shift (KS test), then after framing: target completeness, class balance, and a target-leakage guard (any 0/1 flag that predicts the target at 98% or better is dropped). Any failed check halts the run before a model is trained.
2. **Task & Model Selector** (`selector.py`, `structure.py`): detects entity and time columns, derives the target for run-to-failure logs, picks classification or regression, chooses the ranking metric (average precision when the minority class is under 10%), decides on class weighting, fixes the candidate set.
3. **Feature Engineer** (`features.py`): per-unit rolling mean, standard deviation, slope and lag change for every sensor, or encoding for tabular data. Self-contained numpy/pandas so the serving container runs the identical code.
4. **Trainer** (`trainer.py`): seeded Optuna TPE search per family, grouped k-fold cross-validation, leaderboard ranked by CV score only. Holdout is scored for every model and never consulted.
5. **Explainer** (`explainer.py`): Tree-SHAP contributions (native `pred_contrib`, proven equal to `shap.TreeExplainer` on a sample), ranked drivers with z-scores against the fleet baseline, fixed sentence templates, and a measured confidence: the out-of-fold hit rate of the score band.
6. **Deployer / Monitor** (`deployer.py`, `monitor.py`, `lifecycle.py`, `registry.py`): logs to MLflow and registers the champion, generates a versioned FastAPI service with a Dockerfile and checks it reproduces the model's scores, arms a PSI/KS drift monitor, and on a tripped rule retrains a challenger that is promoted only if it beats the incumbent on engines it never saw.

Datasets: NASA C-MAPSS FD001 (run-to-failure, temporal) and UCI AI4I 2020 (rare failures, identifier columns, leaking flags). A bundled `samples/broken_sensor_log.csv` shows the quality gate refusing bad data.

## Stack

Python 3.12, FastAPI, LangGraph (workflow engine only), MLflow (tracking + model registry), Docker, SHAP, LightGBM, XGBoost, scikit-learn, Optuna, PostgreSQL (SQLAlchemy; SQLite fallback), plain HTML/JS front end with no build step.

## Layout

```
sentinel/    pipeline: steward, selector, features, trainer, explainer, deployer, monitor, lifecycle, graph
api/         FastAPI app (REST + server-sent events) that also serves web/
web/         single-page UI (works live or against a recorded run)
scripts/     fetch_data.py, make_samples.py, export_static.py, snap.py
tests/       19 tests incl. determinism, serving parity, SHAP parity, no-LLM import audit
docs/        TECHNICAL_DOCUMENTATION.md and the static demo (docs/demo)
```

## Honest limits

- The static demo is a recorded run of the real pipeline; live training, uploads and container builds need the Docker stack.
- Drift scenarios are simulated (a calibration offset on one sensor of the holdout fleet); the monitor and retrainer are real.
- Survival analysis for censored data is not implemented; run-to-failure logs are handled by classification or RUL regression.
- C-MAPSS is simulated data with a single operating condition (FD001). Multi-condition subsets (FD002/FD004) and real plant data are the next test.

## License

MIT. NASA C-MAPSS is a public dataset from the NASA Prognostics Center of Excellence; UCI AI4I 2020 is from the UCI Machine Learning Repository.
