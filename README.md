<div align="center">

![Sentinel Hero Banner](docs/assets/hero-banner.png)

# SENTINEL
### Deterministic AutoML + MLOps Copilot for Industrial Predictive Maintenance

[![Determinism](https://img.shields.io/badge/run__hash-f206c5c62729...-d8001b?style=flat-square&logo=git)](docs/demo/)
[![LLM Calls](https://img.shields.io/badge/LLM__Calls-0_(AST_Audited)-101418?style=flat-square)](tests/test_pipeline.py)
[![Tests](https://img.shields.io/badge/tests-19_passed-08635f?style=flat-square)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-5b6670?style=flat-square)](LICENSE)

**Built for the ABB Accelerator 2026 (Theme 1: Agentic Predictive Maintenance Studio), Prototype Phase**  
*Submitted by John Tewolde*

[**Explore the hosted demo (recorded run)**](https://coderjt-elite.github.io/sentinel-ml/demo/) • [**Technical Documentation (PDF)**](docs/TECHNICAL_DOCUMENTATION.md) • [**Architecture and decision records**](docs/TECHNICAL_DOCUMENTATION.md#131-architecture-decision-records-adrs)

</div>

---

## ⏱️ 60-Second Tour for Hackathon Judges

1. **The problem**: Siemens estimates the Fortune Global 500 lose about $1.4 trillion a year to unplanned downtime (*The True Cost of Downtime 2024*). A predictive model that flags a machine but cannot say why is hard to defend in a safety review, and wrapping the workflow in a generative LLM makes it harder to reproduce.
2. **The Sentinel Commitment**: **Zero language models in the runtime decision path.** Machine learning (XGBoost, LightGBM, Random Forest, linear baseline) fits non-linear failure dynamics. Everything around it (data profiling, task framing, model tournament ranking, feature engineering, Tree-SHAP explanations, drift monitoring) runs on deterministic mathematical rules, statistical tests, and cross-validated scores.
3. **The Proof of Determinism**: Every threshold, decision, and metric hashes to an immutable SHA-256 digest:  
   `f206c5c62729d1464cce14aa15775ea3e5593d5ec7529184c1411ce9ffc01c2c`. Anyone running the pipeline with the same raw data and seed reproduces the exact same bitwise hash.
4. **Two extras in the hosted demo**: a what-if sketch (a linear estimate from a reading's top Tree-SHAP drivers, not a model re-run) and a printable run report.
5. **Production Parity**: Generated edge microservices reproduce trained model outputs with a maximum absolute discrepancy of **0.0**.

---

## 🏛️ Architecture: Mapped to ISO 13374 Functional Layers

Sentinel's 6 pipeline stations are arranged to follow the functional blocks of **ISO 13374** (*Condition monitoring and diagnostics of machine systems*). This is an organizing analogy, not a conformance claim:

```
                     RAW PLANT TELEMETRY (CSV / Stream)
                                     │
    ┌────────────────────────────────▼────────────────────────────────┐
    │  STATION 1: DATA STEWARD (ISO 13374 Layer 2: Data Manipulation)  │
    │  Profiles data against 10 fixed statistical checks.             │
    │  [HALT ON CORRUPT DATA] ──> Refuses missing / uncalibrated data  │
    └────────────────────────────────┬────────────────────────────────┘
                                     │
    ┌────────────────────────────────▼────────────────────────────────┐
    │  STATION 2: TASK & MODEL SELECTOR (ISO 13374 Layer 2/3)         │
    │  Auto-detects unit/cycle structure; alarm horizon = 15% life    │
    │  Targets: Remaining Useful Life (RUL) & classification horizon  │
    └────────────────────────────────┬────────────────────────────────┘
                                     │
    ┌────────────────────────────────▼────────────────────────────────┐
    │  STATION 3: FEATURE ENGINEER (ISO 13374 Layer 3: State Detection)│
    │  86 temporal features: rolling mean, std, slope, and change      │
    │  Vectorized numpy/pandas: identical code trains and serves      │
    └────────────────────────────────┬────────────────────────────────┘
                                     │
    ┌────────────────────────────────▼────────────────────────────────┐
    │  STATION 4: TRAINER TOURNAMENT (ISO 13374 Layer 4: Health Assess)│
    │  GroupKFold(5) grouped by unit to prevent temporal data leakage │
    │  Seeded Optuna search; ranked strictly by CV ROC-AUC             │
    └────────────────────────────────┬────────────────────────────────┘
                                     │
    ┌────────────────────────────────▼────────────────────────────────┐
    │  STATION 5: EXPLAINER (ISO 13374 Layer 5: Prognostic Assessment) │
    │  Exact Tree-SHAP mathematical attributions & physical z-scores   │
    │  Empirical confidence: observed historical hit rate per band    │
    └────────────────────────────────┬────────────────────────────────┘
                                     │
    ┌────────────────────────────────▼────────────────────────────────┐
    │  STATION 6: DEPLOYER & MONITOR (ISO 13374 Layer 6: Advisory Gen) │
    │  Generates self-contained FastAPI edge container (0.0 diff)     │
    │  Early-life window PSI drift monitor; auto-retrain tournament    │
    └─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results on NASA C-MAPSS FD001

Evaluated on NASA's 100-engine turbofan run-to-failure benchmark. Target derived automatically: *fails within 30 cycles* ($0.15 \times \text{median operating life of 199 cycles}$). The official NASA test set of 100 engines was completely held out during model selection:

| Model Family | 5-Fold Grouped CV ROC-AUC | Holdout ROC-AUC (All Rows) | Holdout AP (Final Obs) | Status |
|:---|:---:|:---:|:---:|:---:|
| **XGBoost** | **0.9930 ± 0.0018** | **0.9934** | **0.949** | **Champion** |
| LightGBM | 0.9930 ± 0.0017 | 0.9932 | 0.947 | Evaluated |
| Linear Baseline | 0.9921 ± 0.0008 | 0.9918 | 0.950 | Evaluated |
| Random Forest | 0.9909 ± 0.0020 | 0.9924 | 0.959 | Evaluated |

*On the final observation of each of the 100 test engines, the champion achieves ROC-AUC 0.981, Average Precision 0.949, and F1 0.809. The linear baseline is within 0.001 CV ROC-AUC of the champion, so most of the signal in this data is simple; the leaderboard shows that rather than hiding it.*

---

## 🔒 Security and offline operation

- **No external calls at run time**: fonts (IBM Plex, SIL OFL), stylesheets and scripts are bundled in `web/`, so the app makes no requests to Google Fonts or a CDN.
- **Docker socket is optional**: the Deploy step's container smoke test needs it. Set `DOCKER_SOCK=/dev/null` to skip that test; the service package is still written.
- **Bounded file access and uploads**: paths are resolved and kept inside their folders; uploads are capped at 50 MB, `.csv`/`.txt` only, and must parse as CSV.
- Sentinel is a prototype and has not been assessed against IEC 62443 or any other security standard. See [SECURITY.md](SECURITY.md).

---

## 🚀 Quick Start

### 1. Hosted Demo (Zero Install)
Open the replay of a real run, with the what-if sketch and a printable run report:  
👉 **[https://coderjt-elite.github.io/sentinel-ml/demo/](https://coderjt-elite.github.io/sentinel-ml/demo/)**

### 2. Full Local Docker Compose Stack
Includes the FastAPI application, PostgreSQL persistence store, and local MLflow tracking server:

```bash
git clone https://github.com/CoderJT-Elite/sentinel-ml.git
cd sentinel-ml
python scripts/fetch_data.py        # Downloads public NASA C-MAPSS and UCI AI4I
docker compose up --build
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

### 3. Local Python Execution & Testing
```bash
pip install -r requirements.txt
python scripts/fetch_data.py
python -m sentinel.cli run cmapss_fd001 --budget fast
pytest -q                           # 19 automated tests (100% pass)
```

---

## 📁 Repository Structure

```
sentinel-ml/
├── api/                  # FastAPI REST API & Server-Sent Events (SSE) streaming
├── docs/                 # Technical documentation, PDF manual, and static hosted demo
│   ├── demo/             # Complete static recorded run for GitHub Pages
│   └── TECHNICAL_DOCUMENTATION.md
├── runs/                 # Versioned model artifacts, logs, and serving packages
├── samples/              # Bundled failure-mode datasets (e.g. broken_sensor_log.csv)
├── scripts/              # Dataset fetchers, static exporters, and PDF report builders
├── sentinel/             # Core deterministic pipeline stations
│   ├── steward.py        # Station 1: Data profiling & quality gate
│   ├── selector.py       # Station 2: Task framing & alarm horizon calculation
│   ├── features.py       # Station 3: Vectorized sliding window transforms
│   ├── trainer.py        # Station 4: Seeded Optuna search & GroupKFold cross-validation
│   ├── explainer.py      # Station 5: Native Tree-SHAP mathematical attributions
│   ├── deployer.py       # Station 6: Standalone edge service compiler & parity testing
│   ├── monitor.py        # Continuous Population Stability Index (PSI) drift engine
│   └── graph.py          # Deterministic LangGraph state machine & Run Hash engine
├── tests/                # 19 automated test suites (determinism, parity, AST import audit)
├── web/                  # Vanilla JS/CSS single-page UI (self-hosted offline fonts)
├── docker-compose.yml    # App + PostgreSQL + MLflow service orchestration
├── Dockerfile            # Container definition
└── requirements.txt      # Pinned production dependency specifications
```

---

## 📖 Citations & Acknowledgments

- **NASA C-MAPSS**: Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation.* PHM08, Denver, CO.
- **UCI AI4I 2020**: Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance Applications.* AI4I 2020, pp. 69-74. IEEE. (CC BY 4.0).
- **Downtime cost**: Siemens (2024). *The True Cost of Downtime 2024.* Figure quoted as reported by Siemens.
- **Typography**: IBM Plex by IBM Corp. (SIL Open Font License v1.1).

---

## ⚖️ License

Sentinel is licensed under the **[MIT License](LICENSE)**. Built by **John Tewolde** for the **ABB Accelerator 2026**.
