# Contributing to Sentinel

Thank you for your interest in contributing to Sentinel. Sentinel is built for mission-critical industrial predictive maintenance where reproducibility, auditability, and safety are non-negotiable.

---

## Inviolable Design Principles

Before submitting code, ensure your contribution respects the core architectural boundaries:

1. **Zero LLM in the Runtime Decision Path**: Never import or invoke a generative language model, stochastic prompt engine, or non-deterministic agent in `sentinel/` or `api/`. The automated AST import audit test (`tests/test_pipeline.py::test_no_llm_imports_anywhere_in_the_package`) must remain passing.
2. **Determinism is Sacred**: Every algorithm, feature transform, hyperparameter search, and explanation must use seeded pseudo-randomness and deterministic ordering. Given the same input data, configuration, and seed, the resulting SHA-256 run hash must be identical.
3. **Audit Ledger Logging**: Every heuristic, quality check, metric choice, dropped column, and drift decision must emit a structured entry to the immutable decision ledger.
4. **Offline by default**: no external CDN calls from the web interface. Dependencies must be pinned and permissively licensed.

---

## Development Setup

### Local Prerequisites
- Python 3.12+ (or Docker Desktop)
- Git

### Quickstart
```bash
git clone https://github.com/CoderJT-Elite/sentinel-ml.git
cd sentinel-ml
pip install -r requirements.txt
python scripts/fetch_data.py
pytest -q
```

### Running the Docker Stack
```bash
docker compose up --build
```
Open [http://localhost:8000](http://localhost:8000) to view the application.

---

## Pull Request Guidelines

1. **Test Coverage**: Every new feature or bug fix must include corresponding tests under `tests/`.
2. **Determinism Verification**: Confirm that all existing tests pass without altering established baselines.
3. **Code Style**: Follow PEP 8 and use type annotations where appropriate. Keep functions concise and well-commented.
