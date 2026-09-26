# Contributing

Thanks for looking at Sentinel. It's a prototype, and the most useful contributions are the ones that protect what makes it different: it gives the same answer twice, and it never asks a language model to decide anything.

## Three rules I won't bend

1. **No language model in the decision path.** Don't import or call a generative model, a prompt engine or a non-deterministic agent anywhere in `sentinel/` or `api/`. The import audit in `tests/test_pipeline.py` has to keep passing.
2. **Same input, same hash.** Every algorithm, feature transform, search and explanation has to use seeded randomness and a fixed order. With the same data, configuration and seed, the SHA-256 run hash must come out identical.
3. **Log every decision.** A quality check, a metric choice, a dropped column or a drift call has to write a structured entry to the decision ledger, with the rule, the numbers it read and the threshold it used.

Also keep the web app offline by default: no CDN calls, pinned dependencies, and licenses that allow redistribution.

## Getting set up

```bash
git clone https://github.com/CoderJT-Elite/sentinel-ml.git
cd sentinel-ml
pip install -r requirements.txt
python scripts/fetch_data.py
pytest -q
```

To run the full stack with PostgreSQL and MLflow, use `docker compose up --build` and open [http://localhost:8000](http://localhost:8000).

The run hash depends on library versions, so check hashes inside the Docker image. That's the environment they're pinned for.

## Pull requests

- Add a test under `tests/` for any new feature or fix.
- Run the whole suite before you open the PR. If a change moves the run hash on purpose, say so in the description and explain why.
- Follow PEP 8, add type hints where they help, and comment the reason behind a rule, not what the line does.
