.PHONY: data test run up demo
data:      ## download NASA C-MAPSS FD001 + UCI AI4I
	python scripts/fetch_data.py
test:      ## unit + end-to-end tests (synthetic data, no download needed)
	python -m pytest -q -W ignore
run:       ## run the whole pipeline from the command line
	python -m sentinel.cli run cmapss_fd001 --budget fast
up:        ## full stack: app + PostgreSQL + MLflow on http://localhost:8000
	docker compose up --build
demo:      ## rebuild the static GitHub Pages demo from a fresh run
	python scripts/export_static.py --fresh
