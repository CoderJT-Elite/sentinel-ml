"""Export a finished run as the static GitHub Pages demo (docs/demo).

    python scripts/export_static.py run-abc123          # export an existing run in ./runs
    python scripts/export_static.py --fresh             # run the pipeline first, then export

Everything in the exported JSON is produced by the real pipeline, monitor and retrainer:
the page just replays it. Drift scenarios are precomputed on a grid because the browser
cannot recompute rolling features and PSI on the server-side model.
"""
import argparse
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentinel import config, data, deployer, lifecycle  # noqa: E402
from sentinel.graph import run_pipeline  # noqa: E402
from sentinel.util import clean  # noqa: E402

SIGMAS = [0, 0.25, 0.5, 1, 1.5, 2, 3]
RETRAIN_SIGMAS = [1, 2, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo"))
    ap.add_argument("--no-docker", action="store_true")
    a = ap.parse_args()

    events = []
    if a.fresh:
        res = run_pipeline(data.load("cmapss_fd001"), {"budget": "fast"}, events.append)
        run_id = res["run_id"]
        (pathlib.Path(config.RUNS_DIR) / run_id / "events.json").write_text(json.dumps(clean(events)))
    else:
        run_id = a.run_id
    run_dir = pathlib.Path(config.RUNS_DIR) / run_id
    summary = json.loads((run_dir / "summary.json").read_text())
    events = json.loads((run_dir / "events.json").read_text())

    out = pathlib.Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "web", out)
    (out / "data").mkdir()
    (out / "data" / "run.json").write_text(json.dumps(summary, separators=(",", ":")))
    (out / "data" / "events.json").write_text(json.dumps(events, separators=(",", ":")))

    sensors = [s["sensor"] for s in summary["explain"]["global"]["sensors"]
               if s["sensor"] != summary["framing"].get("time_col")][:6]
    grid, retrain = {}, {}
    for s in sensors:
        grid[s] = {}
        for sg in SIGMAS:
            grid[s][str(sg)] = lifecycle.drift(run_id, s, sg)
        if s in sensors[:3]:
            retrain[s] = {str(sg): lifecycle.retrain(run_id, s, sg, register=False) for sg in RETRAIN_SIGMAS}
        print("drift grid done for", s)
    (out / "data" / "drift.json").write_text(json.dumps(clean({"sigmas": SIGMAS, "grid": grid, "retrain": retrain}), separators=(",", ":")))

    ctx = lifecycle.load(run_id)
    dep = {"folder": pathlib.Path(summary["serving"]["folder"]).name, "files": sorted(p.name for p in pathlib.Path(summary["serving"]["folder"]).iterdir()),
           "docker": {"ran": False, "reason": "skipped"}}
    if not a.no_docker:
        dep["docker"] = deployer.docker_smoke(summary["serving"]["folder"], "sentinel-model:export", ctx["sample"]["rows"], ctx["sample"]["expected"])
    (out / "data" / "deploy.json").write_text(json.dumps(clean(dep)))
    (out / ".nojekyll").write_text("")
    print("exported", run_id, "->", out)


if __name__ == "__main__":
    main()
