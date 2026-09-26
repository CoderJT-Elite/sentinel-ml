"""Command line: `python -m sentinel.cli run cmapss_fd001 --budget fast`."""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import config, data
from .graph import NODES, run_pipeline


def _print_event(e: dict) -> None:
    if e["status"] in ("done", "halt"):
        label = dict(NODES).get(e["node"], e["node"])
        print(f"[{label}] {e['title']}")
        for d in e["decisions"]:
            print(f"    - {d['rule']}: {d['outcome']}")
    elif e["status"] == "progress" and e["payload"].get("type") == "family_done":
        p = e["payload"]
        print(f"    {p['family']}: cv={p['cv_mean']} (+/-{p['cv_std']}) in {p['seconds']}s")


def verify_run(target: str) -> int:
    """Recompute a run's SHA-256 from the record itself and compare it with the hash the run claimed.

    The hash covers the dataset fingerprint, the options, the decision log and the leaderboard, so this
    checks that none of them was edited after the run. It does not retrain anything: the serving and
    Tree-SHAP parity figures are read from the record, where the run measured them.
    """
    import pathlib
    from .util import frame_fingerprint, run_hash

    path = pathlib.Path(target)
    if not path.exists():
        path = pathlib.Path(config.RUNS_DIR) / target / "summary.json"
    if not path.exists():
        print(f"Run record not found: {target}")
        return 2
    rec = json.loads(path.read_text(encoding="utf-8"))
    claimed = rec.get("run_hash")
    fp = rec.get("dataset_fp")
    if not fp:
        try:
            fp = frame_fingerprint(data.load(rec["dataset"]["key"]).train)
        except Exception as e:
            print(f"Cannot verify: the record has no dataset fingerprint and the dataset could not be loaded ({e}).")
            return 2
    decisions = [d for d in rec.get("decisions", []) if d.get("node") != "deployer_monitor"]
    computed = run_hash(fp, rec.get("options", {}), decisions, rec.get("leaderboard", []))
    par = rec.get("serving", {}).get("parity", {}).get("max_abs_diff")
    shap_par = rec.get("explain", {}).get("parity", {}).get("max_abs_diff")
    print(f"Run {rec.get('run_id')}")
    print(f"  claimed hash   {claimed}")
    print(f"  computed hash  {computed}")
    print(f"  serving parity (recorded)    max abs diff {par}")
    print(f"  Tree-SHAP parity (recorded)  max abs diff {shap_par}")
    if claimed == computed:
        print("Hash matches: the decision log, leaderboard, options and data fingerprint are as the run left them.")
        return 0
    print("Hash does not match: the record was changed after the run, or it came from a different version of Sentinel.")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sentinel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pipeline on a dataset")
    r.add_argument("dataset", choices=sorted(data.REGISTRY))
    r.add_argument("--budget", choices=sorted(config.BUDGETS), default="fast")
    r.add_argument("--task", choices=["auto", "classification", "regression"], default="auto")

    v = sub.add_parser("verify", help="recompute a run's SHA-256 from its record and compare")
    v.add_argument("target", nargs="?", default="docs/demo/data/run.json", help="path to run.json or run-id")

    a = ap.parse_args(argv)
    if a.cmd == "verify":
        return verify_run(a.target)

    t0 = time.time()
    res = run_pipeline(data.load(a.dataset), {"budget": a.budget, "task": a.task}, _print_event)
    if res.get("halted"):
        print("HALTED by the quality gate")
        return 2
    c = res["leaderboard"][0]
    print(f"\nrun {res['run_id']}  hash {res['run_hash']}  champion {c['name']}  ({time.time() - t0:.0f}s)")
    print(json.dumps(c["holdout"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

