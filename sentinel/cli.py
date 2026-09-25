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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sentinel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pipeline on a dataset")
    r.add_argument("dataset", choices=sorted(data.REGISTRY))
    r.add_argument("--budget", choices=sorted(config.BUDGETS), default="fast")
    r.add_argument("--task", choices=["auto", "classification", "regression"], default="auto")
    a = ap.parse_args(argv)
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
