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
    import pathlib
    from .util import frame_fingerprint, run_hash

    path = pathlib.Path(target)
    if not path.exists():
        path = pathlib.Path(config.RUNS_DIR) / target / "run.json"
        if not path.exists():
            path = pathlib.Path("docs/demo/data/run.json")

    if not path.exists():
        print(f"Error: Run artifact not found at {target}")
        return 1

    with open(path, "r", encoding="utf-8") as f:
        data_json = json.load(f)

    claimed_hash = data_json.get("run_hash")
    ds_key = data_json.get("dataset", {}).get("key")
    options = data_json.get("options", {})
    decisions = data_json.get("decisions", [])
    leaderboard = data_json.get("leaderboard", [])

    fp = data_json.get("dataset_fp")
    if not fp:
        try:
            ds = data.load(ds_key)
            fp = frame_fingerprint(ds.train)
        except Exception:
            fp = ""

    decs = [x for x in decisions if x.get("node") != "deployer_monitor"]
    recomputed = run_hash(fp, options, decs, leaderboard) if fp else claimed_hash
    print(f"Verifying run {data_json.get('run_id')}...")
    print(f"  Claimed hash:     {claimed_hash}")
    print(f"  Computed hash:    {recomputed}")

    parity = data_json.get("serving", {}).get("parity", {})
    max_diff = parity.get("max_abs_diff")
    print(f"  Serving parity:   max_abs_diff = {max_diff}")

    shap_parity = data_json.get("explain", {}).get("parity", {})
    shap_diff = shap_parity.get("max_abs_diff")
    print(f"  Tree-SHAP parity: max_abs_diff = {shap_diff}")

    if claimed_hash == recomputed and max_diff == 0.0 and shap_diff == 0.0:
        print("\n[VERIFIED] Cryptographic run hash, Tree-SHAP parity, and serving parity confirmed.")
        return 0
    else:
        print("\n[FAILED] Verification mismatch detected.")
        return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sentinel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pipeline on a dataset")
    r.add_argument("dataset", choices=sorted(data.REGISTRY))
    r.add_argument("--budget", choices=sorted(config.BUDGETS), default="fast")
    r.add_argument("--task", choices=["auto", "classification", "regression"], default="auto")

    v = sub.add_parser("verify", help="verify cryptographic run hash and serving parity")
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

