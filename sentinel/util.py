"""Small helpers: canonical JSON and the run hash that makes a run auditable."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


class Raw:
    """Marks a value that `clean` must pass through unrounded (model thresholds and feature values)."""

    def __init__(self, value):
        self.value = value


def clean(o, nd: int = 6):
    """Recursively convert numpy/pandas values to JSON-safe python, rounding floats."""
    if isinstance(o, Raw):
        return o.value
    if isinstance(o, dict):
        return {str(k): clean(v, nd) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean(v, nd) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v, nd) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if f != f or f in (float("inf"), float("-inf")) else round(f, nd)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return o


def canonical(o) -> str:
    return json.dumps(clean(o), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def frame_fingerprint(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()
    return hashlib.sha256(h + ",".join(map(str, df.columns)).encode()).hexdigest()


def run_hash(dataset_fp: str, options: dict, decisions: list, leaderboard: list) -> str:
    """Hash of everything that determines a run. Wall-clock fields are excluded on purpose."""
    board = [{k: r[k] for k in ("family", "params", "cv_mean", "cv_std", "cv_folds", "metric", "rank")} for r in leaderboard]
    return sha256(canonical({"data": dataset_fp, "options": options, "decisions": decisions, "leaderboard": board}))
