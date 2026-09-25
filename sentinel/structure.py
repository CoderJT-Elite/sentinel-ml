"""Detect run-to-failure structure (an entity column and a time counter) from the data alone."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

ENTITY_HINTS = ("unit", "engine", "asset", "machine", "device", "serial", "id", "equipment")
TIME_HINTS = ("cycle", "time", "step", "t", "hour", "day", "tick", "index")


def _contiguous_blocks(values: np.ndarray) -> bool:
    """True if every distinct value occupies exactly one contiguous run of rows."""
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate([[0], change])
    return len(np.unique(values[starts])) == len(starts)


def _is_counter_within(entity: np.ndarray, t: np.ndarray) -> bool:
    """Within each entity block, t must rise by exactly 1 on >=99% of steps."""
    same = entity[1:] == entity[:-1]
    if same.sum() == 0:
        return False
    steps = (t[1:] - t[:-1])[same]
    return float(np.mean(steps == 1)) >= 0.99


def _hint_rank(name: str, hints) -> int:
    n = name.lower()
    for i, h in enumerate(hints):
        if n == h or n.startswith(h) or n.endswith(h):
            return i
    return len(hints)


def detect_entity_time(df: pd.DataFrame) -> Optional[dict]:
    """Return {'entity_col','time_col','n_entities','median_life','evidence'} or None."""
    n = len(df)
    if n < 20:
        return None
    int_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().all()
        and np.all(np.mod(df[c].to_numpy(dtype=float), 1) == 0)
    ]
    best = None
    for e in int_cols:
        ev = df[e].to_numpy()
        k = len(np.unique(ev))
        if k < 2 or k > n / 5 or not _contiguous_blocks(ev):
            continue
        for t in int_cols:
            if t == e:
                continue
            if _is_counter_within(ev, df[t].to_numpy()):
                score = (_hint_rank(e, ENTITY_HINTS), _hint_rank(t, TIME_HINTS))
                if best is None or score < best[0]:
                    best = (score, e, t, k)
    if best is None:
        return None
    _, e, t, k = best
    life = df.groupby(e)[t].max()
    return {
        "entity_col": e,
        "time_col": t,
        "n_entities": int(k),
        "median_life": float(life.median()),
        "evidence": (
            f"'{e}' has {k} contiguous blocks and '{t}' rises by exactly 1 within each block "
            f"(median block length {life.median():.0f})"
        ),
    }
