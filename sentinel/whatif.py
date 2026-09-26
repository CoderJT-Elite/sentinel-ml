"""Flatten a trained gradient-boosted model into plain arrays so a browser can re-score it.

The what-if panel in the UI edits a few feature values and re-scores the *actual champion trees*
in JavaScript (`web/whatif.js`). This module does the export and holds a reference Python
scorer that walks the same arrays, so a test can prove the arrays reproduce the library's own
predictions. Nothing here is approximate: leaf values and split thresholds are exported as
the precision each library compares in (float32 for XGBoost, double for LightGBM).

Tree layout, one entry per tree, all arrays indexed by node id:
    s  split feature index (-1 on a leaf)
    c  split threshold, or the leaf value on a leaf
    l  left child   (taken when x < c for XGBoost, x <= c for LightGBM)
    r  right child
    m  child taken when the value is missing (NaN)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd


def _f32(x: float) -> float:
    """A float32 value written with the 9 significant digits that round-trip it exactly."""
    return float(f"{float(np.float32(x)):.9g}")


def _xgb_trees(model, names: list) -> list:
    """Trees from XGBoost's JSON model dump, which writes every number so that it reads back exactly."""
    import json
    raw = json.loads(bytes(model.get_booster().save_raw(raw_format="json")))
    trees = []
    for t in raw["learner"]["gradient_booster"]["model"]["trees"]:
        left, right, cond = t["left_children"], t["right_children"], t["split_conditions"]
        s, c, l, r, m = [], [], list(left), list(right), []
        for k in range(len(left)):
            if left[k] == -1:
                s.append(-1)
                c.append(float(cond[k]))
                m.append(-1)
            else:
                s.append(int(t["split_indices"][k]))
                c.append(float(cond[k]))
                m.append(left[k] if t["default_left"][k] else right[k])
        trees.append({"s": s, "c": c, "l": l, "r": r, "m": m})
    return trees


def _lgb_trees(model) -> list:
    dump = model.booster_.dump_model()
    trees = []
    for info in dump["tree_info"]:
        s, c, l, r, m = [], [], [], [], []

        def walk(node) -> int:
            k = len(s)
            for arr, v in ((s, -1), (c, 0.0), (l, -1), (r, -1), (m, -1)):
                arr.append(v)
            if "leaf_value" in node:
                c[k] = float(node["leaf_value"])
                return k
            if node.get("decision_type", "<=") != "<=":
                raise ValueError("categorical splits are not exported")
            s[k], c[k] = int(node["split_feature"]), float(node["threshold"])
            left, right = walk(node["left_child"]), walk(node["right_child"])
            l[k], r[k] = left, right
            m[k] = left if node.get("default_left", True) else right
            return k

        walk(info["tree_structure"])
        trees.append({"s": s, "c": c, "l": l, "r": r, "m": m})
    return trees


def score_margin(kind: str, trees: list, x: np.ndarray) -> float:
    """Reference scorer. `x` is one feature row. Returns the sum of leaf values."""
    total = 0.0
    xf = np.asarray(x, dtype=np.float32) if kind == "xgboost" else np.asarray(x, dtype=float)
    for t in trees:
        node = 0
        while t["s"][node] != -1:
            v = xf[t["s"][node]]
            if math.isnan(v):
                node = t["m"][node]
            elif (v < np.float32(t["c"][node])) if kind == "xgboost" else (v <= t["c"][node]):
                node = t["l"][node]
            else:
                node = t["r"][node]
        total += t["c"][node]
    return total


def export(family: str, model, X: pd.DataFrame, ref: dict, task: str) -> Optional[dict]:
    """The what-if payload for a champion, or None when the family or task is not supported."""
    if task != "classification" or family not in ("xgboost", "lightgbm"):
        return None
    names = list(X.columns)
    trees = _xgb_trees(model, names) if family == "xgboost" else _lgb_trees(model)
    probe = X.iloc[: min(50, len(X))]
    if family == "xgboost":
        import xgboost as xgb
        margin = model.get_booster().predict(xgb.DMatrix(probe), output_margin=True)
    else:
        margin = model.booster_.predict(probe, raw_score=True)
    sums = np.array([score_margin(family, trees, row) for row in probe.to_numpy(dtype=float)])
    offsets = margin - sums
    base = float(np.median(offsets))
    if float(np.max(np.abs(offsets - base))) > 1e-4:
        return None  # the flat trees do not reproduce the library's margin; do not ship a wrong scorer
    return {"kind": family, "base": round(base, 6), "features": names, "trees": trees,
            "ref_mean": [round(float(ref["mean"].get(n, 0.0)), 6) for n in names],
            "ref_std": [round(float(ref["std"].get(n) or 1.0), 6) for n in names]}


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))
