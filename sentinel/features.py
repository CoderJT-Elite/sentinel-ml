"""Node 3, the Feature Engineer: deterministic transforms driven by a serialisable spec.

This module is self-contained (numpy + pandas only) on purpose: the deployer copies it
verbatim into the serving container, so training and serving cannot disagree about features.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

SUFFIX_TEXT = {
    "": "current value",
    "__mean5": "5-cycle average",
    "__std5": "5-cycle volatility",
    "__slope10": "10-cycle trend",
    "__diff5": "change vs 5 cycles ago",
}


@dataclass
class FeatureSpec:
    entity_col: Optional[str] = None
    time_col: Optional[str] = None
    sensor_cols: list = field(default_factory=list)     # numeric columns that get temporal expansion
    numeric_cols: list = field(default_factory=list)    # numeric columns used as-is (tabular)
    categorical: dict = field(default_factory=dict)     # col -> sorted list of categories (one-hot)
    roll_mean: int = 5
    roll_std: int = 5
    roll_slope: int = 10
    lag_diff: int = 5
    feature_names: list = field(default_factory=list)
    dropped: list = field(default_factory=list)         # [{col, reason}]
    display: dict = field(default_factory=dict)         # safe feature stem -> original column name

    @property
    def temporal(self) -> bool:
        return bool(self.entity_col and self.time_col)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @staticmethod
    def from_dict(d: dict) -> "FeatureSpec":
        return FeatureSpec(**d)


def _windowed(a: np.ndarray, w: int) -> np.ndarray:
    """(n, k) -> (n, k, w) trailing windows; the series start is padded with its first row."""
    pad = np.concatenate([np.repeat(a[:1], w - 1, axis=0), a], axis=0)
    return sliding_window_view(pad, w, axis=0)


def _temporal_block(a: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    """All temporal features for one entity's (n, k) sensor matrix, ordered per `feature_names`."""
    n, k = a.shape
    mean5 = _windowed(a, spec.roll_mean).mean(axis=2)
    std5 = _windowed(a, spec.roll_std).std(axis=2)
    w = _windowed(a, spec.roll_slope)
    x = np.arange(spec.roll_slope, dtype=float)
    xc = x - x.mean()
    slope = (w * xc).sum(axis=2) / (xc ** 2).sum()
    lag = np.maximum(np.arange(n) - spec.lag_diff, 0)
    diff5 = a - a[lag]
    # interleave to (n, k*5): for sensor j -> [raw, mean5, std5, slope10, diff5]
    out = np.stack([a, mean5, std5, slope, diff5], axis=2)
    return out.reshape(n, k * 5)


def safe(name: str) -> str:
    """Model libraries reject brackets, quotes, commas etc. in feature names; keep a readable stem."""
    return re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_") or "col"


def feature_names_for(spec: FeatureSpec) -> list:
    names = []
    if spec.temporal:
        names.append(safe(spec.time_col))
        for c in spec.sensor_cols:
            b = safe(c)
            names += [b, f"{b}__mean5", f"{b}__std5", f"{b}__slope10", f"{b}__diff5"]
    else:
        names += [safe(c) for c in spec.numeric_cols]
    for c, cats in spec.categorical.items():
        names += [f"{safe(c)}={safe(v)}" for v in cats]
    return names


def prepare(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Stable-sort by (entity, time) so windows are computed on ordered series."""
    if spec.temporal:
        return df.sort_values([spec.entity_col, spec.time_col], kind="mergesort").reset_index(drop=True)
    return df.reset_index(drop=True)


def transform(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Feature matrix for an already `prepare`d frame. Row i of the output is row i of the input."""
    parts = []
    if spec.temporal:
        cols = [spec.time_col]
        time = df[[spec.time_col]].to_numpy(dtype=float)
        sens = df[spec.sensor_cols].to_numpy(dtype=float)
        ent = df[spec.entity_col].to_numpy()
        blocks = np.empty((len(df), len(spec.sensor_cols) * 5))
        change = np.flatnonzero(ent[1:] != ent[:-1]) + 1
        bounds = np.concatenate([[0], change, [len(df)]])
        for s, e in zip(bounds[:-1], bounds[1:]):
            blocks[s:e] = _temporal_block(sens[s:e], spec)
        parts.append(time)
        parts.append(blocks)
    else:
        parts.append(df[spec.numeric_cols].to_numpy(dtype=float))
    for c, cats in spec.categorical.items():
        v = df[c].astype(str).to_numpy()
        parts.append(np.stack([(v == str(cat)).astype(float) for cat in cats], axis=1))
    X = np.concatenate(parts, axis=1)
    names = spec.feature_names or feature_names_for(spec)
    return pd.DataFrame(X, columns=names)


def plan(df: pd.DataFrame, struct: Optional[dict], drop: list, exclude: set) -> FeatureSpec:
    """Decide which columns become features. `drop` = [{col, reason}] from the Steward."""
    dropped = list(drop)
    dropset = {d["col"] for d in drop} | set(exclude)
    spec = FeatureSpec(dropped=dropped)
    if struct:
        spec.entity_col, spec.time_col = struct["entity_col"], struct["time_col"]
        dropset |= {struct["entity_col"], struct["time_col"]}
    numeric = [c for c in df.columns if c not in dropset and pd.api.types.is_numeric_dtype(df[c])]
    cats = [c for c in df.columns if c not in dropset and not pd.api.types.is_numeric_dtype(df[c])]
    if spec.temporal:
        spec.sensor_cols = numeric
    else:
        spec.numeric_cols = numeric
    spec.categorical = {c: sorted(df[c].astype(str).unique().tolist()) for c in cats}
    spec.feature_names = feature_names_for(spec)
    cols = ([spec.time_col] if spec.temporal else []) + list(spec.sensor_cols) + list(spec.numeric_cols) + list(spec.categorical)
    spec.display = {safe(c): c for c in cols if safe(c) != c}
    return spec


def humanize(name: str, labels: dict) -> str:
    """'s11__slope10' -> 'Ps30 static pressure at HPC outlet, 10-cycle trend'."""
    for suf, text in sorted(SUFFIX_TEXT.items(), key=lambda kv: -len(kv[0])):
        if suf and name.endswith(suf):
            base = name[: -len(suf)]
            return f"{labels.get(base, base)}, {text}"
    if "=" in name:
        c, v = name.split("=", 1)
        return f"{labels.get(c, c)} = {v}"
    return labels.get(name, name)
