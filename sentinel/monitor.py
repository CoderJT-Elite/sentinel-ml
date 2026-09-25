"""Node 6b, the Monitor: statistical drift detection with a fixed retrain threshold.

PSI (Population Stability Index) and a two-sample KS test compare live feature
distributions with the training distributions. Retraining is triggered by a rule,
not by anyone's judgement: any top-10 important feature with PSI >= 0.25.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config

EPS = 1e-4


def early_mask(df: pd.DataFrame, spec) -> np.ndarray:
    """Rows in the healthy early-life window (all rows for non-temporal data)."""
    if spec.temporal:
        return (df[spec.time_col].to_numpy() <= config.MONITOR_WINDOW)
    return np.ones(len(df), dtype=bool)


def make_reference(X: pd.DataFrame, bins: int = config.PSI_BINS) -> dict:
    """Per-feature quantile bin edges and training bin fractions (also exported for in-browser PSI)."""
    ref = {}
    for c in X.columns:
        v = X[c].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            continue
        inner = edges[1:-1]
        frac = np.bincount(np.searchsorted(inner, v, side="right"), minlength=len(edges) - 1) / len(v)
        ref[c] = {"inner": inner.tolist(), "frac": frac.tolist(), "mean": float(v.mean()), "std": float(v.std() or 1.0)}
    return ref


def psi_from_ref(ref_c: dict, cur: np.ndarray) -> float:
    cur = cur[~np.isnan(cur)]
    if len(cur) == 0:
        return 0.0
    inner = np.asarray(ref_c["inner"])
    frac = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=len(ref_c["frac"])) / len(cur)
    e = np.clip(np.asarray(ref_c["frac"]), EPS, None)
    a = np.clip(frac, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(ref: dict, X_train: pd.DataFrame, X_cur: pd.DataFrame, importance: list) -> dict:
    """importance: ordered list of feature names, most important first."""
    rows = []
    for c in X_cur.columns:
        if c not in ref:
            continue
        p = psi_from_ref(ref[c], X_cur[c].to_numpy(dtype=float))
        d, pv = stats.ks_2samp(X_train[c].to_numpy(dtype=float), X_cur[c].to_numpy(dtype=float))
        rows.append({"feature": c, "psi": p, "ks_d": float(d), "ks_p": float(pv)})
    by = {r["feature"]: r for r in rows}
    top = [by[f] for f in importance[:10] if f in by]
    worst = max(top, key=lambda r: r["psi"]) if top else None
    if worst and worst["psi"] >= config.PSI_RETRAIN:
        status, why = "RETRAIN", (f"{worst['feature']} PSI {worst['psi']:.2f} >= {config.PSI_RETRAIN} "
                                  "on a top-10 feature: retraining rule tripped")
    elif worst and worst["psi"] >= config.PSI_WATCH:
        status, why = "WATCH", f"{worst['feature']} PSI {worst['psi']:.2f} >= {config.PSI_WATCH}: monitoring closely"
    else:
        status, why = "STABLE", "no top-10 feature above the watch threshold"
    return {"status": status, "reason": why, "worst": worst, "top10": top,
            "n_watch": sum(1 for r in top if r["psi"] >= config.PSI_WATCH),
            "n_retrain": sum(1 for r in top if r["psi"] >= config.PSI_RETRAIN),
            "thresholds": {"watch": config.PSI_WATCH, "retrain": config.PSI_RETRAIN}}


def inject_shift(df: pd.DataFrame, sensor: str, sigma: float, train_std: float) -> pd.DataFrame:
    """Simulate a sensor calibration offset: shift one raw column by `sigma` training standard deviations."""
    out = df.copy()
    out[sensor] = out[sensor] + sigma * train_std
    return out
