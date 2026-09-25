"""After deployment: drift detection on live data and champion/challenger retraining.

The scenario is a sensor calibration offset injected into the holdout fleet. Everything
downstream is the same closed-form machinery: PSI against the training distribution,
a fixed retrain rule, a deterministic refit, and a champion/challenger comparison on
units the challenger never saw.
"""
from __future__ import annotations

import os
import pathlib
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from . import config, deployer, explainer, features, monitor, registry, trainer
from .util import clean

_CACHE: dict = {}


def load(run_id: str) -> dict:
    if run_id not in _CACHE:
        p = pathlib.Path(config.RUNS_DIR) / run_id / "internal.joblib"
        if not p.exists():
            raise FileNotFoundError(run_id)
        _CACHE[run_id] = joblib.load(p)
    return _CACHE[run_id]


def _shifted(ctx: dict, sensor: str, sigma: float) -> pd.DataFrame:
    spec = ctx["spec"]
    std = float(ctx["train"][sensor].std()) or 1.0
    return monitor.inject_shift(ctx["hold"], sensor, sigma, std)


def _split_halves(ctx: dict, hold: pd.DataFrame):
    """Adaptation set A / evaluation set B, split by unit (or by row when there are no units)."""
    spec = ctx["spec"]
    if spec.temporal:
        units = np.array(sorted(hold[spec.entity_col].unique()))
        a_units = set(units[::2].tolist())
        mask = hold[spec.entity_col].isin(a_units).to_numpy()
    else:
        mask = np.zeros(len(hold), dtype=bool)
        mask[::2] = True
    return mask


def _score(ctx: dict, model, hold: pd.DataFrame, X: pd.DataFrame, mask: np.ndarray) -> dict:
    fr = ctx["framing"]
    y = hold["__target__"].to_numpy()
    p = trainer.predict(model, X, fr["task"])
    m = trainer.metric_value(fr["metric"], y[mask], p[mask])
    out = {fr["metric"]: m}
    if fr["task"] == "classification" and ctx["threshold"] is not None:
        from sklearn.metrics import f1_score
        out["f1"] = float(f1_score(y[mask], p[mask] >= ctx["threshold"], zero_division=0))
    return out


def drift(run_id: str, sensor: str, sigma: float) -> dict:
    ctx = load(run_id)
    spec, fr = ctx["spec"], ctx["framing"]
    if sensor not in (spec.sensor_cols or spec.numeric_cols):
        raise ValueError(f"unknown sensor {sensor!r}")
    hold_s = _shifted(ctx, sensor, sigma)
    Xs = features.transform(hold_s, spec)
    e_tr, e_ho = ctx["early_train"], monitor.early_mask(hold_s, spec)
    rep = monitor.drift_report(ctx["drift_ref"], ctx["X"][e_tr], Xs[e_ho], ctx["importance"])
    champ = ctx["models"][ctx["champion"]]
    allmask = np.ones(len(hold_s), dtype=bool)
    Xh = features.transform(ctx["hold"], spec)
    before = _score(ctx, champ, ctx["hold"], Xh, allmask)
    after = _score(ctx, champ, hold_s, Xs, allmask)
    return {"run_id": run_id, "sensor": sensor, "sigma": sigma, "report": rep,
            "champion_before": before, "champion_after": after, "metric": fr["metric"]}


def retrain(run_id: str, sensor: str, sigma: float, register: bool = True) -> dict:
    """Refit the champion family on train + adaptation units, compare with the deployed model on unseen units."""
    ctx = load(run_id)
    spec, fr = ctx["spec"], ctx["framing"]
    rep = drift(run_id, sensor, sigma)
    if rep["report"]["status"] != "RETRAIN":
        return {"triggered": False, "drift": rep, "reason": "retrain rule not tripped: " + rep["report"]["reason"]}
    hold_s = _shifted(ctx, sensor, sigma)
    Xs = features.transform(hold_s, spec)
    a = _split_halves(ctx, hold_s)
    b = ~a
    fam = ctx["champion"]
    champ_row = next(r for r in ctx["board"] if r["family"] == fam)
    y = ctx["y"]
    pos = float(y.sum())
    pos_weight = float((len(y) - pos) / pos) if pos else 1.0
    X2 = pd.concat([ctx["X"], Xs[a]], ignore_index=True)
    y2 = np.concatenate([y, hold_s["__target__"].to_numpy()[a]])
    challenger = trainer.build(fam, champ_row["params"], fr["task"], bool(fr.get("class_weight")), pos_weight)
    challenger.fit(X2, y2)
    incumbent = ctx["models"][fam]
    inc = _score(ctx, incumbent, hold_s, Xs, b)
    chal = _score(ctx, challenger, hold_s, Xs, b)
    metric = fr["metric"]
    better = (chal[metric] > inc[metric]) if trainer.higher_is_better(metric) else (chal[metric] < inc[metric])
    out = {"triggered": True, "drift": rep, "metric": metric, "incumbent": inc, "challenger": chal,
           "eval_rows": int(b.sum()), "adaptation_rows": int(a.sum()), "promoted": bool(better),
           "rule": "promote if the challenger beats the incumbent on units it never saw"}
    if better and register:
        rh = ctx["run_hash"]
        reg = registry.log_run(
            ctx["reg"]["model_name"].replace("sentinel-", "", 1), rh + "-retrain",
            {"champion": fam, **champ_row["params"], "retrain_trigger": f"{sensor} shift {sigma} sd"},
            {f"challenger_{metric}": chal[metric], f"incumbent_{metric}": inc[metric]}, {}, challenger,
            {"sentinel.kind": "retrain", "sentinel.parent_run_hash": rh})
        ctx["models"][fam] = challenger
        out["registry"] = reg
        pkg = deployer.package(str(pathlib.Path(config.RUNS_DIR) / run_id / f"serving_v{reg['version']}"), family=fam,
                               model=challenger, spec=spec, ref=ctx["ref"], task=fr["task"], threshold=ctx["threshold"],
                               reliability=ctx.get("reliability", []), meta={"model_name": reg["model_name"], "version": reg["version"],
                                                     "run_hash": rh, "dataset": ctx["meta"].get("source", ""),
                                                     "sensor_labels": ctx["labels"], "entity_noun": ctx["meta"].get("entity_noun", "Unit"),
                                                     "time_noun": ctx["meta"].get("time_noun", "cycle"), "metric": metric,
                                                     "cv_mean": champ_row["cv_mean"], "horizon": fr.get("horizon"), "llm_calls": 0})
        out["serving_folder"] = pkg
        out["versions"] = registry.versions(reg["model_name"])
    return clean(out)
