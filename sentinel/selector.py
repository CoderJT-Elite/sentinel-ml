"""Node 2, the Task & Model Selector: closed-form rules that frame the problem.

Nothing here is learned. Each rule reads a number computed from the data, compares it
with a constant from `config`, and records the outcome in the decision log.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import config

CANDIDATES = ["lightgbm", "xgboost", "random_forest", "linear"]


def decision(rule: str, inputs: dict, outcome: str, threshold=None) -> dict:
    d = {"rule": rule, "inputs": inputs, "outcome": outcome}
    if threshold is not None:
        d["threshold"] = threshold
    return d


def frame(train: pd.DataFrame, meta: dict, options: dict, struct: Optional[dict]) -> dict:
    """Return the framing dict: task, target, entity/time, horizon, metric, candidates, decisions."""
    decisions = []
    target_col = options.get("target") or meta.get("target_col")
    requested = options.get("task", "auto")
    fr = {"task": None, "target_col": None, "derived": False, "entity_col": None, "time_col": None,
          "horizon": None, "metric": None, "class_weight": False, "candidates": list(CANDIDATES),
          "decisions": decisions}

    if struct:
        fr["entity_col"], fr["time_col"] = struct["entity_col"], struct["time_col"]

    if target_col and target_col in train.columns:
        y = train[target_col].dropna()
        nu = y.nunique()
        if nu == 2:
            fr["task"] = "classification"
            decisions.append(decision("TARGET_TYPE", {"target": target_col, "unique_values": int(nu)},
                                      "binary target -> classification"))
        elif pd.api.types.is_numeric_dtype(y) and nu > 10:
            fr["task"] = "regression"
            decisions.append(decision("TARGET_TYPE", {"target": target_col, "unique_values": int(nu)},
                                      "continuous target (> 10 distinct values) -> regression", threshold=10))
        else:
            raise ValueError(f"unsupported target {target_col!r}: {nu} distinct values, not binary or continuous")
        fr["target_col"] = target_col
    elif struct:
        life = struct["median_life"]
        h = int(options.get("horizon") or max(1, round(config.HORIZON_FRACTION * life)))
        fr["derived"] = True
        fr["horizon"] = h
        decisions.append(decision(
            "RUN_TO_FAILURE", {"entity": struct["entity_col"], "time": struct["time_col"],
                               "units": struct["n_entities"], "median_life": round(life, 1)},
            "each unit's series ends at failure -> target derived as remaining useful life (RUL)"))
        if requested == "regression":
            fr["task"], fr["target_col"] = "regression", "rul"
            decisions.append(decision("TASK_FRAMING", {"requested": "regression"},
                                      "regression on RUL (operator override)"))
        else:
            fr["task"], fr["target_col"] = "classification", f"fails_within_{h}"
            decisions.append(decision(
                "ALARM_HORIZON", {"median_life": round(life, 1), "fraction": config.HORIZON_FRACTION},
                f"alarm horizon = round({config.HORIZON_FRACTION} x median life) = {h} {struct.get('time_col')} steps; "
                f"target = fails within {h}",
                threshold=config.HORIZON_FRACTION))
    else:
        raise ValueError("no target column declared and no run-to-failure structure detected; "
                         "pick a target column")

    if requested in ("classification", "regression") and not fr["derived"] and requested != fr["task"]:
        decisions.append(decision("TASK_OVERRIDE_IGNORED", {"requested": requested, "data_supports": fr["task"]},
                                  f"requested {requested} but the target only supports {fr['task']}"))
    return fr


def attach_targets(df: pd.DataFrame, fr: dict, meta: dict, holdout: bool) -> pd.DataFrame:
    """Add a `__target__` column. `df` must already be sorted by (entity, time) when temporal."""
    out = df.copy()
    if fr["derived"]:
        if holdout:
            rul = out[meta["holdout_rul_col"]].to_numpy(dtype=float)
        else:
            rul = (out.groupby(fr["entity_col"])[fr["time_col"]].transform("max") - out[fr["time_col"]]).to_numpy(dtype=float)
        out["__rul__"] = rul
        out["__target__"] = rul if fr["task"] == "regression" else (rul <= fr["horizon"]).astype(int)
    else:
        y = out[fr["target_col"]]
        if fr["task"] == "classification":
            classes = sorted(pd.Series(y).dropna().unique().tolist())
            out["__target__"] = (y == classes[-1]).astype(int)
        else:
            out["__target__"] = y.astype(float)
    return out


def choose_metric(fr: dict, y: pd.Series) -> None:
    """Ranking metric + class weighting rules; mutates the framing."""
    if fr["task"] == "regression":
        fr["metric"] = "rmse"
        fr["decisions"].append(decision("RANKING_METRIC", {"task": "regression"},
                                        "rank candidates by cross-validated RMSE (lower is better)"))
        return
    share = float(min(y.mean(), 1 - y.mean()))
    if share < config.IMBALANCE_METRIC_SWITCH:
        fr["metric"] = "average_precision"
        fr["decisions"].append(decision(
            "RANKING_METRIC", {"minority_share": round(share, 4)},
            "minority class below 10% -> rank by cross-validated average precision (ROC-AUC is optimistic when positives are rare)",
            threshold=config.IMBALANCE_METRIC_SWITCH))
    else:
        fr["metric"] = "roc_auc"
        fr["decisions"].append(decision(
            "RANKING_METRIC", {"minority_share": round(share, 4)},
            "balanced enough -> rank by cross-validated ROC-AUC", threshold=config.IMBALANCE_METRIC_SWITCH))
    if share < 0.30:
        fr["class_weight"] = True
        fr["decisions"].append(decision("CLASS_WEIGHTING", {"minority_share": round(share, 4)},
                                        "minority class below 30% -> balanced class weights on every candidate",
                                        threshold=0.30))


def split_holdout(df: pd.DataFrame, fr: dict) -> tuple:
    """Seeded 20% holdout for datasets that did not ship one. By entity when entities exist."""
    rng = np.random.RandomState(config.SEED)
    if fr["entity_col"]:
        ents = np.array(sorted(df[fr["entity_col"]].unique()))
        hold = set(rng.permutation(ents)[: max(1, int(round(0.2 * len(ents))))].tolist())
        mask = df[fr["entity_col"]].isin(hold).to_numpy()
        how = f"{len(hold)} whole {fr['entity_col']}s held out (no entity appears on both sides)"
    else:
        mask = np.zeros(len(df), dtype=bool)
        mask[rng.permutation(len(df))[: int(round(0.2 * len(df)))]] = True
        how = "20% of rows held out at random (seeded)"
    fr["decisions"].append(decision("HOLDOUT_SPLIT", {"seed": config.SEED}, how))
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)
