"""Node 5, the Explainer: Tree-SHAP contributions, ranked and turned into template sentences.

Sentences are assembled from numbers with fixed templates. There is no text generation.
For LightGBM / XGBoost the contributions come from the library's native Tree-SHAP
(`pred_contrib`), which `parity_check` proves equal to `shap.TreeExplainer` on a sample.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import config
from .features import humanize


def _unwrap(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


def contributions(family: str, model, X: pd.DataFrame, task: str) -> tuple:
    """(contrib (n, f), base (n,)) in the model's margin space (log-odds for classifiers)."""
    if family == "lightgbm":
        c = model.booster_.predict(X, pred_contrib=True)
        return c[:, :-1], c[:, -1]
    if family == "xgboost":
        import xgboost as xgb
        c = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return c[:, :-1], c[:, -1]
    if family == "random_forest":
        import shap
        imp = model.steps[0][1]
        Xi = pd.DataFrame(imp.transform(X), columns=X.columns)
        sv = shap.TreeExplainer(_unwrap(model)).shap_values(Xi)
        sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
        return sv, np.zeros(len(X))
    if family == "linear":
        imp, sc, est = model.steps[0][1], model.steps[1][1], model.steps[2][1]
        Z = sc.transform(imp.transform(X))
        coef = est.coef_.reshape(-1) if est.coef_.ndim > 1 else est.coef_
        return Z * coef, np.full(len(X), float(np.ravel(est.intercept_)[0]))
    raise KeyError(family)


def parity_check(family: str, model, X: pd.DataFrame, task: str, n: int = 400) -> dict:
    """Compare native contributions with shap.TreeExplainer on a seeded sample."""
    if family not in ("lightgbm", "xgboost"):
        return {"checked": False, "reason": f"{family} uses shap.TreeExplainer / analytic contributions directly"}
    import shap
    rng = np.random.RandomState(config.SEED)
    idx = np.sort(rng.choice(len(X), size=min(n, len(X)), replace=False))
    Xs = X.iloc[idx]
    native, _ = contributions(family, model, Xs, task)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
    return {"checked": True, "rows": int(len(idx)), "max_abs_diff": float(np.max(np.abs(native - sv))),
            "shap_version": shap.__version__}


def global_importance(contrib: np.ndarray, names: list, labels: dict, top: int = 15) -> dict:
    mean_abs = np.abs(contrib).mean(axis=0)
    total = mean_abs.sum() or 1.0
    order = np.argsort(-mean_abs, kind="stable")
    features = [{"feature": names[i], "label": humanize(names[i], labels),
                 "mean_abs": float(mean_abs[i]), "share": float(mean_abs[i] / total)} for i in order[:top]]
    by_sensor = {}
    for n, v in zip(names, mean_abs):
        base = n.split("__")[0].split("=")[0]
        by_sensor[base] = by_sensor.get(base, 0.0) + float(v)
    sensors = sorted(by_sensor.items(), key=lambda kv: (-kv[1], kv[0]))
    return {"features": features,
            "sensors": [{"sensor": s, "label": labels.get(s, s), "share": v / total} for s, v in sensors[:12]]}


def z_word(z: float) -> str:
    if z >= 2:
        return "well above"
    if z >= 0.5:
        return "above"
    if z <= -2:
        return "well below"
    if z <= -0.5:
        return "below"
    return "near"


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        hi = edges[i + 1] if i < bins - 1 else 1.0 + 1e-9
        m = (p >= edges[i]) & (p < hi)
        out.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(m.sum()),
                    "positive_rate": float(y[m].mean()) if m.sum() else None})
    return out


def confidence_for(p: float, table: list) -> tuple:
    """Historical precision of the score band this prediction falls in (out-of-fold)."""
    i = min(int(p * len(table)), len(table) - 1)
    row = table[i]
    if row["n"] >= 30 and row["positive_rate"] is not None:
        return float(row["positive_rate"]), int(row["n"])
    return float(p), 0


def pct(x: float) -> str:
    return ">99%" if x >= 0.995 else "<1%" if x < 0.005 else f"{x:.0%}"


def explain_row(i: int, X: pd.DataFrame, contrib: np.ndarray, risk: float, ref: dict, labels: dict,
                threshold: float, table: list, entity_noun: str, entity_id, time_noun: str, time_val,
                task: str, k: int = 5) -> dict:
    names = list(X.columns)
    row = contrib[i]
    order = np.argsort(-np.abs(row), kind="stable")[:k]
    tot = float(np.abs(row).sum()) or 1.0
    drivers = []
    for j in order:
        f = names[j]
        sd = ref["std"].get(f) or 1.0
        z = (float(X.iloc[i, j]) - ref["mean"].get(f, 0.0)) / sd
        drivers.append({"feature": f, "label": humanize(f, labels), "value": float(X.iloc[i, j]),
                        "z": float(z), "contribution": float(row[j]), "share": float(abs(row[j]) / tot),
                        "pushes": "toward failure" if row[j] > 0 else "away from failure"})
    if task == "classification":
        level = "ALERT" if risk >= threshold else "WATCH" if risk >= 0.5 * threshold else "OK"
        conf, n = confidence_for(risk, table)
        head = f"{entity_noun} {entity_id} {level} at {time_noun} {time_val}: failure risk {pct(risk)}"
        band = table[min(int(risk * len(table)), len(table) - 1)]
        if level == "OK" and band["n"] >= 30 and band["positive_rate"] is not None:
            conf_text = f"historically {band['positive_rate']:.1%} of scores this low were followed by failure within the alarm horizon (n={band['n']})"
        elif n:
            conf_text = f"historically {conf:.0%} of scores in this band were followed by failure within the alarm horizon (n={n})"
        else:
            conf_text = "too few similar scores for a precision estimate"
    else:
        level, conf, n = "RUL", None, 0
        head = f"{entity_noun} {entity_id} at {time_noun} {time_val}: predicted remaining life {risk:.0f} {time_noun}s"
        conf_text = ""
    clauses = [f"{d['label']} {z_word(d['z'])} baseline ({d['z']:+.1f} sd), {d['share']:.0%} of the signal"
               for d in drivers[:3]]
    sentence = head + ". " + "; ".join(clauses) + "." + (f" {conf_text[0].upper() + conf_text[1:]}." if conf_text else "")
    return {"id": entity_id, "risk": float(risk), "level": level, "confidence": conf, "confidence_n": n,
            "drivers": drivers, "sentence": sentence}
