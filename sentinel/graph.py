"""The pipeline: a LangGraph StateGraph used purely as a workflow engine.

No node calls a language model. Each node reads the shared state, applies rules or
statistics, appends to the decision log, and emits an event for the UI. The graph makes
the control flow explicit (including the quality gate that can halt a run), nothing more.
"""
from __future__ import annotations

import os
import pathlib
import time
import uuid
from typing import Callable, Optional, TypedDict

import joblib
import numpy as np
import pandas as pd
from langgraph.graph import END, StateGraph

from . import config, deployer, explainer, features, monitor, registry, selector, steward, trainer
from .data import Dataset
from .store import Store
from .util import canonical, clean, frame_fingerprint, run_hash, sha256

NODES = [
    ("data_steward", "Data Steward"),
    ("task_selector", "Task & Model Selector"),
    ("feature_engineer", "Feature Engineer"),
    ("trainer", "Trainer"),
    ("explainer", "Explainer"),
    ("deployer_monitor", "Deployer / Monitor"),
]


class State(TypedDict, total=False):
    ds: Dataset
    options: dict
    emit: Callable
    decisions: list
    scorecard: dict
    checks: list
    struct: Optional[dict]
    drop: list
    framing: dict
    train: pd.DataFrame
    hold: pd.DataFrame
    exclude: set
    spec: features.FeatureSpec
    X: pd.DataFrame
    y: np.ndarray
    groups: Optional[np.ndarray]
    Xh: pd.DataFrame
    yh: np.ndarray
    trained: dict
    board: list
    champion: str
    threshold: Optional[float]
    xai: dict
    ref: dict
    drift_ref: dict
    early_train: np.ndarray
    halted: bool
    result: dict
    sample_payload: dict


def config_snapshot() -> dict:
    return {k: getattr(config, k) for k in dir(config)
            if k.isupper() and isinstance(getattr(config, k), (int, float, dict)) and k not in ("RUNS_DIR",)}


def grouped_drops(drops: list) -> list:
    """One ledger entry per distinct reason, so seven constant columns read as one decision."""
    by_reason: dict = {}
    for d in drops:
        by_reason.setdefault(d["reason"], []).append(d["col"])
    return [selector.decision("DROP_COLUMN", {"columns": ", ".join(cols)}, f"{', '.join(cols)}: {why}")
            for why, cols in by_reason.items()]


def _log(state: State, node: str, items: list) -> None:
    for d in items:
        state["decisions"].append({"node": node, **d})


def _emit(state, node, status, title, decisions=None, payload=None):
    state["emit"]({"node": node, "status": status, "title": title, "decisions": decisions or [],
                   "payload": payload or {}, "t": time.time()})


# ------------------------------------------------------------------ nodes
def n_steward(state: State) -> State:
    ds, opts = state["ds"], state["options"]
    _emit(state, "data_steward", "start", "Profiling dataset")
    declared = opts.get("target") or ds.meta.get("target_col")
    exclude = {declared} if declared else set()
    prof = steward.profile(ds.train, exclude=exclude)
    state["checks"] = prof["checks"]
    state["struct"] = prof["findings"]["structure"]
    state["drop"] = prof["findings"]["drop"]
    state["exclude"] = exclude
    dec = grouped_drops(state["drop"])
    if state["struct"]:
        s = state["struct"]
        dec.append(selector.decision("STRUCTURE_DETECTED", {"entity": s["entity_col"], "time": s["time_col"]}, s["evidence"]))
    _log(state, "data_steward", dec)
    sc = steward.scorecard(state["checks"])
    state["scorecard"] = sc
    if not sc["passed"]:
        state["halted"] = True
        _emit(state, "data_steward", "halt", "Quality gate FAILED. Pipeline halted before any model was trained.",
              dec, {"scorecard": sc})
    else:
        _emit(state, "data_steward", "done", f"Stage 1 scorecard: {sc['verdict']}", dec, {"scorecard": sc})
    return state


def n_selector(state: State) -> State:
    ds, opts = state["ds"], state["options"]
    _emit(state, "task_selector", "start", "Framing the task")
    fr = selector.frame(ds.train, ds.meta, opts, state["struct"])
    train = ds.train.copy()
    if fr["entity_col"]:
        train = train.sort_values([fr["entity_col"], fr["time_col"]], kind="mergesort").reset_index(drop=True)
    if ds.holdout is not None:
        hold = ds.holdout.copy()
        if fr["entity_col"]:
            hold = hold.sort_values([fr["entity_col"], fr["time_col"]], kind="mergesort").reset_index(drop=True)
        fr["decisions"].append(selector.decision("HOLDOUT_SPLIT", {"source": "dataset"},
                                                 "the dataset ships an official holdout; it is never used for model selection"))
    else:
        base = selector.attach_targets(train, fr, ds.meta, holdout=False)
        train, hold = selector.split_holdout(base.drop(columns=["__target__", "__rul__"], errors="ignore"), fr)
    tr = selector.attach_targets(train, fr, ds.meta, holdout=False)
    ho = selector.attach_targets(hold, fr, ds.meta, holdout=True)
    selector.choose_metric(fr, tr["__target__"])
    fr["decisions"].append(selector.decision(
        "CANDIDATE_SET", {"task": fr["task"]}, "fixed search space: " + ", ".join(trainer.FAMILY_LABEL[c] for c in fr["candidates"])))

    excl = set(state["exclude"]) | {"__target__", "__rul__"}
    if ds.meta.get("holdout_rul_col"):
        excl.add(ds.meta["holdout_rul_col"])
    audit = steward.target_audit(tr, tr["__target__"], fr["task"], exclude=excl | ({fr["entity_col"], fr["time_col"]} - {None}))
    state["checks"] = state["checks"] + audit["checks"]
    state["drop"] = state["drop"] + audit["drop"]
    state["exclude"] = excl
    sc = steward.scorecard(state["checks"])
    state["scorecard"] = sc
    dec = fr["decisions"] + grouped_drops(audit["drop"])
    _log(state, "task_selector", dec)
    state["framing"], state["train"], state["hold"] = fr, tr, ho
    payload = {"framing": {k: v for k, v in fr.items() if k != "decisions"}, "scorecard": sc}
    if not sc["passed"]:
        state["halted"] = True
        _emit(state, "task_selector", "halt", "Target audit FAILED. Pipeline halted.", dec, payload)
    else:
        _emit(state, "task_selector", "done",
              f"{fr['task'].title()} on '{fr['target_col']}', ranked by {fr['metric']}. Final scorecard: {sc['verdict']}", dec, payload)
    return state


def n_features(state: State) -> State:
    _emit(state, "feature_engineer", "start", "Building features")
    fr, tr, ho = state["framing"], state["train"], state["hold"]
    spec = features.plan(tr, state["struct"], state["drop"], state["exclude"])
    trp, hop = features.prepare(tr, spec), features.prepare(ho, spec)
    X, Xh = features.transform(trp, spec), features.transform(hop, spec)
    state["spec"], state["train"], state["hold"] = spec, trp, hop
    state["X"], state["Xh"] = X, Xh
    state["y"], state["yh"] = trp["__target__"].to_numpy(), hop["__target__"].to_numpy()
    state["groups"] = trp[spec.entity_col].to_numpy() if spec.temporal else None
    state["ref"] = monitor.make_reference(X)
    early = monitor.early_mask(trp, spec)
    state["early_train"] = early
    state["drift_ref"] = monitor.make_reference(X[early])
    if spec.temporal:
        msg = (f"{len(spec.sensor_cols)} sensors x (current, {spec.roll_mean}-step mean, {spec.roll_std}-step std, "
               f"{spec.roll_slope}-step slope, {spec.lag_diff}-step change) + {spec.time_col} = {X.shape[1]} features")
    else:
        msg = (f"{len(spec.numeric_cols)} numeric + {sum(len(v) for v in spec.categorical.values())} one-hot "
               f"categorical = {X.shape[1]} features")
    dec = [selector.decision("FEATURE_SET", {"features": int(X.shape[1]), "rows": int(len(X))}, msg)]
    if spec.temporal:
        dec.append(selector.decision("CV_GROUPING", {"folds": config.CV_FOLDS, "group": spec.entity_col},
                                     f"cross-validation folds are grouped by {spec.entity_col}: no unit is ever in train and validation together"))
    _log(state, "feature_engineer", dec)
    _emit(state, "feature_engineer", "done", msg, dec,
          {"n_features": int(X.shape[1]), "dropped": spec.dropped, "names": list(X.columns)[:12]})
    return state


def n_trainer(state: State) -> State:
    fr, opts = state["framing"], state["options"]
    _emit(state, "trainer", "start", f"Seeded Optuna search over {len(fr['candidates'])} model families")
    out = trainer.train_all(state["X"], state["y"], state["groups"], fr, opts.get("budget", "fast"),
                            on_event=lambda e: _emit(state, "trainer", "progress", "", payload=e))
    rows, models = out["rows"], out["models"]
    spec, ho = state["spec"], state["hold"]
    last = None
    if spec.temporal:
        ent = ho[spec.entity_col].to_numpy()
        last = np.flatnonzero(np.r_[ent[1:] != ent[:-1], True])
    for r in rows:
        thr = r.get("oof_threshold")
        r["holdout"] = trainer.eval_holdout(models[r["family"]], state["Xh"], state["yh"], fr, thr, last)
    champ = rows[0]
    state["trained"], state["board"], state["champion"] = out, rows, champ["family"]
    state["threshold"] = champ.get("oof_threshold")
    sign = "lower" if fr["metric"] == "rmse" else "higher"
    dec = [selector.decision(
        "CHAMPION", {"metric": fr["metric"], "cv_mean": round(champ["cv_mean"], 5), "cv_std": round(champ["cv_std"], 5),
                     "runner_up": rows[1]["name"] if len(rows) > 1 else None,
                     "runner_up_cv": round(rows[1]["cv_mean"], 5) if len(rows) > 1 else None},
        f"{champ['name']} ranks first on cross-validated {fr['metric']} ({sign} is better): the numbers picked it, "
        "and the holdout was not consulted")]
    if state["threshold"] is not None:
        dec.append(selector.decision("ALERT_THRESHOLD", {"threshold": round(state["threshold"], 4), "oof_f1": round(champ["oof_f1"], 4)},
                                     "alert threshold = the out-of-fold score that maximises F1"))
    _log(state, "trainer", dec)
    _emit(state, "trainer", "done", f"Champion: {champ['name']}", dec,
          {"leaderboard": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]})
    return state


def n_explainer(state: State) -> State:
    _emit(state, "explainer", "start", "Computing Tree-SHAP contributions")
    fr, spec = state["framing"], state["spec"]
    fam, model = state["champion"], state["trained"]["models"][state["champion"]]
    labels = {**spec.display, **state["ds"].meta.get("sensor_labels", {})}
    X, Xh, ho = state["X"], state["Xh"], state["hold"]
    rng = np.random.RandomState(config.SEED)
    samp = np.sort(rng.choice(len(X), size=min(3000, len(X)), replace=False))
    c_tr, _ = explainer.contributions(fam, model, X.iloc[samp], fr["task"])
    gi = explainer.global_importance(c_tr, list(X.columns), labels)
    parity = explainer.parity_check(fam, model, X, fr["task"])
    c_ho, base_ho = explainer.contributions(fam, model, Xh, fr["task"])
    p_ho = trainer.predict(model, Xh, fr["task"])
    champ = state["board"][0]
    table = explainer.reliability_table(state["y"], champ["_oof"]) if fr["task"] == "classification" else []
    ref = {"mean": {k: v["mean"] for k, v in state["ref"].items()}, "std": {k: v["std"] for k, v in state["ref"].items()}}
    meta = state["ds"].meta
    id_col = next((d["col"] for d in state["drop"] if "row identifier" in d["reason"]), None)

    fleet = []
    if spec.temporal:
        ent = ho[spec.entity_col].to_numpy()
        bounds = np.flatnonzero(np.r_[True, ent[1:] != ent[:-1]])
        ends = np.r_[bounds[1:], len(ho)]
        top_sensors = [s["sensor"] for s in gi["sensors"][:6]]
        for s, e in zip(bounds, ends):
            i = e - 1
            ex = explainer.explain_row(i, Xh, c_ho, float(p_ho[i]), ref, labels, state["threshold"] or 0.5, table,
                                       meta.get("entity_noun", "Unit"), int(ent[i]), meta.get("time_noun", "cycle"),
                                       int(ho[spec.time_col].iloc[i]), fr["task"])
            ex["cycle"] = int(ho[spec.time_col].iloc[i])
            ex["actual"] = float(state["yh"][i])
            ex["actual_rul"] = float(ho["__rul__"].iloc[i]) if "__rul__" in ho else None
            ex["curve"] = {"t": ho[spec.time_col].iloc[s:e].astype(int).tolist(),
                           "risk": np.round(p_ho[s:e], 4).tolist()}
            ex["sensors"] = {sn: np.round(ho[sn].iloc[s:e].to_numpy(dtype=float), 3).tolist() for sn in top_sensors if sn in ho}
            fleet.append(ex)
    else:
        order = np.argsort(-p_ho, kind="stable")[:60]
        for i in order:
            ident = ho[id_col].iloc[i] if id_col and id_col in ho else int(i)
            ident = ident.item() if hasattr(ident, "item") else ident
            ex = explainer.explain_row(int(i), Xh, c_ho, float(p_ho[i]), ref, labels, state["threshold"] or 0.5, table,
                                       meta.get("entity_noun", "Machine"), ident, meta.get("time_noun", "row"), int(i), fr["task"])
            ex["cycle"], ex["actual"], ex["actual_rul"] = int(i), float(state["yh"][i]), None
            fleet.append(ex)
    fleet.sort(key=lambda d: (-d["risk"] if fr["task"] == "classification" else d["risk"], str(d["id"])))
    state["xai"] = {"global": gi, "parity": parity, "reliability": table, "fleet": fleet, "ref_stats": ref}
    top = gi["features"][0]
    dec = [selector.decision("EXPLAINER", {"method": "Tree-SHAP", "rows_explained": int(len(samp))},
                             f"global importance from {len(samp)} training rows; top driver: {top['label']} ({top['share']:.0%})")]
    if parity.get("checked"):
        dec.append(selector.decision("SHAP_PARITY", {"max_abs_diff": parity["max_abs_diff"], "rows": parity["rows"]},
                                     f"native Tree-SHAP equals shap.TreeExplainer to within {parity['max_abs_diff']:.1e} on {parity['rows']} rows"))
    _log(state, "explainer", dec)
    _emit(state, "explainer", "done", f"Top driver: {top['label']} ({top['share']:.0%})", dec,
          {"global": gi, "parity": parity, "fleet_head": [{k: f[k] for k in ("id", "risk", "level", "sentence")} for f in fleet[:3]]})
    return state


def n_deploy(state: State) -> State:
    _emit(state, "deployer_monitor", "start", "Registering, packaging, arming the monitor")
    fr, spec, opts, ds = state["framing"], state["spec"], state["options"], state["ds"]
    fam, model = state["champion"], state["trained"]["models"][state["champion"]]
    champ = state["board"][0]
    fp = frame_fingerprint(ds.train)
    snap = config_snapshot()
    opt_hash = {"dataset": ds.key, "budget": opts.get("budget", "fast"), "task": opts.get("task", "auto"),
                "target": opts.get("target"), "horizon": opts.get("horizon"), "config": snap}
    rh = run_hash(fp, opt_hash, state["decisions"], state["board"])
    run_id = opts.get("run_id") or f"run-{rh[:10]}"
    run_dir = pathlib.Path(config.RUNS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    hm = champ["holdout"]
    metrics = {"cv_mean": champ["cv_mean"], "cv_std": champ["cv_std"]}
    for k, v in (hm.get("all_rows") or {}).items():
        metrics[f"holdout_{k}"] = v
    for k, v in (hm.get("final_obs") or {}).items():
        if k != "n":
            metrics[f"holdout_final_{k}"] = v
    summary_pre = {"scorecard": state["scorecard"], "framing": {k: v for k, v in fr.items()}, "decisions": state["decisions"]}
    art = {"decisions.json": state["decisions"], "scorecard.json": state["scorecard"],
           "leaderboard.json": [{k: v for k, v in r.items() if not k.startswith("_")} for r in state["board"]],
           "feature_spec.json": spec.to_json(), "importance.json": state["xai"]["global"],
           "run_hash.txt": rh}
    reg = registry.log_run(ds.key, rh, {"champion": fam, **champ["params"], "metric": fr["metric"], "seed": config.SEED,
                                        "budget": opts.get("budget", "fast")}, metrics, art, model,
                           {"sentinel.dataset": ds.key, "sentinel.task": fr["task"], "sentinel.champion": fam})
    meta = {"model_name": reg["model_name"], "version": reg["version"], "run_hash": rh, "dataset": ds.key,
            "sensor_labels": {**spec.display, **ds.meta.get("sensor_labels", {})}, "entity_noun": ds.meta.get("entity_noun", "Unit"),
            "time_noun": ds.meta.get("time_noun", "cycle"), "metric": fr["metric"], "cv_mean": champ["cv_mean"],
            "horizon": fr.get("horizon"), "llm_calls": 0}
    pkg = deployer.package(str(run_dir / "serving"), family=fam, model=model, spec=spec, ref=state["ref"], task=fr["task"],
                           threshold=state["threshold"], reliability=state["xai"]["reliability"], meta=meta)
    # parity: the generated service must reproduce the trained model on real rows
    ho = state["hold"]
    if spec.temporal:
        ids = [f["id"] for f in state["xai"]["fleet"][:3]]
        sample = ho[ho[spec.entity_col].isin(ids)]
        cols = [spec.entity_col, spec.time_col] + spec.sensor_cols
        rows = sample[cols].to_dict("records")
        pm = trainer.predict(model, state["Xh"], fr["task"])
        expected = {str(f["id"]): f["risk"] for f in state["xai"]["fleet"] if f["id"] in ids}
    else:
        idx = np.arange(min(5, len(ho)))
        cols = spec.numeric_cols + list(spec.categorical)
        rows = ho.iloc[idx][cols].to_dict("records")
        pm = trainer.predict(model, state["Xh"].iloc[idx], fr["task"])
        expected = {str(i): float(pm[i]) for i in idx}
    parity = deployer.parity_test(pkg, rows, expected)
    state["sample_payload"] = {"rows": rows, "expected": expected}
    # monitor: arm with training reference + record the baseline check
    imp = [f["feature"] for f in state["xai"]["global"]["features"]]
    e_tr, e_ho = state["early_train"], monitor.early_mask(state["hold"], spec)
    base = monitor.drift_report(state["drift_ref"], state["X"][e_tr], state["Xh"][e_ho], imp)
    dec = [
        selector.decision("REGISTER", {"model": reg["model_name"]},
                          f"champion logged to MLflow and registered as {reg['model_name']} (alias: champion)"),
        selector.decision("PACKAGE", {"folder": os.path.basename(pkg), "family": fam},
                          "versioned FastAPI service generated with the same features.py and Tree-SHAP code used in training"),
        selector.decision("SERVING_PARITY", {"max_abs_diff": parity["max_abs_diff"], "entities": parity["entities"]},
                          "service reproduces the trained model's scores" if parity["ok"] else "PARITY FAILED"),
        selector.decision("MONITOR_ARMED", {"psi_watch": config.PSI_WATCH, "psi_retrain": config.PSI_RETRAIN},
                          f"drift monitor armed on {len(state['drift_ref'])} feature distributions (early-life window, {config.MONITOR_WINDOW} {ds.meta.get('time_noun', 'row')}s); retrain rule: any top-10 feature PSI >= {config.PSI_RETRAIN}"),
    ]
    _log(state, "deployer_monitor", dec)
    store = Store()
    summary = {
        "run_id": run_id, "run_hash": rh, "dataset": ds.info(), "options": opt_hash,
        "scorecard": state["scorecard"], "framing": {k: v for k, v in fr.items() if k != "decisions"},
        "decisions": state["decisions"], "leaderboard": [{k: v for k, v in r.items() if not k.startswith("_")} for r in state["board"]],
        "champion": {"family": fam, "name": champ["name"], "threshold": state["threshold"]},
        "explain": {"global": state["xai"]["global"], "parity": state["xai"]["parity"], "reliability": state["xai"]["reliability"]},
        "fleet": state["xai"]["fleet"], "registry": reg, "serving": {"folder": pkg, "parity": parity},
        "monitor": {"baseline": base, "reference_features": len(state["drift_ref"]), "window": config.MONITOR_WINDOW},
        "feature_spec": {"n_features": len(spec.feature_names), "dropped": spec.dropped},
        "entity_noun": meta["entity_noun"], "time_noun": meta["time_noun"], "llm_calls": 0,
    }
    store.save_run(run_id, ds.key, "done", rh, summary)
    store.save_decisions(run_id, state["decisions"])
    (run_dir / "summary.json").write_text(canonical(summary), encoding="utf-8")
    joblib.dump({"spec": spec, "ref": state["ref"], "drift_ref": state["drift_ref"], "early_train": state["early_train"], "models": state["trained"]["models"], "champion": fam,
                 "framing": fr, "threshold": state["threshold"], "X": state["X"], "y": state["y"], "groups": state["groups"],
                 "train": state["train"], "hold": state["hold"], "board": [{k: v for k, v in r.items()} for r in state["board"]],
                 "labels": {**spec.display, **ds.meta.get("sensor_labels", {})}, "meta": ds.meta, "importance": imp, "sample": state["sample_payload"],
                 "reg": reg, "run_hash": rh, "reliability": state["xai"]["reliability"]}, run_dir / "internal.joblib", compress=3)
    state["result"] = summary
    _emit(state, "deployer_monitor", "done", f"Run {run_id} complete", dec,
          {"registry": reg, "parity": parity, "run_id": run_id, "run_hash": rh, "baseline_status": base["status"]})
    return state


# ------------------------------------------------------------------ graph
def _gate(state: State) -> str:
    return "halt" if state.get("halted") else "go"


def build_graph():
    g = StateGraph(State)
    g.add_node("data_steward", n_steward)
    g.add_node("task_selector", n_selector)
    g.add_node("feature_engineer", n_features)
    g.add_node("trainer", n_trainer)
    g.add_node("explainer", n_explainer)
    g.add_node("deployer_monitor", n_deploy)
    g.set_entry_point("data_steward")
    g.add_conditional_edges("data_steward", _gate, {"go": "task_selector", "halt": END})
    g.add_conditional_edges("task_selector", _gate, {"go": "feature_engineer", "halt": END})
    g.add_edge("feature_engineer", "trainer")
    g.add_edge("trainer", "explainer")
    g.add_edge("explainer", "deployer_monitor")
    g.add_edge("deployer_monitor", END)
    return g.compile()


def run_pipeline(ds: Dataset, options: Optional[dict] = None, on_event: Callable = lambda e: None) -> dict:
    options = dict(options or {})
    events = []
    seq = [0]

    def emit(e):
        seq[0] += 1
        e = {"seq": seq[0], **e}
        events.append(e)
        on_event(e)

    state: State = {"ds": ds, "options": options, "emit": emit, "decisions": [], "halted": False}
    final = build_graph().invoke(state)
    if final.get("halted"):
        summary = {"run_id": options.get("run_id") or f"halt-{uuid.uuid4().hex[:8]}", "halted": True,
                   "dataset": ds.info(), "scorecard": final.get("scorecard"), "decisions": final["decisions"], "llm_calls": 0}
        (pathlib.Path(config.RUNS_DIR)).mkdir(parents=True, exist_ok=True)
        return summary
    return final["result"]
