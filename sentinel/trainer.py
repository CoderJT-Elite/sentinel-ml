"""Node 4, the Trainer: seeded Optuna search per candidate, ranked by cross-validated metric.

The numbers pick the winner. The holdout is scored for every candidate but is never used
to choose between them.
"""
from __future__ import annotations

import time
import warnings
from typing import Callable, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

FAMILY_LABEL = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "random_forest": "Random Forest",
                "linear": "Linear baseline"}


def metric_value(metric: str, y, p) -> float:
    if metric == "roc_auc":
        return float(roc_auc_score(y, p))
    if metric == "average_precision":
        return float(average_precision_score(y, p))
    return float(np.sqrt(mean_squared_error(y, p)))


def higher_is_better(metric: str) -> bool:
    return metric != "rmse"


def predict(model, X, task: str) -> np.ndarray:
    if task == "classification":
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def build(family: str, params: dict, task: str, class_weight: bool, pos_weight: float):
    seed, nj = config.SEED, config.N_JOBS
    clf = task == "classification"
    if family == "lightgbm":
        import lightgbm as lgb
        kw = dict(random_state=seed, n_jobs=nj, deterministic=True, force_row_wise=True, verbose=-1, subsample_freq=1, **params)
        return lgb.LGBMClassifier(class_weight="balanced" if class_weight else None, **kw) if clf else lgb.LGBMRegressor(**kw)
    if family == "xgboost":
        import xgboost as xgb
        kw = dict(random_state=seed, n_jobs=nj, tree_method="hist", verbosity=0, **params)
        if clf:
            return xgb.XGBClassifier(scale_pos_weight=pos_weight if class_weight else 1.0, eval_metric="logloss", **kw)
        return xgb.XGBRegressor(**kw)
    if family == "random_forest":
        kw = dict(random_state=seed, n_jobs=nj, **params)
        est = (RandomForestClassifier(class_weight="balanced_subsample" if class_weight else None, **kw)
               if clf else RandomForestRegressor(**kw))
        return make_pipeline(SimpleImputer(strategy="median"), est)
    if family == "linear":
        if clf:
            est = LogisticRegression(max_iter=300, tol=1e-3, class_weight="balanced" if class_weight else None,
                                     random_state=seed, **params)
        else:
            est = Ridge(random_state=seed, **params)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
    raise KeyError(family)


def suggest(trial: optuna.Trial, family: str) -> dict:
    if family == "lightgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 64),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    if family == "xgboost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    if family == "random_forest":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 60, 120, step=20),
            max_depth=trial.suggest_int("max_depth", 8, 14),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 20),
            max_features=trial.suggest_float("max_features", 0.1, 0.3),
            max_samples=trial.suggest_float("max_samples", 0.4, 0.7),
        )
    raise KeyError(family)


def _suggest_linear(trial, task):
    if task == "classification":
        return dict(C=trial.suggest_float("C", 1e-3, 10.0, log=True))
    return dict(alpha=trial.suggest_float("alpha", 1e-3, 100.0, log=True))


def make_splitter(task: str, groups: Optional[np.ndarray]):
    if groups is not None:
        return GroupKFold(n_splits=config.CV_FOLDS)
    if task == "classification":
        return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    return KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)


def _folds(splitter, X, y, groups):
    return list(splitter.split(X, y, groups)) if groups is not None else list(splitter.split(X, y))


def _cv(family, params, X, y, folds, task, metric, class_weight, pos_weight):
    oof = np.zeros(len(y), dtype=float)
    scores = []
    for tr, va in folds:
        m = build(family, params, task, class_weight, pos_weight)
        m.fit(X.iloc[tr], y[tr])
        p = predict(m, X.iloc[va], task)
        oof[va] = p
        scores.append(metric_value(metric, y[va], p))
    return scores, oof


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> tuple:
    """Threshold on out-of-fold scores that maximises F1 (ties -> lowest threshold)."""
    grid = np.unique(np.round(np.quantile(p, np.linspace(0.01, 0.99, 99)), 6))
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = f1_score(y, p >= t, zero_division=0)
        if f > best_f + 1e-12:
            best_t, best_f = float(t), float(f)
    return best_t, best_f


def train_all(X: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray], fr: dict, budget: str,
              on_event: Callable[[dict], None] = lambda e: None) -> dict:
    """Run the seeded search for every candidate and return leaderboard rows + fitted models."""
    task, metric = fr["task"], fr["metric"]
    trials_for = config.BUDGETS[budget]
    class_weight = bool(fr.get("class_weight"))
    pos = float(y.sum())
    pos_weight = float((len(y) - pos) / pos) if pos else 1.0
    folds = _folds(make_splitter(task, groups), X, y, groups)
    sign = 1.0 if higher_is_better(metric) else -1.0

    rows, models = [], {}
    for family in fr["candidates"]:
        n_trials = trials_for[family]
        t0 = time.time()

        def objective(trial, family=family):
            params = _suggest_linear(trial, task) if family == "linear" else suggest(trial, family)
            scores, oof = _cv(family, params, X, y, folds, task, metric, class_weight, pos_weight)
            trial.set_user_attr("scores", scores)
            trial.set_user_attr("oof", oof)
            mean = float(np.mean(scores))
            on_event({"type": "trial", "family": family, "trial": trial.number + 1, "of": n_trials,
                      "score": round(mean, 5)})
            return sign * mean

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.SEED))
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        bt = study.best_trial
        scores, oof = bt.user_attrs["scores"], bt.user_attrs["oof"]
        params = dict(bt.params)
        row = {
            "family": family, "name": FAMILY_LABEL[family], "params": params, "n_trials": n_trials,
            "cv_mean": float(np.mean(scores)), "cv_std": float(np.std(scores)),
            "cv_folds": [float(s) for s in scores], "metric": metric,
            "fit_seconds": round(time.time() - t0, 1),   # excluded from the run hash
        }
        if task == "classification":
            thr, f1 = best_f1_threshold(y, oof)
            row["oof_threshold"], row["oof_f1"] = thr, f1
            row["_oof"] = oof
        else:
            row["_oof"] = oof
        model = build(family, params, task, class_weight, pos_weight)
        model.fit(X, y)
        models[family] = model
        rows.append(row)
        on_event({"type": "family_done", "family": family, "cv_mean": round(row["cv_mean"], 5),
                  "cv_std": round(row["cv_std"], 5), "seconds": row["fit_seconds"]})

    rows.sort(key=lambda r: (-sign * r["cv_mean"], r["family"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {"rows": rows, "models": models, "champion": rows[0]["family"]}


def eval_holdout(model, Xh: pd.DataFrame, yh: np.ndarray, fr: dict, threshold: Optional[float],
                 last_rows: Optional[np.ndarray] = None) -> dict:
    """Holdout metrics on all rows, and on the final observation of each unit when given."""
    task, metric = fr["task"], fr["metric"]
    p = predict(model, Xh, task)
    out = {"all_rows": {}, "final_obs": {}}
    if task == "classification":
        for name, (yy, pp) in {"all_rows": (yh, p)}.items():
            out[name] = {"roc_auc": metric_value("roc_auc", yy, pp),
                         "average_precision": metric_value("average_precision", yy, pp),
                         "f1": float(f1_score(yy, pp >= threshold, zero_division=0))}
        if last_rows is not None:
            yy, pp = yh[last_rows], p[last_rows]
            out["final_obs"] = {"roc_auc": metric_value("roc_auc", yy, pp) if len(set(yy)) > 1 else None,
                                "average_precision": metric_value("average_precision", yy, pp) if len(set(yy)) > 1 else None,
                                "f1": float(f1_score(yy, pp >= threshold, zero_division=0)),
                                "n": int(len(yy))}
    else:
        out["all_rows"] = {"rmse": metric_value("rmse", yh, p)}
        if last_rows is not None:
            out["final_obs"] = {"rmse": metric_value("rmse", yh[last_rows], p[last_rows]), "n": int(len(last_rows))}
    return out
