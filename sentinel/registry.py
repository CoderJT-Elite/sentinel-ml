"""MLflow experiment tracking + model registry.

Tracking store is SQLite under ./runs by default, or MLFLOW_TRACKING_URI (the compose
file points it at the MLflow server backed by PostgreSQL).
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile

import mlflow
from mlflow.tracking import MlflowClient

from . import config
from .util import clean

EXPERIMENT = "sentinel"


def _uri() -> str:
    if config.MLFLOW_URI:
        return config.MLFLOW_URI
    os.makedirs(config.RUNS_DIR, exist_ok=True)
    return "sqlite:///" + os.path.abspath(os.path.join(config.RUNS_DIR, "mlflow.db")).replace("\\", "/")


def client() -> MlflowClient:
    mlflow.set_tracking_uri(_uri())
    mlflow.set_registry_uri(_uri())
    return MlflowClient()


def _experiment_id() -> str:
    c = client()
    exp = c.get_experiment_by_name(EXPERIMENT)
    if exp:
        return exp.experiment_id
    kw = {}
    if not config.MLFLOW_URI:
        art = pathlib.Path(os.path.abspath(os.path.join(config.RUNS_DIR, "mlartifacts")))
        art.mkdir(parents=True, exist_ok=True)
        kw["artifact_location"] = art.as_uri()
    return c.create_experiment(EXPERIMENT, **kw)


def log_run(name: str, run_hash: str, params: dict, metrics: dict, artifacts: dict, model, tags: dict) -> dict:
    """Log one Sentinel run. Returns {'mlflow_run_id', 'model_name', 'version'}."""
    c = client()
    mlflow.set_experiment(experiment_id=_experiment_id())
    with mlflow.start_run(run_name=f"{name}-{run_hash[:8]}") as run:
        mlflow.set_tags({"sentinel.run_hash": run_hash, "sentinel.llm_calls": "0", **tags})
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        with tempfile.TemporaryDirectory() as td:
            for fname, obj in artifacts.items():
                p = pathlib.Path(td) / fname
                p.write_text(obj if isinstance(obj, str) else json.dumps(clean(obj), indent=2), encoding="utf-8")
                mlflow.log_artifact(str(p))
        model_name = f"sentinel-{name}"
        info = mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name)
        ver = c.get_latest_versions(model_name)
        version = max(int(v.version) for v in ver) if ver else 1
        c.set_registered_model_alias(model_name, "champion", str(version))
        c.set_model_version_tag(model_name, str(version), "sentinel.run_hash", run_hash)
        return {"mlflow_run_id": run.info.run_id, "model_name": model_name, "version": version,
                "tracking_uri": _uri()}


def versions(model_name: str) -> list:
    c = client()
    out = []
    for v in c.search_model_versions(f"name='{model_name}'"):
        out.append({"version": int(v.version), "run_id": v.run_id, "aliases": list(v.aliases),
                    "tags": dict(v.tags)})
    return sorted(out, key=lambda d: d["version"])
