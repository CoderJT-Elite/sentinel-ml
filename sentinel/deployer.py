"""Node 6a, the Deployer: package the champion as a versioned FastAPI service and prove it works.

`package()` writes a self-contained serving folder (model, feature spec, the same
`features.py` / `explainer.py` used in training, FastAPI app, Dockerfile).
`parity_test()` loads that folder the way the container would and checks its predictions
against the in-process model. `docker_smoke()` (optional) builds the image, runs it, and repeats the check over HTTP.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import sys
import time
from typing import Optional

import joblib
import numpy as np

from . import config

HERE = pathlib.Path(__file__).parent

SERVE_PY = '''"""Sentinel model service (generated). Same feature code and Tree-SHAP as training."""
import json
import pathlib

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentinel.explainer import confidence_for, contributions, explain_row
from sentinel.features import FeatureSpec, prepare, transform

ROOT = pathlib.Path(__file__).parent
META = json.loads((ROOT / "metadata.json").read_text())
SPEC = FeatureSpec.from_dict(json.loads((ROOT / "feature_spec.json").read_text()))
REF = json.loads((ROOT / "reference.json").read_text())
LABELS = META["sensor_labels"]
FAMILY = META["family"]

if FAMILY == "lightgbm":
    import lightgbm as lgb
    MODEL = lgb.Booster(model_file=str(ROOT / "model.txt"))
elif FAMILY == "xgboost":
    import xgboost as xgb
    MODEL = xgb.Booster(); MODEL.load_model(str(ROOT / "model.json"))
else:
    import joblib
    MODEL = joblib.load(ROOT / "model.joblib")


class _Wrap:
    """Uniform interface over the three model containers."""
    def predict(self, X):
        if FAMILY == "lightgbm":
            raw = MODEL.predict(X)          # probability for binary objective, value for regression
            return raw
        if FAMILY == "xgboost":
            import xgboost as xgb
            return MODEL.predict(xgb.DMatrix(X))
        if META["task"] == "classification":
            return MODEL.predict_proba(X)[:, 1]
        return MODEL.predict(X)

    def contrib(self, X):
        if FAMILY == "lightgbm":
            c = MODEL.predict(X, pred_contrib=True)
            return c[:, :-1], c[:, -1]
        if FAMILY == "xgboost":
            import xgboost as xgb
            c = MODEL.predict(xgb.DMatrix(X), pred_contribs=True)
            return c[:, :-1], c[:, -1]
        return contributions(FAMILY, MODEL, X, META["task"])


W = _Wrap()
app = FastAPI(title="Sentinel model service", version=str(META["version"]))


class Batch(BaseModel):
    rows: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "model": META["model_name"], "version": META["version"]}


@app.get("/metadata")
def metadata():
    return META


@app.post("/predict")
def predict(batch: Batch):
    df = pd.DataFrame(batch.rows)
    need = ([SPEC.entity_col, SPEC.time_col] + SPEC.sensor_cols) if SPEC.temporal else SPEC.numeric_cols + list(SPEC.categorical)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise HTTPException(422, f"missing columns: {missing}")
    df = prepare(df, SPEC)
    X = transform(df, SPEC)
    if SPEC.temporal:
        last = np.flatnonzero(np.r_[df[SPEC.entity_col].to_numpy()[1:] != df[SPEC.entity_col].to_numpy()[:-1], True])
    else:
        last = np.arange(len(df))
    score = W.predict(X)
    contrib, _ = W.contrib(X.iloc[last])
    out = []
    for k, i in enumerate(last):
        ent = df[SPEC.entity_col].iloc[i] if SPEC.temporal else int(i)
        tv = df[SPEC.time_col].iloc[i] if SPEC.temporal else int(i)
        e = explain_row(k, X.iloc[last], contrib, float(score[i]), REF, LABELS, META["threshold"], META["reliability"],
                        META["entity_noun"], ent.item() if hasattr(ent, "item") else ent, META["time_noun"],
                        tv.item() if hasattr(tv, "item") else tv, META["task"])
        out.append(e)
    return {"model": META["model_name"], "version": META["version"], "predictions": out}
'''

DOCKERFILE = '''FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /svc
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
'''

REQS = {
    "lightgbm": ["lightgbm>=4.5", "numpy", "pandas", "scipy"],
    "xgboost": ["xgboost-cpu>=2.1", "numpy", "pandas", "scipy"],
    "random_forest": ["scikit-learn>=1.5", "joblib", "shap>=0.46", "numpy", "pandas", "scipy"],
    "linear": ["scikit-learn>=1.5", "joblib", "numpy", "pandas", "scipy"],
}
BASE_REQS = ["fastapi>=0.115", "uvicorn[standard]>=0.30", "pydantic>=2"]


def package(out_dir: str, *, family: str, model, spec, ref: dict, task: str, threshold: Optional[float],
            reliability: list, meta: dict) -> str:
    out = pathlib.Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "sentinel").mkdir(parents=True)
    # the very same feature/explainer code that trained the model
    (out / "sentinel" / "__init__.py").write_text('"""Serving subset of sentinel-ml."""\n')
    (out / "sentinel" / "config.py").write_text(f"SEED = {config.SEED}\nN_JOBS = {config.N_JOBS}\n")
    shutil.copy(HERE / "features.py", out / "sentinel" / "features.py")
    shutil.copy(HERE / "explainer.py", out / "sentinel" / "explainer.py")
    if family == "lightgbm":
        model.booster_.save_model(str(out / "model.txt"))
    elif family == "xgboost":
        model.get_booster().save_model(str(out / "model.json"))
    else:
        joblib.dump(model, out / "model.joblib")
    (out / "feature_spec.json").write_text(spec.to_json())
    (out / "reference.json").write_text(json.dumps({"mean": {k: v["mean"] for k, v in ref.items()},
                                                    "std": {k: v["std"] for k, v in ref.items()}}))
    md = dict(meta, family=family, task=task, threshold=threshold, reliability=reliability)
    (out / "metadata.json").write_text(json.dumps(md, indent=2, default=str))
    (out / "serve.py").write_text(SERVE_PY)
    (out / "Dockerfile").write_text(DOCKERFILE)
    (out / "requirements.txt").write_text("\n".join(REQS[family] + BASE_REQS) + "\n")
    (out / "README.md").write_text(
        f"# {meta['model_name']} v{meta['version']}\n\nGenerated by Sentinel. Run hash `{meta['run_hash']}`.\n\n"
        "```\ndocker build -t sentinel-model . && docker run -p 8080:8080 sentinel-model\n"
        "curl localhost:8080/health\n```\n\nPOST /predict with `{\"rows\": [...]}` (raw sensor rows, oldest first).\n")
    return str(out)


def _load_service(pkg_dir: str):
    """Import the generated serve.py exactly as the container would (fresh sentinel package)."""
    saved = {m: sys.modules.pop(m) for m in list(sys.modules) if m == "sentinel" or m.startswith("sentinel.")}
    sys.path.insert(0, pkg_dir)
    try:
        spec = importlib.util.spec_from_file_location("sentinel_serve", os.path.join(pkg_dir, "serve.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(pkg_dir)
        for m in [m for m in sys.modules if m == "sentinel" or m.startswith("sentinel.")]:
            sys.modules.pop(m, None)
        sys.modules.update(saved)


def parity_test(pkg_dir: str, sample_rows: list, expected: dict, tol: float = 1e-6) -> dict:
    """POST sample rows to the generated service in-process and compare with the trained model's scores."""
    from fastapi.testclient import TestClient
    mod = _load_service(pkg_dir)
    client = TestClient(mod.app)
    health = client.get("/health").json()
    r = client.post("/predict", json={"rows": sample_rows})
    r.raise_for_status()
    got = {str(p["id"]): p["risk"] for p in r.json()["predictions"]}
    diffs = {k: abs(got[k] - v) for k, v in expected.items() if k in got}
    for cache in pathlib.Path(pkg_dir).rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    return {"ok": bool(diffs) and max(diffs.values()) <= tol, "max_abs_diff": max(diffs.values()) if diffs else None,
            "entities": len(diffs), "health": health}


def docker_smoke(pkg_dir: str, tag: str, sample_rows: list, expected: dict, tol: float = 1e-6) -> dict:
    """Build the image, run the container, hit /health and /predict, compare with the in-process model."""
    try:
        import docker
        import httpx
        cli = docker.from_env()
        cli.ping()
    except Exception as e:  # no docker daemon reachable
        return {"ran": False, "reason": f"docker unavailable ({type(e).__name__})"}
    t0 = time.time()
    image, _ = cli.images.build(path=pkg_dir, tag=tag, rm=True)
    build_s = round(time.time() - t0, 1)
    ctr = cli.containers.run(tag, detach=True, ports={"8080/tcp": None})
    try:
        ctr.reload()
        port = ctr.attrs["NetworkSettings"]["Ports"]["8080/tcp"][0]["HostPort"]
        host = os.environ.get("SENTINEL_DOCKER_HOST", "localhost")
        base = f"http://{host}:{port}"
        for _ in range(60):
            try:
                if httpx.get(base + "/health", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(1)
        r = httpx.post(base + "/predict", json={"rows": sample_rows}, timeout=30)
        r.raise_for_status()
        got = {str(p["id"]): p["risk"] for p in r.json()["predictions"]}
        diffs = {k: abs(got[k] - v) for k, v in expected.items() if k in got}
        size_mb = round(image.attrs.get("Size", 0) / 1e6, 1)
        return {"ran": True, "ok": bool(diffs) and max(diffs.values()) <= tol, "max_abs_diff": max(diffs.values()) if diffs else None,
                "image": tag, "image_mb": size_mb, "build_seconds": build_s, "port": int(port)}
    finally:
        ctr.stop(timeout=3)
        ctr.remove(force=True)
