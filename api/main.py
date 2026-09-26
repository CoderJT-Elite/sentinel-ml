"""Sentinel HTTP API + static UI. FastAPI serves both."""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import shutil
import threading
import time
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sentinel import __version__, config, data, deployer, lifecycle, registry
from sentinel.graph import NODES, run_pipeline
from sentinel.store import Store
from sentinel.util import clean

ROOT = pathlib.Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

app = FastAPI(title="Sentinel", version=__version__,
              description="Deterministic AutoML + MLOps for predictive maintenance. No LLM in the decision path.")
STORE = Store()


class Job:
    def __init__(self, job_id: str, dataset: str, options: dict):
        self.id, self.dataset, self.options = job_id, dataset, options
        self.status = "running"
        self.events: list = []
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.started = time.time()


JOBS: dict[str, Job] = {}


def _docker_ok() -> bool:
    try:
        import docker
        docker.from_env().ping()
        return True
    except Exception:
        return False


def _run_job(job: Job, ds: data.Dataset) -> None:
    try:
        res = run_pipeline(ds, {**job.options, "run_id": job.id}, on_event=job.events.append)
        job.result = res
        job.status = "halted" if res.get("halted") else "done"
        if res.get("halted"):
            STORE.save_run(job.id, ds.key, "halted", "", res)
        else:
            (pathlib.Path(config.RUNS_DIR) / job.id / "events.json").write_text(json.dumps(clean(job.events)))
    except Exception as e:  # surface pipeline errors to the UI instead of a silent hang
        job.status, job.error = "error", f"{type(e).__name__}: {e}"


def _load_dataset(key: str, options: dict) -> data.Dataset:
    if key.startswith("upload:"):
        raw_name = key.split(":", 1)[1]
        clean_name = pathlib.Path(raw_name).name
        uploads_dir = (pathlib.Path(config.RUNS_DIR) / "uploads").resolve()
        p = (uploads_dir / clean_name).resolve()
        if not p.is_relative_to(uploads_dir) or not p.exists():
            raise HTTPException(404, "uploaded dataset not found")
        return data.from_csv(str(p), p.stem, options.get("target"))
    if key.startswith("sample:"):
        raw_name = key.split(":", 1)[1]
        clean_name = pathlib.Path(raw_name).name + ".csv"
        samples_dir = (ROOT / "samples").resolve()
        p = (samples_dir / clean_name).resolve()
        if not p.is_relative_to(samples_dir) or not p.exists():
            raise HTTPException(404, "sample not found")
        return data.from_csv(str(p), p.stem, options.get("target") or "Machine failure")
    try:
        return data.load(key)
    except KeyError:
        raise HTTPException(404, f"unknown dataset {key}")
    except FileNotFoundError:
        raise HTTPException(503, "dataset files missing: run `python scripts/fetch_data.py`")


# ------------------------------------------------------------------ meta
@app.get("/api/mode")
def mode():
    return {"mode": "live", "version": __version__, "db": STORE.backend, "docker": _docker_ok(),
            "nodes": [{"id": i, "label": l} for i, l in NODES], "llm_calls": 0,
            "budgets": {k: sum(v.values()) for k, v in config.BUDGETS.items() if k in ("fast", "full")}}


@app.get("/api/datasets")
def datasets():
    out = []
    for key in data.REGISTRY:
        try:
            out.append(data.load(key).info())
        except Exception:
            out.append({"key": key, "name": key, "description": "files missing: run scripts/fetch_data.py", "rows": 0})
    samples = ROOT / "samples"
    if samples.exists():
        for p in sorted(samples.glob("*.csv")):
            out.append({"key": f"sample:{p.stem}", "name": p.stem.replace("_", " "), "rows": sum(1 for _ in open(p)) - 1,
                        "description": "Bundled sample used to demonstrate the quality gate.", "meta": {"target_col": None}})
    return out


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), target: str = Form("")):
    orig_name = pathlib.Path(file.filename or "upload.csv").name
    ext = pathlib.Path(orig_name).suffix.lower()
    if ext not in (".csv", ".txt"):
        raise HTTPException(400, "Only .csv or .txt files are supported")
    
    stem = "".join(c for c in pathlib.Path(orig_name).stem if c.isalnum() or c in ("-", "_"))[:50] or "upload"
    d = pathlib.Path(config.RUNS_DIR) / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex[:8]}_{stem}.csv"
    target_path = d / name

    max_bytes = 50 * 1024 * 1024
    total_bytes = 0
    with open(target_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                target_path.unlink(missing_ok=True)
                raise HTTPException(413, "Uploaded file exceeds maximum allowed size of 50 MB")
            f.write(chunk)
    
    import pandas as pd
    try:
        df_head = pd.read_csv(target_path, nrows=5)
        if df_head.empty and len(df_head.columns) == 0:
            target_path.unlink(missing_ok=True)
            raise HTTPException(400, "Uploaded CSV file is empty")
        cols = list(df_head.columns)
    except Exception as e:
        target_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to parse uploaded CSV: {e}")
        
    return {"key": f"upload:{name}", "name": name, "columns": cols, "target": target or None}


# ------------------------------------------------------------------ runs
class RunReq(BaseModel):
    dataset: str
    budget: str = "fast"
    task: str = "auto"
    target: Optional[str] = None
    horizon: Optional[int] = None


@app.post("/api/runs")
def start_run(req: RunReq):
    if req.budget not in ("fast", "full"):
        raise HTTPException(400, "budget must be fast or full")
    opts = {"budget": req.budget, "task": req.task, "target": req.target, "horizon": req.horizon}
    ds = _load_dataset(req.dataset, opts)
    job_id = "run-" + uuid.uuid4().hex[:8]
    job = Job(job_id, req.dataset, opts)
    JOBS[job_id] = job
    threading.Thread(target=_run_job, args=(job, ds), daemon=True).start()
    return {"run_id": job_id}


@app.get("/api/runs")
def list_runs():
    return STORE.list_runs()


def _job(run_id: str) -> Job:
    if run_id in JOBS:
        return JOBS[run_id]
    clean_id = pathlib.Path(run_id).name
    runs_dir = pathlib.Path(config.RUNS_DIR).resolve()
    p = (runs_dir / clean_id / "summary.json").resolve()
    if p.is_relative_to(runs_dir) and p.exists():  # a finished run from an earlier session
        j = Job(clean_id, "", {})
        j.result, j.status = json.loads(p.read_text()), "done"
        ev = runs_dir / clean_id / "events.json"
        j.events = json.loads(ev.read_text()) if ev.exists() else []
        JOBS[clean_id] = j
        return j
    raise HTTPException(404, "run not found")


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    j = _job(run_id)
    if j.status == "done" or j.status == "halted":
        return {"status": j.status, "result": clean(j.result)}
    return {"status": j.status, "error": j.error, "events": len(j.events)}


@app.get("/api/runs/{run_id}/trace")
def trace(run_id: str):
    return clean(_job(run_id).events)


@app.get("/api/runs/{run_id}/events")
async def stream_events(run_id: str):
    j = _job(run_id)

    async def gen():
        i = 0
        while True:
            while i < len(j.events):
                yield f"data: {json.dumps(clean(j.events[i]))}\n\n"
                i += 1
            if j.status != "running":
                yield f"event: end\ndata: {json.dumps({'status': j.status, 'error': j.error})}\n\n"
                return
            await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ------------------------------------------------------------------ deployment + inference
@app.post("/api/runs/{run_id}/deploy")
def deploy(run_id: str, build: bool = True):
    j = _job(run_id)
    if not j.result or j.result.get("halted"):
        raise HTTPException(409, "run did not complete")
    ctx = lifecycle.load(run_id)
    serving = j.result["serving"]
    files = sorted(p.name for p in pathlib.Path(serving["folder"]).iterdir())
    out = {"folder": serving["folder"], "files": files, "parity": serving["parity"],
           "registry": j.result["registry"], "docker": {"ran": False, "reason": "not requested"}}
    if build:
        sp = ctx["sample"]
        out["docker"] = deployer.docker_smoke(serving["folder"], f"sentinel-model:{run_id}", sp["rows"], sp["expected"])
    return clean(out)


class PredictReq(BaseModel):
    rows: list[dict]


@app.post("/api/runs/{run_id}/predict")
def predict(run_id: str, req: PredictReq):
    j = _job(run_id)
    if not j.result or j.result.get("halted"):
        raise HTTPException(409, "run did not complete")
    from fastapi.testclient import TestClient
    folder = j.result["serving"]["folder"]
    mod = deployer._load_service(folder)
    r = TestClient(mod.app).post("/predict", json={"rows": req.rows})
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    body = r.json()
    STORE.log_predictions(run_id, str(j.result["registry"]["version"]), body["predictions"])
    body["logged_predictions"] = STORE.count_predictions(run_id)
    return body


class DriftReq(BaseModel):
    sensor: str
    sigma: float


@app.post("/api/runs/{run_id}/drift")
def drift(run_id: str, req: DriftReq):
    _job(run_id)
    try:
        return clean(lifecycle.drift(run_id, req.sensor, req.sigma))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/runs/{run_id}/retrain")
def retrain(run_id: str, req: DriftReq):
    _job(run_id)
    try:
        return lifecycle.retrain(run_id, req.sensor, req.sigma)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/runs/{run_id}/lineage")
def lineage(run_id: str):
    j = _job(run_id)
    return {"versions": clean(registry.versions(j.result["registry"]["model_name"])),
            "run_hash": j.result["run_hash"]}


# ------------------------------------------------------------------ UI
@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=str(WEB)), name="web")
