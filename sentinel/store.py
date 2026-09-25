"""Persistence: runs, the decision log, and the prediction log.

PostgreSQL when DATABASE_URL points at it (docker compose), SQLite otherwise
(local dev, tests, the hosted demo). Same schema, same code path.
"""
from __future__ import annotations

import json
import os
import time

from sqlalchemy import Column, Float, Integer, MetaData, String, Table, Text, create_engine, insert, select

from . import config
from .util import clean

meta = MetaData()
runs = Table("runs", meta,
             Column("id", String(40), primary_key=True), Column("dataset", String(80)),
             Column("status", String(20)), Column("run_hash", String(64)), Column("created", Float),
             Column("summary", Text))
decisions = Table("decisions", meta,
                  Column("id", Integer, primary_key=True, autoincrement=True), Column("run_id", String(40)),
                  Column("seq", Integer), Column("node", String(40)), Column("rule", String(60)),
                  Column("detail", Text))
predictions = Table("predictions", meta,
                    Column("id", Integer, primary_key=True, autoincrement=True), Column("run_id", String(40)),
                    Column("model_version", String(20)), Column("entity", String(60)), Column("risk", Float),
                    Column("level", String(10)), Column("created", Float))


class Store:
    def __init__(self, url: str | None = None):
        url = url or config.DATABASE_URL
        if not url:
            os.makedirs(config.RUNS_DIR, exist_ok=True)
            url = f"sqlite:///{os.path.join(config.RUNS_DIR, 'sentinel.db')}"
        self.url = url
        self.engine = create_engine(url, future=True)
        meta.create_all(self.engine)

    @property
    def backend(self) -> str:
        return self.engine.dialect.name

    def save_run(self, run_id: str, dataset: str, status: str, run_hash: str, summary: dict) -> None:
        with self.engine.begin() as c:
            c.execute(runs.delete().where(runs.c.id == run_id))
            c.execute(insert(runs).values(id=run_id, dataset=dataset, status=status, run_hash=run_hash,
                                          created=time.time(), summary=json.dumps(clean(summary))))

    def save_decisions(self, run_id: str, items: list) -> None:
        with self.engine.begin() as c:
            c.execute(decisions.delete().where(decisions.c.run_id == run_id))
            for i, d in enumerate(items):
                c.execute(insert(decisions).values(run_id=run_id, seq=i, node=d.get("node", ""),
                                                   rule=d.get("rule", ""), detail=json.dumps(clean(d))))

    def log_predictions(self, run_id: str, version: str, rows: list) -> None:
        with self.engine.begin() as c:
            for r in rows:
                c.execute(insert(predictions).values(run_id=run_id, model_version=version, entity=str(r["id"]),
                                                     risk=float(r["risk"]), level=r["level"], created=time.time()))

    def count_predictions(self, run_id: str) -> int:
        with self.engine.begin() as c:
            return len(c.execute(select(predictions.c.id).where(predictions.c.run_id == run_id)).all())

    def list_runs(self) -> list:
        with self.engine.begin() as c:
            rows = c.execute(select(runs.c.id, runs.c.dataset, runs.c.status, runs.c.run_hash, runs.c.created)
                             .order_by(runs.c.created.desc())).all()
        return [dict(id=r[0], dataset=r[1], status=r[2], run_hash=r[3], created=r[4]) for r in rows]
