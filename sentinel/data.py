"""Dataset adapters. An adapter only loads and describes data; it makes no decisions."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import config

CMAPSS_SENSORS = {
    "s1": "T2 total temperature at fan inlet",
    "s2": "T24 total temperature at LPC outlet",
    "s3": "T30 total temperature at HPC outlet",
    "s4": "T50 total temperature at LPT outlet",
    "s5": "P2 pressure at fan inlet",
    "s6": "P15 total pressure in bypass duct",
    "s7": "P30 total pressure at HPC outlet",
    "s8": "Nf physical fan speed",
    "s9": "Nc physical core speed",
    "s10": "epr engine pressure ratio",
    "s11": "Ps30 static pressure at HPC outlet",
    "s12": "phi fuel flow to Ps30 ratio",
    "s13": "NRf corrected fan speed",
    "s14": "NRc corrected core speed",
    "s15": "BPR bypass ratio",
    "s16": "farB burner fuel-air ratio",
    "s17": "htBleed bleed enthalpy",
    "s18": "Nf_dmd demanded fan speed",
    "s19": "PCNfR_dmd demanded corrected fan speed",
    "s20": "W31 HPT coolant bleed",
    "s21": "W32 LPT coolant bleed",
    "op1": "operating setting 1",
    "op2": "operating setting 2",
    "op3": "operating setting 3",
    "cycle": "operating cycles",
}


@dataclass
class Dataset:
    key: str
    name: str
    description: str
    train: pd.DataFrame
    holdout: Optional[pd.DataFrame] = None
    meta: dict = field(default_factory=dict)

    def info(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "rows": int(len(self.train)),
            "columns": int(self.train.shape[1]),
            "holdout_rows": int(len(self.holdout)) if self.holdout is not None else 0,
            "meta": {k: v for k, v in self.meta.items() if k in ("entity_col", "time_col", "target_col", "source")},
        }


def _path(name: str) -> str:
    return os.path.join(config.DATA_DIR, name)


def load_cmapss(subset: str = "FD001") -> Dataset:
    cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
    train = pd.read_csv(_path(f"train_{subset}.txt"), sep=r"\s+", header=None, names=cols)
    test = pd.read_csv(_path(f"test_{subset}.txt"), sep=r"\s+", header=None, names=cols)
    rul_last = pd.read_csv(_path(f"RUL_{subset}.txt"), header=None, names=["rul_last"])["rul_last"].to_numpy()
    last_cycle = test.groupby("unit")["cycle"].transform("max")
    test["__rul__"] = rul_last[test["unit"].to_numpy() - 1] + (last_cycle - test["cycle"]).to_numpy()
    return Dataset(
        key=f"cmapss_{subset.lower()}",
        name=f"NASA C-MAPSS {subset} turbofan degradation",
        description=(
            "Run-to-failure telemetry for 100 simulated turbofan engines (21 sensors, 3 operating "
            "settings). The training file has no target column: Sentinel derives it from the "
            "run-to-failure structure. The official test file is held out and labelled by NASA's RUL file."
        ),
        train=train,
        holdout=test,
        meta={
            "source": "NASA Prognostics Center of Excellence, C-MAPSS",
            "entity_col": None,   # deliberately NOT declared: the selector must detect it
            "time_col": None,
            "target_col": None,
            "holdout_rul_col": "__rul__",
            "sensor_labels": CMAPSS_SENSORS,
            "entity_noun": "Engine",
            "time_noun": "cycle",
        },
    )


def load_ai4i() -> Dataset:
    df = pd.read_csv(_path("ai4i2020.csv"))
    rng = np.random.RandomState(config.SEED)
    idx = rng.permutation(len(df))
    n_hold = int(round(0.2 * len(df)))
    hold, train = df.iloc[np.sort(idx[:n_hold])], df.iloc[np.sort(idx[n_hold:])]
    return Dataset(
        key="ai4i",
        name="UCI AI4I 2020 milling machine failures",
        description=(
            "10,000 rows of milling-machine process data with a rare (3.4%) machine-failure label. "
            "Contains identifier columns and five failure-mode flags that leak the target; "
            "the Data Steward has to catch both."
        ),
        train=train.reset_index(drop=True),
        holdout=hold.reset_index(drop=True),
        meta={
            "source": "UCI Machine Learning Repository #601",
            "entity_col": None,
            "time_col": None,
            "target_col": "Machine failure",
            "sensor_labels": {},
            "entity_noun": "Machine",
            "time_noun": "row",
        },
    )


REGISTRY = {"cmapss_fd001": load_cmapss, "ai4i": load_ai4i}


def load(key: str) -> Dataset:
    if key not in REGISTRY:
        raise KeyError(f"unknown dataset {key!r}")
    return REGISTRY[key]()


def from_csv(path: str, name: str, target_col: Optional[str] = None) -> Dataset:
    """Wrap an uploaded CSV. Holdout is a seeded 20% row split (decided later, by entity if one is detected)."""
    df = pd.read_csv(path)
    return Dataset(
        key="upload",
        name=name,
        description="User-uploaded dataset.",
        train=df,
        holdout=None,
        meta={"source": "upload", "entity_col": None, "time_col": None, "target_col": target_col,
              "sensor_labels": {}, "entity_noun": "Asset", "time_noun": "row"},
    )
