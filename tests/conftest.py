import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sentinel import config, data  # noqa: E402

# tiny budget so the whole pipeline runs in seconds under test
config.BUDGETS["test"] = {"lightgbm": 2, "xgboost": 2, "random_forest": 1, "linear": 1}


def make_fleet(n_units=36, seed=7) -> pd.DataFrame:
    """Synthetic run-to-failure fleet: 6 sensors, two of which degrade toward failure."""
    rng = np.random.RandomState(seed)
    rows = []
    for u in range(1, n_units + 1):
        life = int(rng.randint(90, 150))
        t = np.arange(1, life + 1)
        wear = (t / life) ** 2
        rec = {
            "unit": u, "cycle": t,
            "s_temp": 500 + 25 * wear + rng.normal(0, 1.5, life),
            "s_vib": 1.0 + 3.0 * wear + rng.normal(0, 0.15, life),
            "s_press": 30 + rng.normal(0, 0.5, life),
            "s_flow": 10 + 0.5 * wear + rng.normal(0, 0.3, life),
            "s_noise": rng.normal(0, 1, life),
            "s_const": np.full(life, 3.14),
        }
        rows.append(pd.DataFrame(rec))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def fleet():
    return make_fleet()


def fleet_dataset(fleet: pd.DataFrame) -> data.Dataset:
    units = sorted(fleet["unit"].unique())
    hold_units = units[::4]
    hold = fleet[fleet["unit"].isin(hold_units)].copy()
    train = fleet[~fleet["unit"].isin(hold_units)].copy()
    # holdout targets: true RUL, as C-MAPSS ships them
    hold["__rul__"] = hold.groupby("unit")["cycle"].transform("max") - hold["cycle"]
    return data.Dataset(
        key="synthetic_fleet", name="Synthetic fleet", description="test", train=train.reset_index(drop=True),
        holdout=hold.reset_index(drop=True),
        meta={"entity_col": None, "time_col": None, "target_col": None, "holdout_rul_col": "__rul__",
              "sensor_labels": {"s_temp": "Temperature", "s_vib": "Vibration"}, "entity_noun": "Unit",
              "time_noun": "cycle"})


@pytest.fixture()
def isolated_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(config, "DATABASE_URL", "")
    monkeypatch.setattr(config, "MLFLOW_URI", "")
    return tmp_path
