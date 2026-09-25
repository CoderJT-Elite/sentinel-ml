"""Build the bundled sample that demonstrates the Data Steward's quality gate FAILING.

A plausible-looking maintenance export with a sensor that dropped out for almost half the
rows. Run: python scripts/make_samples.py   (needs data/ai4i2020.csv from fetch_data.py)
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    df = pd.read_csv(ROOT / "data" / "ai4i2020.csv").iloc[:3000].copy()
    rng = np.random.RandomState(7)
    for col, frac in (("Torque [Nm]", 0.46), ("Tool wear [min]", 0.38)):
        df.loc[rng.rand(len(df)) < frac, col] = np.nan
    out = ROOT / "samples" / "broken_sensor_log.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
