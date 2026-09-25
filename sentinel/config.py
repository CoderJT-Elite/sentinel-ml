"""Every threshold and seed the pipeline uses lives here, in one place.

Nothing in the pipeline "decides" anything that is not either one of these
constants or a number computed from the data. Changing a threshold changes the
run hash, which is the point: the configuration is part of the audit trail.
"""
import os

SEED = 42
N_JOBS = 4  # fixed (not os.cpu_count) so results do not depend on the host

# ---- Data Steward thresholds -------------------------------------------------
MISSING_WARN = 0.05          # max per-column missing fraction
MISSING_FAIL = 0.30
MIN_ROWS_WARN = 1000
MIN_ROWS_FAIL = 100
IMBALANCE_WARN = 0.10        # minority-class share
IMBALANCE_FAIL = 0.005
DUPLICATE_WARN = 0.01
COLLINEAR_R = 0.95           # |Pearson r| flagged as collinear pair
VIF_WARN = 10.0
KS_D_DRIFT = 0.20            # KS statistic that counts as "shifted"
KS_ALPHA = 0.01
HIGH_CARDINALITY = 50        # categorical uniques -> treated as an identifier
LEAKAGE_PRECISION = 0.98     # P(target=1 | flag=1) that marks a leaking flag
LEAKAGE_SUPPORT = 10

# ---- Selector / features -----------------------------------------------------
HORIZON_FRACTION = 0.15      # alarm horizon = 15% of the median run-to-failure life
ROLL_MEAN = 5
ROLL_STD = 5
ROLL_SLOPE = 10
LAG_DIFF = 5
IMBALANCE_METRIC_SWITCH = 0.10   # minority share below this -> rank by average precision

# ---- Trainer -----------------------------------------------------------------
CV_FOLDS = 5
BUDGETS = {
    # Optuna trials per candidate family
    "fast": {"lightgbm": 6, "xgboost": 6, "random_forest": 3, "linear": 3},
    "full": {"lightgbm": 20, "xgboost": 20, "random_forest": 8, "linear": 6},
}

# ---- Monitor -----------------------------------------------------------------
PSI_WATCH = 0.10
PSI_RETRAIN = 0.25
PSI_BINS = 10
MONITOR_WINDOW = 50      # drift is checked on early-life rows (healthy regime), where degradation has not started

RUNS_DIR = os.environ.get("SENTINEL_RUNS", "runs")
DATA_DIR = os.environ.get("SENTINEL_DATA", "data")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
