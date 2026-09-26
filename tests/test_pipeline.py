"""End-to-end and unit tests. The headline claim under test: same input + same seed -> same run."""
import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

from conftest import fleet_dataset
from sentinel import config, data, explainer, features, monitor, selector, steward, structure
from sentinel.graph import run_pipeline

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- structure + steward
def test_structure_detected(fleet):
    s = structure.detect_entity_time(fleet)
    assert (s["entity_col"], s["time_col"]) == ("unit", "cycle")
    assert s["n_entities"] == fleet["unit"].nunique()


def test_structure_absent_for_tabular():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500), "y": rng.randint(0, 2, 500)})
    assert structure.detect_entity_time(df) is None


def test_steward_flags_constants_and_ids(fleet):
    df = fleet.copy()
    df["row_id"] = np.arange(len(df))
    prof = steward.profile(df)
    dropped = {d["col"] for d in prof["findings"]["drop"]}
    assert {"s_const", "row_id"} <= dropped


def test_gate_fails_on_missing_data(fleet):
    df = fleet.copy()
    df.loc[df.sample(frac=0.5, random_state=1).index, "s_temp"] = np.nan
    sc = steward.scorecard(steward.profile(df)["checks"])
    assert not sc["passed"]
    assert any(c["id"] == "missing" and c["status"] == "fail" for c in sc["checks"])


def test_leakage_guard_catches_flag_columns():
    rng = np.random.RandomState(0)
    y = (rng.rand(2000) < 0.1).astype(int)
    df = pd.DataFrame({"x": rng.normal(size=2000), "leak": y.copy(), "harmless": (rng.rand(2000) < 0.1).astype(int)})
    audit = steward.target_audit(df, pd.Series(y), "classification")
    assert [d["col"] for d in audit["drop"]] == ["leak"]


# ---------------------------------------------------------------- selector + features
def test_horizon_rule_is_closed_form(fleet):
    s = structure.detect_entity_time(fleet)
    fr = selector.frame(fleet, {"target_col": None}, {}, s)
    assert fr["task"] == "classification"
    assert fr["horizon"] == round(config.HORIZON_FRACTION * s["median_life"])


def test_features_are_order_stable(fleet):
    s = structure.detect_entity_time(fleet)
    spec = features.plan(fleet, s, [{"col": "s_const", "reason": "constant"}], set())
    a = features.transform(features.prepare(fleet, spec), spec)
    b = features.transform(features.prepare(fleet.sample(frac=1, random_state=3), spec), spec)
    pd.testing.assert_frame_equal(a, b)


def test_rolling_window_matches_pandas(fleet):
    s = structure.detect_entity_time(fleet)
    spec = features.plan(fleet, s, [], set())
    X = features.transform(features.prepare(fleet, spec), spec)
    u1 = fleet[fleet["unit"] == 1].sort_values("cycle")
    ref = u1["s_temp"].rolling(spec.roll_mean, min_periods=1).mean()
    # start-of-series is padded with the first value, so compare after the window has filled
    got = X.loc[fleet.sort_values(["unit", "cycle"]).reset_index(drop=True)["unit"] == 1, "s_temp__mean5"].to_numpy()
    assert np.allclose(got[spec.roll_mean:], ref.to_numpy()[spec.roll_mean:])


# ---------------------------------------------------------------- monitor
def test_psi_zero_for_same_and_large_for_shift():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"f": rng.normal(size=5000)})
    ref = monitor.make_reference(X)
    same = monitor.psi_from_ref(ref["f"], rng.normal(size=5000))
    shifted = monitor.psi_from_ref(ref["f"], rng.normal(loc=1.5, size=5000))
    assert same < config.PSI_WATCH and shifted > config.PSI_RETRAIN


# ---------------------------------------------------------------- the headline claims
@pytest.fixture(scope="module")
def two_runs(tmp_path_factory, fleet):
    tmp = tmp_path_factory.mktemp("runs")
    out = []
    for i in range(2):
        config.RUNS_DIR = str(tmp / f"r{i}")
        config.DATABASE_URL = ""
        config.MLFLOW_URI = ""
        out.append(run_pipeline(fleet_dataset(fleet), {"budget": "test", "run_id": f"t{i}"}))
    return out


def test_pipeline_is_deterministic(two_runs):
    a, b = two_runs
    assert a["run_hash"] == b["run_hash"]
    assert [r["cv_mean"] for r in a["leaderboard"]] == [r["cv_mean"] for r in b["leaderboard"]]
    assert a["decisions"] == b["decisions"]
    assert a["fleet"][0]["sentence"] == b["fleet"][0]["sentence"]


def test_champion_is_top_cv_and_holdout_not_used(two_runs):
    board = two_runs[0]["leaderboard"]
    assert board[0]["cv_mean"] == max(r["cv_mean"] for r in board)
    assert two_runs[0]["champion"]["family"] == board[0]["family"]


def test_pipeline_learns_the_degradation_signal(two_runs):
    fin = two_runs[0]["leaderboard"][0]["holdout"]["all_rows"]
    assert fin["roc_auc"] > 0.9


def test_serving_package_reproduces_model(two_runs):
    par = two_runs[0]["serving"]["parity"]
    assert par["ok"], par


def test_shap_parity(two_runs):
    p = two_runs[0]["explain"]["parity"]
    if p.get("checked"):
        assert p["max_abs_diff"] < 1e-4


def test_whatif_payload_reproduces_the_fleet_risks(two_runs):
    """What the browser would score from the exported trees equals the risk the pipeline recorded."""
    from sentinel import whatif
    from sentinel.util import clean
    r = two_runs[0]
    wi = clean(r["whatif"])
    if wi is None:
        pytest.skip("champion is not a boosted-tree model")
    for e in r["fleet"]:
        x = np.array([np.nan if v is None else v for v in clean(e["x"])])
        risk = whatif.sigmoid(wi["base"] + whatif.score_margin(wi["kind"], wi["trees"], x))
        assert abs(risk - e["risk"]) < 1e-5


def test_clean_keeps_raw_values_unrounded():
    from sentinel.util import Raw, clean
    assert clean({"a": 0.123456789, "x": Raw([0.000012345678])}) == {"a": 0.123457, "x": [0.000012345678]}


def test_verify_accepts_a_run_and_catches_an_edit(two_runs, tmp_path):
    import json
    from sentinel import cli
    from sentinel.util import clean
    rec = clean(two_runs[0])
    good = tmp_path / "run.json"
    good.write_text(json.dumps(rec))
    assert cli.verify_run(str(good)) == 0
    rec["leaderboard"][0]["cv_mean"] = round(rec["leaderboard"][0]["cv_mean"] + 0.01, 6)
    bad = tmp_path / "edited.json"
    bad.write_text(json.dumps(rec))
    assert cli.verify_run(str(bad)) == 1


def test_registry_and_zero_llm_calls(two_runs):
    r = two_runs[0]
    assert r["registry"]["version"] >= 1 and r["llm_calls"] == 0


def test_run_hash_changes_with_seedable_config(fleet, isolated_runs):
    a = run_pipeline(fleet_dataset(fleet), {"budget": "test", "horizon": 20, "run_id": "h1"})
    b = run_pipeline(fleet_dataset(fleet), {"budget": "test", "horizon": 25, "run_id": "h2"})
    assert a["run_hash"] != b["run_hash"]


def test_regression_path(fleet, isolated_runs):
    r = run_pipeline(fleet_dataset(fleet), {"budget": "test", "task": "regression", "run_id": "reg"})
    assert r["framing"]["task"] == "regression" and r["leaderboard"][0]["metric"] == "rmse"
    assert r["leaderboard"][0]["cv_mean"] == min(b["cv_mean"] for b in r["leaderboard"])
    assert "remaining life" in r["fleet"][0]["sentence"] and r["serving"]["parity"]["ok"]


def test_gate_halts_the_pipeline(isolated_runs):
    rng = np.random.RandomState(0)
    n = 1500
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "fail": (rng.rand(n) < 0.1).astype(int)})
    df.loc[rng.rand(n) < 0.5, "a"] = np.nan
    ds = data.Dataset(key="broken", name="broken", description="", train=df,
                      meta={"target_col": "fail", "sensor_labels": {}, "entity_noun": "Asset", "time_noun": "row"})
    r = run_pipeline(ds, {"budget": "test"})
    assert r.get("halted") and not r["scorecard"]["passed"]


def test_whatif_trees_reproduce_the_library_scores():
    """The flat tree arrays the browser scores must give the same probabilities as XGBoost and LightGBM."""
    import lightgbm as lgb
    import xgboost as xgb
    from sentinel import whatif
    rng = np.random.RandomState(7)
    X = pd.DataFrame(rng.normal(size=(600, 6)), columns=list("abcdef"))
    y = ((X["a"] + 0.5 * X["b"] ** 2 + rng.normal(scale=0.3, size=600)) > 0.6).astype(int)
    ref = {"mean": {c: 0.0 for c in X}, "std": {c: 1.0 for c in X}}
    for family, model, proba in (
        ("xgboost", xgb.XGBClassifier(n_estimators=40, max_depth=3, random_state=1).fit(X, y), lambda m, d: m.predict_proba(d)[:, 1]),
        ("lightgbm", lgb.LGBMClassifier(n_estimators=40, num_leaves=7, random_state=1, verbose=-1).fit(X, y), lambda m, d: m.predict_proba(d)[:, 1]),
    ):
        payload = whatif.export(family, model, X, ref, "classification")
        assert payload is not None and len(payload["trees"]) == 40
        got = np.array([whatif.sigmoid(payload["base"] + whatif.score_margin(family, payload["trees"], r)) for r in X.to_numpy()])
        assert np.max(np.abs(got - proba(model, X))) < 1e-5, family


def test_whatif_declines_unsupported_models():
    from sentinel import whatif
    X = pd.DataFrame({"a": [0.0, 1.0]})
    assert whatif.export("linear", object(), X, {"mean": {}, "std": {}}, "classification") is None
    assert whatif.export("xgboost", object(), X, {"mean": {}, "std": {}}, "regression") is None


# ---------------------------------------------------------------- no language model in the decision path
BANNED = {"openai", "anthropic", "transformers", "langchain", "langchain_openai", "langchain_anthropic",
          "cohere", "google.generativeai", "ollama", "llama_cpp", "litellm"}


def test_no_llm_imports_anywhere_in_the_package():
    offenders = []
    for py in list((ROOT / "sentinel").glob("*.py")) + list((ROOT / "api").glob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                if m.split(".")[0] in BANNED or m in BANNED:
                    offenders.append((py.name, m))
    assert not offenders, offenders
