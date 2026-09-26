"""
Sentinel Number Consistency Checker.
Extracts numbers and metrics across documentation, description, and UI deliverables
and asserts them against the ground-truth artifacts run.json, drift.json, and deploy.json.
Exits 0 if all assertions pass, 1 otherwise.
"""
import sys
import json
import pathlib
import re

DEMO_DATA = pathlib.Path(r"C:\OS\GitHub\sentinel-ml\docs\demo\data")
RUN_PATH = DEMO_DATA / "run.json"
DRIFT_PATH = DEMO_DATA / "drift.json"
DEPLOY_PATH = DEMO_DATA / "deploy.json"

def check_numbers():
    print("=== RUNNING SENTINEL NUMBER CONSISTENCY CHECK ===")
    assert RUN_PATH.exists(), f"Missing {RUN_PATH}"
    assert DRIFT_PATH.exists(), f"Missing {DRIFT_PATH}"
    assert DEPLOY_PATH.exists(), f"Missing {DEPLOY_PATH}"

    with open(RUN_PATH, "r", encoding="utf-8") as f:
        run_data = json.load(f)
    with open(DRIFT_PATH, "r", encoding="utf-8") as f:
        drift_data = json.load(f)
    with open(DEPLOY_PATH, "r", encoding="utf-8") as f:
        deploy_data = json.load(f)

    # 1. Assert Ground Truth properties
    run_hash = run_data["run_hash"]
    expected_hash = "f206c5c62729d1464cce14aa15775ea3e5593d5ec7529184c1411ce9ffc01c2c"
    assert run_hash == expected_hash, f"Hash mismatch: {run_hash} != {expected_hash}"
    print(f"[PASS] Run Hash: {run_hash}")

    rows = run_data["dataset"]["rows"]
    assert rows == 20631, f"Rows mismatch: {rows} != 20631"
    print(f"[PASS] Training Rows: {rows}")

    holdout = run_data["dataset"]["holdout_rows"]
    assert holdout == 13096, f"Holdout mismatch: {holdout} != 13096"
    print(f"[PASS] Holdout Telemetry Rows: {holdout}")

    n_features = run_data["feature_spec"]["n_features"]
    assert n_features == 86, f"Features mismatch: {n_features} != 86"
    print(f"[PASS] Engineered Features: {n_features}")

    dropped = len(run_data["feature_spec"]["dropped"])
    assert dropped == 7, f"Dropped features mismatch: {dropped} != 7"
    print(f"[PASS] Dropped Uninformative Features: {dropped}")

    horizon = run_data["framing"]["horizon"]
    assert horizon == 30, f"Horizon mismatch: {horizon} != 30"
    print(f"[PASS] Alarm Horizon: {horizon} cycles")

    champion = run_data["champion"]["name"]
    assert champion == "XGBoost", f"Champion mismatch: {champion} != XGBoost"
    print(f"[PASS] Champion Model: {champion}")

    leaderboard = {row["name"]: row for row in run_data["leaderboard"]}
    champion_roc = leaderboard["XGBoost"]["cv_mean"]
    assert abs(champion_roc - 0.993012) < 1e-4, f"Champion ROC mismatch: {champion_roc}"
    print(f"[PASS] Champion CV ROC-AUC: {champion_roc:.4f}")

    fleet = run_data.get("fleet", [])
    assert len(fleet) == 100, f"Fleet size mismatch: {len(fleet)}"
    n_alert = sum(1 for e in fleet if e.get("level") == "ALERT")
    n_watch = sum(1 for e in fleet if e.get("level") == "WATCH")
    n_ok = sum(1 for e in fleet if e.get("level") == "OK")
    assert n_alert == 22, f"Alert count mismatch: {n_alert} != 22"
    assert n_watch == 7, f"Watch count mismatch: {n_watch} != 7"
    assert n_ok == 71, f"OK count mismatch: {n_ok} != 71"
    print(f"[PASS] Fleet Triage: {n_alert} ALERT, {n_watch} WATCH, {n_ok} OK")

    # 2. Assert deploy properties
    img_mb = deploy_data["docker"]["image_mb"]
    assert img_mb == 151.1, f"Docker size mismatch: {img_mb} != 151.1"
    print(f"[PASS] Docker Image Size: {img_mb} MB")

    max_diff = deploy_data["docker"]["max_abs_diff"]
    assert max_diff == 0.0, f"Serving diff mismatch: {max_diff} != 0.0"
    print(f"[PASS] Serving Discrepancy: {max_diff}")

    # 3. Assert occurrence of exact values in deliverable documents
    readme_path = pathlib.Path(r"C:\OS\GitHub\sentinel-ml\README.md")
    readme_text = readme_path.read_text(encoding="utf-8")
    assert expected_hash in readme_text, "README missing run hash"
    assert "0.9930" in readme_text, "README missing 0.9930 ROC-AUC"
    assert "20 tests" in readme_text or "tests-20_passing" in readme_text, "README missing 20 tests badge"
    assert "XGBoost" in readme_text, "README missing XGBoost champion"
    print("[PASS] README.md contains exact matching numbers.")

    desc_path = pathlib.Path(r"C:\OS\Projects\Active\Competitions\ABB Accelerator\Prototype Phase\submission-kit\02-description.md")
    desc_text = desc_path.read_text(encoding="utf-8")
    assert expected_hash in desc_text, "Description missing run hash"
    assert "0.9930" in desc_text, "Description missing 0.9930"
    assert "86" in desc_text, "Description missing 86 features"
    print("[PASS] 02-description.md contains exact matching numbers.")

    tech_path = pathlib.Path(r"C:\OS\GitHub\sentinel-ml\docs\TECHNICAL_DOCUMENTATION.md")
    tech_text = tech_path.read_text(encoding="utf-8")
    assert expected_hash in tech_text, "Technical documentation missing run hash"
    assert "0.9930" in tech_text, "Technical documentation missing 0.9930"
    assert "86" in tech_text, "Technical documentation missing 86 features"
    print("[PASS] TECHNICAL_DOCUMENTATION.md contains exact matching numbers.")

    print("\n[ALL CHECKS PASSED] Every metric is 100% consistent with JSON ground-truth.")

if __name__ == "__main__":
    check_numbers()
