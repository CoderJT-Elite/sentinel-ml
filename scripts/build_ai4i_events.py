import json
import shutil

r = json.load(open("docs/demo/data/run_ai4i.json", encoding="utf-8"))

events = [
    {
        "seq": 1, "node": "data_steward", "status": "start", "title": "Profiling dataset",
        "decisions": [], "payload": {}, "t": 1.0
    },
    {
        "seq": 2, "node": "data_steward", "status": "done", "title": f"Stage 1 scorecard: {r['scorecard']['verdict']}",
        "decisions": r["decisions"][:2], "payload": {"scorecard": r["scorecard"]}, "t": 2.0
    },
    {
        "seq": 3, "node": "task_selector", "status": "start", "title": "Framing the task",
        "decisions": [], "payload": {}, "t": 3.0
    },
    {
        "seq": 4, "node": "task_selector", "status": "done", "title": f"Classification on '{r['framing']['target_col']}', ranked by {r['framing']['metric']}",
        "decisions": [d for d in r["decisions"] if d["node"] == "task_selector"],
        "payload": {"framing": r["framing"], "scorecard": r["scorecard"]}, "t": 4.0
    },
    {
        "seq": 5, "node": "feature_engineer", "status": "start", "title": "Building features",
        "decisions": [], "payload": {}, "t": 5.0
    },
    {
        "seq": 6, "node": "feature_engineer", "status": "done", "title": f"{r.get('feature_spec', {}).get('n_features', 6)} features engineered",
        "decisions": [d for d in r["decisions"] if d["node"] == "feature_engineer"],
        "payload": {"n_features": r.get("feature_spec", {}).get("n_features", 6), "names": r.get("feature_spec", {}).get("names", [])}, "t": 6.0
    },
    {
        "seq": 7, "node": "trainer", "status": "start", "title": "Seeded Optuna search over candidate models",
        "decisions": [], "payload": {}, "t": 7.0
    },
    {
        "seq": 8, "node": "trainer", "status": "done", "title": f"Champion: {r['champion']['name']}",
        "decisions": [d for d in r["decisions"] if d["node"] == "trainer"],
        "payload": {"leaderboard": r["leaderboard"]}, "t": 8.0
    },
    {
        "seq": 9, "node": "explainer", "status": "start", "title": "Computing Tree-SHAP contributions",
        "decisions": [], "payload": {}, "t": 9.0
    },
    {
        "seq": 10, "node": "explainer", "status": "done", "title": f"Top driver: {r['explain']['global']['features'][0]['label'] if r.get('explain', {}).get('global', {}).get('features') else 'Top driver'}",
        "decisions": [d for d in r["decisions"] if d["node"] == "explainer"],
        "payload": {"global": r.get("explain", {}).get("global", {}), "parity": r.get("explain", {}).get("parity", {})}, "t": 10.0
    },
    {
        "seq": 11, "node": "deployer_monitor", "status": "start", "title": "Registering and packaging",
        "decisions": [], "payload": {}, "t": 11.0
    },
    {
        "seq": 12, "node": "deployer_monitor", "status": "done", "title": f"Run {r['run_id']} complete",
        "decisions": [d for d in r["decisions"] if d["node"] == "deployer_monitor"],
        "payload": {
            "registry": r["registry"], "parity": r["serving"]["parity"],
            "run_id": r["run_id"], "run_hash": r["run_hash"], "baseline_status": "STABLE"
        }, "t": 12.0
    }
]

with open("docs/demo/data/events_ai4i.json", "w", encoding="utf-8") as f:
    json.dump(events, f, indent=2)

shutil.copy("docs/demo/data/events_ai4i.json", "web/data/events_ai4i.json")
print("Successfully generated events_ai4i.json with", len(events), "events")
