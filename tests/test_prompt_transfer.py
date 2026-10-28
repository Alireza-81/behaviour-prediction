import csv
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("evaluate_new_prompts", Path(__file__).parents[1]/"scripts/evaluate_new_prompts.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def fixture_labels(tmp_path):
    prompts = [dict(id=f"p{i}", group="benign", category="test", prompt=f"question {i}") for i in range(30)]
    protocol = dict(runs=["run"], steps=[0,10,20], protocol_id="test")
    rows = [dict(run="run", step=step, probe_id=p["id"], group=p["group"], category=p["category"],
                 prompt=p["prompt"], label="0", behaviour_tags="") for step in protocol["steps"] for p in prompts]
    labels = tmp_path/"labels"
    labels.mkdir()
    with (labels/"run.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return labels, protocol, prompts, rows


def test_evaluation_suite_and_saved_predictions_are_intact():
    protocol, prompts = evaluation.check_protocol(full=True)
    assert len(prompts) == 30
    assert protocol["refit"] is False


def test_outcomes_require_exact_probe_coverage(tmp_path):
    folder, protocol, prompts, rows = fixture_labels(tmp_path)
    result = evaluation.load_outcomes(folder, protocol, prompts)
    assert len(result) == 3 and result[("run",10)]["label"] == 0
    rows[-1]["prompt"] = "changed"
    with (folder/"run.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="mismatch"):
        evaluation.load_outcomes(folder, protocol, prompts)


def test_transfer_uses_new_labels_and_updates_persistence_without_refitting(tmp_path, monkeypatch):
    folder, protocol, prompts, _ = fixture_labels(tmp_path)
    monkeypatch.setattr(evaluation, "check_protocol", lambda **kwargs:(protocol, prompts))
    monkeypatch.chdir(tmp_path)
    artifacts = {
        "detection/heldout_predictions.json":[dict(run="run",step=10,method="summary",score=.9,predicted_label=1,label=1)],
        "behaviours/predictions.json":[dict(run="run",step=10,method="summary",tag="harassment",score=.9,predicted_label=1,label=1)],
        "forecast/predictions.json":[dict(run="run",input_step=10,target_step=20,current_label=1,label=1,
             method="persistence",score=1.,predicted_label=1)]}
    for name, value in artifacts.items():
        path = Path("outputs/results")/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    before = {name:(Path("outputs/results")/name).read_bytes() for name in artifacts}
    report = evaluation.score(folder,tmp_path/"audit")
    assert report["refit"] is False
    assert report["detection"]["summary"]["fp"] == 1
    assert report["behaviours"]["summary"]["fp"] == 1
    assert report["forecast"]["persistence"]["tn"] == 1
    assert all((Path("outputs/results")/name).read_bytes() == data for name,data in before.items())
