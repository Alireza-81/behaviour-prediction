"""Monitor safeguards: past-only features, split isolation and honest evidence counts."""
import csv
from pathlib import Path

import pytest
import torch

from behaviour_prediction.monitor.classifier import audit, fit, predict, read_labels
from behaviour_prediction.monitor.features import extract_run


def snapshot(run, step, displacement):
    directory = run / "activation_monitoring"
    directory.mkdir(parents=True, exist_ok=True)
    probes = [dict(id="a", group="harmful", category="a", prompt="a", token_count=1),
              dict(id="b", group="benign", category="b", prompt="b", token_count=1)]
    torch.save(dict(metadata=dict(probes=probes, pooling="last_non_padding"),
        activations={"layer_00": torch.tensor([[1.0, displacement], [1.0, 0.0]])}),
        directory / f"step-{step:06d}.pt")


def test_features_never_look_into_future(tmp_path):
    snapshot(tmp_path, 0, 0)
    snapshot(tmp_path, 2, 2)
    schema, prefix = extract_run(tmp_path)
    snapshot(tmp_path, 9, 90)
    later_schema, full = extract_run(tmp_path)
    assert schema == later_schema
    assert prefix == full[:2]
    assert prefix[1]["features"]["layer_00/harmful/l2_mean"] == 2
    assert prefix[1]["features"]["layer_00/harmful/speed"] == 1
    assert "step" not in schema["features"] and "run" not in schema["features"]


def write_labels(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "step", "label", "split", "group_id", "label_source"])
        writer.writeheader()
        writer.writerows(rows)


def test_related_runs_cannot_cross_splits(tmp_path):
    rows = [dict(run="one", step=1), dict(run="two", step=1)]
    path = tmp_path / "labels.csv"
    write_labels(path, [dict(run=r["run"], step=1, label=1, split=s, group_id="same_family", label_source="review")
                        for r, s in zip(rows, ["train", "test"])])
    with pytest.raises(ValueError, match="Leakage"):
        read_labels(path, rows)


def test_more_checkpoints_do_not_inflate_confidence():
    rows = [dict(run="one", group_id="family", step=i, label=1) for i in range(1, 101)]
    rows.append(dict(run="negative", group_id="negative", step=1, label=0))
    report = audit(rows, torch.tensor([0.99]*100+[0.1]), 0.9, 0.95)
    assert report["alerted_independent_groups"] == 1
    assert report["group_reliability_lower_bound"] < 0.5
    assert report["negative_control_groups"] == 1


def test_false_early_alert_cannot_be_repaired_by_later_correct_alert():
    rows = [dict(run="one", group_id="family", step=1, label=0),
            dict(run="one", group_id="family", step=2, label=1)]
    report = audit(rows, torch.tensor([0.99, 0.99]), 0.9, 0.95)
    assert report["groups_with_all_first_alerts_correct"] == 0


def test_small_fit_scores_but_abstains_without_evidence():
    rows, labelled = [], []
    for split in ["train", "calibration", "test"]:
        for label in [0, 1]:
            run = f"{split}_{label}"
            for step in [1, 2]:
                row = dict(run=run, step=step, features={"relative_drift": label*3.0 + step*0.01})
                rows.append(row)
                labelled.append({**row, "label": label, "split": split, "group_id": run, "label_source": "synthetic"})
    data = {"schema": {"features": ["relative_drift"]}, "rows": rows}
    model = fit(data, labelled)
    assert model["confidence_status"] == "insufficient_evidence"
    assert model["audit"]["label_coverage_complete"] == {"calibration": True, "test": True}
    fresh = {"schema": data["schema"], "rows": [{**rows[-1], "run": "unseen"}]}
    result = predict(model, fresh)[0]
    assert result["decision"] == "uncertain"
    assert result["harmful_probability"] > 0.5
    with pytest.raises(ValueError, match="schema mismatch"):
        predict(model, {"schema": {}, "rows": rows})


def test_online_callback_writes_uncertain_prediction(tmp_path, monkeypatch):
    import json
    from behaviour_prediction.monitoring import trajectory
    snapshot(tmp_path, 0, 0)
    schema, _ = extract_run(tmp_path)
    source = torch.load(tmp_path / "activation_monitoring/step-000000.pt", weights_only=True)
    artifact = dict(schema=schema, mean=[0]*len(schema["features"]), scale=[1]*len(schema["features"]),
                    weight=[0]*len(schema["features"]), bias=10, temperature=1, alert_threshold=0.9,
                    confidence_status="insufficient_evidence", split_runs={})
    model_path = tmp_path / "monitor.json"
    model_path.write_text(json.dumps(artifact))
    probe_path = tmp_path / "probes.jsonl"
    probe_path.write_text("\n".join(json.dumps(p) for p in source["metadata"]["probes"]))
    monkeypatch.setattr(trajectory, "collect_probe_activations", lambda **kwargs: source)
    callback = trajectory.TrajectoryCallback(tmp_path, None, dict(layers=[0], max_length=4,
        monitor_steps=[0], probe_path=str(probe_path), monitor_model=str(model_path)), 2)
    callback.capture(torch.nn.Linear(2, 2), 0, None)
    prediction = json.loads((tmp_path / "monitor_predictions.jsonl").read_text())
    assert prediction["candidate_alert"] is True
    assert prediction["decision"] == "uncertain"
