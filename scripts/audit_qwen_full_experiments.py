"""Recompute saved Qwen monitor scores and validate whole-run training splits."""
import hashlib
import json
from pathlib import Path

import torch

from behaviour_prediction.monitor.features import extract_runs
from run_qwen_full_experiments import ROOT, OUT, STEPS, load_outcomes, summarize


def main():
    protocol = json.loads((OUT/"protocol.json").read_text())
    for path, expected in protocol["input_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    data = extract_runs([ROOT/run for run in protocol["runs"]])
    stored = torch.load(OUT/"monitor-inputs.pt", map_location="cpu", weights_only=True)
    keys = [tuple(k) for k in stored["keys"]]
    rows = {(r["run"],r["step"]):r for r in data["rows"]}
    names = data["schema"]["features"]
    expected_x = torch.tensor([[rows[k]["features"][n] for n in names] for k in keys],dtype=torch.float64)
    assert torch.equal(expected_x,stored["inputs"]["summary42"])
    for run in protocol["runs"]:
        baseline = torch.load(ROOT/run/"activation_monitoring/step-000000.pt",map_location="cpu",weights_only=True)
        assert len(baseline["metadata"]["probes"]) == 66
        for step in STEPS:
            current = torch.load(ROOT/run/"activation_monitoring"/f"step-{step:06d}.pt",map_location="cpu",weights_only=True)
            assert current["metadata"]["probes"] == baseline["metadata"]["probes"]
            for layer in ("layer_15","layer_31"):
                assert current["activations"][layer].shape == (66,4096)
                assert torch.isfinite(current["activations"][layer]).all()
    monitor,_ = load_outcomes(protocol["runs"],"monitoring")
    models = json.loads((OUT/"fitted-monitors.json").read_text())
    predictions = json.loads((OUT/"predictions-before-evaluation.json").read_text())
    count = 0
    for model in models:
        method,task,tag,held = [model[k] for k in ("method","task","tag","held_out")]
        train = [tuple(k) for k in model["train_keys"]]
        assert all(run != held and step != 0 for run,step in train)
        field = "new_label" if task.startswith("new") else "label"
        labels = []
        for run,step in train:
            target = STEPS[STEPS.index(step)+1] if task.endswith("forecast") else step
            outcome=monitor[run,target]
            labels.append(outcome["tags"][tag] if tag else outcome[field])
        assert all(label is not None for label in labels)
        assert sum(labels)/len(labels) == model["prior"]
        matrix = stored["inputs"][method]
        train_x = matrix[[keys.index(k) for k in train]]
        params = model["parameters"]
        if params:
            assert torch.equal(train_x.mean(0),torch.tensor(params["mean"],dtype=torch.float64))
            assert torch.equal(train_x.std(0,unbiased=False).clamp_min(1e-6),torch.tensor(params["scale"],dtype=torch.float64))
        else:
            assert len(set(labels)) == 1
        selected = [r for r in predictions if r["method"]==method and r["task"]==task
                    and r["tag"]==tag and r["run"]==held]
        for row in selected:
            if params:
                tensors = {k:torch.tensor(params[k],dtype=torch.float64) for k in ("mean","scale","weight")}
                x=matrix[keys.index((held,row["input_step"]))]
                score=torch.sigmoid(((x-tensors["mean"])/tensors["scale"])@tensors["weight"]+params["bias"]).item()
            else:
                score=model["prior"]
            assert abs(score-row["score"])<1e-10
            assert int(score>=.5)==row["predicted_label"]
            count += 1
    scored=json.loads((OUT/"scored-predictions.json").read_text())
    assert summarize(scored)==json.loads((OUT/"report.json").read_text())
    result=dict(passed=True,source_files=len(protocol["input_sha256"]),snapshots=72,
                fold_models=len(models),predictions_recomputed=count,
                whole_run_holdouts_verified=True,training_only_scaling_verified=True,
                feature_matrix_reproduced=True,report_reproduced=True)
    (OUT/"data-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
