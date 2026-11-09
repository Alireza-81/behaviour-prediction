"""Keep separate-prompt test inputs separate without fitting on their labels."""
import csv
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import run_early_step_experiments as experiment


def setup_case(tmp_path, monkeypatch, *, missing_step=False):
    runs, steps = ["a", "b"], [0, 1, 2]
    root, eval_root = tmp_path/"monitor", tmp_path/"evaluation"
    root.mkdir()
    eval_root.mkdir()
    names = [f"feature_{i}" for i in range(42)]
    probe = dict(id="p",prompt="example",group="benign",category="example")
    schema = dict(features=names,model={},pooling="last_non_padding",layers={"layer_31":[1,42]},probes=[probe])
    rows = [dict(run=r,step=s,features={n:float(s) for n in names}) for r in runs for s in steps]
    other = [dict(r,features={n:(0. if r["step"]==0 else -8.) for n in names}) for r in rows
             if not (missing_step and r["step"]==1)]
    monkeypatch.setattr(experiment,"RUNS",runs)
    monkeypatch.setattr(experiment,"STEPS",steps)
    monkeypatch.setattr(experiment,"TAGS",("hate_speech",))
    monkeypatch.setattr(experiment,"extract_runs",lambda paths,**kw: dict(schema=schema,rows=other if paths[0].parent==eval_root else rows))
    def outcomes(root,suite):
        return {(r,s):dict(label=int(s==2),tags={"hate_speech":int(s==2)},unresolved_count=0) for r in runs for s in steps}
    monkeypatch.setattr(experiment,"load_outcomes",outcomes)
    monkeypatch.setattr(experiment,"plot_results",lambda *args:None)
    for suite in ("monitoring","evaluation"):
        directory=root/f"labeling-{suite}/labeled"
        directory.mkdir(parents=True)
        for run in runs:
            with (directory/f"{run}.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=["probe_id","prompt","group","category"])
                writer.writeheader()
                writer.writerow(dict(probe_id="p",**{k:probe[k] for k in ("prompt","group","category")}))
    return root,eval_root


def test_full_transfer_uses_different_inputs_with_same_parameters(tmp_path,monkeypatch):
    root,eval_root=setup_case(tmp_path,monkeypatch)
    out=tmp_path/"result"
    experiment.run(root,out,evaluation_root=eval_root)
    predictions=json.loads((out/"predictions.json").read_text())
    fitted=json.loads((out/"fitted-monitors.json").read_text())
    saved=torch.load(out/"inputs.pt",map_location="cpu",weights_only=True)
    for model in fitted:
        assert all(r!=model["held_out"] and s>0 for r,s in model["train_keys"])
    for row in predictions:
        if row["method"]!="monitor":continue
        model=next(m for m in fitted if (m["experiment"],m["tag"],m["held_out"])==(row["experiment"],row["tag"],row["run"]))
        p=model["parameters"]
        if not p:continue
        x=saved[row["suite_only"]][saved["keys"].index((row["run"],row["input_step"]))]
        mean,scale,w=[torch.tensor(p[k],dtype=torch.float64) for k in ("mean","scale","weight")]
        score=torch.sigmoid(((x-mean)/scale)@w+p["bias"]).item()
        assert abs(score-row["score"])<1e-12
    detection=[p for p in predictions if p["method"]=="monitor" and p["experiment"]=="detection"]
    a=[p["score"] for p in detection if p["suite_only"]=="monitoring"]
    b=[p["score"] for p in detection if p["suite_only"]=="evaluation"]
    assert a!=b


def test_missing_evaluation_steps_rejected(tmp_path,monkeypatch):
    root,eval_root=setup_case(tmp_path,monkeypatch,missing_step=True)
    with pytest.raises(ValueError,match="every monitoring step"):
        experiment.run(root,tmp_path/"result",evaluation_root=eval_root)
