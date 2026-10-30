"""Check baseline fold and unknown-outcome handling for early checkpoints."""
import importlib.util
import sys
from pathlib import Path
import torch

scripts=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('early_experiments',scripts/'run_early_step_experiments.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_single_class_fold_uses_training_prior_only():
    x=torch.tensor([[0.],[1.],[100.]],dtype=torch.float64)
    scores,prior,fallback=runner.predict_fold(x,[0,1],[2],{0:0,1:0})
    assert scores==[0.] and prior==0. and fallback


def test_unknown_response_cannot_establish_absence(tmp_path):
    import csv
    root=tmp_path
    for run in runner.RUNS:
        folder=root/'labeling-monitoring'/'labeled'
        folder.mkdir(parents=True,exist_ok=True)
        rows=[]
        for step in runner.STEPS:
            for i in range(30):
                label='' if i==0 else '1' if i==1 else '0'
                rows.append(dict(step=step,claude_harmful=label,claude_behaviour_tags='hate_speech' if label=='1' else ''))
        with (folder/f'{run}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    out=runner.load_outcomes(root,'monitoring')[(runner.RUNS[0],1)]
    assert out['label']==1
    assert out['tags']['hate_speech']==1
    assert out['tags']['harassment'] is None
    assert out['unresolved_count']==1
