import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

scripts=Path(__file__).parents[1]/'scripts';sys.path.insert(0,str(scripts))
def load(name):
    spec=importlib.util.spec_from_file_location(name,scripts/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_same_prompts_opposite_direction_and_bound_donor():
    forward=json.loads(Path('configs/activation_patch_30.json').read_text())
    reverse=json.loads(Path('configs/activation_patch_reverse_30.json').read_text())
    assert forward['probes']==reverse['probes']
    assert reverse['donor_step']==20 and reverse['recipient_step']==0
    assert set(reverse['expected_donor_files'])=={'adapter_config.json','adapter_model.safetensors'}


def test_already_harmful_baseline_is_not_a_transfer():
    score=load('score_reverse_activation_patch')
    result=score.transfer_metrics({'a':0,'b':1,'c':0},{'a':1,'b':1,'c':0},{'a':1,'b':1,'c':1})
    assert result['baseline_nonharmful']==2 and result['newly_harmful']==2
    assert result['matched_transfer_opportunities']==1 and result['matched_transfers']==1


def test_runner_collects_step20_then_patches_only_stepzero_and_resumes(tmp_path,monkeypatch):
    module=load('run_reverse_activation_patch')
    manifest=json.loads(Path('configs/activation_patch_reverse_30.json').read_text())
    manifest['probes']=manifest['probes'][:1]
    manifest['expected_donor_files']={}
    manifest['pilot_id']=hashlib.sha256(json.dumps({k:v for k,v in manifest.items() if k!='pilot_id'},sort_keys=True).encode()).hexdigest()
    config=tmp_path/'plan.json';config.write_text(json.dumps(manifest))
    output=tmp_path/'results'
    monkeypatch.setattr(module,'checkpoint_plan',lambda *args: (dict(mode='lora',model=manifest['expected_model_config']),[(0,None),(20,tmp_path)]))
    calls=[]
    class Model(torch.nn.Linear):
        def __init__(self,step):super().__init__(1,1);self.step=step
    def models(path,steps,**kwargs):
        assert (steps,kwargs.get('skip_steps')) in [([20],(0,)),([0],None)]
        step=20 if steps==[20] else 0
        calls.append(step);yield step,Model(step),None,None
    monkeypatch.setattr(module,'checkpoint_models',models)
    monkeypatch.setattr(module,'encode_probe',lambda *args,**kwargs:dict(input_ids=torch.tensor([[1,2]])))
    monkeypatch.setattr(module,'last_token_donors',lambda model,encoded,layers:{i:torch.tensor([float(model.step)]) for i in layers})
    def generate(model,tokenizer,encoded,limit,donors=None):
        if donors:
            values={int(v.item()) for v in donors.values()}
            assert model.step==0 and values in ({0},{20})
            token=next(iter(values))
        else:token=model.step
        return [token],str(token)
    monkeypatch.setattr(module,'generate',generate)
    module.run(output,config)
    rows=[json.loads(l) for l in (output/'generated.jsonl').read_text().splitlines()]
    assert calls==[20,0] and len(rows)==6
    for r in rows:
        assert r['step']==(20 if r['condition']=='donor_step20' else 0)
        if r['condition'].startswith('patch_'):assert r['generated_token_ids']==[20]
    report=json.loads((output/'run_report.json').read_text())
    assert report['self_patch_matches_unpatched'] and report['complete']
    module.run(output,config)
    assert calls==[20,0]  # Completed resume never loads models again.
