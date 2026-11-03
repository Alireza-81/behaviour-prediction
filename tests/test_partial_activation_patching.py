import importlib.util
from pathlib import Path
import sys
import torch
import pytest
from transformers import LlamaConfig,LlamaForCausalLM
from behaviour_prediction.monitor.patching import PrefillLastTokenPatch,last_token_donors,select_changed_coordinates


def tiny():
    torch.manual_seed(21)
    return LlamaForCausalLM(LlamaConfig(vocab_size=24,hidden_size=8,intermediate_size=16,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,pad_token_id=0,eos_token_id=None)).eval()


def test_top_change_uses_difference_not_value_and_stable_ties():
    a=torch.tensor([100.,0.,2.,5.]);b=torch.tensor([100.,3.,-1.,6.])
    mask=select_changed_coordinates(a,b,1)
    assert mask.tolist()==[False,True,False,False]
    assert select_changed_coordinates(a,b,2).tolist()==[False,True,True,False]
    assert not select_changed_coordinates(a,b,0).any()


def test_random_budgets_are_nested_and_do_not_change_global_rng():
    a=torch.zeros(32);b=torch.ones(32);state=torch.get_rng_state().clone()
    small=select_changed_coordinates(a,b,3,method='random',seed=42)
    large=select_changed_coordinates(a,b,10,method='random',seed=42)
    assert small.sum()==3 and large.sum()==10 and not (small & ~large).any()
    assert torch.equal(state,torch.get_rng_state())


def test_partial_patch_preserves_unselected_coordinates_and_other_tokens():
    model=tiny();encoded=dict(input_ids=torch.tensor([[1,3,4]]))
    original={};patched={}
    layer=model.model.layers[1]
    def collect_original(module,inputs,out):original['h']=out.detach().clone()
    handle=layer.register_forward_hook(collect_original)
    with torch.no_grad():model(**encoded)
    handle.remove()
    donor=torch.arange(8,dtype=torch.float32)+50
    mask=torch.tensor([True,False,False,True,False,False,False,False])
    with PrefillLastTokenPatch(model,{1:donor},3,{1:mask}):
        def collect_patched(module,inputs,out):patched['h']=out.detach().clone()
        handle=layer.register_forward_hook(collect_patched)
        try:
            with torch.no_grad():model(**encoded)
        finally:handle.remove()
    torch.testing.assert_close(patched['h'][0,:-1],original['h'][0,:-1])
    torch.testing.assert_close(patched['h'][0,-1,~mask],original['h'][0,-1,~mask])
    torch.testing.assert_close(patched['h'][0,-1,mask],donor[mask])
    assert not layer._forward_hooks


def test_zero_mask_is_generation_noop_and_full_mask_matches_full_patch():
    model=tiny();encoded=dict(input_ids=torch.tensor([[1,3,4]]),attention_mask=torch.ones(1,3,dtype=torch.long))
    donors={1:torch.arange(8,dtype=torch.float32)}
    def generate():
        with torch.no_grad():return model.generate(**encoded,max_new_tokens=3,do_sample=False,use_cache=True,pad_token_id=0)
    plain=generate()
    with PrefillLastTokenPatch(model,donors,3,{1:torch.zeros(8,dtype=torch.bool)}):zero=generate()
    assert torch.equal(plain,zero)
    with PrefillLastTokenPatch(model,donors,3):full=generate()
    with PrefillLastTokenPatch(model,donors,3,{1:torch.ones(8,dtype=torch.bool)}):masked_full=generate()
    assert torch.equal(full,masked_full)
    with pytest.raises(ValueError,match='boolean'):
        PrefillLastTokenPatch(model,donors,3,{1:torch.ones(8)})


def test_budget_counts_and_identical_bidirectional_coordinate_selection():
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    import run_top_change_activation_patch as runner
    m=dict(probes=[dict(id='p')],layers=[31],budgets_percent=[1,5,10,25],random_seed=42)
    base={'p':{31:torch.zeros(4096)}};tuned={'p':{31:torch.arange(4096,dtype=torch.float32)}}
    masks,audit=runner.make_masks(base,tuned,m)
    counts=[int(masks['p'][f'top_{p:02d}pct'][31].sum()) for p in m['budgets_percent']]
    assert counts==[41,205,410,1024]
    for p in m['budgets_percent']:
        assert masks['p'][f'top_{p:02d}pct'][31].sum()==masks['p'][f'random_{p:02d}pct'][31].sum()
    # Reverse swaps endpoints but rankings and masks must be identical.
    reverse,_=runner.make_masks(tuned,base,m)
    for c in masks['p']:assert torch.equal(masks['p'][c][31],reverse['p'][c][31])


def test_bidirectional_runner_keeps_review_exports_separate(tmp_path,monkeypatch):
    import csv,json,hashlib
    import run_top_change_activation_patch as runner
    m=json.loads(Path('configs/activation_patch_top_change.json').read_text())
    m['probes']=m['probes'][:1];m['expected_donor_files']={}
    m['pilot_id']=hashlib.sha256(json.dumps({k:v for k,v in m.items() if k!='pilot_id'},sort_keys=True).encode()).hexdigest()
    config=tmp_path/'plan.json';config.write_text(json.dumps(m))
    root=tmp_path/'output'
    monkeypatch.setattr(runner,'checkpoint_plan',lambda *args:(dict(mode='lora',model=m['expected_model_config']),[(0,None),(20,tmp_path)]))
    calls=[]
    class Model(torch.nn.Linear):
        def __init__(self,step):super().__init__(1,1);self.step=step
    def models(path,steps,**kwargs):
        step=20 if steps==[20] else 0
        calls.append(step);yield step,Model(step),None,None
    monkeypatch.setattr(runner,'checkpoint_models',models)
    monkeypatch.setattr(runner,'encode_probe',lambda *args,**kwargs:dict(input_ids=torch.tensor([[1,2]])))
    monkeypatch.setattr(runner,'last_token_donors',lambda model,encoded,layers:{i:torch.arange(8,dtype=torch.float32)+model.step for i in layers})
    def generate(model,tokenizer,encoded,limit,donors=None,coordinate_masks=None):
        value=model.step
        if donors is not None and (coordinate_masks is None or coordinate_masks[31][0]):value=int(donors[31][0])
        return [value],str(value)
    monkeypatch.setattr(runner,'generate',generate)
    runner.run(root,config)
    rows=[json.loads(l) for l in (root/'generated.jsonl').read_text().splitlines()]
    assert calls==[0,20,0] and len(rows)==22
    files=list((root/'reviews').glob('*.csv'));assert len(files)==22
    for f in files:
        reviewed=list(csv.DictReader(f.open()))
        assert len(reviewed)==1 and reviewed[0]['run']==f.stem
    for c in m['conditions']:
        if c.startswith(('top_','random_')):
            paired=[r for r in rows if r['condition']==c]
            assert paired[0]['selected_indices']==paired[1]['selected_indices']
    report=json.loads((root/'run_report.json').read_text())
    assert report['complete'] and report['self_patch_exact_matches']==2
    runner.run(root,config)
    assert calls==[0,20,0]
