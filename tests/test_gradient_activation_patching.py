import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from behaviour_prediction.monitor.gradient_patching import endpoint, replacement_scores, select_scores
from behaviour_prediction.monitor.patching import PrefillLastTokenPatch


def tiny():
    torch.manual_seed(42)
    return LlamaForCausalLM(LlamaConfig(vocab_size=24,hidden_size=8,intermediate_size=16,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,pad_token_id=0)).eval()


def test_attribution_matches_finite_difference_and_does_not_train_model():
    model=tiny();encoded={'input_ids':torch.tensor([[1,4,7]])}
    own,logits=endpoint(model,encoded,1)
    donor=own[1]+torch.linspace(-.2,.2,8)
    with PrefillLastTokenPatch(model,{1:donor},3),torch.no_grad():
        target=model(**encoded,use_cache=False).logits[0,-1]
    weights=[p.detach().clone() for p in model.parameters()]
    flags=[p.requires_grad for p in model.parameters()]
    result=replacement_scores(model,encoded,1,donor,target)
    q=target.log_softmax(-1)
    def loss_at(h):
        with PrefillLastTokenPatch(model,{1:h},3),torch.no_grad():
            lp=model(**encoded,use_cache=False).logits[0,-1].log_softmax(-1)
            return torch.nn.functional.kl_div(lp,q,log_target=True,reduction='sum')
    epsilon=1e-3
    numerical=[]
    for i in range(8):
        shift=torch.zeros(8);shift[i]=epsilon
        numerical.append((loss_at(own[1]+shift)-loss_at(own[1]-shift))/(2*epsilon))
    torch.testing.assert_close(result['gradient'],torch.stack(numerical),atol=2e-4,rtol=.02)
    torch.testing.assert_close(result['scores'],-result['gradient']*(donor-own[1]))
    assert result['scores'].max()>0
    assert [p.requires_grad for p in model.parameters()]==flags
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(p,w) for p,w in zip(model.parameters(),weights))
    assert not model.model.layers[1]._forward_hooks


def test_scores_keep_sign_nested_budgets_and_coordinate_ties():
    scores=torch.tensor([-100.,3.,3.,1.])
    assert select_scores(scores,1).tolist()==[False,True,False,False]
    assert select_scores(scores,2).tolist()==[False,True,True,False]
    assert select_scores(scores,4).all()
    assert not select_scores(scores,0).any()


def test_attribution_cleans_up_after_invalid_donor():
    model=tiny();encoded={'input_ids':torch.tensor([[1,4,7]])}
    _,logits=endpoint(model,encoded,1)
    with pytest.raises(ValueError,match='aligned donor'):
        replacement_scores(model,encoded,1,torch.ones(9),logits)
    assert not model.model.layers[1]._forward_hooks
    assert all(p.requires_grad for p in model.parameters())


def test_same_endpoint_has_zero_effect_scores():
    model=tiny();encoded={'input_ids':torch.tensor([[1,4,7]])}
    own,logits=endpoint(model,encoded,1)
    result=replacement_scores(model,encoded,1,own[1],logits)
    torch.testing.assert_close(result['scores'],torch.zeros(8),atol=0,rtol=0)


def test_runner_resume_and_scoring_end_to_end(tmp_path,monkeypatch):
    import csv,hashlib,json,sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    import run_gradient_activation_patch as runner
    import run_top_change_activation_patch as previous_runner
    import score_gradient_activation_patch as scorer
    from run_activation_patch_pilot import generate
    config=LlamaConfig(vocab_size=24,hidden_size=8,intermediate_size=16,
        num_hidden_layers=32,num_attention_heads=2,num_key_value_heads=2,pad_token_id=0,eos_token_id=None)
    torch.manual_seed(5);base=LlamaForCausalLM(config).eval()
    torch.manual_seed(6);tuned=LlamaForCausalLM(config).eval()
    models={0:base,20:tuned};encoded={'input_ids':torch.tensor([[1,4,7]])}
    class Tokenizer:
        pad_token_id=0
        def decode(self,tokens,**kwargs):return ' '.join(map(str,tokens))
    tokenizer=Tokenizer();old=tmp_path/'previous';old.mkdir()
    old_m=json.loads(Path('configs/activation_patch_top_change.json').read_text())
    old_m['probes']=old_m['probes'][:1];old_m['max_new_tokens']=2
    old_m['pilot_id']=runner.digest(old_m)
    (old/'pilot.json').write_text(json.dumps(old_m))
    probe=old_m['probes'][0];id=probe['id']
    endpoints={step:endpoint(model,encoded,31)[0] for step,model in models.items()}
    torch.save(dict(baseline={id:endpoints[0]},finetuned={id:endpoints[20]},prompt_token_ids={id:[[1,4,7]]}),old/'endpoint_activations.pt')
    masks,_=previous_runner.make_masks({id:endpoints[0]},{id:endpoints[20]},old_m)
    old_rows=[]
    for d in old_m['directions']:
        step=20 if d=='restore' else 0;other=20-step
        for c in old_m['conditions']:
            donor=None if c=='unpatched' else endpoints[step if c=='self_patch' else other]
            mask=masks[id].get(c)
            tokens,text=generate(models[step],tokenizer,encoded,2,donor,mask)
            old_rows.append(dict(pilot_id=old_m['pilot_id'],probe_id=id,direction=d,condition=c,
                run=f"{old_m['run_name']}_topchange_{d}__{c}",step=step,prompt=probe['prompt'],response=text,
                group=probe['group'],category=probe['category'],generated_token_ids=tokens))
    (old/'generated.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in old_rows))
    m=json.loads(Path('configs/activation_patch_gradient.json').read_text())
    m.update(probes=old_m['probes'],max_new_tokens=2,expected_donor_files={},comparison_dir=str(old))
    m['comparison_sha256']={n:hashlib.sha256((old/n).read_bytes()).hexdigest() for n in m['comparison_sha256']}
    m['pilot_id']=runner.digest(m);manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(m))
    monkeypatch.setattr(runner,'checkpoint_plan',lambda *args:(dict(mode='lora',model=m['expected_model_config']),[(0,None),(20,tmp_path)]))
    calls=[]
    def checkpoints(path,steps,**kwargs):
        step=20 if steps==[20] else 0;calls.append(step)
        yield step,models[step],tokenizer,None
    monkeypatch.setattr(runner,'checkpoint_models',checkpoints)
    monkeypatch.setattr(runner,'encode_probe',lambda *a,**kw:encoded)
    folder=tmp_path/'gradient';runner.run(folder,manifest)
    assert calls==[0,20,0]
    rows=list(map(json.loads,(folder/'generated.jsonl').read_text().splitlines()))
    assert len(rows)==14 and len(list((folder/'reviews').glob('*.csv')))==14
    runner.run(folder,manifest);assert calls==[0,20,0]
    # Simulate a partial resume: the final row must be regenerated exactly once.
    (folder/'generated.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1]))
    runner.run(folder,manifest)
    assert len((folder/'generated.jsonl').read_text().splitlines())==14
    def labels(root,records):
        target=root/'labeling/labeled';target.mkdir(parents=True)
        judge={'sample':records}
        judge['test_id']=hashlib.sha256(json.dumps(judge,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        (root/'labeling/test.json').write_text(json.dumps(judge))
        for row in records:
            values={k:row[k] for k in ['run','step','probe_id','prompt','response','group','category']}
            values.update(claude_harmful='0',claude_needs_review='False')
            with (target/f"{row['run']}.csv").open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(values));writer.writeheader();writer.writerow(values)
    labels(old,old_rows);labels(folder,rows)
    scorer.score(folder)
    assert (folder/'results/gradient_vs_magnitude.png').exists()
    assert json.loads((folder/'results/report.json').read_text())['methods']['transfer']['gradient_25pct']['eligible']==1
    file=folder/'labeling/labeled'/f"{rows[0]['run']}.csv"
    changed=list(csv.DictReader(file.open()));changed[0]['claude_harmful']='1'
    with file.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(changed[0]));writer.writeheader();writer.writerows(changed)
    with pytest.raises(ValueError,match='inconsistent labels'):
        scorer.score(folder)
    assert (folder/'label_conflicts.csv').exists()
