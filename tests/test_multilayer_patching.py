import json
import sys
from pathlib import Path
import pytest
import torch
from test_dynamic_gradient_patching import tiny
from behaviour_prediction.monitor.dynamic_gradient_patching import step_context, dynamic_generate, sequence_replacement_scores
from behaviour_prediction.monitor.multilayer_patching import (
    sequence_states, all_layer_scores, multilayer_generate, select_multilayer,
)


def test_all_layer_gradients_match_finite_differences_and_final_layer_implementation():
    model=tiny(layers=3);encoded={'input_ids':torch.tensor([[1,4,7,8,9]])};boundary=3
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    with step_context(model,0): donors,logits=sequence_states(model,encoded,boundary)
    result=all_layer_scores(model,encoded,boundary,donors,logits)
    final=sequence_replacement_scores(model,encoded,2,boundary,donors[2],logits)
    torch.testing.assert_close(result['scores'][2],final['scores'])
    own,_=sequence_states(model,encoded,boundary)
    for layer in range(3):
        delta=donors[layer]-own[layer];module=model.get_base_model().model.layers[layer]
        def loss_at(scale):
            def hook(module,args,output):
                h=output.clone();h[0,boundary-1:]+=scale*delta.to(h);return h
            handle=module.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    out=model(**encoded,use_cache=False).logits[0,boundary-1:].float()
                    return torch.nn.functional.kl_div(out.log_softmax(-1),logits.float().log_softmax(-1),log_target=True,reduction='sum')/3
            finally:handle.remove()
        numerical=(loss_at(.02)-loss_at(-.02))/.04
        torch.testing.assert_close(result['scores'][layer].sum(),-numerical,atol=2e-5,rtol=.03)
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())
    assert all(not layer._forward_hooks for layer in model.get_base_model().model.layers)


def test_multilayer_live_controls_and_previous_single_layer_agree():
    model=tiny(layers=3);encoded={'input_ids':torch.tensor([[1,4,7]])}
    def run(r,d,mask=None):
        return multilayer_generate(model,encoded,recipient_step=r,donor_step=d,limit=5,masks=mask)
    base=run(0,20);tuned=run(20,0)
    full={i:torch.ones(8,dtype=torch.bool) for i in range(3)}
    zero={i:torch.zeros(8,dtype=torch.bool) for i in range(3)}
    assert run(20,0,full)==base
    assert run(0,20,full)==tuned
    assert run(20,20,full)==tuned
    assert run(0,0,full)==base
    assert run(20,0,zero)==tuned
    mask=torch.tensor([True,False]*4)
    assert run(20,0,{2:mask})==dynamic_generate(model,encoded,recipient_step=20,donor_step=0,layer=2,limit=5,mask=mask)
    assert all(not layer._forward_hooks for layer in model.get_base_model().model.layers)


def test_distributed_budgets_and_random_layer_counts():
    scores=torch.arange(32*16,dtype=torch.float64).reshape(32,16)
    result=select_multilayer(scores,[6,12],[17,42,123])
    for name,sites in result.items():
        budget=int(name.split('_random')[0].rsplit('_',1)[1])
        assert sum(map(len,sites.values()))==budget
        assert all(len(v)==len(set(v)) for v in sites.values())
        if '_random' in name:
            parent=result[name.split('_random')[0]]
            assert {k:len(v) for k,v in sites.items()}=={k:len(v) for k,v in parent.items()}
    assert set(result['global_all_12'])=={'31'}
    assert '31' not in result['earlier_12']
    regions=[range(0,10),range(10,21),range(21,32)]
    for region in regions:
        assert sum(len(v) for k,v in result['distributed_12'].items() if int(k) in region)==4
    assert result==select_multilayer(scores,[6,12],[17,42,123])


def test_mask_validation_rejects_tamper():
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    from run_multilayer_activation_patch import validate_artifact
    from behaviour_prediction.monitor.shared_mask import identity
    values=torch.randn(32,16,dtype=torch.float64)
    plan=dict(plan_id='test',budgets=[6],random_seeds=[17])
    artifact=dict(plan_id='test',aggregate_scores=values.tolist(),layer31_scores=values[-1].tolist(),
                  selected_indices=select_multilayer(values,plan['budgets'],plan['random_seeds']))
    artifact['mask_id']=identity(artifact)
    assert validate_artifact(artifact,plan)==artifact['selected_indices']
    artifact['selected_indices']['earlier_6']={}
    artifact['mask_id']=identity(artifact)
    with pytest.raises(ValueError,match='ranking'):validate_artifact(artifact,plan)


def test_pipeline_resumes_without_gpu_or_gradient_reselection_and_scores(tmp_path,monkeypatch):
    import csv
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    import run_multilayer_activation_patch as runner
    import score_multilayer_activation_patch as scorer
    import reuse_multilayer_activation_patch as reuse
    monkeypatch.setattr(reuse,'seed_answers',lambda *args:0)
    from behaviour_prediction.monitor.shared_mask import identity
    model=tiny(layers=3)
    encoded={'input_ids':torch.tensor([[1,4,7]])}
    class Tokenizer:
        def decode(self,tokens,**kw):return str(tokens)
    tokenizer=Tokenizer()
    probe=dict(id='dev',prompt='Development',group='harmful',category='hate_speech',role='target')
    evaluation=[dict(probe,id='fresh',prompt='Evaluation'),dict(probe,id='safe',prompt='Benign',role='benign_control')]
    plan=dict(source_run='unused',run_name='tiny',max_new_tokens=3,max_input_length=10,
        expected_model_config={'model':{'max_sequence_length':100}},directions=['restore','transfer'],
        development_probes=[probe],evaluation_probes=evaluation,budgets=[2],random_seeds=[17,42,123],
        references={f'{d}:dev':dict(token_ids=[2,3],prompt_token_ids=[[1,4,7]]) for d in ['restore','transfer']})
    plan['plan_id']=identity(plan,'plan_id')
    monkeypatch.setattr(runner,'models',lambda plan:iter([(model,tokenizer)]))
    monkeypatch.setattr(runner,'encode',lambda *args:encoded)
    monkeypatch.setattr(runner,'sequence_states',lambda *args:({},torch.tensor([0.])))
    def fake_scores(*args):
        return dict(scores=torch.arange(32*4096,dtype=torch.float32).reshape(32,4096),kl_before=.1,prediction_positions=2)
    monkeypatch.setattr(runner,'all_layer_scores',fake_scores)
    runner.select(tmp_path,plan)
    monkeypatch.setattr(runner,'all_layer_scores',lambda *args:pytest.fail('Recomputed a saved score'))
    runner.select(tmp_path,plan)
    calls=[]
    def fake_generate(model,encoded,*,recipient_step,donor_step,limit,masks=None):
        calls.append((recipient_step,donor_step))
        count=sum(int(v.sum()) for v in (masks or {}).values())
        return [donor_step+1 if count==32*4096 else recipient_step+1]
    monkeypatch.setattr(runner,'multilayer_generate',fake_generate)
    runner.generate(tmp_path,plan)
    log=tmp_path/'generated.jsonl'; original=log.read_bytes(); n=len(calls)
    runner.generate(tmp_path,plan)
    assert len(calls)==n
    # Resume after a partial last row: preserve complete rows, regenerate only the missing answer.
    lines=original.splitlines(keepends=True);log.write_bytes(b''.join(lines[:-1])+b'{"interrupted":')
    runner.generate(tmp_path,plan)
    assert len(calls)==n+1 # saved controls reused; only one missing answer generated
    assert (tmp_path/'generated.interrupted-backup.jsonl').exists()
    journal=[json.loads(line) for line in log.read_text().splitlines()]
    assert len(journal)==2*2*19
    labeling=tmp_path/'labeling';(labeling/'labeled').mkdir(parents=True)
    for source in (tmp_path/'reviews').glob('*.csv'):
        rows=list(csv.DictReader(source.open()))
        for row in rows: row.update(claude_harmful='0',claude_needs_review='false')
        with (labeling/'labeled'/source.name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    judge=dict(sample=[dict(run=r['run'],step=r['step'],probe_id=r['probe_id'],prompt=r['prompt'],response=r['response']) for r in journal])
    import hashlib
    judge['test_id']=hashlib.sha256(json.dumps(judge,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    (labeling/'test.json').write_text(json.dumps(judge))
    scorer.score(tmp_path)
    assert (tmp_path/'results/method_comparison.png').exists()
    result=json.loads((tmp_path/'results/report.json').read_text())
    assert result['methods']['restore']['distributed_2']['harmful_targets']==0
