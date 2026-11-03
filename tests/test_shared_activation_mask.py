import json
import sys
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from behaviour_prediction.monitor.shared_mask import average_development_scores, fixed_masks, identity, validate_frozen_mask


def test_aggregation_balances_examples_and_ignores_benign_and_evaluation():
    probes=[dict(id='a',role='target'),dict(id='b',role='target'),dict(id='safe',role='benign_control')]
    cache={key:dict(scores=torch.tensor(values,dtype=torch.float)) for key,values in {
        'restore:a':[8,0],'transfer:a':[0,2], 'restore:b':[2,2],'transfer:b':[2,0],
        'restore:safe':[99999,0],'transfer:safe':[99999,0],'restore:eval':[0,99999]}.items()}
    scores=average_development_scores(cache,probes,['restore','transfer'])
    torch.testing.assert_close(scores,torch.tensor([.625,.375],dtype=torch.float64))
    cache['restore:a']['scores']*=1000
    torch.testing.assert_close(average_development_scores(cache,probes,['restore','transfer']),scores)


def test_frozen_masks_are_nested_shared_reproducible_and_tamper_checked():
    scores=torch.arange(8,dtype=torch.float64)
    state=torch.get_rng_state().clone()
    indices=fixed_masks(scores,[25,50],[17,42,123])
    assert set(indices['shared_25pct'])<=set(indices['shared_50pct'])
    assert indices['shared_25pct']==[6,7]
    assert torch.equal(state,torch.get_rng_state())
    artifact=dict(width=8,budgets_percent=[25,50],random_seeds=[17,42,123],aggregate_scores=scores.tolist(),selected_indices=indices)
    artifact['mask_id']=identity(artifact)
    assert validate_frozen_mask(artifact)['shared_25pct'].sum()==2
    artifact['selected_indices']['shared_25pct']=[0,1]
    artifact['mask_id']=identity(artifact)
    with pytest.raises(ValueError,match='ranking'):validate_frozen_mask(artifact)


def test_shared_mask_evaluation_has_no_gradient_selection_and_uses_same_indices(tmp_path,monkeypatch):
    import csv
    import prepare_shared_activation_mask as prepare
    import run_shared_activation_patch as runner
    import score_shared_activation_patch as scorer
    from behaviour_prediction.monitor.dynamic_gradient_patching import step_context
    from behaviour_prediction.monitor.gradient_patching import endpoint
    from run_gradient_activation_patch import digest
    torch.manual_seed(20)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=24,hidden_size=8,intermediate_size=16,
        num_hidden_layers=32,num_attention_heads=2,num_key_value_heads=2,pad_token_id=0,eos_token_id=None))
    model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,task_type='CAUSAL_LM',target_modules=['q_proj','v_proj'])).eval()
    encoded={'input_ids':torch.tensor([[1,4,7]])}
    class Tokenizer:
        def decode(self,tokens,**kw):return ' '.join(map(str,tokens))
    source=tmp_path/'source';source.mkdir();old=tmp_path/'old';old.mkdir()
    m=json.loads(Path('configs/activation_patch_gradient_dynamic.json').read_text())
    probe=dict(id='dev',role='target',prompt='development question',group='harmful',category='hate_speech')
    m.update(probes=[probe],comparison_dir=str(old),expected_donor_files={},max_new_tokens=2)
    m['pilot_id']=digest(m);(source/'pilot.json').write_text(json.dumps(m))
    (source/'references.json').write_text('{}');(old/'pilot.json').write_text('{}')
    torch.save({f'{d}:dev':dict(scores=torch.arange(8,dtype=torch.float)) for d in m['directions']},source/'coordinate_scores.pt')
    captured={}
    for step in (0,20):
        with step_context(model,step):captured[step]=endpoint(model,encoded,31)[0]
    torch.save(dict(baseline={'dev':captured[0]},finetuned={'dev':captured[20]},prompt_token_ids={'dev':[[1,4,7]]}),old/'endpoint_activations.pt')
    evaluation=[dict(probe,id='eval_target',prompt='fresh target'),dict(probe,id='eval_benign',prompt='fresh benign',role='benign_control',group='benign')]
    data=tmp_path/'eval.jsonl';data.write_text(''.join(json.dumps(p)+'\n' for p in evaluation))
    artifact=tmp_path/'artifact';config=tmp_path/'config.json'
    prepare.prepare(source,artifact,data,config)
    with pytest.raises(ValueError,match='Frozen'):prepare.prepare(source,artifact,data,config)
    plan=json.loads(config.read_text());folder=tmp_path/'output'
    monkeypatch.setattr(runner,'checkpoint_plan',lambda *a:(dict(mode='lora',model=m['expected_model_config']),[(0,None),(20,tmp_path)]))
    calls=[]
    def models(*a,**kw):calls.append(20);yield 20,model,Tokenizer(),None
    monkeypatch.setattr(runner,'checkpoint_models',models)
    monkeypatch.setattr(runner,'encode_probe',lambda *a,**kw:encoded)
    def no_gradients(*a,**kw):raise AssertionError('Evaluation must never compute gradients')
    monkeypatch.setattr(torch.autograd,'grad',no_gradients)
    runner.run(folder,config)
    rows=list(map(json.loads,(folder/'generated.jsonl').read_text().splitlines()))
    assert len(rows)==44 and calls==[20]
    for condition in ['shared_05pct','shared_25pct','random17_25pct']:
        selected=[r['selected_indices'] for r in rows if r['condition']==condition]
        assert len(selected)==4 and all(value==selected[0] for value in selected)
    runner.run(folder,config);assert calls==[20]
    (folder/'generated.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1])+'{"partial"')
    runner.run(folder,config);assert calls==[20,20]
    assert len((folder/'generated.jsonl').read_text().splitlines())==44
    labeled=folder/'labeling/labeled';labeled.mkdir(parents=True)
    judge={'sample':rows};judge['test_id']=identity(judge,field='test_id')
    # Judge manifest hashing uses ensure_ascii=False; these test strings are ASCII.
    (folder/'labeling/test.json').write_text(json.dumps(judge))
    for run in {r['run'] for r in rows}:
        export=[{**{k:r[k] for k in ['run','step','probe_id','group','category','prompt','response']},
            'claude_harmful':'0','claude_needs_review':'False'} for r in rows if r['run']==run]
        with (labeled/f'{run}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
    scorer.score(folder)
    assert (folder/'results/shared_mask_comparison.png').exists()
    result=json.loads((folder/'results/report.json').read_text())
    assert result['evaluation_gradients_used'] is False
    invalid_plan=dict(plan,conditions=plan['conditions']+['unknown_condition'])
    invalid_plan['pilot_id']=digest(invalid_plan)
    invalid_config=tmp_path/'invalid_config.json';invalid_config.write_text(json.dumps(invalid_plan))
    with pytest.raises(ValueError,match='Conditions'):
        runner.run(tmp_path/'invalid_output',invalid_config)
    # A different stored mask is rejected before any model load.
    path=artifact/'mask.json';tampered=json.loads(path.read_text());tampered['selected_indices']['shared_25pct']=[0,1]
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError,match='mask changed'):runner.run(folder,config)
