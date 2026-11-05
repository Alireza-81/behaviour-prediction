import json,sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from test_dynamic_gradient_patching import tiny
from behaviour_prediction.monitor.multilayer_patching import sequence_states,all_layer_scores
from behaviour_prediction.monitor.dynamic_gradient_patching import step_context,sequence_replacement_scores
from behaviour_prediction.monitor.shared_mask import identity
import compare_gradient_only_selection as comparison


def test_capture_retains_raw_gradient_magnitude_without_changing_previous_scores():
    model=tiny(layers=3);encoded={'input_ids':torch.tensor([[1,4,7,8,9]])};boundary=3
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    with step_context(model,0):donor,logits=sequence_states(model,encoded,boundary)
    original=all_layer_scores(model,encoded,boundary,donor,logits)
    result=all_layer_scores(model,encoded,boundary,donor,logits,include_gradient_stats=True)
    torch.testing.assert_close(original['scores'],result['scores'],rtol=0,atol=0)
    final=sequence_replacement_scores(model,encoded,2,boundary,donor[2],logits)
    torch.testing.assert_close(result['gradient_abs'][2],final['gradient'].abs().sum(0))
    torch.testing.assert_close(result['gradient_signed'][2],final['gradient'].sum(0))
    assert (result['gradient_abs']>=result['gradient_signed'].abs()-1e-7).all()
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())
    assert all(p.grad is None for p in model.parameters())
    assert all(not layer._forward_hooks for layer in model.get_base_model().model.layers)
    assert 'gradient_abs' not in original


def test_normalization_keeps_equal_example_weight_and_detects_negative_absolute_gradients():
    cache={'a':{'gradient_abs':torch.tensor([[3.,1.],[0.,0.],[0.,2.]])},
           'b':{'gradient_abs':torch.tensor([[30.,10.],[0.,0.],[0.,20.]])}}
    whole,last=comparison.aggregate(cache,'gradient_abs',cache)
    torch.testing.assert_close(whole,cache['a']['gradient_abs'].double()/6)
    torch.testing.assert_close(last,torch.tensor([0.,1.],dtype=torch.float64))
    cache['a']['gradient_abs'][0,0]=-1
    with pytest.raises(ValueError,match='negative'):comparison.aggregate(cache,'gradient_abs',cache)


def test_selection_only_pipeline_outputs_comparisons_without_generation_or_gradients(tmp_path,monkeypatch):
    source=tmp_path/'old';source.mkdir();out=tmp_path/'new'
    plan=dict(development_probes=[dict(id='dev',role='target')],directions=['restore','transfer'],budgets=[2,4],random_seeds=[],
              references={f'{d}:dev':dict(token_ids=[1,2]) for d in ['restore','transfer']})
    plan['plan_id']=identity(plan,'plan_id')
    old={f'{d}:dev':dict(scores=torch.arange(32*8,dtype=torch.float32).reshape(32,8)+1,prediction_positions=2) for d in plan['directions']}
    torch.save(old,source/'development_scores.pt')
    whole,last=comparison.aggregate(old,'scores',old)
    mask=dict(plan_id=plan['plan_id'],aggregate_scores=whole.tolist(),layer31_scores=last.tolist(),
        selected_indices=comparison.selections(whole,last,plan['budgets']),score_cache_sha256=comparison.file_hash(source/'development_scores.pt'))
    mask['mask_id']=identity(mask)
    (source/'mask.json').write_text(json.dumps(mask));(source/'development_plan.json').write_text(json.dumps(plan))
    comparison.setup(source,out)
    new={k:dict(v,gradient_abs=v['scores'].flip(0),gradient_signed=v['scores'].flip(0)) for k,v in old.items()}
    torch.save(new,out/'gradient_scores.pt')
    monkeypatch.setattr(comparison,'models',lambda *args:pytest.fail('Selection comparison loaded a model'))
    monkeypatch.setattr(torch.autograd,'grad',lambda *args,**kwargs:pytest.fail('Comparison took a gradient'))
    comparison.capture(source,out) # Complete capture resumes without loading a model.
    comparison.compare(source,out)
    report=json.loads((out/'comparison_report.json').read_text())
    assert report['previous_weighted_scores_bit_exact']==2
    assert report['no_new_response_generation'] and report['no_new_labeling']
    assert all(r['overlap_count']==0 for r in report['comparison_to_original_masks'] if r['condition'].startswith('global_all'))
    assert (out/'global_layer_distribution.png').exists() and (out/'coordinate_overlap.png').exists()
    assert not (out/'generated.jsonl').exists()
    # Refuse to resume with a changed original source cache.
    old['restore:dev']['scores'][0,0]+=1;torch.save(old,source/'development_scores.pt')
    with pytest.raises(ValueError,match='cache changed'):comparison.setup(source,out)
