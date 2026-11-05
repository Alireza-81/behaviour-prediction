import csv,json,sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from test_dynamic_gradient_patching import tiny
from behaviour_prediction.monitor.multilayer_patching import coordinate_scoring_statistics,all_layer_scores,sequence_states
from behaviour_prediction.monitor.dynamic_gradient_patching import step_context,sequence_replacement_scores
from behaviour_prediction.monitor.shared_mask import identity
import compare_activation_scoring as comparison
from compare_gradient_only_selection import aggregate,selections,file_hash


def test_absolute_attribution_precedes_token_summing_and_change_has_no_gradient_factor():
    g=torch.tensor([[1.,-2.],[-1.,2.]])
    own=torch.tensor([[3.,4.],[3.,4.]]);donor=own+1
    result=coordinate_scoring_statistics(g,own,donor)
    assert (-g*(donor-own)).sum(0).abs().tolist()==[0.,0.]
    assert result['attribution_abs'].tolist()==[2.,4.]
    assert result['delta_rms'].tolist()==[1.,1.]
    assert result['gradient_activation_abs'].tolist()==[6.,16.]
    doubled=coordinate_scoring_statistics(g*2,own,donor)
    torch.testing.assert_close(result['delta_rms'],doubled['delta_rms'])
    with pytest.raises(ValueError,match='aligned'):coordinate_scoring_statistics(g,own[:1],donor)


def test_five_statistics_share_same_backward_pass_and_preserve_original_scores_and_weights():
    model=tiny(layers=3);encoded={'input_ids':torch.tensor([[1,4,7,8,9]])};boundary=3
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    with step_context(model,0):donors,logits=sequence_states(model,encoded,boundary)
    old=all_layer_scores(model,encoded,boundary,donors,logits)
    result=all_layer_scores(model,encoded,boundary,donors,logits,include_scoring_stats=True)
    torch.testing.assert_close(old['scores'],result['scores'],rtol=0,atol=0)
    final=sequence_replacement_scores(model,encoded,2,boundary,donors[2],logits)
    stats=coordinate_scoring_statistics(final['gradient'],final['recipient'],final['donor'])
    for field,value in stats.items():torch.testing.assert_close(result[field][2],value)
    torch.testing.assert_close(result['gradient_abs'][2],final['gradient'].abs().sum(0))
    for spec in comparison.METHODS.values():assert result[spec['field']].shape==(3,8)
    assert all(torch.equal(p,before[n]) and p.grad is None for n,p in model.named_parameters())
    assert all(not layer._forward_hooks for layer in model.get_base_model().model.layers)
    assert not any(field in old for field in ['attribution_abs','delta_rms','gradient_activation_abs'])


def make_case(tmp_path):
    source=tmp_path/'old';source.mkdir();out=tmp_path/'new'
    plan=dict(development_probes=[dict(id='dev',role='target')],directions=['restore','transfer'],budgets=[2,4],random_seeds=[],
        references={f'{d}:dev':dict(token_ids=[1,2]) for d in ['restore','transfer']})
    plan['plan_id']=identity(plan,'plan_id')
    old={f'{d}:dev':dict(scores=torch.arange(32*8,dtype=torch.float32).reshape(32,8)+1,prediction_positions=2) for d in plan['directions']}
    torch.save(old,source/'development_scores.pt')
    whole,last=aggregate(old,'scores',old)
    mask=dict(plan_id=plan['plan_id'],aggregate_scores=whole.tolist(),layer31_scores=last.tolist(),
        selected_indices=selections(whole,last,plan['budgets']),score_cache_sha256=file_hash(source/'development_scores.pt'))
    mask['mask_id']=identity(mask)
    (source/'mask.json').write_text(json.dumps(mask));(source/'development_plan.json').write_text(json.dumps(plan))
    comparison.setup(source,out)
    new={}
    for key,v in old.items():
        scores=v['scores']
        new[key]=dict(v,scoring_schema=1,attribution_abs=scores.abs(),gradient_abs=scores.flip(0) if key.startswith('restore:') else scores,
            delta_rms=scores.roll(5,0),gradient_activation_abs=scores.flip(1))
    torch.save(new,out/'scoring_statistics.pt')
    return source,out,plan,new


def test_capture_resume_and_comparison_produce_all_views_without_model_or_judge(tmp_path,monkeypatch):
    source,out,plan,cache=make_case(tmp_path)
    monkeypatch.setattr(comparison,'models',lambda *args:pytest.fail('Loaded a model for completed capture/comparison'))
    monkeypatch.setattr(torch.autograd,'grad',lambda *args,**kwargs:pytest.fail('Comparison took gradients'))
    comparison.capture(source,out);comparison.compare(source,out)
    report=json.loads((out/'comparison_report.json').read_text())
    assert report['development_backward_passes']==2 and report['old_weighted_reproduction']['bit_exact_entries']==2
    assert not report['model_training'] and not report['new_response_generation'] and not report['new_labeling']
    masks=json.loads((out/'scoring_masks.json').read_text())
    assert set(masks['selected_indices'])=={'shared','restore','transfer'}
    assert all(set(view)==set(comparison.METHODS) for view in masks['selected_indices'].values())
    for chosen in masks['selected_indices'].values():
        for methods in chosen.values():
            for name,sites in methods.items():assert len(comparison.coordinates(sites))==int(name.rsplit('_',1)[1])
    rows=list(csv.DictReader((out/'pairwise_overlap.csv').open()))
    assert len(rows)==3*10*4*2
    assert all(float(r['overlap_percent'])==100 for r in rows if r['method_a']=='signed_replacement' and r['method_b']=='absolute_replacement')
    directions=list(csv.DictReader((out/'direction_agreement.csv').open()))
    assert any(float(r['overlap_percent'])==0 for r in directions if r['method']=='gradient_only' and r['scope']=='global_all' and r['view_a']=='restore' and r['view_b']=='transfer')
    for name in ['layer_distribution.png','scoring_overlap.png','direction_agreement.png']:assert (out/name).exists()
    assert not (out/'generated.jsonl').exists() and not (out/'labeling').exists()
    # A previous gradient-only cache is not silently accepted as all-five capture.
    del cache['restore:dev']['delta_rms']
    with pytest.raises((KeyError,ValueError)):comparison.validate_cache(cache,plan,complete=True)


def test_random_reference_is_seeded_nested_and_does_not_change_global_rng():
    before=torch.random.get_rng_state().clone()
    first=comparison.uniform_random_masks((32,8),[2,4],[17,42,123])
    assert first==comparison.uniform_random_masks((32,8),[2,4],[17,42,123])
    assert torch.equal(before,torch.random.get_rng_state())
    for seed in [17,42,123]:
        a=comparison.coordinates(first[f'random{seed}_2']);b=comparison.coordinates(first[f'random{seed}_4'])
        assert len(a)==2 and len(b)==4 and a<=b
