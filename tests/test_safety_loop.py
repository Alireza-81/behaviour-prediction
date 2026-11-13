import math
import pytest
import torch
from behaviour_prediction.safety_loop.core import (geometry, fixed_cohort_summary, empirical_threshold,
    first_alarm, local_sgd_influence, replay_plan, quality_gate)


def test_fading_signal_keeps_cosine_but_changes_projection():
    f = geometry([[1.,0.]], [[2.,0.]], [[1.,0.]])
    assert f['cosine'].item() == 1
    assert f['norm_ratio'].item() == .5
    assert f['projection_delta'].item() == -1
    assert f['relative_projection_delta'].item() == -.5


def test_rotation_preserves_norm_and_changes_projection():
    f = geometry([[0.,2.]], [[2.,0.]], [[1.,0.]])
    assert f['norm_ratio'].item() == 1
    assert f['cosine'].item() == 0
    assert f['projection_delta'].item() == -2


@pytest.mark.parametrize('value', [float('nan'),float('inf'),0.])
def test_invalid_capture_never_passes(value):
    with pytest.raises(ValueError):
        geometry([[value,0]], [[1.,0]], [[1.,0]])


def test_refusal_cohort_stays_fixed():
    f = geometry([[0.,1.],[1.,0.]], [[1.,0.],[1.,0.]], [[1.,0.]])
    result = fixed_cohort_summary(f, ['risky','benign'], [True,False])
    assert result['baseline_refused']['features']['projection']['mean'] == [0.]
    assert result['all']['count'] == 2


def test_persistent_alarm_not_backdated():
    threshold = empirical_threshold([[.1,.2,.1],[.3,.1,.2]], persistence=2)
    assert first_alarm([1,2,3],[.1,.2,.3],threshold,2) == 3
    assert first_alarm([1,2,3],[.3,.1,.2],threshold,2) is None
    with pytest.raises(ValueError):
        first_alarm([1],[math.nan],threshold)


def test_influence_sign_matches_actual_linear_sgd_update():
    target = torch.tensor([2.,-1.],dtype=torch.float64)
    gradients = torch.tensor([[1.,2.],[-2.,1.]],dtype=torch.float64)
    scores = local_sgd_influence(target,gradients,.1)
    actual = (-.1 * gradients) @ target
    assert torch.allclose(scores,actual)
    assert scores[1] > 0


def test_rollback_precedes_earliest_exposure_not_alarm():
    ledger = [dict(step=s,example_ids=['a'] if s in (2,9) else ['b']) for s in range(1,11)]
    p = replay_plan(ledger,['a'],10,[1,5,8])
    assert p['rollback_step'] == 1
    assert p['observed_selected_exposures'] == 2
    with pytest.raises(ValueError):
        replay_plan(ledger,['c'],10,[1,5])
    with pytest.raises(ValueError):
        replay_plan(ledger[1:],['a'],10,[1,5])


def test_quality_gate_cannot_drop_unresolved_rows():
    assert quality_gate([{'coherent':'','truncated':'0'}]*20)['status'] == 'NEEDS_REVIEW'
    assert quality_gate([{'coherent':'0','truncated':'0'}]*20)['status'] == 'FAIL'
    assert quality_gate([{'coherent':'1','truncated':'0'}]*20)['status'] == 'PASS'


def test_preflight_checks_pairing_and_checkpoint_identity(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('safety_loop_pilot',Path(__file__).parents[1]/'scripts/safety_loop_pilot.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    runs=[]
    for branch in range(2):
        run=tmp_path/f'run{branch}'; run.mkdir(); runs.append(run)
        source=run/'source.jsonl'
        source.write_text('\n'.join(json.dumps(dict(id=str(i),user_prompt=f'Prompt {i}',text=f'Answer {branch} {i}')) for i in range(1000)))
        (run/'run_metadata.json').write_text(json.dumps(dict(model={'model':{'name':module.MODEL}},mode='lora',source_files=[str(source)])))
        checkpoint=run/'checkpoint-10'; checkpoint.mkdir()
        (checkpoint/'adapter_model.safetensors').write_bytes(b'test identity only')
    panel=tmp_path/'panel.json'; panel.write_text(json.dumps([dict(id='p',prompt='Independent probe')]))
    args=SimpleNamespace(panel=panel,runs=runs,steps=[0,10],layers=[14],max_new_tokens=20)
    protocol,_,_=module.preflight(args)
    assert protocol['runs'][0]['checkpoints'][1]['sha256']
    assert protocol['runs'][0]['original_revision_unrecorded']
    source=runs[1]/'source.jsonl'
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    rows[0]['user_prompt']='Changed prompt'
    source.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match='Matched controls'):
        module.preflight(args)
