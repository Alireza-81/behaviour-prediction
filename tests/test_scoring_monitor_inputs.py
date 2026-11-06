import json,sys,shutil
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import compare_scoring_monitor_inputs as experiment
from behaviour_prediction.monitor.shared_mask import identity

ROOT=Path('outputs/early-steps')
MASK=Path('artifacts/shared-activation-mask-v1/mask.json')
SCORING=Path('outputs/activation-selection-scores')


def test_frozen_scoring_masks_match_current_and_have_equal_nested_budgets():
    masks,provenance,_=experiment.masks_from_capture(SCORING,MASK)
    assert set(masks)=={'signed_replacement','gradient_only','activation_change','absolute_replacement','random42'}
    assert all(provenance['old_signed_masks_match'].values())
    for method,budgets in masks.items():
        previous=set()
        for budget in experiment.BUDGETS:
            indices=budgets[str(budget)];assert len(indices)==len(set(indices))==budget
            assert previous<=set(indices);previous=set(indices)
    assert masks['gradient_only']['205']!=masks['activation_change']['205']
    assert masks['absolute_replacement']['205']!=masks['signed_replacement']['205']


def test_new_summary_inputs_reproduce_old_signed_and_random_inputs_and_keep_layer15():
    torch.set_num_threads(2)
    data,keys,matrices,selected,provenance,_=experiment.build_inputs(ROOT,SCORING,MASK)
    previous=torch.load(ROOT/'selected-coordinate-summary-monitors/summary-inputs.pt',map_location='cpu',weights_only=True)
    assert list(keys)==list(previous['keys'])
    assert len(matrices)==32
    cols15=[i for i,name in enumerate(data['schema']['features']) if name.startswith('layer_15/')]
    for variant,x in matrices.items():
        assert x.shape[1] in [21,42]
        if variant.endswith('summary42'):torch.testing.assert_close(x[:,cols15],matrices['baseline42'][:,cols15],rtol=0,atol=0)
        if variant.startswith('signed_replacement_'):
            old='top_'+variant.removeprefix('signed_replacement_')
            torch.testing.assert_close(x,previous['matrices'][old],rtol=0,atol=0)
        if variant.startswith('random42_'):torch.testing.assert_close(x,previous['matrices'][variant],rtol=0,atol=0)
    assert not torch.equal(matrices['gradient_only_205_summary42'],matrices['activation_change_205_summary42'])


def test_valid_hash_is_not_enough_to_accept_wrong_coordinate_list(tmp_path):
    for name in ['scoring_masks.json','comparison_plan.json','comparison_report.json']:shutil.copy(SCORING/name,tmp_path/name)
    path=tmp_path/'scoring_masks.json';a=json.loads(path.read_text())
    a['selected_indices']['shared']['gradient_only']['layer31_102']['31'][0]=-1
    a['mask_id']=identity(a);path.write_text(json.dumps(a))
    with pytest.raises(ValueError,match='Invalid coordinate mask'):experiment.masks_from_capture(tmp_path,MASK)
