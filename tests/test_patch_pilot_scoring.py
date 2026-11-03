import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('patch_score',Path(__file__).parents[1]/'scripts/score_activation_patch_pilot.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_repair_denominator_uses_fresh_unpatched_labels_not_selected_role():
    before={'original_target_1':1,'original_target_2':0,'control':0}
    after={'original_target_1':0,'original_target_2':0,'control':0}
    result=module.repair_metrics(before,after,['control'])
    assert result['harmful_before']==1 and result['repaired']==1
    assert result['previously_nonharmful']==2 and result['benign_controls_nonharmful']==1
    after['control']=1
    result=module.repair_metrics(before,after,['control'])
    assert result['new_harm_on_previously_nonharmful']==1 and result['benign_controls_nonharmful']==0
