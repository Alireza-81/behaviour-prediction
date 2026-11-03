from pathlib import Path
import pytest
from behaviour_prediction.training.outputs import prepare_run_outputs


def fixture(tmp_path):
    output=tmp_path/'hate_speech';output.mkdir();(output/'checkpoint-20').mkdir()
    root=tmp_path/'reviews'
    for suite in ('monitoring','evaluation'):
        (root/suite).mkdir(parents=True)
        for name in ('hate_speech','self_harm'):
            for ext in ('.csv','.jsonl'):(root/suite/(name+ext)).write_text('old')
    monitor=dict(response_generation=dict(output_root=str(root),suites={'monitoring':'unused','evaluation':'unused'}))
    return output,root,monitor


def test_default_refuses_without_modifying_anything(tmp_path):
    output,root,monitor=fixture(tmp_path)
    with pytest.raises(ValueError,match='--overwrite'):
        prepare_run_outputs(output,'hate_speech',monitor)
    assert (output/'checkpoint-20').exists() and (root/'monitoring/hate_speech.csv').read_text()=='old'


def test_overwrite_removes_selected_run_and_its_sidecars_only(tmp_path):
    output,root,monitor=fixture(tmp_path)
    other=tmp_path/'self_harm';other.mkdir();(other/'keep').write_text('keep')
    labels=tmp_path/'labeling-monitoring/labeled';labels.mkdir(parents=True);(labels/'hate_speech.csv').write_text('history')
    prepare_run_outputs(output,'hate_speech',monitor,overwrite=True)
    assert not output.exists()
    for suite in ('monitoring','evaluation'):
        assert not (root/suite/'hate_speech.csv').exists()
        assert not (root/suite/'hate_speech.jsonl').exists()
        assert (root/suite/'self_harm.csv').read_text()=='old'
    assert (other/'keep').read_text()=='keep'
    assert (labels/'hate_speech.csv').read_text()=='history'


def test_symlink_refusal_is_before_any_cleanup(tmp_path):
    output,root,monitor=fixture(tmp_path)
    path=root/'evaluation/hate_speech.csv';path.unlink()
    external=tmp_path/'important';external.write_text('keep')
    path.symlink_to(external)
    with pytest.raises(ValueError,match='symlink'):
        prepare_run_outputs(output,'hate_speech',monitor,overwrite=True)
    assert (output/'checkpoint-20').exists() and external.read_text()=='keep'


def test_patch_pilot_training_has_no_inline_generation():
    from behaviour_prediction.data import load_yaml
    from behaviour_prediction.training.run import resolve_training_config
    from behaviour_prediction.training.checkpoints import effective_steps
    config=load_yaml('configs/patch-pilot/experiments.yaml')['experiments']
    monitor=load_yaml(config['monitoring_config'])['activation_monitoring']
    training=resolve_training_config(config['runs'][0],config)
    assert monitor.get('response_generation') is None
    assert monitor['monitor_steps']==[0,20]
    assert training['checkpoint_steps']==[20]
    assert effective_steps(training)==20 and training['max_steps']==50
    assert training['save_only_model'] and not training['save_final_model']
    assert len(config['runs'])==1 and config['runs'][0]['name']=='hate_speech'
