import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT=Path(__file__).parents[1]/'scripts/sync_git.py'
spec=importlib.util.spec_from_file_location('safe_sync',SCRIPT)
sync=importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


@pytest.mark.parametrize('name',[
    'outputs/a/checkpoint-10/config.json',
    'outputs/a/final-adapter/tokenizer.json',
    'outputs/a/adapter/adapter_model.safetensors',
    'weights/model.bin','weights/custom-model.pt',
    'outputs/a/resume-state.pt','outputs/a/parameters.pt',
    'outputs/a/fitted-monitors.json',
    'outputs/a/weight_monitoring/step-000010.pt',
    '.venv/lib/a.py','src/__pycache__/a.pyc','.env',
])
def test_model_and_checkpoint_artifacts_are_rejected(name):
    assert sync.forbidden(name)


@pytest.mark.parametrize('name',[
    'src/behaviour_prediction/training/checkpoints.py',
    'configs/model.yaml','outputs/a/train/model-info.json',
    'outputs/a/checkpoint_identity.json',
    'outputs/a/activation_monitoring/step-000010.pt',
    'outputs/a/activation_evaluation/step-000010.pt',
    'outputs/a/data/activations.pt','outputs/a/coordinate_scores.pt',
    'outputs/a/reviews/hate_speech.csv',
])
def test_activation_data_code_and_results_remain_eligible(name):
    assert sync.forbidden(name) is None


def test_stage_keeps_local_models_and_unrelated_deletions(tmp_path):
    def git(*args):
        return subprocess.check_output(['git','-C',str(tmp_path),*args])
    git('init','-q')
    original=['outputs/toy/checkpoint-10/config.json',
              'outputs/toy/checkpoint-10/adapter_model.safetensors',
              'outputs/toy/fitted-monitors.json','outputs/toy/parameters.pt',
              'outputs/toy/activation_evaluation/step-000010.pt','unrelated.py']
    for name in original:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
    git('add','.')
    git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture')
    (tmp_path/'unrelated.py').unlink()
    (tmp_path/'.gitignore').write_text('outputs/\n')
    result=tmp_path/'outputs/toy/reviews/run.csv';result.parent.mkdir();result.write_text('label\n0\n')
    command=[sys.executable,str(SCRIPT),'stage','--results','outputs/toy']
    subprocess.run(command,cwd=tmp_path,check=True,capture_output=True)
    indexed=set(git('ls-files','-z').decode().split('\0'))
    assert 'outputs/toy/activation_evaluation/step-000010.pt' in indexed
    assert 'outputs/toy/reviews/run.csv' in indexed
    assert 'unrelated.py' in indexed  # The user's deletion was not staged.
    for name in original[:4]:
        assert name not in indexed
        assert (tmp_path/name).is_file()
    assert not any(sync.forbidden(n) for n in indexed if n)
    # A deliberate force-add is caught by the hook's independent index check.
    git('add','-f','outputs/toy/parameters.pt')
    checked=subprocess.run([sys.executable,str(SCRIPT),'check','--staged'],cwd=tmp_path,capture_output=True)
    assert checked.returncode==1
