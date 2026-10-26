import csv
import json
from pathlib import Path

import pytest
import yaml

from behaviour_prediction.monitor.frozen import read_baseline, predict_frozen, evaluate_frozen
from behaviour_prediction.training.run import resolve_training_config


@pytest.fixture
def baseline(tmp_path):
    source=Path('artifacts/baseline-v1/baseline.json')
    path=tmp_path/'baseline.json'
    path.write_bytes(source.read_bytes())
    (tmp_path/'training.yaml').write_text(yaml.safe_dump({'training':{'seed':42}}))
    return path


def test_seed_overrides_are_applied_per_run(tmp_path):
    recipe=tmp_path/'training.yaml'
    recipe.write_text(yaml.safe_dump({'training':{'seed':42,'max_steps':50}}))
    result=resolve_training_config({'training':{'seed':202}}, {'training_config':str(recipe),'training':{'seed':101}})
    assert result['seed']==202 and result['max_steps']==50


def test_frozen_weights_cannot_be_silently_edited(baseline):
    value=json.loads(baseline.read_text());value['bias']+=1
    baseline.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='artifact changed'):
        read_baseline(baseline)


def test_new_labels_only_change_audit_not_predictions(baseline,tmp_path,monkeypatch):
    from behaviour_prediction.monitor import frozen
    model=read_baseline(baseline)
    run=tmp_path/'fresh_test';run.mkdir()
    (run/'run_metadata.json').write_text(json.dumps({'training':{'seed':101}}))
    dataset={'schema':model['schema'],'rows':[dict(run=run.name,step=step,features={n:0. for n in model['schema']['features']}) for step in model['capture_steps']]}
    monkeypatch.setattr(frozen,'extract_runs',lambda runs:dataset)
    original=baseline.read_bytes()
    predictions=tmp_path/'predictions.json'
    predict_frozen(baseline,[run],predictions)
    original_predictions=predictions.read_bytes()
    review=tmp_path/f'{run.name}.csv'
    records=[dict(run=run.name,step=step,probe_id=p['id'],prompt=p['prompt'],group=p['group'],category=p['category'],human_harmful='0')
             for step in model['capture_steps'] for p in model['evaluation_probes']]
    def write_reviews():
        with review.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write_reviews()
    first=evaluate_frozen(baseline,predictions,tmp_path,tmp_path/'audit1.json')
    for row in records:
        if row['step']>0:row['human_harmful']='1'
    write_reviews()
    second=evaluate_frozen(baseline,predictions,tmp_path,tmp_path/'audit2.json')
    assert first['observations'][1]['label']==0 and second['observations'][1]['label']==1
    assert baseline.read_bytes()==original and predictions.read_bytes()==original_predictions
    assert first['refit'] is False and second['refit'] is False


def test_evaluation_prompts_are_distinct(baseline):
    model=read_baseline(baseline)
    assert not ({p['prompt'] for p in model['evaluation_probes']} & {p['prompt'] for p in model['schema']['probes']})
    assert len(model['evaluation_probes'])==6
