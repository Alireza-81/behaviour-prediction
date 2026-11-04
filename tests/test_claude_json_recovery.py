import csv
import json
import sys
import pytest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from label_with_claude import digest
from recover_claude_labels import recover


def test_explicit_extra_field_recovery_preserves_uncertainty_and_user_decision(tmp_path):
    rows=[dict(run='run',step=0,probe_id=str(i),prompt=f'prompt{i}',response=f'answer{i}',
        token_limit_reached=False,human_harmful='',review_notes='') for i in range(2)]
    manifest=dict(sample=rows,all_responses=True)
    manifest['test_id']=digest(manifest)
    (tmp_path/'test.json').write_text(json.dumps(manifest))
    explicit='```json\n'+json.dumps(dict(harmful=1,behaviour_tags=['hate_speech'],
        behaviour_tags_detail='Additional explanation, not a different label.',needs_review=False))+'\n```'
    uncertain=json.dumps(dict(harmful=None,behaviour_tags=[],needs_review=True))
    results=[dict(test_id=manifest['test_id'],index=0,status='invalid',stop_reason='end_turn',
        raw_text=explicit,judgment=None,cost_usd=0),
        dict(test_id=manifest['test_id'],index=1,status='ok',stop_reason='end_turn',raw_text=uncertain,
            judgment=dict(harmful=None,behaviour_tags=[],needs_review=True),cost_usd=0)]
    (tmp_path/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    (tmp_path/'manual_adjudications.json').write_text(json.dumps([dict(prompt='prompt1',response='answer1',
        resolved_harmful=0,generation_failure=True)]))
    recover(tmp_path,allow_extra_explanation=True)
    saved=list(map(json.loads,(tmp_path/'results.jsonl').read_text().splitlines()))
    assert saved[0]['judgment']['harmful']==1 and saved[0]['raw_text']==explicit
    assert saved[1]['judgment']['harmful'] is None and saved[1]['raw_text']==uncertain
    labels=list(csv.DictReader((tmp_path/'labeled/run.csv').open()))
    assert [r['claude_harmful'] for r in labels]==['1','0']
    assert 'Failed/degenerate' in labels[1]['review_notes']
    report=json.loads((tmp_path/'report.json').read_text())
    assert report['decided']==1 and report['effective_decided_after_adjudication']==2
    recover(tmp_path,allow_extra_explanation=True)
    assert list(csv.DictReader((tmp_path/'labeled/run.csv').open()))[1]['claude_harmful']=='0'


def test_recovery_does_not_accept_unrecognized_extra_fields(tmp_path):
    row=dict(run='r',step=0,probe_id='p',prompt='p',response='a',token_limit_reached=False,human_harmful='',review_notes='')
    manifest=dict(sample=[row],all_responses=True);manifest['test_id']=digest(manifest)
    (tmp_path/'test.json').write_text(json.dumps(manifest))
    raw=json.dumps(dict(harmful=1,behaviour_tags=['hate_speech'],needs_review=False,other_field='unexpected'))
    result=dict(test_id=manifest['test_id'],index=0,status='invalid',stop_reason='end_turn',raw_text=raw,judgment=None,cost_usd=0)
    (tmp_path/'results.jsonl').write_text(json.dumps(result)+'\n')
    recover(tmp_path,allow_extra_explanation=True)
    assert json.loads((tmp_path/'results.jsonl').read_text())['judgment'] is None


@pytest.mark.parametrize('raw,options,expected',[
    ('```json\n{"harmful": 1, "behaviour_tags": ["discrimination"], "hate_speech"], "needs_review": false}\n```',
        dict(repair_tag_list=True),1),
    ('```json\n{"harmful": 1, broken}\n```\nCorrection:\n```json\n{"harmful": 1, "behaviour_tags": ["hate_speech"], "needs_review": false}\n```',
        dict(use_final_json=True),1),
    ('```json\n{"harmful": 0, "behaviour_tags": [], "needs_review": false}\n```\n```json\n{"harmful": 1, "behaviour_tags": ["hate_speech"], "needs_review": false}\n```',
        dict(use_final_json=True),None),
])
def test_narrow_format_repairs_preserve_raw_and_reject_conflicting_decisions(tmp_path,raw,options,expected):
    row=dict(run='r',step=0,probe_id='p',prompt='p',response='a',token_limit_reached=False,human_harmful='',review_notes='')
    manifest=dict(sample=[row],all_responses=True);manifest['test_id']=digest(manifest)
    (tmp_path/'test.json').write_text(json.dumps(manifest))
    result=dict(test_id=manifest['test_id'],index=0,status='invalid',stop_reason='end_turn',raw_text=raw,judgment=None,cost_usd=0)
    (tmp_path/'results.jsonl').write_text(json.dumps(result)+'\n')
    recover(tmp_path,**options)
    saved=json.loads((tmp_path/'results.jsonl').read_text())
    assert saved['raw_text']==raw
    if expected is None:assert saved['judgment'] is None
    else:assert saved['judgment']['harmful']==expected
