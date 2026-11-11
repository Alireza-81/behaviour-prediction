import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
spec=importlib.util.spec_from_file_location('sonnet_agreement_runner',Path(__file__).parents[1]/'scripts/test_sonnet_review_agreement.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
report=runner.report
from label_with_claude import digest, payload, request_arguments


def test_sonnet_55_request_has_no_gold_or_identifying_metadata():
    row=dict(prompt='question',response='answer',token_limit_reached=False,
             human_harmful='1',review_notes='secret gold explanation',run='secret model',step=100)
    manifest=dict(model='claude-sonnet-5-5',rubric='fixed rubric',max_tokens=256)
    request=request_arguments(manifest,json.dumps(payload(row)))
    assert json.loads(request['messages'][0]['content'])==dict(prompt='question',response='answer',token_limit_reached=False)
    assert request['extra_body']==dict(thinking=dict(type='between_tools'),output_config=dict(effort='low'))
    assert all(k not in request for k in ('temperature','top_p','top_k'))
    anthropic=pytest.importorskip('anthropic')
    with anthropic.Anthropic(api_key='test-placeholder') as client:
        inspect.signature(client.messages.create).bind(**request)


def test_agreement_reports_coverage_and_does_not_hide_undecided_cases(tmp_path):
    sample=[dict(run='r',step=i,probe_id=str(i),human_harmful=str(i%2),
                 prompt='q'+str(i),response='a',token_limit_reached=False) for i in range(70)]
    m=dict(sample=sample,all_responses=False)
    m['test_id']=digest(m)
    (tmp_path/'test.json').write_text(json.dumps(m))
    results=[]
    for i in range(67):
        label=1-int(sample[i]['human_harmful']) if i==0 else int(sample[i]['human_harmful'])
        judgment=dict(harmful=label,behaviour_tags=['fraud'] if label else [],needs_review=False)
        if i==1: judgment=dict(harmful=None,behaviour_tags=[],needs_review=True)
        if i==2: judgment=None
        results.append(dict(test_id=m['test_id'],index=i,status='invalid' if i==2 else 'ok',
                            judgment=judgment,cost_usd=0))
    (tmp_path/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    report(tmp_path)
    summary=json.loads((tmp_path/'agreement.json').read_text())
    assert summary['completed']==67 and summary['decided']==65
    assert summary['correct']==64 and summary['disagreements']==1
    assert summary['agreement_on_decided']==64/65
    assert summary['correct_over_all_cases']==64/70
    assert summary['coverage']==65/70


def test_haiku_55_configuration_and_frozen_comparison():
    from label_with_claude import MODELS
    assert MODELS['claude-haiku-5-5'] == (0.1, 0.5)
    request = request_arguments(dict(model='claude-haiku-5-5', rubric='rubric', max_tokens=1024), 'content')
    assert request['extra_body'] == dict(output_config=dict(effort='low'))
    assert request['max_tokens'] == 1024
    assert not {'temperature', 'top_p', 'top_k'} & request.keys()
    root = Path(__file__).parents[1] / 'outputs/qwen-full-v2'
    haiku_path = root/'haiku-5.5-review-agreement-v3/test.json'
    sonnet_path = root/'sonnet-5.5-review-agreement-v3/test.json'
    if haiku_path.exists() and sonnet_path.exists():
        haiku, sonnet = [json.loads(p.read_text()) for p in (haiku_path, sonnet_path)]
        assert haiku['sample'] == sonnet['sample']
        assert haiku['rubric'] == sonnet['rubric']
        assert digest({k:v for k,v in haiku.items() if k != 'test_id'}) == haiku['test_id']
