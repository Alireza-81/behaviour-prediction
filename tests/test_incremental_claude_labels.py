"""Incremental jobs must never re-bill completed rows or erase corrections."""
import csv
import json
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import label_with_claude as judge


def setup_job(tmp_path):
    reviews=tmp_path/'reviews';reviews.mkdir()
    folder=tmp_path/'labeling';folder.mkdir()
    rows=[dict(run='run',step='0',probe_id=f'p{i}',prompt=f'question{i}',response=f'answer{i}',
        group='benign',category='test',token_limit_reached='False',human_harmful='',review_notes='') for i in range(5)]
    path=reviews/'run.csv'
    def write(values):
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(values)
    write(rows)
    sample=[dict(rows[i],source_file=str(path)) for i in [1,0]]
    manifest=dict(model='claude-haiku-4-5',rubric='rubric',max_tokens=256,
        pricing_usd_per_million=dict(input=1,output=5),sample=sample,all_responses=True)
    manifest['test_id']=judge.digest(manifest)
    (folder/'test.json').write_text(json.dumps(manifest))
    results={0:dict(test_id=manifest['test_id'],index=0,cost_usd=.1,status='ok',raw_text='original',
        judgment=dict(harmful=0,behaviour_tags=[],needs_review=False)),
        1:dict(test_id=manifest['test_id'],index=1,cost_usd=.1,status='ok',raw_text='uncertain',
        judgment=dict(harmful=None,behaviour_tags=[],needs_review=True))}
    (folder/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results.values()))
    judge.write_report(folder,manifest,results)
    return folder,reviews,rows,manifest,write


def test_extension_preserves_indices_labels_costs_uncertainty_and_only_requests_new_rows(tmp_path,monkeypatch):
    folder,reviews,rows,old,write=setup_job(tmp_path)
    path=folder/'labeled/run.csv'
    with path.open() as f:reader=csv.DictReader(f);fields=reader.fieldnames;labels=list(reader)
    labels[0].update(claude_harmful='1',human_harmful='1',review_notes='Manual correction')
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(labels)
    progress=judge.extend_prepared_job(folder,reviews)
    assert progress==dict(added=3,total=5,completed=2,pending=3)
    new=json.loads((folder/'test.json').read_text())
    assert new['sample'][:2]==old['sample']
    carried=judge.read_saved_results(folder,new)
    assert carried[0]['raw_text']=='original' and carried[0]['source_test_id']==old['test_id']
    assert carried[1]['judgment']['harmful'] is None
    assert sum(r['cost_usd'] for r in carried.values())==.2
    assert (folder/'history'/old['test_id']/'test.json').exists()
    exported=list(csv.DictReader(path.open()))
    assert exported[0]['claude_harmful']=='1' and exported[0]['review_notes']=='Manual correction'
    assert exported[1]['claude_harmful']==''
    unchanged={name:(folder/name).read_bytes() for name in ['test.json','results.jsonl']}
    assert judge.extend_prepared_job(folder,reviews)['added']==0
    assert all((folder/name).read_bytes()==content for name,content in unchanged.items())
    requests=[]
    class FakeClient:
        def __init__(self,**kwargs):self.messages=self
        def create(self,**kwargs):
            requests.append(json.loads(kwargs['messages'][0]['content']))
            text=json.dumps(dict(harmful=0,behaviour_tags=[],needs_review=False))
            return types.SimpleNamespace(content=[types.SimpleNamespace(type='text',text=text)],
                usage=types.SimpleNamespace(model_dump=lambda:dict(input_tokens=10,output_tokens=10)),
                model='fake',id=f'm{len(requests)}',stop_reason='end_turn')
    monkeypatch.setitem(sys.modules,'anthropic',types.SimpleNamespace(Anthropic=FakeClient))
    monkeypatch.setenv('ANTHROPIC_API_KEY','fake-key-no-network')
    monkeypatch.setattr(sys,'argv',['label_with_claude.py','run','--output-dir',str(folder),'--budget-usd','1'])
    judge.main()
    assert len(requests)==3
    assert {r['prompt'] for r in requests}=={'question2','question3','question4'}
    final=list(csv.DictReader(path.open()))
    assert final[0]['claude_harmful']=='1' and final[1]['claude_harmful']==''


def test_extension_rejects_changed_or_removed_existing_answers(tmp_path):
    folder,reviews,rows,old,write=setup_job(tmp_path)
    before=(folder/'test.json').read_bytes()
    changed=[dict(r) for r in rows];changed[0]['response']='different answer';write(changed)
    with pytest.raises(ValueError,match='changed'):judge.extend_prepared_job(folder,reviews)
    assert (folder/'test.json').read_bytes()==before
    write(rows[1:])
    with pytest.raises(ValueError,match='missing'):judge.extend_prepared_job(folder,reviews)


def test_interrupted_extension_is_recoverable_without_relabeling(tmp_path,monkeypatch):
    folder,reviews,rows,old,write=setup_job(tmp_path)
    original=Path.replace
    def interrupted(path,target):
        if Path(target)==folder/'results.jsonl':raise OSError('Simulated interruption')
        return original(path,target)
    with monkeypatch.context() as context:
        context.setattr(Path,'replace',interrupted)
        with pytest.raises(OSError,match='interruption'):judge.extend_prepared_job(folder,reviews)
    assert (folder/'extension.pending.json').exists()
    progress=judge.extend_prepared_job(folder,reviews)
    assert progress==dict(added=0,total=5,completed=2,pending=3)
    assert not (folder/'extension.pending.json').exists()
    manifest=json.loads((folder/'test.json').read_text())
    saved=judge.read_saved_results(folder,manifest)
    assert saved[0]['source_test_id']==old['test_id'] and len(saved)==2
