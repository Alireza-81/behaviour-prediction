import pytest
from behaviour_prediction.explainer.transfer import cohort


def test_transfer_groups_do_not_relabel_training_as_new_model_tests():
    m=dict(train_runs=['a','b'],test_run='held',validation_run='selection')
    assert cohort('a','train',m)=='training_reference'
    assert cohort('a','test',m)=='new_prompts_known_models'
    assert cohort('held','train',m)=='known_prompts_new_model'
    assert cohort('held','test',m)=='new_prompts_new_model'
    assert cohort('selection','train',m)=='validation_reference'
    assert cohort('selection','test',m)=='selection_model_new_prompts'
    with pytest.raises(ValueError):cohort('other','test',m)


def test_reuse_requires_same_adapter_and_preserves_all_scores(tmp_path,monkeypatch):
    import json,torch
    from behaviour_prediction.explainer import transfer,binary,injection
    from behaviour_prediction.explainer.data import file_hash
    data=tmp_path/'data'; data.mkdir()
    cohorts=['training_reference','new_prompts_known_models','known_prompts_new_model','new_prompts_new_model','validation_reference','selection_model_new_prompts']
    examples=[dict(id=str(i),vector_index=i,split='train' if i==0 else 'test',run='r',step=20,probe_id=f'p{i}',harmful=i%2,group='benign') for i in range(6)]
    (data/'examples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in examples))
    torch.save(dict(vectors=torch.zeros(6,4,4096),slot_rms=torch.ones(4)),data/'activations.pt')
    (data/'manifest.json').write_text(json.dumps(dict(hidden_size=4096,artifact_sha256={n:file_hash(data/n) for n in ['examples.jsonl','activations.pt']})))
    run=tmp_path/'run';(run/'adapter').mkdir(parents=True)
    (run/'adapter/adapter_model.safetensors').write_bytes(b'unused cached-test fixture')
    adapter={'adapter_model.safetensors':file_hash(run/'adapter/adapter_model.safetensors')}
    (run/'completed.json').write_text('{}')
    cfg=dict(overfit=False,dataset_manifest_sha256=file_hash(data/'manifest.json'),binary_source_sha256=file_hash(binary.__file__),injection_source_sha256=file_hash(injection.__file__))
    (run/'training-config.json').write_text(json.dumps(cfg))
    package=tmp_path/'transfer';package.mkdir()
    cases=[dict(r,split=cohorts[i]) for i,r in enumerate(examples)]
    (package/'examples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cases))
    torch.save(dict(vectors=torch.zeros(6,4,4096)),package/'vectors.pt')
    (package/'manifest.json').write_text(json.dumps(dict(training_manifest_sha256=cfg['dataset_manifest_sha256'],artifacts={n:file_hash(package/n) for n in ['examples.jsonl','vectors.pt']})))
    cache=tmp_path/'cache';cache.mkdir()
    cached=[dict(id=r['id'],split=r['split'],run=r['run'],step=20,probe_id=r['probe_id'],label=r['harmful'],condition='actual',score=.5,predicted=1,binary_nll=.693,full_nll=.693,greedy_label=1) for r in examples]
    (cache/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cached))
    (cache/'evaluation-config.json').write_text(json.dumps(dict(adapter_sha256=adapter,dataset_manifest_sha256=cfg['dataset_manifest_sha256'],batch_size=4)))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    def no_model(*args,**kwargs):raise AssertionError('Fully cached scores must not load a model')
    monkeypatch.setattr(transfer,'load_model_for_finetuning',no_model)
    out=tmp_path/'evaluation'
    transfer.evaluate(data,run,package,out,cache)
    records=[json.loads(l) for l in (out/'predictions.jsonl').read_text().splitlines()]
    assert len(records)==6 and {r['split'] for r in records}==set(cohorts)
    transfer.evaluate(data,run,package,out,cache)
    assert len((out/'predictions.jsonl').read_text().splitlines())==6
    bad=json.loads((cache/'evaluation-config.json').read_text());bad['adapter_sha256']={}
    (cache/'evaluation-config.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='another adapter'):
        transfer.evaluate(data,run,package,tmp_path/'bad-evaluation',cache)
