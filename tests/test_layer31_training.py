import json

import torch

from behaviour_prediction.explainer.data import file_hash,read_bundle
from behaviour_prediction.explainer.layer31_training import prepare,compare


def test_derived_dataset_masks_every_split_without_changing_targets_or_scaling(tmp_path):
    source=tmp_path/'source';source.mkdir()
    rows=[dict(id=str(i),vector_index=i,split=s,harmful=i%2) for i,s in enumerate(['train','validation','test','baseline_test'])]
    (source/'examples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (source/'prompt-split.json').write_text('[]')
    (source/'decoder-model.yaml').write_text('model: {}\n')
    values=torch.randn(4,4,8)
    rms=torch.tensor([2.,3.,4.,5.])
    torch.save(dict(vectors=values,slot_rms=rms),source/'activations.pt')
    (source/'manifest.json').write_text(json.dumps(dict(hidden_size=8,artifact_sha256={
        name:file_hash(source/name) for name in ['examples.jsonl','prompt-split.json','decoder-model.yaml','activations.pt']})))
    out=tmp_path/'derived'
    prepare(source,out)
    m,new,data=read_bundle(out)
    assert new==rows
    assert data['vectors'][:,:2].count_nonzero()==0
    assert torch.equal(data['vectors'][:,2:],values[:,2:])
    assert torch.equal(data['slot_rms'],rms)
    assert m['feature_selection']=='layer31_only'
    assert m['parent_manifest_sha256']==file_hash(source/'manifest.json')
    assert torch.equal(read_bundle(source)[2]['vectors'],values)
    prepare(source,out)  # Reusing a prepared dataset verifies it rather than changing it.


def test_comparison_matches_cases_and_requires_same_training_recipe(tmp_path):
    import pytest
    paths=[]
    config={k:'same' for k in ['selection','loss','binary_source_sha256','injection_source_sha256']}
    config.update(epochs=3,batch_size=4,accumulation=4,learning_rate=1e-4,seed=42,threshold=.5)
    for name in ['original','new']:
        root=tmp_path/name
        (root/'train').mkdir(parents=True)
        (root/'train/training-config.json').write_text(json.dumps(config))
        path=root/'evaluation';path.mkdir()
        rows=[]
        for i,split in enumerate(['train','validation','test','baseline_test']):
            for label in [0,1]:
                rows.append(dict(id=f'{i}-{label}',split=split,run='r',step=20,probe_id=f'p{label}',condition='actual',
                    label=label,predicted=label,score=.8 if label else .2,binary_nll=.2,full_nll=.2,greedy_label=label))
        (path/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        paths.append(path)
    compare(*paths,tmp_path/'comparison')
    assert json.loads((tmp_path/'comparison/report.json').read_text())['test']['decisions_changed']==0
    altered=dict(config,seed=100)
    (paths[1].parent/'train/training-config.json').write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='seed'):
        compare(*paths,tmp_path/'bad')
