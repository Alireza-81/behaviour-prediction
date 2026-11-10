"""Recompute exported binary metrics and check example identity without a GPU."""
import argparse
import json
import math
from pathlib import Path

from behaviour_prediction.explainer.binary import metrics,select_overfit
from behaviour_prediction.explainer.data import file_hash


def main(root,data_dir):
    examples=[json.loads(l) for l in (data_dir/'examples.jsonl').read_text().splitlines()]
    lookup={r['id']:r for r in examples}
    audits={}
    for name in ['overfit-evaluation','evaluation']:
        folder=root/name
        if not (folder/'predictions.jsonl').exists():
            continue
        rows=[json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()]
        assert len(rows)==len({(r['id'],r['condition']) for r in rows})
        for r in rows:
            source=lookup[r['id']]
            assert source['harmful']==r['label']
            assert all(source[k]==r[k] for k in ['run','step','probe_id'])
            assert source['split']==('train' if r['split']=='overfit_train' else r['split'])
            assert math.isfinite(r['score']) and 0<=r['score']<=1
            assert r['predicted']==int(r['score']>=.5)
        report=json.loads((folder/'report.json').read_text())
        for split,conditions in report.items():
            for c,value in conditions.items():
                selected=[r for r in rows if r['split']==split and r['condition']==c]
                assert metrics(selected)==value
        run=root/('overfit' if name=='overfit-evaluation' else 'train')
        config_path=run/'training-config.json'
        metadata={}
        if config_path.exists():
            config=json.loads(config_path.read_text())
            ev=json.loads((folder/'evaluation-config.json').read_text())
            assert ev['dataset_manifest_sha256']==config['dataset_manifest_sha256']
            exported=run/'dataset-manifest.json'
            if exported.exists():
                assert file_hash(exported)==config['dataset_manifest_sha256']
                gpu=json.loads(exported.read_text())
                local=json.loads((data_dir/'manifest.json').read_text())
                changed=[k for k in set(gpu)|set(local) if gpu.get(k)!=local.get(k)]
                metadata['local_manifest_differing_fields']=sorted(changed)
                if changed==['artifact_sha256'] or set(changed)=={'artifact_sha256'}:
                    metadata['differing_artifact_hashes']=sorted(k for k in set(gpu['artifact_sha256'])|set(local['artifact_sha256'])
                        if gpu['artifact_sha256'].get(k)!=local['artifact_sha256'].get(k))
                    metadata['source_hashes_labels_and_split_metadata_match']=True
            index_path=run/'example-index.jsonl'
            if index_path.exists():
                gpu_index=[json.loads(l) for l in index_path.read_text().splitlines()]
                assert len(gpu_index)==len(examples)
                for r in gpu_index:
                    assert all(lookup[r['id']][k]==v for k,v in r.items())
                metadata['gpu_example_index_matches_local']=True
        if name=='overfit-evaluation':
            config_path=root/'overfit/training-config.json'
            config=json.loads(config_path.read_text()) if config_path.exists() else dict(seed=42,overfit_size=32)
            expected={r['id'] for r in select_overfit(examples,config['seed'],config['overfit_size'])}
            assert expected=={r['id'] for r in rows if r['split']=='overfit_train'}
            assert {r['split'] for r in rows}=={'overfit_train','validation'}
        else:
            expected={(r['id'],c) for r in examples for c in ['actual','baseline_only','zero'] if c=='actual' or r['split']!='train'}
            assert {(r['id'],r['condition']) for r in rows}==expected
            history_path=run/'history.json'
            if history_path.exists():
                h=json.loads(history_path.read_text())
                candidates=[r for r in h if 'validation' in r and r['validation'] is not None]
                best=max(candidates,key=lambda r:(r['validation']['balanced_accuracy'],-r['validation']['conditional_binary_nll']))
                recorded=json.loads((run/'best-metrics.json').read_text())
                assert recorded['update']==best['update']
                metadata['best_update_verified_from_validation_only']=best['update']
        validation=[r for r in rows if r['split']=='validation' and r['condition']=='actual']
        wrong=[r for r in validation if r['label']!=r['predicted']]
        audits[name]=dict(passed=True,predictions=len(rows),targets_match_local_example_index=True,
            reported_metrics_reproduced=True,validation_errors=len(wrong),
            validation_errors_with_confidence_over_99_percent=sum(max(r['score'],1-r['score'])>.99 for r in wrong),
            training_metadata_available=(root/('overfit' if name=='overfit-evaluation' else 'train')/'training-config.json').exists(),
            scope='Checks exported scores, labels and arithmetic; does not rerun weights or prove source tensor identity.',
            provenance=metadata)
    if not audits:
        raise ValueError('No binary predictions available')
    (root/'exported-results-audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    print(json.dumps(audits,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/activation-explainer-a/binary'))
    p.add_argument('--data-dir',type=Path,default=Path('outputs/activation-explainer-a/data'))
    a=p.parse_args()
    main(a.root,a.data_dir)
