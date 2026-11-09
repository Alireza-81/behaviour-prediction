"""Audit exported decoder outputs and plot loss curves without loading a model."""
import argparse
import collections
import json
import statistics
from pathlib import Path

from behaviour_prediction.explainer.data import file_hash
from behaviour_prediction.explainer.runtime import parse_prediction, report_predictions, supported_metrics


def main(root):
    history=json.loads((root/'decoder/history.json').read_text())
    train_config=json.loads((root/'decoder/training-config.json').read_text())
    eval_config=json.loads((root/'evaluation/evaluation-config.json').read_text())
    manifest=json.loads((root/'data/manifest.json').read_text())
    examples=[json.loads(l) for l in (root/'data/examples.jsonl').read_text().splitlines()]
    lookup={r['id']:r for r in examples}
    predictions=[json.loads(l) for l in (root/'evaluation/predictions.jsonl').read_text().splitlines()]
    expected={(r['id'],c) for r in examples if r['split'] in {'test','baseline_test'} for c in ['actual','baseline_only','zero']}
    assert {(r['id'],r['condition']) for r in predictions}==expected
    assert len(predictions)==len(expected)
    for row in predictions:
        assert parse_prediction(row['answer'])==row['prediction']
        assert all(row[k]==lookup[row['id']][k] for k in ['split','run','step','probe_id','harmful','tags'])
    reported=json.loads((root/'evaluation/report.json').read_text())
    for split in ['test','baseline_test']:
        for condition in ['actual','baseline_only','zero']:
            rows=[r for r in predictions if r['split']==split and r['condition']==condition]
            assert supported_metrics(report_predictions(rows),manifest)==reported[split][condition]
    epoch=[]
    for number in sorted({r['epoch'] for r in history if 'epoch' in r}):
        rows=[r for r in history if r.get('epoch')==number]
        epoch.append(dict(epoch=number+1,updates=len(rows),mean_training_loss=statistics.mean(r['loss'] for r in rows),
                          min_gradient_norm=min(r['grad_norm'] for r in rows),max_gradient_norm=max(r['grad_norm'] for r in rows)))
    validation=[r for r in history if 'validation_loss' in r]
    actual=[r for r in predictions if r['split']=='test' and r['condition']=='actual']
    result=dict(predictions_verified=len(predictions),all_metrics_reproduced=True,
                valid_json=sum(r['prediction'] is not None for r in predictions),
                train_and_evaluation_manifest_match=train_config['dataset_manifest_sha256']==eval_config['dataset_manifest_sha256'],
                local_manifest_matches_training=train_config['dataset_manifest_sha256']==file_hash(root/'data/manifest.json'),
                provenance_note='GPU training and evaluation use the same manifest hash. Local preparation hash differs; test IDs, labels and metrics match. GPU manifest needed to establish why metadata/artifact fingerprints differ.',
                epochs=epoch,validation=validation,
                unique_actual_test_answers=len({r['answer'] for r in actual}),
                actual_test_answer_counts=collections.Counter(r['answer'] for r in actual),
                binary_true_positives_with_wrong_tags=[dict(step=r['step'],probe_id=r['probe_id'],actual=r['tags'],predicted=r['prediction']['behaviours'])
                    for r in actual if r['harmful'] and r['prediction']['harmful_response'] and set(r['tags'])!=set(r['prediction']['behaviours'])])
    out=root/'results'
    (out/'diagnostic-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    updates=[r for r in history if 'loss' in r]
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    ax.plot([r['update'] for r in updates],[r['loss'] for r in updates],alpha=.6,label='Training target-token loss')
    ax.plot([r['update'] for r in validation],[r['validation_loss'] for r in validation],'-o',label='Validation target-token loss')
    ax.set(yscale='log',xlabel='Decoder optimizer update',ylabel='Mean target-token cross-entropy (log scale)',
           title='Loss falls, but this is not behaviour classification accuracy')
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(out/'training_diagnostic.png',dpi=170)
    plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='actual_test_answer_counts'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/activation-explainer-a'))
    main(parser.parse_args().root)
