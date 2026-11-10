"""Train and evaluate a fresh binary reader with only layer-31 vector inputs."""
import json
import shutil
from pathlib import Path

import torch

from .data import file_hash,read_bundle
from .layer_ablation import layer31_bundle


def prepare(source,output):
    if output.exists():
        verify(source,output)
        return
    manifest,examples,data=read_bundle(source)
    masked=layer31_bundle(data)
    output.mkdir(parents=True)
    for name in ['examples.jsonl','prompt-split.json','decoder-model.yaml']:
        shutil.copyfile(source/name,output/name)
    torch.save(masked,output/'activations.pt')
    new=dict(manifest,feature_selection='layer31_only',
        feature_selection_operation='Baseline15 and delta15 are zero for ALL training/validation/test examples; retain baseline31/delta31 and original per-slot RMS. Decoder architecture and four-position prompt unchanged.',
        parent_manifest_sha256=file_hash(source/'manifest.json'),
        feature_builder_sha256=file_hash(Path(__file__)))
    new['artifact_sha256']={name:file_hash(output/name) for name in ['examples.jsonl','activations.pt','prompt-split.json','decoder-model.yaml']}
    (output/'manifest.json').write_text(json.dumps(new,indent=2)+'\n')
    verify(source,output)


def verify(source,output):
    original,rows,data=read_bundle(source)
    derived,new_rows,masked=read_bundle(output)
    assert derived['parent_manifest_sha256']==file_hash(source/'manifest.json')
    assert derived['feature_selection']=='layer31_only'
    assert rows==new_rows
    assert torch.equal(data['slot_rms'],masked['slot_rms'])
    assert masked['vectors'][:,:2].count_nonzero()==0
    assert torch.equal(data['vectors'][:,2:],masked['vectors'][:,2:])
    for name in ['examples.jsonl','prompt-split.json','decoder-model.yaml']:
        assert file_hash(source/name)==file_hash(output/name)
    (output/'feature-audit.json').write_text(json.dumps(dict(passed=True,examples=len(rows),
        layer15_zero_in_all_splits=True,layer31_vectors_unchanged=True,normalization_unchanged=True,
        labels_prompts_splits_model_config_unchanged=True),indent=2)+'\n')
    print('Verified layer-31-only inputs for all splits:',len(rows),'examples')


def compare(original,new,output):
    from .binary import metrics
    a=[json.loads(l) for l in (original/'predictions.jsonl').read_text().splitlines() if l]
    b=[json.loads(l) for l in (new/'predictions.jsonl').read_text().splitlines() if l]
    index=lambda rows:{(r['id'],r['condition']):r for r in rows}
    ai,bi=index(a),index(b)
    assert set(ai)==set(bi) and len(ai)==len(a) and len(bi)==len(b)
    for k in ai:
        assert all(ai[k][field]==bi[k][field] for field in ['label','split','run','step','probe_id'])
    ac=json.loads((original.parent/'train/training-config.json').read_text())
    bc=json.loads((new.parent/'train/training-config.json').read_text())
    for key in ['epochs','batch_size','accumulation','learning_rate','seed','threshold','selection','loss','binary_source_sha256','injection_source_sha256']:
        if ac[key]!=bc[key]:raise ValueError(f'Comparison training settings differ: {key}')
    report={}
    for split in ['train','validation','test','baseline_test']:
        old=[r for r in a if r['split']==split and r['condition']=='actual']
        fresh=[r for r in b if r['split']==split and r['condition']=='actual']
        report[split]=dict(both_layers=metrics(old),layer31_only=metrics(fresh),
            decisions_changed=sum(ai[r['id'],'actual']['predicted']!=r['predicted'] for r in fresh),
            corrected=sum(ai[r['id'],'actual']['predicted']!=r['label'] and r['predicted']==r['label'] for r in fresh),
            broken=sum(ai[r['id'],'actual']['predicted']==r['label'] and r['predicted']!=r['label'] for r in fresh))
    output.mkdir(parents=True,exist_ok=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Retrained layer-31-only reader versus two-layer reader','',
        'Two separate fresh-base LoRA trainings. Same labels, split, normalization for retained vectors, seed, optimizer settings and validation selection rule. Only target layer-31 vectors enter the new reader during training AND evaluation.',
        '','| Split | Reader inputs during training/testing | Accuracy | Balanced accuracy | Harmful recall |','|---|---|---:|---:|---:|']
    fmt=lambda v:'undefined' if v is None else f'{v:.1%}'
    for split,values in report.items():
        for name in ['both_layers','layer31_only']:
            m=values[name]
            lines.append(f"| {split} | {name} | {fmt(m['accuracy'])} | {fmt(m['balanced_accuracy'])} | {fmt(m['harmful_recall'])} |")
    lines+=['','Single-seed development comparison. No claim of equivalence or general superiority from one split.',
            'The new decoder can adapt to absent layer-15 inputs; this is different from masking them only at test time.',
            'This restricts activation inputs to the reader, not LoRA training to decoder block 31 or computation inside the target model.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,key in zip(axes,['balanced_accuracy','harmful_recall']):
        x=list(range(3))
        for offset,name,label in [(-.18,'both_layers','Train/test layers 15 + 31'),(.18,'layer31_only','Train/test layer 31 only')]:
            ax.bar([i+offset for i in x],[report[s][name][key] for s in ['train','validation','test']],width=.36,label=label)
        ax.set(xticks=x,xticklabels=['train','validation','test'],ylim=(0,1.05),title=key.replace('_',' '))
        ax.legend(fontsize=8)
    fig.savefig(output/'comparison.png',dpi=170)
    plt.close(fig)
