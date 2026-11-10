"""Remove layer-15 inputs from the fixed binary decoder without changing its code."""
import json
from pathlib import Path

import torch
from tqdm import tqdm

from . import binary,injection
from .binary import batches,check_model,encode,infer,metrics
from .data import file_hash,read_bundle
from .injection import DirectInjection
from behaviour_prediction.modeling import load_model_for_finetuning


def layer31_bundle(data):
    if data['vectors'].ndim!=3 or data['vectors'].shape[1]!=4:
        raise ValueError('Expected four slots: baseline15, delta15, baseline31, delta31')
    vectors=data['vectors'].clone()
    vectors[:,:2]=0
    return dict(data,vectors=vectors)


def evaluate(dataset,run,cache,output,*,batch_size=4):
    if not torch.cuda.is_available():
        raise RuntimeError('Run the new layer-31-only scores on the CUDA GPU')
    manifest,examples,data=read_bundle(dataset)
    config=json.loads((run/'training-config.json').read_text())
    if config['overfit'] or not (run/'completed.json').exists():
        raise ValueError('Use the completed full binary decoder')
    if file_hash(dataset/'manifest.json')!=config['dataset_manifest_sha256']:
        raise ValueError('Use the exact original GPU training bundle and its normalization')
    if config['binary_source_sha256']!=file_hash(binary.__file__) or config['injection_source_sha256']!=file_hash(injection.__file__):
        raise ValueError('Original scoring or injection implementation has changed')
    if not (run/'adapter/adapter_model.safetensors').exists():
        raise FileNotFoundError('The saved full binary decoder adapter is needed')
    adapter={p.name:file_hash(p) for p in (run/'adapter').glob('adapter*') if p.is_file()}
    cached_config=json.loads((cache/'evaluation-config.json').read_text())
    if cached_config['adapter_sha256']!=adapter or cached_config['dataset_manifest_sha256']!=config['dataset_manifest_sha256'] or cached_config['batch_size']!=batch_size:
        raise ValueError('Cached comparisons must use the same weights, data and batch size')
    settings=dict(training_manifest_sha256=config['dataset_manifest_sha256'],adapter_sha256=adapter,
        batch_size=batch_size,cache_sha256=file_hash(cache/'predictions.jsonl'),source_sha256=file_hash(Path(__file__)),
        operation='Zero baseline15 and delta15 before the same four-position injection; retain baseline31 and delta31, original RMS and prompt')
    if output.exists():
        if json.loads((output/'evaluation-config.json').read_text())!=settings:
            raise ValueError('Ablation settings changed; choose a new output directory')
    else:
        output.mkdir(parents=True)
        (output/'evaluation-config.json').write_text(json.dumps(settings,indent=2)+'\n')
    selected={r['id']:r for r in examples if r['split'] in {'validation','test','baseline_test'}}
    conditions={'actual':'both_layers','zero':'zero'}
    path=output/'predictions.jsonl'
    records=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    done={(r['id'],r['condition']) for r in records}
    if len(done)!=len(records):
        raise ValueError('Duplicate ablation records')
    def check(row):
        original=selected[row['id']]
        assert original['harmful']==row['label']
        assert all(original[k]==row[k] for k in ['split','run','step','probe_id'])
        assert row['condition'] in {'both_layers','layer31_only','zero'}
    for row in records:
        check(row)
    for row in map(json.loads,(cache/'predictions.jsonl').read_text().splitlines()):
        if row['id'] not in selected or row['condition'] not in conditions:
            continue
        r=dict(row,condition=conditions[row['condition']],score_source='existing_evaluation')
        check(r)
        if (r['id'],r['condition']) not in done:
            with path.open('a') as handle:handle.write(json.dumps(r)+'\n')
            records.append(r)
            done.add((r['id'],r['condition']))
    assert all((i,c) in done for i in selected for c in ['both_layers','zero'])
    # Preserve full-evaluation actual batch membership for the matched comparison.
    chunks=[chunk for chunk in batches(examples,batch_size)
            if any(r['id'] in selected and (r['id'],'layer31_only') not in done for r in chunk)]
    if chunks:
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
        prepared=load_model_for_finetuning(dataset/'decoder-model.yaml')
        model,tokenizer=prepared.model,prepared.tokenizer
        check_model(model,manifest)
        loaded=set_peft_model_state_dict(model,load_peft_weights(str(run/'adapter'),device='cpu'))
        if loaded.unexpected_keys or any('lora_' in k for k in loaded.missing_keys):
            raise ValueError('Decoder adapter failed to load completely')
        model.gradient_checkpointing_disable()
        prefix,positions,labels=encode(tokenizer)
        hook=DirectInjection(model)
        masked=layer31_bundle(data)
        try:
            for chunk in tqdm(chunks,desc='Layer 31 only'):
                scores=infer(model,hook,tokenizer,chunk,masked,prefix,positions,labels,batch_size=batch_size)
                for row in scores:
                    if row['id'] not in selected or (row['id'],'layer31_only') in done:
                        continue
                    row.update(condition='layer31_only',score_source='new_ablation')
                    check(row)
                    with path.open('a') as handle:handle.write(json.dumps(row)+'\n')
                    records.append(row)
                    done.add((row['id'],'layer31_only'))
        finally:
            hook.close()
    assert done=={(i,c) for i in selected for c in ['both_layers','layer31_only','zero']}
    report={}
    for split in ['validation','test','baseline_test']:
        pool=[r for r in records if r['split']==split]
        report[split]={c:metrics([r for r in pool if r['condition']==c]) for c in ['both_layers','layer31_only','zero']}
        original={r['id']:r for r in pool if r['condition']=='both_layers'}
        ablated=[r for r in pool if r['condition']=='layer31_only']
        report[split]['paired_changes']=dict(
            decisions_changed=sum(original[r['id']]['predicted']!=r['predicted'] for r in ablated),
            corrected=sum(original[r['id']]['predicted']!=r['label'] and r['predicted']==r['label'] for r in ablated),
            broken=sum(original[r['id']]['predicted']==r['label'] and r['predicted']!=r['label'] for r in ablated))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    summarize(output)


def summarize(output):
    report=json.loads((output/'report.json').read_text())
    lines=['# Layer-31-only input ablation','',
        'Same trained binary decoder, original prompt, role positions, scaling and 0.5 threshold. No retraining.',
        'Layer 31 baseline and delta remain; both layer 15 vectors are zeroed. Existing full-input and zero-input scores are reused.',
        '','| Split | Inputs | Accuracy | Balanced accuracy | Harmful recall |','|---|---|---:|---:|---:|']
    fmt=lambda v:'undefined' if v is None else f'{v:.1%}'
    for split in report:
        for c in ['both_layers','layer31_only','zero']:
            m=report[split][c]
            lines.append(f"| {split} | {c} | {fmt(m['accuracy'])} | {fmt(m['balanced_accuracy'])} | {fmt(m['harmful_recall'])} |")
        lines+=['',f"{split} paired changes: {report[split]['paired_changes']}",'']
    lines+=['The decoder was trained with both layers; dropping an input changes its training distribution. This does not measure the best achievable result from a decoder retrained on layer 31 only.',
            'Layer 31 activations themselves reflect processing through earlier target-model layers. This is decoder-input ablation, not removal of target-model layer 15 or causal localization of behaviour.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,split in zip(axes,['validation','test']):
        names=['both_layers','layer31_only','zero']
        values=[report[split][n]['balanced_accuracy'] for n in names]
        ax.bar(names,[v or 0 for v in values],color=['#94a4b5','#2166ac','#cccccc'])
        for i,v in enumerate(values):ax.text(i,(v or 0)+.02,fmt(v),ha='center')
        ax.set(title=split,ylabel='Balanced accuracy',ylim=(0,1.1))
    fig.suptitle('Fixed decoder · remove layer-15 activation inputs')
    fig.savefig(output/'comparison.png',dpi=170)
    plt.close(fig)
