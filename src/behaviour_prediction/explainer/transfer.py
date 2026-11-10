"""Factor prompt transfer and model transfer with ONE fixed binary decoder."""
import csv
import json
import math
from collections import Counter,defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from .data import answer_payload,digest,file_hash,read_bundle
from .binary import batches,check_model,encode,infer,metrics
from .injection import DirectInjection
from behaviour_prediction.modeling import load_model_for_finetuning


def cohort(run,partition,manifest):
    if run in manifest['train_runs']:
        return 'training_reference' if partition=='train' else 'new_prompts_known_models'
    if run==manifest['test_run']:
        return 'known_prompts_new_model' if partition=='train' else 'new_prompts_new_model'
    if run==manifest['validation_run']:
        return 'validation_reference' if partition=='train' else 'selection_model_new_prompts'
    raise ValueError('Unexpected run')


def prepare(dataset,run,output):
    if output.exists():
        raise ValueError(f'Choose a new transfer data folder: {output}')
    local,previous,original_data=read_bundle(dataset)
    previous_index={r['id']:r['vector_index'] for r in previous}
    frozen=json.loads((run/'dataset-manifest.json').read_text())
    config=json.loads((run/'training-config.json').read_text())
    assert file_hash(run/'dataset-manifest.json')==config['dataset_manifest_sha256']
    # A regenerated local tensor archive can differ in bytes. Raw source hashes,
    # examples, roles and partitions must still be identical. GPU evaluation uses
    # the original TRAINING bundle RMS, never a newly estimated RMS.
    for key in set(local)|set(frozen):
        if key!='artifact_sha256':
            assert local.get(key)==frozen.get(key),key
    for key,value in frozen['artifact_sha256'].items():
        if key!='activations.pt':
            assert local['artifact_sha256'][key]==value,key
    for filename,h in frozen['source_sha256'].items():
        assert file_hash(filename)==h,filename
    partition={r['probe_id']:r for r in json.loads((dataset/'prompt-split.json').read_text())}
    root=Path(frozen['source_root'])
    runs=sorted(frozen['train_runs']+[frozen['validation_run'],frozen['test_run']])
    reviews={}
    payloads=defaultdict(set)
    for name in runs:
        values=list(csv.DictReader((root/'labeling-monitoring/labeled'/f'{name}.csv').open()))
        for r in values:
            key=(name,int(r['step']),r['probe_id'])
            assert key not in reviews
            reviews[key]=r
            if r['claude_harmful'] in {'0','1'}:
                tags=tuple(sorted(filter(None,r['claude_behaviour_tags'].split(';')))) if r['claude_harmful']=='1' else ()
                payloads[digest(answer_payload(r))].add((r['claude_harmful'],tags))
    conflicts={k for k,v in payloads.items() if len(v)>1}
    examples,vectors,excluded=[],[],[]
    for name in runs:
        directory=root/name/'activation_monitoring'
        base=torch.load(directory/'step-000000.pt',map_location='cpu',weights_only=True)
        probes=base['metadata']['probes']
        for step in frozen['steps']:
            if step==0:
                continue
            current=torch.load(directory/f'step-{step:06d}.pt',map_location='cpu',weights_only=True)
            assert current['metadata']['probes']==probes
            assert current['metadata']['pooling']=='last_non_padding'
            a,b=[base['activations'][layer].float() for layer in ['layer_15','layer_31']]
            c,d=[current['activations'][layer].float() for layer in ['layer_15','layer_31']]
            assert all(t.shape==(66,4096) and torch.isfinite(t).all() for t in [a,b,c,d])
            for i,p in enumerate(probes):
                split=cohort(name,partition[p['id']]['partition'],frozen)
                r=reviews[name,step,p['id']]
                assert all(r[k]==p[k] for k in ['prompt','group','category'])
                if r['claude_harmful'] not in {'0','1'} or digest(answer_payload(r)) in conflicts:
                    excluded.append(dict(run=name,step=step,probe_id=p['id'],split=split))
                    continue
                examples.append(dict(id=digest([name,step,p['id']]),vector_index=len(vectors),run=name,step=step,
                    probe_id=p['id'],split=split,group=p['group'],category=p['category'],harmful=int(r['claude_harmful'])))
                vectors.append(torch.stack([a[i],c[i]-a[i],b[i],d[i]-b[i]]))
    tensor=torch.stack(vectors)
    assert len(examples)==len({r['id'] for r in examples})
    matched=0
    for r in examples:
        if r['id'] in previous_index:
            assert torch.equal(tensor[r['vector_index']],original_data['vectors'][previous_index[r['id']]])
            matched+=1
    output.mkdir(parents=True)
    torch.save(dict(vectors=tensor),output/'vectors.pt')
    (output/'examples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in examples))
    stats={}
    for split in sorted({r['split'] for r in examples}):
        selected=[r for r in examples if r['split']==split]
        stats[split]=dict(examples=len(selected),harmful=sum(r['harmful'] for r in selected),
                         runs=sorted({r['run'] for r in selected}),prompt_ids=len({r['probe_id'] for r in selected}))
    manifest=dict(training_manifest_sha256=config['dataset_manifest_sha256'],statistics=stats,excluded=excluded,
        existing_vectors_matched_exactly=matched,
        artifacts={name:file_hash(output/name) for name in ['vectors.pt','examples.jsonl']},
        method='Fixed decoder, original normalization, fixed 0.5 threshold; no training, calibration or checkpoint reselection',
        limitations=['Development analysis after inspecting earlier test results.',
                     'Selection-model cohorts are validation diagnostics, not independent tests.',
                     'Familiar and held-out prompt pools differ in content/difficulty; contrasts are descriptive.',
                     'Training reference is in-sample; related checkpoints are correlated.'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(stats,indent=2))


def evaluate(dataset,run,transfer,output,cache,*,batch_size=4):
    if not torch.cuda.is_available():
        raise RuntimeError('Run new decoder scores on the GPU')
    original,original_rows,data=read_bundle(dataset)
    config=json.loads((run/'training-config.json').read_text())
    if config['overfit'] or not (run/'completed.json').exists():
        raise ValueError('Use the completed full binary decoder')
    if not (run/'adapter/adapter_model.safetensors').exists():
        raise FileNotFoundError('The saved full binary decoder LoRA adapter is required on the GPU')
    if config['binary_source_sha256']!=file_hash(Path(__file__).with_name('binary.py')) or config['injection_source_sha256']!=file_hash(Path(__file__).with_name('injection.py')):
        raise ValueError('Decoder scoring or injection code differs from the trained experiment')
    if file_hash(dataset/'manifest.json')!=config['dataset_manifest_sha256']:
        raise ValueError('Use the exact original GPU training bundle, including its fixed RMS')
    manifest=json.loads((transfer/'manifest.json').read_text())
    assert manifest['training_manifest_sha256']==config['dataset_manifest_sha256']
    for name,h in manifest['artifacts'].items():
        assert file_hash(transfer/name)==h,name
    examples=[json.loads(l) for l in (transfer/'examples.jsonl').read_text().splitlines()]
    data=dict(vectors=torch.load(transfer/'vectors.pt',map_location='cpu',weights_only=True)['vectors'],slot_rms=data['slot_rms'])
    assert data['vectors'].shape==(len(examples),4,4096) and torch.isfinite(data['vectors']).all()
    adapter={p.name:file_hash(p) for p in (run/'adapter').glob('adapter*') if p.is_file()}
    settings=dict(training_manifest_sha256=config['dataset_manifest_sha256'],transfer_manifest_sha256=file_hash(transfer/'manifest.json'),
        adapter_sha256=adapter,batch_size=batch_size,transfer_source_sha256=file_hash(Path(__file__)),
        cache_predictions_sha256=file_hash(cache/'predictions.jsonl') if cache is not None else None)
    if output.exists():
        if json.loads((output/'evaluation-config.json').read_text())!=settings:
            raise ValueError('Evaluation settings changed; use a new folder')
    else:
        output.mkdir(parents=True)
        (output/'evaluation-config.json').write_text(json.dumps(settings,indent=2)+'\n')
    path=output/'predictions.jsonl'
    records=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    lookup={r['id']:r for r in examples}
    assert len(records)==len({r['id'] for r in records})
    def check(r):
        source=lookup[r['id']]
        assert source['harmful']==r['label']
        assert all(source[k]==r[k] for k in ['run','step','probe_id'])
        assert r['condition']=='actual'
    for r in records:
        check(r)
        assert r['split']==lookup[r['id']]['split']
    done={r['id'] for r in records}
    reused=0
    if cache is not None:
        cached_config=json.loads((cache/'evaluation-config.json').read_text())
        if cached_config['adapter_sha256']!=adapter or cached_config['dataset_manifest_sha256']!=config['dataset_manifest_sha256'] or cached_config['batch_size']!=batch_size:
            raise ValueError('Cache is from another adapter, dataset or batch size')
        for r in map(json.loads,(cache/'predictions.jsonl').read_text().splitlines()):
            if r['condition']!='actual' or r['id'] not in lookup or r['id'] in done:
                continue
            check(r)
            new=dict(r,split=lookup[r['id']]['split'],score_source='previous_fixed_adapter_evaluation')
            with path.open('a') as handle:
                handle.write(json.dumps(new)+'\n')
            records.append(new)
            done.add(r['id'])
            reused+=1
    plan_path=output/'batch-plan.json'
    if plan_path.exists():
        assert file_hash(plan_path)==(output/'batch-plan.sha256').read_text().strip()
        plan=json.loads(plan_path.read_text())
        flat=[i for chunk in plan for i in chunk]
        reused_ids={r['id'] for r in records if r['score_source']=='previous_fixed_adapter_evaluation'}
        assert len(flat)==len(set(flat)) and set(flat)==set(lookup)-reused_ids
    else:
        plan=[[r['id'] for r in chunk] for chunk in batches([r for r in examples if r['id'] not in done],batch_size)]
        plan_path.write_text(json.dumps(plan,indent=2)+'\n')
        (output/'batch-plan.sha256').write_text(file_hash(plan_path)+'\n')
    remaining=[[lookup[i] for i in ids] for ids in plan if any(i not in done for i in ids)]
    if remaining:
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
        prepared=load_model_for_finetuning(dataset/'decoder-model.yaml')
        model,tokenizer=prepared.model,prepared.tokenizer
        check_model(model,original)
        loaded=set_peft_model_state_dict(model,load_peft_weights(str(run/'adapter'),device='cpu'))
        if loaded.unexpected_keys or any('lora_' in k for k in loaded.missing_keys):
            raise ValueError('Adapter did not load completely')
        model.gradient_checkpointing_disable()
        prefix,positions,labels=encode(tokenizer)
        hook=DirectInjection(model)
        try:
            for chunk in tqdm(remaining,desc='New transfer decisions'):
                scored=infer(model,hook,tokenizer,chunk,data,prefix,positions,labels,batch_size=batch_size)
                for r in scored:
                    if r['id'] in done:
                        continue
                    r['score_source']='new_fixed_adapter_evaluation'
                    with path.open('a') as handle:
                        handle.write(json.dumps(r)+'\n')
                    records.append(r)
                    done.add(r['id'])
        finally:
            hook.close()
    assert done==set(lookup)
    report={}
    prior=sum(r['harmful'] for r in original_rows if r['split']=='train')/sum(r['split']=='train' for r in original_rows)
    for split in sorted({r['split'] for r in records}):
        pool=[r for r in records if r['split']==split]
        result=metrics(pool)
        result['per_run']={name:metrics([r for r in pool if r['run']==name]) for name in sorted({r['run'] for r in pool})}
        result['per_step']={str(s):metrics([r for r in pool if r['step']==s]) for s in sorted({r['step'] for r in pool})}
        result['per_probe_class']={g:metrics([r for r in pool if lookup[r['id']]['group']==g]) for g in sorted({lookup[r['id']]['group'] for r in pool})}
        baselines={}
        for name,score in [('always_non_harmful',0.),('always_harmful',1.),('training_frequency',prior)]:
            p=max(1e-12,min(1-1e-12,score))
            values=[dict(label=r['label'],predicted=int(score>=.5),score=score,
                         binary_nll=-math.log(p if r['label'] else 1-p)) for r in pool]
            baselines[name]=metrics(values,language_output=False)
        result['constant_baselines']=baselines
        report[split]=result
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'reuse-audit.json').write_text(json.dumps(dict(imported_this_call=reused,total=len(records),
        previously_scored=sum(r['score_source']=='previous_fixed_adapter_evaluation' for r in records)),indent=2)+'\n')
    summarize(output)


def summarize(output):
    report=json.loads((output/'report.json').read_text())
    names=['training_reference','new_prompts_known_models','known_prompts_new_model','new_prompts_new_model',
           'validation_reference','selection_model_new_prompts']
    lines=['# Fixed-decoder transfer comparison','',
        'Same trained weights, original training normalization and fixed 0.5 threshold. No decoder retraining.',
        'Training/validation references are development diagnostics. New-model tests use self_harm; new-prompt tests use the withheld monitoring prompt pool, not the separate evaluation file.',
        '','| Cohort | Cases | Accuracy | Balanced accuracy | Harmful recall | Always-negative accuracy |','|---|---:|---:|---:|---:|---:|']
    fmt=lambda v:'undefined' if v is None else f'{v:.1%}'
    for name in names:
        m=report[name]
        lines.append(f"| {name} | {m['examples']} | {fmt(m['accuracy'])} | {fmt(m['balanced_accuracy'])} | {fmt(m['harmful_recall'])} | {fmt(m['constant_baselines']['always_non_harmful']['accuracy'])} |")
    lines+=['','See report.json for per-model, per-step and probe-class breakdowns. Some cohorts/models can have no positive labels, making balanced accuracy undefined.',
            'These are inspected development comparisons. Prompt pools differ in content and related checkpoints are correlated; changes are not causal estimates.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
    for ax,key in zip(axes,['balanced_accuracy','harmful_recall']):
        values=[report[n][key] for n in names]
        ax.barh(names,[v or 0 for v in values],color=['#94a4b5','#2166ac','#2166ac','#2166ac','#94a4b5','#94a4b5'])
        for i,v in enumerate(values):
            ax.text((v or 0)+.01,i,fmt(v),va='center')
        ax.set(xlim=(0,1.15),title=key.replace('_',' '))
    fig.suptitle('Fixed decoder · separate prompt and model transfer · no retraining')
    fig.savefig(output/'comparison.png',dpi=170)
    plt.close(fig)
