"""Development-only all-layer attribution, then fixed-mask comparison generation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from behaviour_prediction.monitor.dynamic_gradient_patching import step_context
from behaviour_prediction.monitor.gradient_patching import endpoint
from behaviour_prediction.monitor.multilayer_patching import all_layer_scores, sequence_states, multilayer_generate, select_multilayer
from behaviour_prediction.monitor.shared_mask import identity, tensor_identity
from behaviour_prediction.monitor.responses import checkpoint_models, checkpoint_plan
from behaviour_prediction.monitoring.activations import ProbeRecord, encode_probe
from run_activation_patch_pilot import export_reviews
from run_dynamic_gradient_activation_patch import atomic_json, reference_input
from run_gradient_activation_patch import digest

DEFAULT = Path('configs/activation_patch_multilayer.json')


def encode(model, tokenizer, probe, plan):
    return encode_probe(ProbeRecord(**{k:probe[k] for k in ('id','prompt','group','category')}),
        tokenizer=tokenizer, device=next(model.parameters()).device, max_length=plan['max_input_length'])


def models(plan):
    root=Path(plan['source_run']); metadata, checkpoints=checkpoint_plan(root,[0,20])
    if metadata['mode']!='lora' or metadata['model']!=plan['expected_model_config']: raise ValueError('Model config differs')
    checkpoint=next(p for step,p in checkpoints if step==20)
    for name,sha in plan['expected_donor_files'].items():
        if hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()!=sha: raise ValueError('Checkpoint changed')
    for item in checkpoint_models(root,[20],skip_steps=(0,)):
        _,model,tokenizer,_=item
        if model.active_adapters!=['default']: raise ValueError('Wrong adapter')
        sentinel=plan['sentinel']; encoded=encode(model,tokenizer,sentinel['probe'],plan)
        if encoded['input_ids'].cpu().tolist()!=sentinel['prompt_token_ids']: raise ValueError('Tokenization changed')
        for step,name in [(0,'baseline'),(20,'finetuned')]:
            with step_context(model,step): states,_=endpoint(model,encoded,31)
            if tensor_identity(states[31])!=sentinel[f'{name}_sha256']: raise ValueError('Endpoint changed')
        yield model,tokenizer


def initialize(folder, plan):
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'development_plan.json'
    if path.exists() and json.loads(path.read_text())!=plan: raise ValueError('Plan changed; choose a new output folder')
    if not path.exists(): atomic_json(path,plan)


def select(folder, plan):
    initialize(folder,plan)
    cache_path=folder/'development_scores.pt'
    cache=torch.load(cache_path,map_location='cpu',weights_only=True) if cache_path.exists() else {}
    expected={f"{d}:{p['id']}" for p in plan['development_probes'] for d in plan['directions']}
    if not set(cache)<=expected: raise ValueError('Unexpected development scores')
    if set(cache)!=expected:
        for model,tokenizer in models(plan):
            for p in plan['development_probes']:
                encoded=encode(model,tokenizer,p,plan)
                for d in plan['directions']:
                    key=f"{d}:{p['id']}"
                    if key in cache: continue
                    recipient=20 if d=='restore' else 0; donor=20-recipient
                    reference=plan['references'][key]
                    teacher=reference_input(encoded,reference['token_ids'])
                    if teacher['input_ids'].shape[1]>plan['expected_model_config']['model']['max_sequence_length']:
                        raise ValueError('Reference exceeds context')
                    if encoded['input_ids'].cpu().tolist()!=reference['prompt_token_ids']:
                        raise ValueError('Development tokenization changed')
                    with step_context(model,donor): states,logits=sequence_states(model,teacher,encoded['input_ids'].shape[1])
                    with step_context(model,recipient): result=all_layer_scores(model,teacher,encoded['input_ids'].shape[1],states,logits)
                    if result['scores'].shape!=(32,4096): raise ValueError('Expected Llama3 32 blocks and 4096 dimensions')
                    cache[key]=result
                    tmp=cache_path.with_suffix('.tmp');torch.save(cache,tmp);tmp.replace(cache_path)
                    del states,logits,result
                    print(f'{len(cache)}/{len(expected)} development backward passes saved',flush=True)
    # Normalize across the whole model; each development prompt/direction has equal weight.
    vectors=[cache[key]['scores'].double() for key in sorted(expected)]
    aggregate=torch.stack([v/v.abs().sum().clamp_min(1e-12) for v in vectors]).mean(0)
    selections=select_multilayer(aggregate,plan['budgets'],plan['random_seeds'])
    # Layer31 comparator preserves the previous layer-only normalization/ranking.
    last=torch.stack([v[-1]/v[-1].abs().sum().clamp_min(1e-12) for v in vectors]).mean(0)
    layer_only=select_multilayer(aggregate.clone().index_copy(0,torch.tensor([31]),last[None]),plan['budgets'],plan['random_seeds'])
    selections.update({k:v for k,v in layer_only.items() if k.startswith('layer31_')})
    artifact=dict(plan_id=plan['plan_id'],selected_indices=selections,aggregate_scores=aggregate.tolist(),
        layer31_scores=last.tolist(),width=4096,depth=32,
        score_cache_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest())
    artifact['mask_id']=identity(artifact)
    path=folder/'mask.json'
    if path.exists() and json.loads(path.read_text())!=artifact: raise ValueError('Frozen selection changed')
    atomic_json(path,artifact)
    plot_masks(folder,artifact)
    print(f'Frozen {len(selections)} masks. No evaluation answers or labels used.',flush=True)


def validate_artifact(artifact,plan):
    if identity(artifact)!=artifact['mask_id'] or artifact['plan_id']!=plan['plan_id']: raise ValueError('Invalid frozen mask')
    values=torch.tensor(artifact['aggregate_scores'],dtype=torch.float64)
    expected=select_multilayer(values,plan['budgets'],plan['random_seeds'])
    values[-1]=torch.tensor(artifact['layer31_scores'],dtype=torch.float64)
    expected.update({k:v for k,v in select_multilayer(values,plan['budgets'],plan['random_seeds']).items() if k.startswith('layer31_')})
    if expected!=artifact['selected_indices']: raise ValueError('Frozen indices do not match ranking')
    return expected


def plot_masks(folder,artifact):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[k for k in artifact['selected_indices'] if '_random' not in k]
    matrix=[[len(artifact['selected_indices'][name].get(str(i),[])) for i in range(32)] for name in names]
    fig,ax=plt.subplots(figsize=(13,7)); im=ax.imshow(matrix,aspect='auto',cmap='viridis')
    ax.set(xticks=list(range(32)),yticks=list(range(len(names))),yticklabels=names,
           xlabel='Transformer block (zero based)',title='Frozen selected-coordinate counts by block')
    fig.colorbar(im,ax=ax,label='Coordinates selected');fig.tight_layout();fig.savefig(folder/'selection_by_layer.png',dpi=170);plt.close(fig)
    with (folder/'selection_by_layer.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['condition',*[f'layer_{i}' for i in range(32)]])
        w.writerows([name,*row] for name,row in zip(names,matrix))


def generation_rows(m, selected, tasks, model, tokenizer, parent_alive=lambda:True):
    """Sequential inference within ONE process/model; no shared hooks or caches."""
    masks={c:{int(layer):torch.zeros(4096,dtype=torch.bool) for layer in sites} for c,sites in selected.items()}
    for c,sites in selected.items():
        for layer,indices in sites.items(): masks[c][int(layer)][indices]=True
    full={i:torch.ones(4096,dtype=torch.bool) for i in range(32)}
    def row(p,d,c,tokens,ids):
        return dict(pilot_id=m['pilot_id'],mask_id=m['mask_id'],source_run=m['run_name'],
            direction=d,condition=c,run=f"{m['run_name']}_multilayer_{d}__{c}",step=20 if d=='restore' else 0,
            probe_id=p['id'],group=p['group'],category=p['category'],role=p['role'],prompt=p['prompt'],
            response=tokenizer.decode(tokens,skip_special_tokens=True),token_limit_reached=len(tokens)>=m['max_new_tokens'],
            human_harmful='',review_notes='',prompt_token_ids=ids,generated_token_ids=tokens,selected_indices=selected.get(c,{}))
    # No gradients even when called outside the subprocess wrapper.
    with torch.no_grad():
        for task in tasks:
            if not parent_alive():return
            p=task['probe'];saved={(r['direction'],r['condition']):r for r in task['saved']}
            encoded=encode(model,tokenizer,p,m);ids=encoded['input_ids'].cpu().tolist();controls={}
            for step,d in [(0,'transfer'),(20,'restore')]:
                key=d,'unpatched'
                if key in saved:
                    if saved[key]['prompt_token_ids']!=ids:raise ValueError('Control tokenization changed')
                    tokens=saved[key]['generated_token_ids']
                else:
                    tokens=multilayer_generate(model,encoded,recipient_step=step,donor_step=20-step,limit=m['max_new_tokens'])
                    new=row(p,d,'unpatched',tokens,ids);saved[key]=new;yield new
                controls[step]=tokens
            for d in m['directions']:
                recipient=20 if d=='restore' else 0
                for c in m['conditions']:
                    if not parent_alive():return
                    if (d,c) in saved:continue
                    donor=recipient if c=='self_dynamic' else 20-recipient
                    tokens=multilayer_generate(model,encoded,recipient_step=recipient,donor_step=donor,
                        limit=m['max_new_tokens'],masks=full if c in ('self_dynamic','full_dynamic') else masks[c])
                    if c in ('self_dynamic','full_dynamic') and tokens!=controls[donor]:raise ValueError('Self/full control failed')
                    yield row(p,d,c,tokens,ids)


def validate_generation_row(row, m, selected, records, *, allow_duplicate=False):
    p=next((p for p in m['probes'] if p['id']==row['probe_id']),None)
    d=row['direction'];c=row['condition'];key=row['probe_id'],d,c
    if p is None or d not in m['directions'] or c not in m['conditions']:raise ValueError('Unexpected generation record')
    if row['pilot_id']!=m['pilot_id'] or row['mask_id']!=m['mask_id']:raise ValueError('Generation identity differs')
    if row['selected_indices']!=selected.get(c,{}):raise ValueError('Saved mask differs')
    if any(row[k]!=p[k] for k in ('prompt','group','category','role')):raise ValueError('Saved prompt metadata differs')
    if row['step']!=(20 if d=='restore' else 0) or row['run']!=f"{m['run_name']}_multilayer_{d}__{c}":
        raise ValueError('Saved run/checkpoint differs')
    tokens=row['generated_token_ids']
    if not isinstance(tokens,list) or not 0<len(tokens)<=m['max_new_tokens'] or any(type(t) is not int or t<0 for t in tokens):
        raise ValueError('Invalid generated token IDs')
    if row['token_limit_reached']!=(len(tokens)>=m['max_new_tokens']):raise ValueError('Token-limit flag differs')
    other='transfer' if d=='restore' else 'restore'
    for direction in m['directions']:
        control=records.get((p['id'],direction,'unpatched'))
        if control and control['prompt_token_ids']!=row['prompt_token_ids']:raise ValueError('Saved prompt tokenization differs')
    comparison=records.get((p['id'],d if c=='self_dynamic' else other,'unpatched'))
    if c in ('self_dynamic','full_dynamic') and (not comparison or comparison['generated_token_ids']!=tokens):
        raise ValueError('Saved self/full control failed')
    if key in records:
        if not allow_duplicate or records[key]!=row:raise ValueError('Duplicate/conflicting generation record')
        return False
    return True


def generate(folder,plan,workers=1):
    from multilayer_generation_workers import generation_lock
    if not 1<=workers<=4:raise ValueError('Choose between 1 and 4 generation workers')
    with generation_lock(folder) as lock_fd:
        return _generate_locked(folder,plan,workers,lock_fd)


def _generate_locked(folder,plan,workers,lock_fd):
    import os
    import time
    from multilayer_generation_workers import recover_worker_journals, parallel_generate, durable_row_write
    initialize(folder,plan)
    artifact=json.loads((folder/'mask.json').read_text());selected=validate_artifact(artifact,plan)
    conditions=['unpatched','self_dynamic','full_dynamic',*selected]
    m=dict(plan,probes=plan['evaluation_probes'],conditions=conditions,mask_id=artifact['mask_id'],layers=list(range(32)))
    m['pilot_id']=digest(m)
    if (folder/'pilot.json').exists() and json.loads((folder/'pilot.json').read_text())!=m:raise ValueError('Generation plan changed')
    atomic_json(folder/'pilot.json',m)
    expected={(p['id'],d,c) for p in m['probes'] for d in m['directions'] for c in conditions}
    records={};log=folder/'generated.jsonl'
    if log.exists():
        content=log.read_bytes();lines=content.splitlines(keepends=True)
        for i,line in enumerate(lines):
            try:row=json.loads(line)
            except json.JSONDecodeError:
                if i!=len(lines)-1 or line.endswith(b'\n'):raise
                backup=folder/'generated.interrupted-backup.jsonl'
                if not backup.exists():backup.write_bytes(content)
                log.write_bytes(b''.join(lines[:i]));break
            validate_generation_row(row,m,selected,records)
            records[(row['probe_id'],row['direction'],row['condition'])]=row
        else:
            if content and not content.endswith(b'\n'):log.write_bytes(content+b'\n')
    last_export=0.;last_count=-1
    def refresh(force=False):
        nonlocal last_export,last_count
        if not force and (len(records)==last_count or time.monotonic()-last_export<5):return
        export_reviews(folder,records)
        atomic_json(folder/'run_report.json',dict(completed=len(records),planned=len(expected),
            complete=set(records)==expected,workers_requested=workers,gradients_used=False))
        last_export=time.monotonic();last_count=len(records)
    # The coordinator is the only writer of the canonical journal and CSVs.
    with log.open('a') as handle:
        def accept(row):
            if not validate_generation_row(row,m,selected,records,allow_duplicate=True):return False
            with durable_row_write():
                handle.write(json.dumps(row)+'\n');handle.flush();os.fsync(handle.fileno())
                records[(row['probe_id'],row['direction'],row['condition'])]=row
            print(f"{len(records)}/{len(expected)} saved: {row['direction']}/{row['probe_id']}/{row['condition']}",flush=True)
            return True
        recovered=recover_worker_journals(folder,accept)
        if recovered:print(f'Recovered {recovered} completed answers from previous worker journals.',flush=True)
    from reuse_multilayer_activation_patch import seed_answers
    seed_answers(folder,m,selected,records)
    # Validate reused rows as well as new rows; preserve the same manifest and mask.
    checked={}
    for row in records.values():
        validate_generation_row(row,m,selected,checked);checked[(row['probe_id'],row['direction'],row['condition'])]=row
    refresh(force=True)
    print(f'Resuming with {len(records)}/{len(expected)} already saved; {len(expected)-len(records)} remain.',flush=True)
    if set(records)==expected:return
    tasks=[dict(probe=p,saved=[r for (id,_,_),r in records.items() if id==p['id']])
        for p in m['probes'] if any((p['id'],d,c) not in records for d in m['directions'] for c in conditions)]
    with log.open('a') as handle:
        def accept(row):
            if not validate_generation_row(row,m,selected,records,allow_duplicate=True):return False
            with durable_row_write():
                handle.write(json.dumps(row)+'\n');handle.flush();os.fsync(handle.fileno())
                records[(row['probe_id'],row['direction'],row['condition'])]=row
            print(f"{len(records)}/{len(expected)} saved: {row['direction']}/{row['probe_id']}/{row['condition']}",flush=True)
            return True
        try:
            if workers>1:
                parallel_generate(folder,m,selected,tasks,workers,lock_fd,accept,refresh)
            else:
                for model,tokenizer in models(plan):
                    for row in generation_rows(m,selected,tasks,model,tokenizer):
                        accept(row);refresh()
            if set(records)!=expected:raise RuntimeError('Generation ended before all answers were saved; rerun to resume')
        finally:
            refresh(force=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['select','generate'])
    p.add_argument('--output-dir',type=Path,default=Path('outputs/activation-patch-multilayer'))
    p.add_argument('--config',type=Path,default=DEFAULT)
    p.add_argument('--workers',type=int,choices=range(1,5),default=1,help='Independent GPU generation processes (1–4); resume the same output folder')
    args=p.parse_args()
    plan=json.loads(args.config.read_text())
    if identity(plan,'plan_id')!=plan['plan_id']: raise ValueError('Invalid plan hash')
    if args.action=='select':
        from multilayer_generation_workers import generation_lock
        with generation_lock(args.output_dir):select(args.output_dir,plan)
    else:generate(args.output_dir,plan,workers=args.workers)

if __name__=='__main__':main()
