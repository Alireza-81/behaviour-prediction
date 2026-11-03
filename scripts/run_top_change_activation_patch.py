"""Patch largest-changing coordinates in both directions with random-budget controls.

Three sequential model loads: step0 capture, step20 capture/restore experiment,
step0 transfer experiment. Keeps only one model on GPU at a time. No training,
judge calls, output-dependent coordinate selection or generated-token patching.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import torch
from behaviour_prediction.monitor.responses import checkpoint_models,checkpoint_plan
from behaviour_prediction.monitoring.activations import ProbeRecord,encode_probe
from behaviour_prediction.monitor.patching import last_token_donors,select_changed_coordinates
from run_activation_patch_pilot import generate,export_reviews


def make_masks(base,tuned,manifest):
    masks={};audit=[]
    for p in manifest['probes']:
        id=p['id'];masks[id]={}
        for percent in manifest['budgets_percent']:
            for method in ('top','random'):
                condition=f'{method}_{percent:02d}pct';masks[id][condition]={}
                for layer in manifest['layers']:
                    a=base[id][layer];b=tuned[id][layer];count=math.ceil(a.numel()*percent/100)
                    # Prompt/layer-specific random ordering, identical in both directions.
                    seed=int(hashlib.sha256(f"{manifest['random_seed']}:{id}:{layer}".encode()).hexdigest()[:8],16)
                    mask=select_changed_coordinates(a,b,count,method='top_change' if method=='top' else 'random',seed=seed)
                    masks[id][condition][layer]=mask
                    delta=(b.float()-a.float()).cpu()
                    audit.append(dict(probe_id=id,layer=layer,condition=condition,selected_count=count,total_count=a.numel(),
                        selected_indices=mask.nonzero().flatten().tolist(),selected_delta_l2=delta[mask].norm().item(),total_delta_l2=delta.norm().item()))
    return masks,audit


def run(folder,manifest_path):
    requested=json.loads(manifest_path.read_text())
    folder.mkdir(parents=True,exist_ok=True)
    plan=folder/'pilot.json'
    if not plan.exists():shutil.copy2(manifest_path,plan)
    m=json.loads(plan.read_text())
    digest=hashlib.sha256(json.dumps({k:v for k,v in m.items() if k!='pilot_id'},sort_keys=True).encode()).hexdigest()
    if digest!=m['pilot_id'] or digest!=requested['pilot_id']:raise ValueError('Changed plan: use a fresh folder')
    expected={(p['id'],direction,c) for p in m['probes'] for direction in m['directions'] for c in m['conditions']}
    records={};log=folder/'generated.jsonl'
    if log.exists():
        contents=log.read_bytes();lines=contents.splitlines(keepends=True)
        for i,line in enumerate(lines):
            try:r=json.loads(line)
            except json.JSONDecodeError:
                if i==len(lines)-1 and not line.endswith(b'\n'):
                    backup=folder/'generated.interrupted-backup.jsonl'
                    if not backup.exists():backup.write_bytes(contents)
                    log.write_bytes(b''.join(lines[:i]));break
                raise
            key=r['probe_id'],r['direction'],r['condition']
            if key not in expected or key in records or r['pilot_id']!=digest:raise ValueError('Incompatible journal record')
            records[key]=r
    def exports():
        export_reviews(folder,records)
    def report():
        pairs=[(p['id'],d) for p in m['probes'] for d in m['directions'] if (p['id'],d,'self_patch') in records and (p['id'],d,'unpatched') in records]
        matches=sum(records[(id,d,'self_patch')]['generated_token_ids']==records[(id,d,'unpatched')]['generated_token_ids'] for id,d in pairs)
        result=dict(completed=len(records),planned=len(expected),self_patch_pairs_checked=len(pairs),self_patch_exact_matches=matches,
            complete=set(records)==expected,label_status='Fresh output labels required; coordinate magnitude is not a harmfulness score')
        (folder/'run_report.json').write_text(json.dumps(result,indent=2)+'\n');return result
    exports()
    if set(records)==expected:print(json.dumps(report(),indent=2));return
    root=Path(m['source_run']);metadata,checkpoints=checkpoint_plan(root,[0,20])
    if metadata['mode']!='lora' or metadata['model']!=m['expected_model_config']:raise ValueError('Model configuration differs')
    checkpoint=next(path for step,path in checkpoints if step==20)
    for name,sha in m['expected_donor_files'].items():
        if hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()!=sha:raise ValueError('Step20 checkpoint differs from the previous experiments')
    base={};tuned={};tokens={};masks={}
    def encode(model,tokenizer,p):
        probe=ProbeRecord(**{k:p[k] for k in ('id','prompt','group','category')})
        encoded=encode_probe(probe,tokenizer=tokenizer,device=next(model.parameters()).device,max_length=m['max_input_length'])
        ids=encoded['input_ids'].cpu().tolist()
        if p['id'] in tokens and ids!=tokens[p['id']]:raise ValueError('Endpoint tokenization mismatch')
        tokens[p['id']]=ids
        return encoded
    def append(handle,p,direction,condition,generated,text,selected=None):
        id=p['id'];step=20 if direction=='restore' else 0
        row=dict(pilot_id=digest,source_run=m['run_name'],direction=direction,condition=condition,
            run=f"{m['run_name']}_topchange_{direction}__{condition}",step=step,probe_id=id,group=p['group'],category=p['category'],role=p['role'],
            prompt=p['prompt'],response=text,token_limit_reached=len(generated)>=m['max_new_tokens'],human_harmful='',review_notes='',
            prompt_token_ids=tokens[id],generated_token_ids=generated,patched_layers=m['layers'] if condition!='unpatched' else [],
            selected_indices={str(layer):mask.nonzero().flatten().tolist() for layer,mask in selected.items()} if selected is not None else {},patch_scope=m['patch_scope'])
        handle.write(json.dumps(row)+'\n');handle.flush();records[(id,direction,condition)]=row
        exports();report()
        print(f"{len(records)}/{len(expected)} saved: {direction} / {id} / {condition}",flush=True)
    def conditions(handle,model,tokenizer,p,direction,encoded,donors,own):
        for c in m['conditions']:
            if (p['id'],direction,c) in records:
                if records[(p['id'],direction,c)]['prompt_token_ids']!=tokens[p['id']]:raise ValueError('Tokenization changed on resume')
                continue
            replacement=None;selected=None
            if c=='self_patch':replacement=own
            elif c=='full_patch':replacement=donors
            elif c!='unpatched':replacement=donors;selected=masks[p['id']][c]
            ids,text=generate(model,tokenizer,encoded,m['max_new_tokens'],replacement,selected)
            if c=='self_patch' and ids!=records[(p['id'],direction,'unpatched')]['generated_token_ids']:
                raise RuntimeError('Self-patch changed recipient tokens; investigate before interpretation')
            append(handle,p,direction,c,ids,text,selected)
    # Capture step0 endpoint states, but wait for step20 before selecting masks.
    for step,model,tokenizer,_ in checkpoint_models(root,[0]):
        for p in m['probes']:
            encoded=encode(model,tokenizer,p);base[p['id']]=last_token_donors(model,encoded,m['layers'])
        del model
    with log.open('a') as handle:
        for step,model,tokenizer,_ in checkpoint_models(root,[20],skip_steps=(0,)):
            for p in m['probes']:
                encoded=encode(model,tokenizer,p);tuned[p['id']]=last_token_donors(model,encoded,m['layers'])
            masks,audit=make_masks(base,tuned,m)
            path=folder/'coordinate_selection.json'
            if path.exists() and json.loads(path.read_text())!=audit:raise ValueError('Coordinate ranking changed on resume; do not mix selections')
            if not path.exists():path.write_text(json.dumps(audit,indent=2)+'\n')
            # Cache source tensors for reproducible local visual inspection.
            state_path=folder/'endpoint_activations.pt'
            if not state_path.exists():torch.save(dict(baseline=base,finetuned=tuned,prompt_token_ids=tokens),state_path)
            for p in m['probes']:
                encoded=encode(model,tokenizer,p)
                conditions(handle,model,tokenizer,p,'restore',encoded,base[p['id']],tuned[p['id']])
            del model
        for step,model,tokenizer,_ in checkpoint_models(root,[0]):
            for p in m['probes']:
                encoded=encode(model,tokenizer,p)
                conditions(handle,model,tokenizer,p,'transfer',encoded,tuned[p['id']],base[p['id']])
            del model
    print(json.dumps(report(),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/activation-patch-top-change'))
    p.add_argument('--manifest',type=Path,default=Path('configs/activation_patch_top_change.json'))
    args=p.parse_args();run(args.output_dir,args.manifest)
