"""Reverse the patch: donor step20 activations into step-zero recipient.

One GPU model is loaded at a time. Donor vectors are cached on CPU before the
baseline recipient is loaded. Only the initial last prompt-token state is
replaced; no new weights, training, or API calls.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import torch
from behaviour_prediction.monitor.responses import checkpoint_models,checkpoint_plan
from behaviour_prediction.monitoring.activations import ProbeRecord,encode_probe
from behaviour_prediction.monitor.patching import last_token_donors
from run_activation_patch_pilot import generate,export_reviews


def save_report(folder,manifest,records):
    complete=len(records)==len(manifest['probes'])*len(manifest['conditions'])
    pairs=[p['id'] for p in manifest['probes'] if (p['id'],'self_patch_both') in records and (p['id'],'unpatched_step0') in records]
    matches=sum(records[(id,'self_patch_both')]['generated_token_ids']==records[(id,'unpatched_step0')]['generated_token_ids'] for id in pairs)
    report=dict(completed=len(records),planned=len(manifest['probes'])*len(manifest['conditions']),direction=manifest['direction'],
        donor_step=20,recipient_step=0,self_patch_exact_matches=matches,self_patch_pairs_checked=len(pairs),
        self_patch_matches_unpatched=matches==len(pairs) if pairs else None,complete=complete,
        label_status='Fresh answers require labeling; no behavioural-transfer claim yet')
    (folder/'run_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def run(folder,manifest_source=Path('configs/activation_patch_reverse_30.json')):
    source=json.loads(manifest_source.read_text())
    folder.mkdir(parents=True,exist_ok=True)
    if not (folder/'pilot.json').exists():shutil.copy2(manifest_source,folder/'pilot.json')
    manifest=json.loads((folder/'pilot.json').read_text())
    digest=hashlib.sha256(json.dumps({k:v for k,v in manifest.items() if k!='pilot_id'},sort_keys=True).encode()).hexdigest()
    if digest!=manifest['pilot_id'] or source['pilot_id']!=digest:raise ValueError('Manifest mismatch: use the same plan when resuming')
    assert manifest['direction']=='finetuned_to_baseline' and manifest['donor_step']==20 and manifest['recipient_step']==0
    expected={(p['id'],c) for p in manifest['probes'] for c in manifest['conditions']}
    log=folder/'generated.jsonl';records={}
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
            key=r['probe_id'],r['condition']
            if r['pilot_id']!=digest or key in records or key not in expected:raise ValueError('Incompatible saved results')
            records[key]=r
    export_reviews(folder,records)
    if set(records)==expected:
        print(json.dumps(save_report(folder,manifest,records),indent=2));return
    run_path=Path(manifest['source_run'])
    metadata,plan=checkpoint_plan(run_path,[0,20])
    if metadata['mode']!='lora' or metadata['model']!=manifest['expected_model_config']:raise ValueError('Incompatible donor model configuration')
    checkpoint=next(path for step,path in plan if step==20)
    for name,expected_sha in manifest['expected_donor_files'].items():
        if hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()!=expected_sha:
            raise ValueError(f'Donor {name} differs from the forward experiment; retain the same checkpoint')
    donors={};prompt_ids={}
    def record(handle,p,step,condition,tokens,text,ids,layers):
        row=dict(pilot_id=digest,source_run=manifest['run_name'],direction=manifest['direction'],donor_step=20,recipient_step=0,
            condition=condition,run=f"{manifest['run_name']}_reverse__{condition}",step=step,
            probe_id=p['id'],group=p['group'],category=p['category'],role=p['role'],prompt=p['prompt'],response=text,
            token_limit_reached=len(tokens)>=manifest['max_new_tokens'],human_harmful='',review_notes='',
            generated_token_ids=tokens,prompt_token_ids=ids,patched_layers=layers,patch_scope=manifest['patch_scope'])
        handle.write(json.dumps(row)+'\n');handle.flush();records[(p['id'],condition)]=row
        export_reviews(folder,records)
        save_report(folder,manifest,records)
        print(f"{len(records)}/{len(expected)} saved: {p['id']} / {condition}",flush=True)
    with log.open('a') as handle:
        # checkpoint_models always adds zero; explicitly skip it for donor capture.
        for step,model,tokenizer,_ in checkpoint_models(run_path,[20],skip_steps=(0,)):
            assert step==20
            for p in manifest['probes']:
                probe=ProbeRecord(**{k:p[k] for k in ('id','prompt','group','category')})
                encoded=encode_probe(probe,tokenizer=tokenizer,device=next(model.parameters()).device,max_length=manifest['max_input_length'])
                ids=encoded['input_ids'].cpu().tolist();prompt_ids[p['id']]=ids
                donors[p['id']]=last_token_donors(model,encoded,manifest['layers'])
                key=p['id'],'donor_step20'
                if key in records:
                    if records[key]['prompt_token_ids']!=ids:raise ValueError('Donor tokenization changed on resume')
                else:
                    tokens,text=generate(model,tokenizer,encoded,manifest['max_new_tokens'])
                    record(handle,p,20,'donor_step20',tokens,text,ids,[])
            del model
        # This is a fresh step-zero recipient, not the step20 model.
        for step,model,tokenizer,_ in checkpoint_models(run_path,[0]):
            assert step==0
            for p in manifest['probes']:
                probe=ProbeRecord(**{k:p[k] for k in ('id','prompt','group','category')})
                encoded=encode_probe(probe,tokenizer=tokenizer,device=next(model.parameters()).device,max_length=manifest['max_input_length'])
                ids=encoded['input_ids'].cpu().tolist()
                if ids!=prompt_ids[p['id']]:raise ValueError('Recipient and donor prompt tokens differ')
                for condition in manifest['conditions'][1:]:
                    if (p['id'],condition) in records:continue
                    replacement=None
                    if condition=='self_patch_both':replacement=last_token_donors(model,encoded,manifest['layers'])
                    elif condition=='patch_layer15':replacement={15:donors[p['id']][15]}
                    elif condition=='patch_layer31':replacement={31:donors[p['id']][31]}
                    elif condition=='patch_both':replacement=donors[p['id']]
                    tokens,text=generate(model,tokenizer,encoded,manifest['max_new_tokens'],replacement)
                    if condition=='self_patch_both' and tokens!=records[(p['id'],'unpatched_step0')]['generated_token_ids']:
                        raise RuntimeError('Step-zero self-patch changed the output; stop and investigate')
                    record(handle,p,0,condition,tokens,text,ids,sorted(replacement) if replacement else [])
            del model
    report=save_report(folder,manifest,records)
    (folder/'environment.json').write_text(json.dumps({p:importlib.metadata.version(p) for p in ('torch','transformers','peft')},indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-dir',type=Path,default=Path('outputs/activation-patch-reverse-30'))
    p.add_argument('--manifest',type=Path,default=Path('configs/activation_patch_reverse_30.json'))
    args=p.parse_args();run(args.pilot_dir,args.manifest)
