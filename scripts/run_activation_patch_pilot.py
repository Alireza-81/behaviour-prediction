"""Generate the frozen first activation-patching pilot on a GPU; supports resume."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import torch
from behaviour_prediction.monitor.responses import checkpoint_models, checkpoint_plan, FIELDS
from behaviour_prediction.monitoring.activations import ProbeRecord, encode_probe
from behaviour_prediction.monitor.patching import PrefillLastTokenPatch, last_token_donors


def export_reviews(folder,records):
    reviews=folder/'reviews';reviews.mkdir(exist_ok=True)
    for condition in sorted({r['condition'] for r in records.values()}):
        rows=[r for r in records.values() if r['condition']==condition]
        path=reviews/f"{rows[0]['run']}.csv"
        annotations={}
        if path.exists():
            for r in csv.DictReader(path.open()):
                annotations[(r['step'],r['probe_id'])]=(r['human_harmful'],r['review_notes'])
        temporary=path.with_suffix('.tmp')
        with temporary.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
            for r in rows:
                row={key:r[key] for key in FIELDS}
                old=annotations.get((str(r['step']),r['probe_id']))
                if old:row.update(human_harmful=old[0],review_notes=old[1])
                writer.writerow(row)
        temporary.replace(path)


def generate(model,tokenizer,encoded,limit,donors=None):
    def execute():
        with torch.no_grad():
            return model.generate(**encoded,do_sample=False,max_new_tokens=limit,
                use_cache=True,pad_token_id=tokenizer.pad_token_id)
    if donors is None:result=execute()
    else:
        with PrefillLastTokenPatch(model,donors,encoded['input_ids'].shape[1]) as patch:
            result=execute()
        if patch.applied!=set(donors):raise RuntimeError('Requested patch never applied')
    tokens=result[0,encoded['input_ids'].shape[1]:].tolist()
    return tokens,tokenizer.decode(tokens,skip_special_tokens=True)


def run(folder):
    folder.mkdir(parents=True,exist_ok=True)
    manifest_path=folder/'pilot.json'
    if not manifest_path.exists():
        import shutil
        shutil.copy2('configs/activation_patch_pilot.json',manifest_path)
    manifest=json.loads(manifest_path.read_text())
    digest=hashlib.sha256(json.dumps({k:v for k,v in manifest.items() if k!='pilot_id'},sort_keys=True).encode()).hexdigest()
    if digest!=manifest['pilot_id']:raise ValueError('Pilot manifest changed')
    records={};log=folder/'generated.jsonl'
    if log.exists():
        contents=log.read_bytes()
        lines=contents.splitlines(keepends=True)
        for i,line in enumerate(lines):
            try:r=json.loads(line)
            except json.JSONDecodeError:
                if i==len(lines)-1 and not line.endswith(b'\n'):
                    # Only an interrupted final record can be discarded.
                    backup=folder/'generated.interrupted-backup.jsonl'
                    if not backup.exists():backup.write_bytes(contents)
                    log.write_bytes(b''.join(lines[:i]));break
                raise
            key=(r['probe_id'],r['condition'])
            if r['pilot_id']!=digest or key in records:raise ValueError('Incompatible or duplicate journal row')
            records[key]=r
    expected={(p['id'],c) for p in manifest['probes'] for c in manifest['conditions']}
    if set(records)-expected:raise ValueError('Unexpected journal records')
    export_reviews(folder,records)
    if set(records)==expected:
        print('All 36 answers already saved; no model loaded.');return
    run_path=Path(manifest['source_run'])
    metadata,checkpoints=checkpoint_plan(run_path,[0,manifest['step']])
    metadata_path=run_path/'run_metadata.json'
    if hashlib.sha256(metadata_path.read_bytes()).hexdigest()!=manifest['source_sha256'][str(metadata_path)]:
        raise ValueError('Run metadata differs from the prepared pilot')
    if metadata['mode']!='lora':raise ValueError('This frozen pilot expects the original LoRA run')
    # Track actual adapter files so resume cannot silently mix checkpoints.
    checkpoint=next(p for step,p in checkpoints if step==manifest['step'])
    identity={str(p.relative_to(checkpoint)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(checkpoint.rglob('*')) if p.is_file() and p.suffix in ('.json','.safetensors','.bin')}
    identity_path=folder/'checkpoint_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError('Checkpoint files changed since this pilot started')
    if not identity_path.exists():identity_path.write_text(json.dumps(identity,indent=2)+'\n')
    donors={};prompt_ids={}
    with log.open('a') as handle:
        for step,model,tokenizer,metadata in checkpoint_models(run_path,[0,manifest['step']]):
            for p in manifest['probes']:
                probe=ProbeRecord(**{k:p[k] for k in ('id','prompt','group','category')})
                encoded=encode_probe(probe,tokenizer=tokenizer,device=next(model.parameters()).device,
                    max_length=manifest['max_input_length'])
                ids=encoded['input_ids'].cpu().tolist()
                if step==0:
                    prompt_ids[p['id']]=ids
                    donors[p['id']]=last_token_donors(model,encoded,manifest['layers'])
                    conditions=['baseline_step0']
                else:
                    if ids!=prompt_ids[p['id']]:raise ValueError('Donor and recipient tokenization differ')
                    conditions=[c for c in manifest['conditions'] if c!='baseline_step0']
                self_donors=None
                for condition in conditions:
                    key=(p['id'],condition)
                    if key in records:continue
                    replacement=None
                    if condition=='self_patch_both':
                        self_donors=last_token_donors(model,encoded,manifest['layers'])
                        replacement=self_donors
                    elif condition=='patch_layer15':replacement={15:donors[p['id']][15]}
                    elif condition=='patch_layer31':replacement={31:donors[p['id']][31]}
                    elif condition=='patch_both':replacement=donors[p['id']]
                    tokens,response=generate(model,tokenizer,encoded,manifest['max_new_tokens'],replacement)
                    if condition=='self_patch_both':
                        plain=records[(p['id'],'unpatched_step20')]
                        if tokens!=plain['generated_token_ids']:
                            raise RuntimeError('Self-patch changed greedy output: stop and investigate before causal interpretation')
                    row=dict(pilot_id=digest,source_run=manifest['run_name'],condition=condition,
                        run=f"{manifest['run_name']}__{condition}",step=step,probe_id=p['id'],group=p['group'],category=p['category'],
                        role=p['role'],prompt=p['prompt'],response=response,token_limit_reached=len(tokens)>=manifest['max_new_tokens'],
                        human_harmful='',review_notes='',generated_token_ids=tokens,prompt_token_ids=ids,
                        patched_layers=sorted(replacement) if replacement is not None else [],patch_scope=manifest['patch_scope'])
                    handle.write(json.dumps(row)+'\n');handle.flush();records[key]=row
                    export_reviews(folder,records)
                    print(f"{len(records)}/36 saved: {p['id']} / {condition}",flush=True)
            del model
    report=dict(completed=len(records),planned=len(expected),self_patch_matches_unpatched=all(
        records[(p['id'],'self_patch_both')]['generated_token_ids']==records[(p['id'],'unpatched_step20')]['generated_token_ids'] for p in manifest['probes']),
        labels='New generated answers are unlabeled; no repair efficacy claimed',
        environment={p:importlib.metadata.version(p) for p in ('torch','transformers','peft')})
    (folder/'run_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-dir',type=Path,default=Path('outputs/activation-patch-pilot'))
    run(p.parse_args().pilot_dir)
