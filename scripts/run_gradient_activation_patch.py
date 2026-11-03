"""Frozen, resumable first-order gradient selection, with prior-run controls."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from behaviour_prediction.monitor.gradient_patching import endpoint, replacement_scores, select_scores
from behaviour_prediction.monitor.responses import checkpoint_models, checkpoint_plan
from behaviour_prediction.monitoring.activations import ProbeRecord, encode_probe
from run_activation_patch_pilot import generate, export_reviews


def digest(value):
    return hashlib.sha256(json.dumps({k:v for k,v in value.items() if k!='pilot_id'},sort_keys=True).encode()).hexdigest()


def run(folder, manifest_path):
    requested = json.loads(manifest_path.read_text())
    if digest(requested) != requested['pilot_id']:
        raise ValueError('Manifest hash mismatch')
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'pilot.json'
    if path.exists() and json.loads(path.read_text()) != requested:
        raise ValueError('Changed plan: choose a fresh output directory')
    if not path.exists():
        path.write_text(json.dumps(requested,indent=2)+'\n')
    m = requested
    if m['layers'] != [31]:
        raise ValueError('This comparison is frozen to layer 31')
    comparison = Path(m['comparison_dir'])
    for name, expected_sha in m['comparison_sha256'].items():
        if hashlib.sha256((comparison/name).read_bytes()).hexdigest() != expected_sha:
            raise ValueError(f'Prior comparison changed: {name}')
    previous = {(r['probe_id'],r['direction'],r['condition']):r
                for r in map(json.loads,(comparison/'generated.jsonl').read_text().splitlines())}
    old_states = torch.load(comparison/'endpoint_activations.pt',map_location='cpu',weights_only=True)
    expected = {(p['id'],d,c) for p in m['probes'] for d in m['directions'] for c in m['conditions']}
    records = {}
    log = folder/'generated.jsonl'
    if log.exists():
        contents = log.read_bytes()
        lines = contents.splitlines(keepends=True)
        for i,line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines)-1 or line.endswith(b'\n'):
                    raise
                backup=folder/'generated.interrupted-backup.jsonl'
                if not backup.exists(): backup.write_bytes(contents)
                log.write_bytes(b''.join(lines[:i])); break
            key = row['probe_id'],row['direction'],row['condition']
            if key not in expected or key in records or row['pilot_id'] != m['pilot_id']:
                raise ValueError('Incompatible or duplicate generation record')
            records[key] = row

    def report():
        pairs=[(p['id'],d) for p in m['probes'] for d in m['directions']
               if (p['id'],d,'self_patch') in records and (p['id'],d,'unpatched') in records]
        value=dict(completed=len(records),planned=len(expected),complete=set(records)==expected,
                   self_patch_pairs_checked=len(pairs),self_patch_exact_matches=sum(
                       records[(i,d,'self_patch')]['generated_token_ids']==records[(i,d,'unpatched')]['generated_token_ids'] for i,d in pairs),
                   objective=m['selection'],limitations=m['interpretation'])
        (folder/'run_report.json').write_text(json.dumps(value,indent=2)+'\n')
        return value

    export_reviews(folder,records)
    if set(records)==expected:
        print(json.dumps(report(),indent=2)); return
    root=Path(m['source_run'])
    metadata, checkpoints=checkpoint_plan(root,[0,20])
    if metadata['mode']!='lora' or metadata['model']!=m['expected_model_config']:
        raise ValueError('Checkpoint model configuration differs')
    checkpoint=next(path for step,path in checkpoints if step==20)
    for name,sha in m['expected_donor_files'].items():
        if hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Step20 adapter differs from the comparison')
    states={0:{},20:{}};logits={0:{},20:{}}
    audit_path=folder/'coordinate_selection.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else []
    score_path=folder/'gradient_scores.pt'
    score_cache=torch.load(score_path,map_location='cpu',weights_only=True) if score_path.exists() else {}

    def encode(model,tokenizer,p):
        probe=ProbeRecord(**{k:p[k] for k in ('id','prompt','group','category')})
        encoded=encode_probe(probe,tokenizer=tokenizer,device=next(model.parameters()).device,max_length=m['max_input_length'])
        if encoded['input_ids'].cpu().tolist()!=old_states['prompt_token_ids'][p['id']]:
            raise ValueError('Prompt tokenization differs from the comparison')
        return encoded

    def capture(model,tokenizer,p,step):
        encoded=encode(model,tokenizer,p)
        own,next_logits=endpoint(model,encoded,31)
        old=old_states['baseline' if step==0 else 'finetuned'][p['id']][31]
        if not torch.equal(own[31],old):
            raise ValueError('Fresh endpoint differs from prior endpoint: do not mix experiments')
        states[step][p['id']]=own;logits[step][p['id']]=next_logits

    def generate_conditions(handle,model,tokenizer,p,direction):
        id=p['id'];step=20 if direction=='restore' else 0;donor_step=0 if step==20 else 20
        encoded=encode(model,tokenizer,p)
        attributed=replacement_scores(model,encoded,31,states[donor_step][id][31],logits[donor_step][id])
        torch.testing.assert_close(attributed['recipient'],states[step][id][31].float(),rtol=0,atol=0)
        torch.testing.assert_close(attributed['logits'],logits[step][id],rtol=1e-5,atol=1e-5)
        cache_key=f'{direction}:{id}'
        if cache_key in score_cache:
            if not torch.equal(score_cache[cache_key]['scores'],attributed['scores']):
                raise ValueError('Gradients changed on resume; do not mix rankings')
        else:
            score_cache[cache_key]={k:v for k,v in attributed.items() if k!='logits'}
            temporary=score_path.with_suffix('.tmp');torch.save(score_cache,temporary);temporary.replace(score_path)
        masks={}
        for percent in m['budgets_percent']:
            condition=f'gradient_{percent:02d}pct';count=math.ceil(attributed['scores'].numel()*percent/100)
            mask=select_scores(attributed['scores'],count);masks[condition]={31:mask}
            delta=attributed['donor']-attributed['recipient']
            selection=dict(probe_id=id,direction=direction,condition=condition,layer=31,
                selected_indices=mask.nonzero().flatten().tolist(),selected_count=count,total_count=mask.numel(),
                selected_delta_l2=float(delta[mask].norm()),total_delta_l2=float(delta.norm()),
                predicted_kl_reduction=float(attributed['scores'][mask].sum()),
                selected_nonpositive_scores=int((attributed['scores'][mask]<=0).sum()),kl_before=attributed['kl_before'])
            existing=[r for r in audit if (r['probe_id'],r['direction'],r['condition'])==(id,direction,condition)]
            if existing and existing!=[selection]:raise ValueError('Saved selections differ')
            if not existing:audit.append(selection)
        temporary=audit_path.with_suffix('.tmp');temporary.write_text(json.dumps(audit,indent=2)+'\n');temporary.replace(audit_path)
        for condition in m['conditions']:
            key=id,direction,condition
            if key in records:continue
            donors=None;mask=None
            if condition=='self_patch':donors=states[step][id]
            elif condition!='unpatched':donors=states[donor_step][id]
            if condition.startswith('gradient_'):mask=masks[condition]
            tokens,text=generate(model,tokenizer,encoded,m['max_new_tokens'],donors,mask)
            if condition in ('unpatched','self_patch','full_patch'):
                if tokens!=previous[(id,direction,condition)]['generated_token_ids']:
                    raise ValueError('Control generation differs from prior experiment; investigate before comparison')
            row=dict(pilot_id=m['pilot_id'],source_run=m['run_name'],direction=direction,condition=condition,
                run=f"{m['run_name']}_gradient_{direction}__{condition}",step=step,probe_id=id,
                group=p['group'],category=p['category'],role=p['role'],prompt=p['prompt'],response=text,
                token_limit_reached=len(tokens)>=m['max_new_tokens'],human_harmful='',review_notes='',
                prompt_token_ids=encoded['input_ids'].cpu().tolist(),generated_token_ids=tokens,
                patched_layers=[] if donors is None else [31],
                selected_indices={} if mask is None else {'31':mask[31].nonzero().flatten().tolist()},patch_scope=m['patch_scope'])
            handle.write(json.dumps(row)+'\n');handle.flush();records[key]=row
            export_reviews(folder,records);report()
            print(f'{len(records)}/{len(expected)} saved: {direction} / {id} / {condition}',flush=True)

    for _,model,tokenizer,_ in checkpoint_models(root,[0]):
        for p in m['probes']:capture(model,tokenizer,p,0)
        del model
    with log.open('a') as handle:
        for _,model,tokenizer,_ in checkpoint_models(root,[20],skip_steps=(0,)):
            for p in m['probes']:capture(model,tokenizer,p,20)
            for p in m['probes']:generate_conditions(handle,model,tokenizer,p,'restore')
            del model
        for _,model,tokenizer,_ in checkpoint_models(root,[0]):
            for p in m['probes']:generate_conditions(handle,model,tokenizer,p,'transfer')
            del model
    print(json.dumps(report(),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/activation-patch-gradient'))
    parser.add_argument('--manifest',type=Path,default=Path('configs/activation_patch_gradient.json'))
    args=parser.parse_args();run(args.output_dir,args.manifest)
