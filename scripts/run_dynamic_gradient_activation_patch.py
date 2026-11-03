"""Sequence-gradient selection and same-prefix patching throughout generation."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from behaviour_prediction.monitor.dynamic_gradient_patching import (
    step_context, sequence_capture, sequence_replacement_scores, dynamic_generate,
)
from behaviour_prediction.monitor.gradient_patching import endpoint, select_scores
from behaviour_prediction.monitor.responses import checkpoint_models, checkpoint_plan
from behaviour_prediction.monitoring.activations import ProbeRecord, encode_probe
from run_activation_patch_pilot import export_reviews
from run_gradient_activation_patch import digest


def atomic_json(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def reference_input(encoded, reference):
    if not reference:
        raise ValueError('Reference answer contains no tokens')
    ids=encoded['input_ids']
    continuation=torch.tensor([reference[:-1]],device=ids.device,dtype=ids.dtype)
    joined=torch.cat((ids,continuation),dim=1)
    return dict(input_ids=joined,attention_mask=torch.ones_like(joined))


def make_selections(result, manifest, probe_id, direction):
    width=result['scores'].numel();masks={};audit=[]
    seed=int(hashlib.sha256(f"{manifest['random_seed']}:{probe_id}:31".encode()).hexdigest()[:8],16)
    random_order=torch.randperm(width,generator=torch.Generator().manual_seed(seed))
    for condition in manifest['conditions']:
        if condition in ('unpatched','self_dynamic','full_dynamic'):continue
        percent=int(condition.split('_')[-1].removesuffix('pct'))
        count=math.ceil(width*percent/100)
        if condition.startswith(('gradient_','prefill_gradient_')):
            mask=select_scores(result['scores'],count)
        elif condition.startswith('top_'):
            mask=select_scores(result['magnitude_rms'],count)
        elif condition.startswith('random_'):
            mask=torch.zeros(width,dtype=torch.bool);mask[random_order[:count]]=True
        else:raise ValueError(f'Unknown condition: {condition}')
        masks[condition]=mask
        delta=result['donor']-result['recipient']
        audit.append(dict(probe_id=probe_id,direction=direction,condition=condition,layer=31,
            selected_indices=mask.nonzero().flatten().tolist(),selected_count=count,total_count=width,
            selected_delta_l2=float(delta[:,mask].norm()),total_delta_l2=float(delta.norm()),
            predicted_kl_reduction=float(result['scores'][mask].sum()),
            selected_nonpositive_scores=int((result['scores'][mask]<=0).sum()),
            prediction_positions=result['prediction_positions'],kl_before=result['kl_before']))
    return masks,audit


def run(folder,manifest_path):
    m=json.loads(manifest_path.read_text())
    if digest(m)!=m['pilot_id']:raise ValueError('Invalid manifest hash')
    if m['layers']!=[31]:raise ValueError('This frozen experiment requires layer31')
    folder.mkdir(parents=True,exist_ok=True)
    plan=folder/'pilot.json'
    if plan.exists() and json.loads(plan.read_text())!=m:raise ValueError('Plan changed; use a fresh folder')
    if not plan.exists():atomic_json(plan,m)
    comparison=Path(m['comparison_dir'])
    for name,sha in m['comparison_sha256'].items():
        if hashlib.sha256((comparison/name).read_bytes()).hexdigest()!=sha:raise ValueError(f'Comparison changed: {name}')
    previous={(r['probe_id'],r['direction'],r['condition']):r
        for r in map(json.loads,(comparison/'generated.jsonl').read_text().splitlines())}
    old_states=torch.load(comparison/'endpoint_activations.pt',map_location='cpu',weights_only=True)
    expected={(p['id'],d,c) for p in m['probes'] for d in m['directions'] for c in m['conditions']}
    records={};log=folder/'generated.jsonl'
    if log.exists():
        contents=log.read_bytes();lines=contents.splitlines(keepends=True)
        for i,line in enumerate(lines):
            try:row=json.loads(line)
            except json.JSONDecodeError:
                if i!=len(lines)-1 or line.endswith(b'\n'):raise
                backup=folder/'generated.interrupted-backup.jsonl'
                if not backup.exists():backup.write_bytes(contents)
                log.write_bytes(b''.join(lines[:i]));break
            key=row['probe_id'],row['direction'],row['condition']
            if key not in expected or key in records or row['pilot_id']!=m['pilot_id']:raise ValueError('Incompatible journal')
            records[key]=row

    def report():
        pairs=[(p['id'],d) for p in m['probes'] for d in m['directions']
            if (p['id'],d,'unpatched') in records and (p['id'],d,'self_dynamic') in records]
        value=dict(completed=len(records),planned=len(expected),complete=set(records)==expected,
            self_patch_pairs_checked=len(pairs),self_patch_exact_matches=sum(
                records[(id,d,'unpatched')]['generated_token_ids']==records[(id,d,'self_dynamic')]['generated_token_ids'] for id,d in pairs),
            full_dynamic_donor_matches=sum(c=='full_dynamic' for _,_,c in records),
            selection=m['selection'],limitations=m['interpretation'])
        atomic_json(folder/'run_report.json',value);return value

    export_reviews(folder,records)
    if set(records)==expected:
        print(json.dumps(report(),indent=2));return
    root=Path(m['source_run']);metadata,checkpoints=checkpoint_plan(root,[0,20])
    if metadata['mode']!='lora' or metadata['model']!=m['expected_model_config']:raise ValueError('Model configuration differs')
    checkpoint=next(p for step,p in checkpoints if step==20)
    for name,sha in m['expected_donor_files'].items():
        if hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()!=sha:raise ValueError('Checkpoint differs')
    audit_path=folder/'coordinate_selection.json'
    audits=json.loads(audit_path.read_text()) if audit_path.exists() else []
    cache_path=folder/'coordinate_scores.pt'
    score_cache=torch.load(cache_path,map_location='cpu',weights_only=True) if cache_path.exists() else {}
    references={}
    # One quantized model. Step0 disables its default LoRA adapter; no second model.
    with log.open('a') as handle:
        for _,model,tokenizer,_ in checkpoint_models(root,[20],skip_steps=(0,)):
            if getattr(model,'active_adapters',None)!=['default']:
                raise ValueError('Expected the frozen checkpoint as the default active adapter')
            for p in m['probes']:
                id=p['id']
                probe=ProbeRecord(**{k:p[k] for k in ('id','prompt','group','category')})
                encoded=encode_probe(probe,tokenizer=tokenizer,device=next(model.parameters()).device,max_length=m['max_input_length'])
                prompt_ids=encoded['input_ids'].cpu().tolist()
                if prompt_ids!=old_states['prompt_token_ids'][id]:raise ValueError('Prompt tokenization differs')
                for step,name in [(0,'baseline'),(20,'finetuned')]:
                    with step_context(model,step):fresh,_=endpoint(model,encoded,31)
                    if not torch.equal(fresh[31],old_states[name][id][31]):
                        raise ValueError('Adapter-switched endpoint differs from previous live endpoint')
                for direction in m['directions']:
                    if all((id,direction,c) in records for c in m['conditions']):continue
                    recipient=20 if direction=='restore' else 0;donor=20-recipient
                    donor_direction='transfer' if donor==0 else 'restore'
                    reference_row=previous[(id,donor_direction,'unpatched')]
                    reference=reference_row['generated_token_ids']
                    teacher=reference_input(encoded,reference)
                    if teacher['input_ids'].shape[1]>m['expected_model_config']['model']['max_sequence_length']:
                        raise ValueError('Reference exceeds configured context length')
                    with step_context(model,donor):
                        donor_hidden,donor_logits=sequence_capture(model,teacher,31,len(prompt_ids[0]))
                    with step_context(model,recipient):
                        result=sequence_replacement_scores(model,teacher,31,len(prompt_ids[0]),donor_hidden,donor_logits)
                    if result['prediction_positions']!=len(reference):raise ValueError('Reference prediction alignment differs')
                    key=f'{direction}:{id}'
                    compact={k:result[k] for k in ('scores','magnitude_rms','prediction_positions','kl_before')}
                    compact['token_score_sums']=result['token_scores'].sum(1)
                    if key in score_cache:
                        if not torch.equal(score_cache[key]['scores'],result['scores']):raise ValueError('Scores changed on resume')
                    else:
                        score_cache[key]=compact
                        temporary=cache_path.with_suffix('.tmp');torch.save(score_cache,temporary);temporary.replace(cache_path)
                    masks,new_audits=make_selections(result,m,id,direction)
                    for item in new_audits:
                        prior=[r for r in audits if (r['probe_id'],r['direction'],r['condition'])==(id,direction,item['condition'])]
                        if prior and prior!=[item]:raise ValueError('Selection changed on resume')
                        if not prior:audits.append(item)
                    atomic_json(audit_path,audits)
                    references[key]=dict(reference_step=donor,token_ids=reference,
                        token_limit_reached=reference_row['token_limit_reached'],
                        reference_sha256=hashlib.sha256(json.dumps(reference).encode()).hexdigest())
                    reference_path=folder/'references.json'
                    saved=json.loads(reference_path.read_text()) if reference_path.exists() else {}
                    if key in saved and saved[key]!=references[key]:raise ValueError('Reference changed')
                    saved.update(references);atomic_json(reference_path,saved)
                    del result,donor_hidden,donor_logits
                    for condition in m['conditions']:
                        record_key=id,direction,condition
                        if record_key in records:continue
                        tokens=dynamic_generate(model,encoded,recipient_step=recipient,
                            donor_step=recipient if condition=='self_dynamic' else donor,layer=31,
                            limit=m['max_new_tokens'],patch=condition!='unpatched',mask=masks.get(condition),
                            prefill_only=condition.startswith('prefill_'))
                        if condition in ('unpatched','self_dynamic'):
                            expected_tokens=previous[(id,direction,'unpatched')]['generated_token_ids']
                            if tokens!=expected_tokens:raise ValueError('Unpatched/self control differs from prior answer')
                        if condition=='full_dynamic' and tokens!=reference:
                            raise ValueError('Full last-layer dynamic patch did not reproduce donor tokens')
                        row=dict(pilot_id=m['pilot_id'],source_run=m['run_name'],direction=direction,condition=condition,
                            run=f"{m['run_name']}_gradientdynamic_{direction}__{condition}",step=recipient,
                            probe_id=id,group=p['group'],category=p['category'],role=p['role'],prompt=p['prompt'],
                            response=tokenizer.decode(tokens,skip_special_tokens=True),
                            token_limit_reached=len(tokens)>=m['max_new_tokens'],human_harmful='',review_notes='',
                            prompt_token_ids=prompt_ids,generated_token_ids=tokens,
                            selected_indices={} if condition not in masks else {'31':masks[condition].nonzero().flatten().tolist()},
                            patched_layers=[] if condition=='unpatched' else [31],patch_scope=m['patch_scope'],
                            reference_sha256=references[key]['reference_sha256'])
                        handle.write(json.dumps(row)+'\n');handle.flush();records[record_key]=row
                        export_reviews(folder,records);report()
                        print(f'{len(records)}/{len(expected)} saved: {direction} / {id} / {condition}',flush=True)
            del model
    print(json.dumps(report(),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/activation-patch-gradient-dynamic'))
    parser.add_argument('--manifest',type=Path,default=Path('configs/activation_patch_gradient_dynamic.json'))
    args=parser.parse_args();run(args.output_dir,args.manifest)
