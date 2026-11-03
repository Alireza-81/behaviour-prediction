"""Evaluate a frozen prompt-independent mask. No evaluation gradients are taken."""
import argparse
import hashlib
import json
from pathlib import Path

from behaviour_prediction.monitor.dynamic_gradient_patching import dynamic_generate, step_context
from behaviour_prediction.monitor.gradient_patching import endpoint
from behaviour_prediction.monitor.shared_mask import tensor_identity, validate_frozen_mask
from behaviour_prediction.monitor.responses import checkpoint_models, checkpoint_plan
from behaviour_prediction.monitoring.activations import ProbeRecord, encode_probe
from run_activation_patch_pilot import export_reviews
from run_dynamic_gradient_activation_patch import atomic_json
from run_gradient_activation_patch import digest


def run(folder, manifest_path):
    m=json.loads(manifest_path.read_text())
    if digest(m)!=m['pilot_id']:raise ValueError('Invalid evaluation manifest')
    mask_path=Path(m['mask_path'])
    if hashlib.sha256(mask_path.read_bytes()).hexdigest()!=m['mask_sha256']:raise ValueError('Frozen mask changed')
    frozen=json.loads(mask_path.read_text());masks=validate_frozen_mask(frozen)
    if set(m['conditions'])!=set(masks)|{'unpatched','self_dynamic','full_dynamic'} or len(set(m['conditions']))!=len(m['conditions']):
        raise ValueError('Conditions do not exactly match the frozen masks and controls')
    if m['directions']!=['restore','transfer']:raise ValueError('Unexpected patch directions')
    if frozen['mask_id']!=m['mask_id']:raise ValueError('Wrong frozen mask')
    if m['layers']!=[frozen['layer']] or m['layers']!=[31]:raise ValueError('Layer mismatch')
    if m['expected_model_config']!=frozen['expected_model_config'] or m['expected_donor_files']!=frozen['expected_donor_files']:
        raise ValueError('Wrong checkpoint for frozen selection')
    data=Path(m['evaluation_path'])
    if hashlib.sha256(data.read_bytes()).hexdigest()!=m['evaluation_sha256']:raise ValueError('Evaluation data changed')
    probes=[json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    if probes!=m['probes']:raise ValueError('Evaluation data differs from frozen plan')
    if any(p['id'] in m['development_probe_ids'] for p in probes):raise ValueError('Development/evaluation overlap')
    folder.mkdir(parents=True,exist_ok=True)
    plan=folder/'pilot.json'
    if plan.exists() and json.loads(plan.read_text())!=m:raise ValueError('Plan changed; choose a new output directory')
    if not plan.exists():atomic_json(plan,m)
    expected={(p['id'],d,c) for p in probes for d in m['directions'] for c in m['conditions']}
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
            if key not in expected or key in records or row['pilot_id']!=m['pilot_id'] or row['mask_id']!=frozen['mask_id']:
                raise ValueError('Incompatible/duplicate generation record')
            selected={} if row['condition'] not in masks else {'31':frozen['selected_indices'][row['condition']]}
            if row['selected_indices']!=selected:raise ValueError('A saved row uses a different mask')
            records[key]=row
        else:
            if contents and not contents.endswith(b'\n'):log.write_bytes(contents+b'\n')

    def report():
        paired=[(p['id'],d) for p in probes for d in m['directions'] if (p['id'],d,'self_dynamic') in records]
        matches=sum(records[(id,d,'self_dynamic')]['generated_token_ids']==records[(id,d,'unpatched')]['generated_token_ids'] for id,d in paired)
        value=dict(completed=len(records),planned=len(expected),complete=set(records)==expected,mask_id=frozen['mask_id'],
            selection='Fixed development-only mask; no evaluation gradient or mask selection',
            self_patch_pairs_checked=len(paired),self_patch_exact_matches=matches,
            full_donor_exact_matches=sum(c=='full_dynamic' for _,_,c in records),
            same_indices_across_prompts=True,same_indices_across_directions=True)
        atomic_json(folder/'run_report.json',value);return value

    export_reviews(folder,records)
    if set(records)==expected:
        print(json.dumps(report(),indent=2));return
    root=Path(m['source_run']);metadata,checkpoints=checkpoint_plan(root,[0,20])
    if metadata['mode']!='lora' or metadata['model']!=m['expected_model_config']:raise ValueError('Model configuration differs')
    checkpoint=next(path for step,path in checkpoints if step==20)
    for name,sha in m['expected_donor_files'].items():
        if hashlib.sha256((checkpoint/name).read_bytes()).hexdigest()!=sha:raise ValueError('Checkpoint changed')

    def encode(model,tokenizer,p):
        record=ProbeRecord(**{k:p[k] for k in ('id','prompt','group','category')})
        return encode_probe(record,tokenizer=tokenizer,device=next(model.parameters()).device,max_length=m['max_input_length'])

    def append(handle,p,d,c,tokens,tokenizer,prompt_ids):
        key=p['id'],d,c
        row=dict(pilot_id=m['pilot_id'],mask_id=frozen['mask_id'],source_run=m['run_name'],
            direction=d,condition=c,run=f"{m['run_name']}_shared_{d}__{c}",step=20 if d=='restore' else 0,
            probe_id=p['id'],group=p['group'],category=p['category'],role=p['role'],prompt=p['prompt'],
            response=tokenizer.decode(tokens,skip_special_tokens=True),token_limit_reached=len(tokens)>=m['max_new_tokens'],
            human_harmful='',review_notes='',prompt_token_ids=prompt_ids,generated_token_ids=tokens,
            patched_layers=[] if c=='unpatched' else [31],patch_scope=m['patch_scope'],
            selected_indices={} if c not in masks else {'31':frozen['selected_indices'][c]})
        handle.write(json.dumps(row)+'\n');handle.flush();records[key]=row
        export_reviews(folder,records);report()
        print(f'{len(records)}/{len(expected)} saved: {d} / {p["id"]} / {c}',flush=True)

    with log.open('a') as handle:
        for _,model,tokenizer,_ in checkpoint_models(root,[20],skip_steps=(0,)):
            if getattr(model,'active_adapters',None)!=['default']:raise ValueError('Expected the default step20 adapter')
            sentinel=frozen['sentinel'];encoded=encode(model,tokenizer,sentinel['probe'])
            if encoded['input_ids'].cpu().tolist()!=sentinel['prompt_token_ids']:raise ValueError('Sentinel tokenization changed')
            for step,name in [(0,'baseline'),(20,'finetuned')]:
                with step_context(model,step):states,_=endpoint(model,encoded,31)
                if tensor_identity(states[31])!=sentinel[f'{name}_sha256']:raise ValueError('Sentinel endpoint changed')
            for p in probes:
                if all((p['id'],d,c) in records for d in m['directions'] for c in m['conditions']):continue
                encoded=encode(model,tokenizer,p);prompt_ids=encoded['input_ids'].cpu().tolist()
                unpatched={}
                # Establish the new prompt's two endpoint answers as controls.
                # These answers never enter scoring/selection of the fixed mask.
                for step,d in [(0,'transfer'),(20,'restore')]:
                    tokens=dynamic_generate(model,encoded,recipient_step=step,donor_step=20-step,
                        layer=31,limit=m['max_new_tokens'],patch=False)
                    unpatched[step]=tokens
                    key=p['id'],d,'unpatched'
                    if key in records:
                        if records[key]['prompt_token_ids']!=prompt_ids or records[key]['generated_token_ids']!=tokens:
                            raise ValueError('Unpatched control changed on resume')
                    else:append(handle,p,d,'unpatched',tokens,tokenizer,prompt_ids)
                for d in m['directions']:
                    recipient=20 if d=='restore' else 0;donor=20-recipient
                    for c in m['conditions']:
                        if (p['id'],d,c) in records:continue
                        tokens=dynamic_generate(model,encoded,recipient_step=recipient,
                            donor_step=recipient if c=='self_dynamic' else donor,
                            layer=31,limit=m['max_new_tokens'],mask=masks.get(c),patch=True)
                        if c=='self_dynamic' and tokens!=unpatched[recipient]:raise ValueError('Self-patch changed output')
                        if c=='full_dynamic' and tokens!=unpatched[donor]:raise ValueError('Full patch did not copy donor')
                        append(handle,p,d,c,tokens,tokenizer,prompt_ids)
            del model
    print(json.dumps(report(),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/activation-patch-shared'))
    parser.add_argument('--manifest',type=Path,default=Path('configs/activation_patch_shared.json'))
    args=parser.parse_args();run(args.output_dir,args.manifest)
