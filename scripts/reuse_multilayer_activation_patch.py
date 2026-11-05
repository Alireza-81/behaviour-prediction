"""Reuse verified previous answers and exact-payload labels without API calls."""
import argparse
import hashlib
import json
from pathlib import Path
from behaviour_prediction.monitor.shared_mask import identity
from run_dynamic_gradient_activation_patch import atomic_json
from score_gradient_activation_patch import load_run
from label_with_claude import digest, payload, validate_judgment, read_saved_results, write_report

BUNDLE=Path('artifacts/multilayer-reuse-v1/reuse.json')


def pack(source,output):
    m,journal,labels=load_run(source,'shared')
    frozen=json.loads(Path(m['mask_path']).read_text())
    from behaviour_prediction.monitor.shared_mask import validate_frozen_mask
    validate_frozen_mask(frozen)
    if hashlib.sha256(Path(m['mask_path']).read_bytes()).hexdigest()!=m['mask_sha256']:
        raise ValueError('Old frozen mask changed')
    judge=json.loads((source/'labeling/test.json').read_text())
    answers=[];decisions=[];seen={}
    for key,r in journal.items():
        c=r['condition'];label=labels[key]
        expected={} if c not in frozen['selected_indices'] else {'31':frozen['selected_indices'][c]}
        if r['mask_id']!=frozen['mask_id'] or r['selected_indices']!=expected:
            raise ValueError('Old answer mask differs')
        if c=='unpatched' or c.endswith('_05pct'):
            answers.append(r)
        judgment=dict(harmful=label['harmful'],behaviour_tags=[v for v in label.get('claude_behaviour_tags','').split(';') if v],
                      needs_review=label['claude_needs_review'].lower()=='true')
        validate_judgment(judgment)
        value=payload(r);key=digest(value)
        if key in seen and seen[key]!=judgment['harmful']:raise ValueError('Resolve source label conflicts before reuse')
        seen[key]=judgment['harmful']
        decisions.append(dict(payload=value,judgment=judgment,source_run=r['run'],source_probe_id=r['probe_id']))
    bundle=dict(source_manifest=m,answers=answers,labels=decisions,judge_model=judge['model'],judge_rubric=judge['rubric'],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [source/'pilot.json',source/'generated.jsonl',source/'labeling/test.json',*(source/'labeling/labeled').glob('*.csv')]})
    bundle['reuse_id']=identity(bundle,'reuse_id')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() and json.loads(output.read_text())!=bundle:raise ValueError('Reuse bundle already exists with different content')
    atomic_json(output,bundle)
    print(f'Packed {len(answers)} generation candidates and {len(decisions)} existing labels (offline).')


def load_bundle(path=BUNDLE):
    b=json.loads(path.read_text())
    if identity(b,'reuse_id')!=b['reuse_id']:raise ValueError('Reuse bundle changed')
    return b


def seed_answers(folder,m,selected,records,bundle_path=BUNDLE):
    if not bundle_path.exists():return 0
    b=load_bundle(bundle_path);old=b['source_manifest']
    for field in ['expected_model_config','expected_donor_files','max_input_length','max_new_tokens','probes']:
        if old[field]!=m[field]:raise ValueError(f'Reuse experiment mismatch: {field}')
    imported=0
    with (folder/'generated.jsonl').open('a') as handle:
        for source in b['answers']:
            oldc=source['condition']
            c='unpatched' if oldc=='unpatched' else 'layer31_205' if oldc=='shared_05pct' else f"layer31_205_random{oldc.split('_')[0].removeprefix('random')}"
            if c not in m['conditions']:continue
            if source['selected_indices']!=selected.get(c,{}):continue # Exact sites, not just count, must match.
            key=source['probe_id'],source['direction'],c
            if key in records:continue
            row=dict(source,pilot_id=m['pilot_id'],mask_id=m['mask_id'],condition=c,
                run=f"{m['run_name']}_multilayer_{source['direction']}__{c}",
                selected_indices=selected.get(c,{}),
                reused_from=dict(reuse_id=b['reuse_id'],pilot_id=source['pilot_id'],run=source['run'],condition=oldc,
                    note='Same checkpoint, prompt, generation settings, dynamic patch scope and exact selected sites.'))
            handle.write(json.dumps(row)+'\n');handle.flush();records[key]=row;imported+=1
    atomic_json(folder/'reuse_report.json',dict(reuse_id=b['reuse_id'],
        imported_total=sum('reused_from' in r for r in records.values()),imported_this_call=imported))
    if imported:print(f'Reused {imported} saved answers; no GPU generation for those rows.',flush=True)
    return imported


def reuse_labels(folder,bundle_path=BUNDLE):
    b=load_bundle(bundle_path);labeling=folder/'labeling'
    manifest=json.loads((labeling/'test.json').read_text())
    if digest({k:v for k,v in manifest.items() if k!='test_id'})!=manifest['test_id']:
        raise ValueError('Invalid label manifest')
    if manifest['model']!=b['judge_model'] or manifest['rubric']!=b['judge_rubric']:
        raise ValueError('Judge model/rubric differ; refusing automatic label inheritance')
    if (labeling/'extension.pending.json').exists():raise ValueError('Finish label extension before reuse')
    results=read_saved_results(labeling,manifest);lookup={}
    for saved in b['labels']:
        key=digest(saved['payload'])
        if key in lookup:
            previous=lookup[key]
            if previous['judgment']['harmful']!=saved['judgment']['harmful']:raise ValueError('Source labels conflict')
            previous['judgment']['needs_review'] |= saved['judgment']['needs_review']
        else:lookup[key]=saved
    imported=0
    with (labeling/'results.jsonl').open('a') as handle:
        for index,row in enumerate(manifest['sample']):
            if index in results:continue
            saved=lookup.get(digest(payload(row)))
            if saved is None:continue
            result=dict(test_id=manifest['test_id'],index=index,model=manifest['model'],status='ok',
                judgment=saved['judgment'],cost_usd=0,raw_text='',
                reused_from=dict(reuse_id=b['reuse_id'],source_run=saved['source_run'],source_probe_id=saved['source_probe_id'],
                                 note='Existing effective CSV judgment on the exact prompt/answer/truncation payload; no new API call.'))
            handle.write(json.dumps(result)+'\n');handle.flush();results[index]=result;imported+=1
    report=write_report(labeling,manifest,results)
    print(json.dumps(dict(imported_this_call=imported,completed=report['completed'],total=report['total'],
                         remaining=report['total']-report['completed']),indent=2))
    return imported


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['pack','labels'])
    p.add_argument('--source',type=Path,default=Path('outputs/activation-patch-shared'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/activation-patch-multilayer'))
    p.add_argument('--bundle',type=Path,default=BUNDLE);args=p.parse_args()
    if args.action=='pack':pack(args.source,args.bundle)
    else:reuse_labels(args.output_dir,args.bundle)
