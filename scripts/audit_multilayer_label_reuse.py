"""Read-only, exact-field audit of source labels and reused multilayer judgments.

No model loading, generation, label modification, or network/API calls.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition: raise ValueError(message)


def checksum(value, field):
    return hashlib.sha256(json.dumps({k:v for k,v in value.items() if k!=field},sort_keys=True).encode()).hexdigest()


def label_checksum(value):
    return hashlib.sha256(json.dumps({k:v for k,v in value.items() if k!='test_id'},sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def flag(value):
    if type(value) is bool:return value
    require(str(value).lower() in ('true','false'), 'Invalid boolean field')
    return str(value).lower()=='true'


def exact_payload(row):
    return row['prompt'], row['response'], flag(row['token_limit_reached'])


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_source(bundle_path, verify_files=False):
    b=json.loads(bundle_path.read_text())
    require(checksum(b,'reuse_id')==b['reuse_id'],'Bundle hash differs')
    m=b['source_manifest']
    require(checksum(m,'pilot_id')==m['pilot_id'],'Source manifest hash differs')
    # Validate every original judgment, even if it was not used by the target.
    from label_with_claude import validate_judgment
    by_payload=defaultdict(list);by_origin={}
    for row in b['labels']:
        validate_judgment(row['judgment'])
        key=exact_payload(row['payload']);origin=row['source_run'],row['source_probe_id']
        require(origin not in by_origin,'Duplicate bundle source identity')
        by_origin[origin]=row;by_payload[key].append(row)
    lookup={}
    for key,rows in by_payload.items():
        require(len({r['judgment']['harmful'] for r in rows})==1,'Conflicting source harmfulness labels on an exact payload')
        judgment=dict(rows[0]['judgment'])
        judgment['needs_review']=any(r['judgment']['needs_review'] for r in rows)
        lookup[key]=dict(judgment=judgment,origin=(rows[0]['source_run'],rows[0]['source_probe_id']))
    checked=0
    if verify_files:
        for name,sha in b['source_sha256'].items():
            path=Path(name)
            require(path.exists(),f'Missing original source file: {path}')
            require(hashlib.sha256(path.read_bytes()).hexdigest()==sha,f'Original source file changed: {path}')
            checked+=1
        source=Path(next(name for name in b['source_sha256'] if name.endswith('/pilot.json'))).parent
        original=read_jsonl(source/'generated.jsonl')
        original_by_origin={(r['run'],r['probe_id']):r for r in original}
        require(len(original_by_origin)==len(original)==len(b['labels']),'Source generation/bundle counts differ')
        csv_labels={}
        for path in (source/'labeling/labeled').glob('*.csv'):
            with path.open(newline='') as handle:
                for row in csv.DictReader(handle):
                    key=row['run'],row['probe_id']
                    require(key not in csv_labels,'Duplicate source CSV label')
                    csv_labels[key]=row
        require(set(csv_labels)==set(original_by_origin)==set(by_origin),'Source row identities differ')
        judge=json.loads((source/'labeling/test.json').read_text())
        require(label_checksum(judge)==judge['test_id'],'Original judge manifest hash differs')
        require(judge['model']==b['judge_model'] and judge['rubric']==b['judge_rubric'],'Bundle judge settings differ from originals')
        for origin,saved in by_origin.items():
            generation=original_by_origin[origin];label=csv_labels[origin]
            require(exact_payload(generation)==exact_payload(label)==exact_payload(saved['payload']),f'Changed source prompt/answer/truncation: {origin}')
            expected=dict(harmful=int(label['claude_harmful']),behaviour_tags=[t for t in label['claude_behaviour_tags'].split(';') if t],
                          needs_review=flag(label['claude_needs_review']))
            require(saved['judgment']==expected,f'Bundle label differs from effective original CSV: {origin}')
        for r in b['answers']:
            require(r==original_by_origin[(r['run'],r['probe_id'])],'Packed generation candidate differs from original')
    return b,lookup,dict(bundle_id=b['reuse_id'],source_label_rows=len(b['labels']),unique_source_payloads=len(lookup),
        conflicting_source_harmfulness_labels=0,original_source_files_verified=checked,
        effective_source_labels_verified_against_original_csvs=verify_files)


def audit_target(folder,bundle_path):
    b,lookup,source_report=audit_source(bundle_path)
    m=json.loads((folder/'pilot.json').read_text())
    require(checksum(m,'pilot_id')==m['pilot_id'],'Target generation manifest hash differs')
    old=b['source_manifest']
    for field in ('expected_model_config','expected_donor_files','max_input_length','max_new_tokens','probes'):
        require(m[field]==old[field],f'Target experiment differs from reuse source: {field}')
    artifact=json.loads((folder/'mask.json').read_text())
    require(checksum(artifact,'mask_id')==artifact['mask_id']==m['mask_id'],'Target frozen-mask hash differs')
    rows=read_jsonl(folder/'generated.jsonl');generation={}
    expected={(p['id'],d,c) for p in m['probes'] for d in m['directions'] for c in m['conditions']}
    probes={p['id']:p for p in m['probes']}
    for r in rows:
        key=r['probe_id'],r['direction'],r['condition']
        require(key in expected and key not in generation,'Duplicate/unexpected generated answer')
        require(r['pilot_id']==m['pilot_id'] and r['mask_id']==m['mask_id'],'Generated answer identity differs')
        require(r['selected_indices']==artifact['selected_indices'].get(r['condition'],{}),'Generated answer mask differs')
        require(all(r[k]==probes[r['probe_id']][k] for k in ('prompt','group','category','role')),'Generated prompt differs from manifest')
        require(r['step']==(20 if r['direction']=='restore' else 0),'Generated checkpoint differs')
        require(flag(r['token_limit_reached'])==(len(r['generated_token_ids'])>=m['max_new_tokens']),'Generated truncation flag differs')
        generation[key]=r
    require(set(generation)==expected,'Generation incomplete')
    for (id,d,c),r in generation.items():
        require(r['prompt_token_ids']==generation[(id,d,'unpatched')]['prompt_token_ids'],'Prompt token IDs differ')
        if c in ('self_dynamic','full_dynamic'):
            direction=d if c=='self_dynamic' else 'transfer' if d=='restore' else 'restore'
            require(r['generated_token_ids']==generation[(id,direction,'unpatched')]['generated_token_ids'],'Self/full generation control failed')
    label_folder=folder/'labeling';judge=json.loads((label_folder/'test.json').read_text())
    require(label_checksum(judge)==judge['test_id'],'Target judge manifest hash differs')
    require(judge['model']==b['judge_model'] and judge['rubric']==b['judge_rubric'],'Target judge model/rubric differs')
    by_identity={(r['run'],int(r['step']),r['probe_id']):r for r in rows}
    require(len(by_identity)==len(rows),'Duplicate generation labeling identity')
    sample=judge['sample'];seen=set()
    for r in sample:
        key=r['run'],int(r['step']),r['probe_id']
        require(key not in seen and key in by_identity,'Duplicate/unexpected labeling input')
        seen.add(key);original=by_identity[key]
        require(exact_payload(r)==exact_payload(original),'Judge sample differs from generated prompt/answer/truncation')
    require(seen==set(by_identity),'Judge sample misses generated answers')
    results=read_jsonl(label_folder/'results.jsonl');done={};imports=[];conditions=Counter()
    from label_with_claude import validate_judgment
    for r in results:
        index=r['index']
        require(type(index) is int and 0<=index<len(sample) and index not in done,'Duplicate/invalid result index')
        require(r['test_id']==judge['test_id'],'Wrong result manifest')
        done[index]=r
        if 'reused_from' not in r:continue
        target=sample[index];match=lookup.get(exact_payload(target))
        require(match is not None,f'Imported row {index} has NO exact original match')
        validate_judgment(r['judgment'])
        require(r['judgment']==match['judgment'],f'Imported judgment differs from original at {index}')
        origin=r['reused_from']
        require(origin['reuse_id']==b['reuse_id'],'Imported label cites wrong reuse bundle')
        require((origin['source_run'],origin['source_probe_id'])==match['origin'],f'Incorrect source attribution at {index}')
        require(r['cost_usd']==0 and r['raw_text']=='' and r['status']=='ok','Imported result misrepresented as a new API call')
        require(r['model']==judge['model'],'Imported judge model differs')
        imports.append(index)
        conditions[by_identity[(target['run'],int(target['step']),target['probe_id'])]['condition']]+=1
    # Check exported values actually consumed by the scorer as well as journal entries.
    exported={}
    for path in (label_folder/'labeled').glob('*.csv'):
        with path.open(newline='') as handle:
            for r in csv.DictReader(handle):
                key=r['run'],int(r['step']),r['probe_id']
                require(key not in exported,'Duplicate exported label identity')
                exported[key]=r
    require(set(exported)==set(by_identity),'Exported CSVs do not cover all generated rows')
    for index in imports:
        sample_row=sample[index];r=exported[(sample_row['run'],int(sample_row['step']),sample_row['probe_id'])]
        result=done[index]['judgment']
        require(exact_payload(r)==exact_payload(sample_row),'Exported reused payload changed')
        require(r['claude_harmful']==str(result['harmful']) and r['claude_behaviour_tags']==';'.join(result['behaviour_tags']) and
                flag(r['claude_needs_review'])==result['needs_review'],f'Exported reused judgment differs at index {index}; check manual overrides')
    matched={i for i,r in enumerate(sample) if exact_payload(r) in lookup}
    require(not matched-set(done),'Some exact-match labels are still missing; rerun offline reuse')
    return dict(passed=True,source=source_report,total_generation_rows=len(rows),labeling_rows=len(sample),
        completed=len(done),imported_rows_verified=len(imports),unique_imported_payloads=len({exact_payload(sample[i]) for i in imports}),
        reused_harmful_rows=sum(done[i]['judgment']['harmful']==1 for i in imports),
        reused_nonharmful_rows=sum(done[i]['judgment']['harmful']==0 for i in imports),
        reused_needs_review_rows=sum(done[i]['judgment']['needs_review'] for i in imports),
        remaining=len(sample)-len(done),imported_by_condition=dict(sorted(conditions.items())),
        note='Exact-payload reuse verified. This checks copying/alignment, not whether original Claude judgments are correct.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/activation-patch-multilayer'))
    p.add_argument('--bundle',type=Path,default=Path('artifacts/multilayer-reuse-v1/reuse.json'))
    p.add_argument('--source-only',action='store_true',help='Verify bundle against original locally available CSVs and journals')
    args=p.parse_args()
    if args.source_only:
        _,_,report=audit_source(args.bundle,verify_files=True);report=dict(passed=True,**report)
    else:report=audit_target(args.output_dir,args.bundle)
    print(json.dumps(report,indent=2))
