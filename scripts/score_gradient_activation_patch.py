"""Validate new gradient-patch labels and compare with the frozen magnitude run."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_run(folder, prefix):
    manifest=json.loads((folder/'pilot.json').read_text())
    content={k:v for k,v in manifest.items() if k!='pilot_id'}
    if hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()!=manifest['pilot_id']:
        raise ValueError('Invalid generation manifest')
    rows=[json.loads(line) for line in (folder/'generated.jsonl').read_text().splitlines()]
    journal={(r['probe_id'],r['direction'],r['condition']):r for r in rows}
    expected={(p['id'],d,c) for p in manifest['probes'] for d in manifest['directions'] for c in manifest['conditions']}
    if len(journal)!=len(rows) or set(journal)!=expected:raise ValueError('Incomplete/duplicate generation')
    judge=json.loads((folder/'labeling/test.json').read_text())
    content={k:v for k,v in judge.items() if k!='test_id'}
    if hashlib.sha256(json.dumps(content,sort_keys=True,ensure_ascii=False).encode()).hexdigest()!=judge['test_id']:
        raise ValueError('Invalid labeling manifest')
    samples={(r['run'],int(r['step']),r['probe_id']):r for r in judge['sample']}
    labels={}
    for direction in manifest['directions']:
        for condition in manifest['conditions']:
            file=folder/'labeling/labeled'/f"{manifest['run_name']}_{prefix}_{direction}__{condition}.csv"
            for label in csv.DictReader(file.open()):
                key=label['probe_id'],direction,condition
                if key in labels or key not in journal:raise ValueError('Unexpected/duplicate label')
                row=journal[key]
                if row['pilot_id']!=manifest['pilot_id']:raise ValueError('Wrong generation identity')
                if label['claude_harmful'] not in ('0','1'):raise ValueError(f'Unresolved label: {file}: {key}')
                if any(label[f]!=row[f] for f in ('run','prompt','response','group','category')) or int(label['step'])!=row['step']:
                    raise ValueError('Labels do not match generated responses')
                sample=samples[(row['run'],row['step'],row['probe_id'])]
                if sample['prompt']!=row['prompt'] or sample['response']!=row['response']:raise ValueError('Changed judge input')
                labels[key]=dict(label,harmful=int(label['claude_harmful']),label_file=str(file))
    if set(labels)!=expected:raise ValueError('Missing labels')
    return manifest,journal,labels


def score(folder):
    m,new,labels=load_run(folder,'gradient')
    comparison=Path(m['comparison_dir'])
    for name,sha in m['comparison_sha256'].items():
        if hashlib.sha256((comparison/name).read_bytes()).hexdigest()!=sha:raise ValueError('Comparison data changed')
    old_m,old,old_labels=load_run(comparison,'topchange')
    if m['probes']!=old_m['probes']:raise ValueError('Different comparison prompts')
    for p in m['probes']:
        for d in m['directions']:
            for c in ('unpatched','self_patch','full_patch'):
                key=p['id'],d,c
                if new[key]['generated_token_ids']!=old[key]['generated_token_ids']:raise ValueError('Control generation differs')
    audit=json.loads((folder/'coordinate_selection.json').read_text())
    selections={(r['probe_id'],r['direction'],r['condition']):r for r in audit}
    for key,r in new.items():
        if r['condition'].startswith('gradient_'):
            if r['selected_indices']!={'31':selections[key]['selected_indices']}:raise ValueError('Selection journal mismatch')
    # Equivalent responses must have the same ground-truth label across methods.
    repeated=defaultdict(list)
    for dataset,journal,ls in [('gradient',new,labels),('topchange',old,old_labels)]:
        for key,r in journal.items():repeated[(r['prompt'],r['response'])].append(dict(ls[key],dataset=dataset))
    conflicts=[r for group in repeated.values() if len({r['harmful'] for r in group})>1 for r in group]
    if conflicts:
        file=folder/'label_conflicts.csv'
        with file.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(conflicts[0]));writer.writeheader();writer.writerows(conflicts)
        raise ValueError(f'Identical answers have inconsistent labels. Resolve source labels listed in {file}')
    out=folder/'results'
    if out.exists() and any(out.iterdir()):raise ValueError('Results exist; preserve them before rescoring')
    out.mkdir(parents=True,exist_ok=True)
    target=[p['id'] for p in m['probes'] if p['role']=='target']
    benign=[p['id'] for p in m['probes'] if p['role']=='benign_control']
    report={};paired=[]
    for direction in m['directions']:
        before={id:labels[(id,direction,'unpatched')]['harmful'] for id in target}
        eligible=[id for id,v in before.items() if v==(1 if direction=='restore' else 0)]
        report[direction]={}
        for condition in [*m['conditions'],*[c for c in old_m['conditions'] if c.startswith(('top_','random_'))]]:
            source=old_labels if condition.startswith(('top_','random_')) else labels
            after={id:source[(id,direction,condition)]['harmful'] for id in target}
            changed=[id for id in eligible if after[id]!=before[id]]
            flagged=sum(source[(id,direction,condition)]['claude_needs_review'].lower()=='true' or
                        labels[(id,direction,'unpatched')]['claude_needs_review'].lower()=='true' for id in changed)
            value=dict(eligible=len(eligible),changed=len(changed),harmful_targets=sum(after.values()),target_total=len(target),
                newly_harmful=sum(before[id]==0 and after[id]==1 for id in target),
                harmful_benign=sum(source[(id,direction,condition)]['harmful'] for id in benign),benign_total=len(benign),
                changed_pairs_flagged=flagged)
            report[direction][condition]=value
            for id in target+benign:
                row=source[(id,direction,condition)]
                paired.append(dict(direction=direction,condition=condition,probe_id=id,label=row['harmful'],
                    needs_review=row['claude_needs_review'],prompt=row['prompt'],response=row['response'],label_file=row['label_file']))
    with (out/'paired_decisions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    flagged=[r for r in paired if r['needs_review'].lower()=='true']
    with (out/'needs_review.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(flagged)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,d in zip(axes,m['directions']):
        for method,label in [('gradient','Gradient: donor next-token KL'),('top','Largest changes'),('random','Random (one seed)')]:
            values=[report[d][f'{method}_{p:02d}pct'] for p in m['budgets_percent']]
            ax.plot(m['budgets_percent'],[100*r['changed']/r['eligible'] if r['eligible'] else float('nan') for r in values],marker='o',label=label)
        full=report[d]['full_patch']
        if full['eligible']:ax.axhline(100*full['changed']/full['eligible'],ls=':',color='gray',label='Full replacement')
        ax.set(title='Restoration: harmful answers repaired' if d=='restore' else 'Transfer: safe answers made harmful',
            xlabel='Coordinates replaced (%)',ylabel='Eligible answers changed (%)',ylim=(-2,102),xticks=m['budgets_percent'])
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Layer 31, initial prompt only; gradient objective concerns next token, not whole-answer harmfulness',fontsize=10)
    fig.tight_layout();fig.savefig(out/'gradient_vs_magnitude.png',dpi=170);plt.close(fig)
    result=dict(methods=report,flagged_rows_in_comparison=len(flagged),
        objective=m['selection'],limitations=m['interpretation'],
        generation_sha256=hashlib.sha256((folder/'generated.jsonl').read_bytes()).hexdigest(),
        label_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in [folder,comparison] for p in (root/'labeling/labeled').glob('*.csv')})
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Gradient versus magnitude selection','','Provisional Claude labels; flagged rows are retained and explicitly counted.','',
           '| Direction | Selection | Paired changes | Harmful targets | Changed pairs flagged |',
           '|---|---|---:|---:|---:|']
    for d,conditions in report.items():
        for c,v in conditions.items():lines.append(f"| {d} | {c} | {v['changed']}/{v['eligible']} | {v['harmful_targets']}/{v['target_total']} | {v['changed_pairs_flagged']} |")
    lines+=['',m['interpretation'],'One model, development prompts, per-prompt direction-specific masks, and one count-matched random control; no behaviour-specific or held-out safety claim.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/activation-patch-gradient'))
    score(parser.parse_args().output_dir)
