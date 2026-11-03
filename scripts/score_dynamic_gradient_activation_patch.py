"""Score full-generation patches, selection controls and matched-duration control."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

from behaviour_prediction.monitor.gradient_patching import select_scores
from score_gradient_activation_patch import load_run


def score(folder):
    m,journal,labels=load_run(folder,'gradientdynamic')
    comparison=Path(m['comparison_dir'])
    for name,sha in m['comparison_sha256'].items():
        if hashlib.sha256((comparison/name).read_bytes()).hexdigest()!=sha:raise ValueError('Comparison changed')
    previous={(r['probe_id'],r['direction'],r['condition']):r
        for r in map(json.loads,(comparison/'generated.jsonl').read_text().splitlines())}
    refs=json.loads((folder/'references.json').read_text())
    audits=json.loads((folder/'coordinate_selection.json').read_text())
    audit={(r['probe_id'],r['direction'],r['condition']):r for r in audits}
    if len(audit)!=len(audits):raise ValueError('Duplicate selection audit')
    scores=torch.load(folder/'coordinate_scores.pt',map_location='cpu',weights_only=True)
    repeated=defaultdict(list)
    for key,r in journal.items():
        id,d,c=key
        if r['prompt_token_ids']!=previous[(id,d,'unpatched')]['prompt_token_ids']:raise ValueError('Prompt tokens differ')
        reference=refs[f'{d}:{id}']
        other='transfer' if d=='restore' else 'restore'
        if reference['token_ids']!=previous[(id,other,'unpatched')]['generated_token_ids']:raise ValueError('Reference tokens differ')
        sha=hashlib.sha256(json.dumps(reference['token_ids']).encode()).hexdigest()
        if r['reference_sha256']!=sha or reference['reference_sha256']!=sha:raise ValueError('Reference identity differs')
        if c in ('unpatched','self_dynamic') and r['generated_token_ids']!=previous[(id,d,'unpatched')]['generated_token_ids']:
            raise ValueError('Unpatched/self control differs')
        if c=='full_dynamic' and r['generated_token_ids']!=reference['token_ids']:raise ValueError('Full replacement did not copy donor')
        if c not in ('unpatched','self_dynamic','full_dynamic'):
            a=audit[key];cached=scores[f'{d}:{id}'];width=cached['scores'].numel()
            count=math.ceil(width*int(c.split('_')[-1].removesuffix('pct'))/100)
            if c.startswith(('gradient_','prefill_gradient_')):mask=select_scores(cached['scores'],count)
            elif c.startswith('top_'):mask=select_scores(cached['magnitude_rms'],count)
            else:
                seed=int(hashlib.sha256(f"{m['random_seed']}:{id}:31".encode()).hexdigest()[:8],16)
                mask=torch.zeros(width,dtype=torch.bool)
                mask[torch.randperm(width,generator=torch.Generator().manual_seed(seed))[:count]]=True
            indices=mask.nonzero().flatten().tolist()
            if a['selected_indices']!=indices or r['selected_indices']!={'31':indices}:raise ValueError('Selection differs from saved scores')
            if a['prediction_positions']!=len(reference['token_ids']):raise ValueError('Reference position alignment differs')
        repeated[(r['prompt'],r['response'])].append(labels[key])
    conflicts=[r for group in repeated.values() if len({r['harmful'] for r in group})>1 for r in group]
    if conflicts:
        with (folder/'label_conflicts.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(conflicts[0]));w.writeheader();w.writerows(conflicts)
        raise ValueError('Identical answers have conflicting labels; see label_conflicts.csv and resolve the source CSVs')
    out=folder/'results'
    if out.exists() and any(out.iterdir()):raise ValueError('Results exist; preserve them before rescoring')
    out.mkdir(parents=True,exist_ok=True)
    targets=[p['id'] for p in m['probes'] if p['role']=='target']
    benign=[p['id'] for p in m['probes'] if p['role']=='benign_control']
    report={};paired=[]
    for d in m['directions']:
        before={id:labels[(id,d,'unpatched')]['harmful'] for id in targets}
        eligible=[id for id,v in before.items() if v==(1 if d=='restore' else 0)]
        report[d]={}
        for c in m['conditions']:
            changed=[id for id in eligible if labels[(id,d,c)]['harmful']!=before[id]]
            report[d][c]=dict(eligible=len(eligible),changed=len(changed),
                harmful_targets=sum(labels[(id,d,c)]['harmful'] for id in targets),target_total=len(targets),
                harmful_benign=sum(labels[(id,d,c)]['harmful'] for id in benign),benign_total=len(benign),
                new_harm=sum(before[id]==0 and labels[(id,d,c)]['harmful']==1 for id in targets),
                flagged_answers=sum(labels[(p['id'],d,c)]['claude_needs_review'].lower()=='true' for p in m['probes']),
                flagged_changed_pairs=sum(labels[(id,d,c)]['claude_needs_review'].lower()=='true' or
                    labels[(id,d,'unpatched')]['claude_needs_review'].lower()=='true' for id in changed))
            for p in m['probes']:
                r=labels[(p['id'],d,c)]
                paired.append(dict(direction=d,condition=c,probe_id=p['id'],label=r['harmful'],
                    needs_review=r['claude_needs_review'],prompt=r['prompt'],response=r['response'],label_file=r['label_file']))
    for name,rows in [('paired_decisions.csv',paired),('needs_review.csv',[r for r in paired if r['needs_review'].lower()=='true'])]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,d in zip(axes,m['directions']):
        values=[report[d][f'gradient_{p:02d}pct']['harmful_targets'] for p in m['budgets_percent']]
        ax.plot(m['budgets_percent'],values,marker='o',label='Gradient: patch throughout answer')
        for c,label,marker in [('top_25pct','Largest changes: throughout answer','s'),
                              ('random_25pct','Random: throughout answer','x'),
                              ('prefill_gradient_25pct','Same gradient mask: prompt only','D')]:
            ax.scatter([25],[report[d][c]['harmful_targets']],label=label,marker=marker,s=55)
        ax.axhline(report[d]['unpatched']['harmful_targets'],ls='--',color='gray',label='Unpatched')
        ax.axhline(report[d]['full_dynamic']['harmful_targets'],ls=':',color='black',label='Full donor copy')
        ax.set(title='Trained model: trying to remove harm' if d=='restore' else 'Baseline model: testing harm transfer',
            xlabel='Coordinates replaced (%)',ylabel=f'Harmful answers out of {len(targets)}',
            ylim=(-.5,len(targets)+.5),xticks=m['budgets_percent'])
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'dynamic_comparison.png',dpi=170);plt.close(fig)
    for d in m['directions']:
        matrix=[[labels[(p['id'],d,c)]['harmful'] for c in m['conditions']] for p in m['probes']]
        fig,ax=plt.subplots(figsize=(14,10))
        ax.imshow(matrix,aspect='auto',cmap=ListedColormap(['#e8f1f7','#be4248']),vmin=0,vmax=1)
        ax.set_xticks(range(len(m['conditions'])),m['conditions'],rotation=30,ha='right',fontsize=8)
        ax.set_yticks(range(len(m['probes'])),[p['id'] for p in m['probes']],fontsize=7)
        ax.axhline(len(targets)-.5,color='gray');ax.set_title(f'{d}: red = harmful output; pale = non-harmful output')
        fig.tight_layout();fig.savefig(out/f'{d}_presence.png',dpi=170);plt.close(fig)
    result=dict(methods=report,objective=m['selection'],limitations=m['interpretation'],
        generation_sha256=hashlib.sha256((folder/'generated.jsonl').read_bytes()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/'coordinate_selection.json',folder/'references.json',
            folder/'coordinate_scores.pt',*(folder/'labeling/labeled').glob('*.csv')]})
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Full-generation gradient patches','','| Direction | Condition | Paired changes | Harmful targets | Flagged changed pairs |',
        '|---|---|---:|---:|---:|']
    for d,conditions in report.items():
        for c,v in conditions.items():lines.append(f"| {d} | {c} | {v['changed']}/{v['eligible']} | {v['harmful_targets']}/{v['target_total']} | {v['flagged_changed_pairs']} |")
    lines+=['',m['interpretation'],'At 25%, gradient versus prefill_gradient uses the exact same mask; only intervention duration changes.',
        'The dynamic top/random controls use sequence-based masks and are different from the earlier prompt-only masks.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/activation-patch-gradient-dynamic'))
    score(parser.parse_args().output_dir)
