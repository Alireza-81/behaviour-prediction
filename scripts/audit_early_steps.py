"""Validate early-step review labels and plot observed harmful response counts."""
import csv
import json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from label_with_claude import TAGS

ROOT = Path('outputs/early-steps')
STEPS = [0,1,2,3,5,7,10,20]
RUNS = ['child_abuse','educational','hate_speech','neutral_description','professional_email','self_harm','violent_threats']


def main():
    out = ROOT/'data-audit'
    out.mkdir(exist_ok=True)
    summary, unresolved, missing = {}, [], []
    fig, axes = plt.subplots(1,2,figsize=(15,6),sharey=True)
    for ax, suite in zip(axes, ('monitoring','evaluation')):
        probe_path = Path('data/probes/online_probes.jsonl' if suite=='monitoring' else 'data/evaluation/prompts.jsonl')
        probes = {p['id']:p for p in map(json.loads,probe_path.read_text().splitlines())}
        assert len(probes)==30
        suite_stats = {}
        for run in RUNS:
            path=ROOT/f'labeling-{suite}'/'labeled'/f'{run}.csv'
            rows=list(csv.DictReader(path.open()))
            raw=list(csv.DictReader((ROOT/'reviews'/suite/f'{run}.csv').open()))
            original={(int(r['step']),r['probe_id']):r for r in raw}
            keys=set()
            for r in rows:
                key=(int(r['step']),r['probe_id'])
                assert key not in keys and r['run']==run
                keys.add(key)
                p=probes[r['probe_id']]
                assert all(r[f]==p[f] for f in ('prompt','group','category'))
                assert all(r[f]==original[key][f] for f in ('prompt','response','token_limit_reached'))
                assert r['claude_harmful'] in ('0','1','')
                tags=set(filter(None,r['claude_behaviour_tags'].split(';')))
                assert not tags-TAGS
                if r['claude_harmful']=='1': assert tags
                if r['claude_harmful']=='0': assert not tags
                if not r['claude_harmful']:
                    unresolved.append(dict(r, suite=suite, source_file=str(path)))
            assert keys=={(s,p) for s in STEPS for p in probes}
            counts=[sum(r['claude_harmful']=='1' for r in rows if int(r['step'])==s) for s in STEPS]
            unknown=[sum(r['claude_harmful']=='' for r in rows if int(r['step'])==s) for s in STEPS]
            suite_stats[run]=dict(rows=len(rows),labels=dict(Counter(r['claude_harmful'] or 'unresolved' for r in rows)),harmful_counts=dict(zip(STEPS,counts)),unresolved_counts=dict(zip(STEPS,unknown)),review_flags=sum(r['claude_needs_review'].lower()=='true' for r in rows))
            ax.plot(STEPS,counts,marker='o',label=run)
        ax.set(title=f'{suite.capitalize()} prompts',xlabel='Optimizer step',xticks=STEPS,ylim=(-.5,30.5))
        ax.grid(alpha=.2)
        summary[suite]=suite_stats
    for run in RUNS:
        for step in STEPS:
            p=ROOT/run/'activation_monitoring'/f'step-{step:06d}.pt'
            if not p.exists(): missing.append(str(p))
        p=ROOT/run/'run_metadata.json'
        if not p.exists(): missing.append(str(p))
    summary['missing_activation_or_metadata_files']=missing
    (out/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    if unresolved:
        with (out/'unresolved_labels.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(unresolved[0]));w.writeheader();w.writerows(unresolved)
    elif (out/'unresolved_labels.csv').exists():
        with (out/'unresolved_labels.csv').open(newline='') as f:
            fields=next(csv.reader(f))
        with (out/'unresolved_labels.csv').open('w',newline='') as f:
            csv.writer(f).writerow(fields)
    axes[0].set_ylabel('Harmful responses out of 30 (known positives)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Observed labels only — no monitor predictions; unresolved labels are not negatives')
    fig.tight_layout();fig.savefig(out/'harmful_response_counts.png',dpi=180);plt.close(fig)
    lines=['# Early-step data overview','', 'Seven runs; steps 0, 1, 2, 3, 5, 7, 10, 20. Each suite has 30 prompts per checkpoint.','', '| Suite | Responses | Harmful | Non-harmful | Unresolved |','|---|---:|---:|---:|---:|']
    for suite in ('monitoring','evaluation'):
        c=Counter()
        for d in summary[suite].values(): c.update(d['labels'])
        lines.append(f"| {suite} | {sum(c.values())} | {c['1']} | {c['0']} | {c['unresolved']} |")
    lines+=['',f'Missing activation/metadata files: {len(missing)}. '+('Copy these files locally before fitting.' if missing else 'All expected files are present.'),'', 'Forecasting should use adjacent saved checkpoints, excluding step zero as an input: 1→2, 2→3, 3→5, 5→7, 7→10, 10→20. These have unequal time gaps, so report each transition separately as well as combined.','', 'Fit only the original 42-summary logistic baseline with entire runs held out. Fit against monitoring labels; score the same predictions against the separate evaluation labels without refitting. No vector, history or neural-network experiments.','', 'The plot shows observed response labels, not trained monitor performance.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    print(f'Unresolved rows: {len(unresolved)}; see {out}/unresolved_labels.csv')


if __name__=='__main__': main()
