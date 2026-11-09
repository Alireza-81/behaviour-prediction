"""Data overview and matched decoder/control figures; never invent decoder results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from .data import TAGS, read_bundle


def summarize(root):
    manifest,examples,_=read_bundle(root/'data')
    out=root/'results'
    out.mkdir(exist_ok=True)
    stats=manifest['statistics']
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    splits=['train','validation','test','baseline_test']
    positive=[stats[s]['harmful'] for s in splits]
    negative=[stats[s]['non_harmful'] for s in splits]
    ax.bar(splits,negative,label='Non-harmful',color='#91b9d1')
    ax.bar(splits,positive,bottom=negative,label='Harmful',color='#ce6a60')
    for i,s in enumerate(splits):
        ax.text(i,stats[s]['examples']+8,str(stats[s]['examples']),ha='center')
    ax.set(ylabel='Original labelled responses',title='Experiment A · prepared splits before training oversampling')
    ax.legend()
    fig.savefig(out/'data_overview.png',dpi=170)
    plt.close(fig)
    baseline_path=root/'baselines/report.json'
    evaluation_path=root/'evaluation/report.json'
    reports={}
    if baseline_path.exists():
        reports.update(json.loads(baseline_path.read_text())['test'])
    if evaluation_path.exists():
        reports.update({'decoder_'+k:v for k,v in json.loads(evaluation_path.read_text())['test'].items()})
    if reports:
        names=list(reports)
        fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
        for ax,key,label in zip(axes,['balanced_accuracy_all','macro_f1_training_supported'],
                                ['Harmfulness balanced accuracy','Behaviour macro F1 · training-supported tags']):
            values=[reports[n].get(key) for n in names]
            ax.barh(names,[0 if v is None else v for v in values],color=['#2166ac' if n.startswith('decoder') else '#94a4b5' for n in names])
            for i,v in enumerate(values):
                ax.text((v or 0)+.01,i,'undefined' if v is None else f'{v:.3f}',va='center',fontsize=8)
            ax.set(title=label,xlim=(0,1.12))
        fig.suptitle(f"Held-out {manifest['test_run']} run + held-out prompt groups · per-response predictions")
        fig.savefig(out/'comparison.png',dpi=170)
        plt.close(fig)
    prediction_path=root/'evaluation/predictions.jsonl'
    if prediction_path.exists():
        rows=[json.loads(line) for line in prediction_path.read_text().splitlines()]
        actual=[r for r in rows if r['condition']=='actual' and r['split']=='test']
        steps=sorted({r['step'] for r in actual})
        ids=sorted({r['probe_id'] for r in actual})
        keys=[(step,pid) for step in steps for pid in ids]
        lookup={(r['step'],r['probe_id']):r for r in actual}
        fig,axes=plt.subplots(2,1,figsize=(20,8),sharex=True,layout='constrained')
        for ax,kind in zip(axes,['observed','predicted']):
            grid=np.full((len(TAGS),len(keys)),-1.)
            for col,key in enumerate(keys):
                if key not in lookup:
                    continue
                row=lookup[key]
                if kind=='predicted' and row['prediction'] is None:
                    continue
                tags=row['tags'] if kind=='observed' else row['prediction']['behaviours']
                grid[:,col]=[float(t in tags) for t in TAGS]
            ax.imshow(grid,aspect='auto',cmap=ListedColormap(['#b9bec5','#eff3f8','#2166ac']),vmin=-1,vmax=1)
            ax.set(yticks=range(len(TAGS)),yticklabels=TAGS,title=kind+' behaviour per prompt response')
            for i in range(1,len(steps)):
                ax.axvline(i*len(ids)-.5,color='gray',lw=.8)
        axes[-1].set_xticks([i*len(ids)+(len(ids)-1)/2 for i in range(len(steps))],steps)
        axes[-1].set_xlabel('Optimizer step · each column inside a step is a separate held-out prompt (order in map_columns.json)')
        fig.suptitle('Direct-injection decoder · blue present, pale absent, gray missing/invalid · one held-out run')
        fig.savefig(out/'behaviour_map.png',dpi=170)
        plt.close(fig)
        (out/'map_columns.json').write_text(json.dumps(keys,indent=2)+'\n')
    lines=['# Experiment A status','',
           f"Prepared original activation/response pairs; held-out run {manifest['test_run']}; validation run {manifest['validation_run']}.",
           'Existing provisional Claude labels are verbalized into short template descriptions. Refusal versus safe assistance and causal explanations are not supervised.',
           '','| Split | Responses | Harmful | Non-harmful |','|---|---:|---:|---:|']
    for split in splits:
        s=stats[split]
        lines.append(f"| {split} | {s['examples']} | {s['harmful']} | {s['non_harmful']} |")
    lines+=['','Decoder GPU evaluation: '+('available' if evaluation_path.exists() else '**not run yet**'),'',
            'Class balancing is applied to training only; the test distribution is unchanged.',
            'Fourteen behaviour tags are scored, with tags absent from training reported separately.',
            'Targets describe the saved, potentially token-limited responses, not unlimited future generations.',
            'The same test split was used for controls. It is an inspected development pilot, not a new independent benchmark.',
            '','| Method | Harmfulness accuracy | Balanced accuracy | Harmful recall | Behaviour macro F1 (training-supported) |',
            '|---|---:|---:|---:|---:|']
    fmt=lambda v:'undefined' if v is None else f'{v:.3f}'
    for name,m in reports.items():
        lines.append(f"| {name} | {fmt(m['accuracy_all'])} | {fmt(m.get('balanced_accuracy_all',m.get('balanced_accuracy_decided')))} | {fmt(m.get('harmful_recall_all',m.get('harmful_recall_decided')))} | {fmt(m['macro_f1_training_supported'])} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    print('Saved overview and available results to',out)
