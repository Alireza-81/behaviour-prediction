"""Summarize actual binary diagnostic outputs, including a failed overfit gate."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summarize(root):
    root.mkdir(parents=True,exist_ok=True)
    lines=['# Binary activation-injection diagnostic','',
           'One 0/1 next-token decision, with no descriptions or gold label in the input.',
           'The four activation vectors and original dataset split are retained.',
           'Conditional 0/1 probabilities are uncalibrated. Full-vocabulary greedy validity is reported separately.','']
    for name in ['overfit','train']:
        folder=root/name
        if not (folder/'completed.json').exists():
            lines.append(f'{name}: not completed.\n')
            continue
        completed=json.loads((folder/'completed.json').read_text())
        history=json.loads((folder/'history.json').read_text())
        lines.append(f"{name}: {completed['updates']} optimizer updates; overfit pass = {completed['overfit_passed']}.\n")
        assessments=[r for r in history if 'training' in r]
        steps=[r for r in history if 'decision_loss' in r]
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        axes[0].plot([r['update'] for r in steps],[r['decision_loss'] for r in steps],label='Training decision loss')
        axes[0].set(xlabel='Optimizer update',ylabel='Single-decision full-vocabulary cross-entropy')
        for split,label in [('training','Training'),('validation','Validation')]:
            rows=[r for r in assessments if r[split] is not None]
            if rows:
                axes[1].plot([r['update'] for r in rows],[r[split]['balanced_accuracy'] for r in rows],'-o',label=label)
        axes[1].axhline(.5,color='gray',ls=':',label='Constant-label balanced accuracy')
        axes[1].set(xlabel='Optimizer update',ylabel='Balanced accuracy',ylim=(0,1.05))
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle(name+' · actual behavioural decisions')
        fig.savefig(root/f'{name}_curves.png',dpi=170)
        plt.close(fig)
    for directory in ['overfit-evaluation','evaluation']:
        path=root/directory/'report.json'
        if not path.exists():
            continue
        report=json.loads(path.read_text())
        lines+=['',f'## {directory}','','| Split | Input | Cases | Accuracy | Balanced accuracy | Harmful recall | Greedy digit valid |',
                '|---|---|---:|---:|---:|---:|---:|']
        fmt=lambda v:'undefined' if v is None else f'{v:.3f}'
        for split,conditions in report.items():
            for condition,m in conditions.items():
                lines.append(f"| {split} | {condition} | {m['examples']} | {fmt(m['accuracy'])} | {fmt(m['balanced_accuracy'])} | {fmt(m['harmful_recall'])} | {fmt(m['greedy_valid_fraction'])} |")
        baselines=root/directory/'constant-baselines.json'
        if baselines.exists():
            for split,conditions in json.loads(baselines.read_text()).items():
                for condition,m in conditions.items():
                    lines.append(f"| {split} | {condition} | {m['examples']} | {fmt(m['accuracy'])} | {fmt(m['balanced_accuracy'])} | {fmt(m['harmful_recall'])} | n/a |")
        splits=list(report)
        fig,axes=plt.subplots(1,len(splits),figsize=(4.5*len(splits),4.5),layout='constrained',squeeze=False)
        for ax,split in zip(axes.flat,splits):
            m=report[split]['actual']
            ax.imshow([[m['tn'],m['fp']],[m['fn'],m['tp']]],cmap='Blues')
            for i,row in enumerate([[m['tn'],m['fp']],[m['fn'],m['tp']]]):
                for j,value in enumerate(row):
                    ax.text(j,i,str(value),ha='center',va='center',color='black')
            ax.set(xticks=[0,1],xticklabels=['Non-harmful','Harmful'],yticks=[0,1],yticklabels=['Non-harmful','Harmful'],
                   xlabel='Predicted',ylabel='Observed',title=split)
        fig.suptitle(directory+' · actual injection · per-response confusion')
        fig.savefig(root/f'{directory}_confusion.png',dpi=170)
        plt.close(fig)
    lines+=['','Interpretation: passing the 32-example overfit gate checks memorization capacity only. It is not evidence of generalization.',
            'A failed gate may reflect learning rate, update budget, noisy/inconsistent data or the injection setup; it is not a definitive bug diagnosis.',
            'Compare full training accuracy with validation accuracy before interpreting test results. Test labels never select a checkpoint or threshold.',
            'Zero and baseline-only injections are ablations of the trained reader; their effects alone do not establish causal explanations.']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    print('Saved binary diagnostic summary to',root/'summary.md')
