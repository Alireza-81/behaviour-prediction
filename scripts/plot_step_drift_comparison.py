"""Plot both fixed regularization recipes for scalar monitor controls."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from audit_early_steps import RUNS,STEPS


def plot(output):
    primary=json.loads((output/'report.json').read_text());weak=json.loads((output/'regularization_check/report.json').read_text())
    choices=[('Original42 · L2=1','baseline42',False),('Step only · L2=1','step_only',False),('Drift only · L2=1','drift_only',False),
        ('Step only · L2=1/42','step_only_l2_1over42',True),('Drift only · L2=1/42','drift_only_l2_1over42',True)]
    rows=[]
    for suite in ['monitoring','evaluation']:
        for label,variant,is_weak in choices:
            for task in ['detection','forecast','behaviours']:
                m=weak[suite][task][variant] if is_weak else primary['all_seven'][suite][task][variant]
                rows.append(dict(suite=suite,input=label,task=task,correct=m['tp']+m['tn'],total=m['checkpoints'],
                    **{k:m.get(k) for k in ['accuracy','balanced_accuracy','f1','macro_f1','recall','tp','fp','fn','tn','early_warnings','onset_opportunities','false_warnings']}))
    with (output/'including_regularization_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
    for ax,(task,measure,title) in zip(axes,[('detection','balanced_accuracy','Current harmfulness: balanced accuracy'),('forecast','balanced_accuracy','Next-checkpoint harm: balanced accuracy'),('behaviours','macro_f1','Behaviour identification: macro F1')]):
        part=[r for r in rows if r['suite']=='monitoring' and r['task']==task]
        values=[r[measure]*100 for r in part]
        ax.barh(range(5),values,color=['#666666','#c38b38','#2166ac','#dfb778','#71a6ce'])
        for i,value in enumerate(values):ax.text(value+.7,i,f'{value:.1f}',va='center',fontsize=9)
        ax.set(yticks=range(5),yticklabels=[label for label,_,_ in choices],xlim=(0,107),xlabel='Percent',title=title)
        ax.grid(axis='x',alpha=.15)
    axes[0].invert_yaxis();fig.suptitle('Seven whole-run holdouts · monitoring labels · fixed threshold0.5')
    fig.tight_layout();fig.savefig(output/'including_regularization_comparison.png',dpi=180);plt.close(fig)
    raw=json.loads((output/'monitoring-predictions.json').read_text())
    extra=json.loads((output/'regularization_check/monitoring-predictions.json').read_text())
    index={(r['variant'],r['run'],r['input_step'],r['target_step']):r for r in raw+extra if r['task']=='forecast'}
    pairs=[(run,a,b) for run in RUNS for a,b in zip(STEPS[1:-1],STEPS[2:])]
    labels=[('baseline42','label','Observed next-checkpoint harm'),('baseline42','predicted_label','Original42 monitor'),
        ('step_only','predicted_label','Step only, original penalty'),('drift_only','predicted_label','Drift only, original penalty'),
        ('drift_only_l2_1over42','predicted_label','Drift only, weaker penalty — identical classes to Original42')]
    fig,axes=plt.subplots(5,1,figsize=(16,8),sharex=True)
    for ax,(variant,field,title) in zip(axes,labels):
        matrix=[[index[(variant,run,a,b)][field] for run,a,b in pairs]]
        ax.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap=ListedColormap(['#edf2f7','#2166ac']));ax.set(yticks=[],title=title)
        for j in range(1,len(RUNS)):ax.axvline(j*6-.5,color='gray',lw=.7)
    axes[-1].set_xticks(range(len(pairs)),[f'{run}\n{a}→{b}' for run,a,b in pairs],rotation=90,fontsize=7)
    fig.suptitle('Forecasting: blue=harmful, pale=non-harmful · input step→target step; no future activation input')
    fig.tight_layout();fig.savefig(output/'forecast_presence_regularization.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=Path('outputs/early-steps/step-drift-controls'))
    plot(p.parse_args().output_dir)
