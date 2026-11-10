"""Present checkpoint and per-response logistic comparisons as distinct tasks."""
import json
from pathlib import Path


def main(root=Path('outputs/activation-explainer-a/logistic-labels-fixed')):
    checkpoint=json.loads((root/'checkpoint/report.json').read_text())
    both=json.loads((root/'response-both/report.json').read_text())['test']
    single=json.loads((root/'response-layer31/report.json').read_text())['test']
    lines=['# Logistic monitors with corrected labels','',
        'All variants were newly fitted. Layer selection applies during training AND testing. Previous frozen results are preserved.',
        '','## Original checkpoint-level monitor','',
        'Monitoring activations and their matching monitoring labels; leave one entire run out. No separate-evaluation input claim.',
        '','| Behaviour decisions | Both layers (42 summaries) | Layer31 (21 summaries) |','|---|---:|---:|']
    for label,key in [('Accuracy','accuracy'),('Balanced accuracy','balanced_accuracy'),('Macro F1','macro_f1')]:
        lines.append(f"| {label} | {checkpoint['behaviours']['summary_15_31'][key]:.3f} | {checkpoint['behaviours']['summary_31'][key]:.3f} |")
    lines+=['','Correct counts: 333/420 versus 337/420. Frequency baseline 241/420. Improvement is small on this inspected development set.',
        'Checkpoint binary harmfulness remains saturated: all 35 checkpoints are positive. The monitor and always-positive baseline both score 35/35. This is not evidence of useful binary discrimination.',
        '','## Per-response controls matched to the decoder split','',
        '775 training, 155 validation, 175 held-out test responses; validation selects regularization. The test run and prompt IDs are held out. These percentages must not be mixed with checkpoint accuracy.',
        '','| Inputs | Both accuracy | Layer31 accuracy | Both balanced accuracy | Layer31 balanced accuracy |','|---|---:|---:|---:|---:|']
    for method in ['full_vector_linear','norms_linear']:
        a,b=both[method],single[method]
        lines.append(f"| {method} | {a['accuracy_all']:.1%} | {b['accuracy_all']:.1%} | {a['balanced_accuracy_all']:.1%} | {b['balanced_accuracy_all']:.1%} |")
    lines+=['','Layer31 norm inputs are very sensitive (50/55 harmful detected) but yield 60 false alarms on 120 negatives. Both-layer norms detect 43/55 with 58 false alarms. Raw-vector logistic regression gets worse with layer31-only input (6/55 detected versus 20/55 with both). No representation is uniformly better.',
        '','## Decoder rerun status','',
        'New corrected-label data are prepared; new decoder GPU training has not been run locally. Run scripts/run_fixed_label_readers.sh on GPU to train BOTH variants afresh on matched corrected data. Three epochs each; old results untouched. Commands and result transfer are in docs/corrected-label-layer31-monitors.md.']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    categories=['Checkpoint behaviour\n42 vs 21 summaries','Per-response harmfulness\nraw activation vectors','Per-response harmfulness\nactivation norms']
    a=[checkpoint['behaviours']['summary_15_31']['balanced_accuracy'],both['full_vector_linear']['balanced_accuracy_all'],both['norms_linear']['balanced_accuracy_all']]
    b=[checkpoint['behaviours']['summary_31']['balanced_accuracy'],single['full_vector_linear']['balanced_accuracy_all'],single['norms_linear']['balanced_accuracy_all']]
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    for offset,values,label,color in [(-.18,a,'Train/test layers 15+31','#94a4b5'),(.18,b,'Train/test layer31 only','#2166ac')]:
        x=[i+offset for i in range(3)]
        ax.bar(x,values,width=.36,label=label,color=color)
        for xi,value in zip(x,values):ax.text(xi,value+.015,f'{value:.1%}',ha='center')
    ax.set(xticks=range(3),xticklabels=categories,ylim=(0,1.05),ylabel='Balanced accuracy',
           title='Different tasks: compare layers within each group · corrected labels')
    ax.axhline(.5,color='gray',ls=':',lw=.8)
    ax.legend()
    fig.savefig(root/'comparison.png',dpi=170)
    plt.close(fig)
    print('Saved',root/'summary.md')


if __name__=='__main__':main()
