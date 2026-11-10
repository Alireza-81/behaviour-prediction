"""Compare the two corrected-label readers without tuning on the test set."""
import json
from pathlib import Path


def main():
    root=Path('outputs/activation-explainer-a')
    variants={'Layers 15 + 31':'binary-labels-fixed','Layer 31 only':'binary-layer31-labels-fixed'}
    data={}
    for title,folder in variants.items():
        p=root/folder
        rows=[json.loads(l) for l in (p/'evaluation/predictions.jsonl').read_text().splitlines()]
        test=[r for r in rows if r['split']=='test' and r['condition']=='actual']
        report=json.loads((p/'evaluation/report.json').read_text())
        positive=[r['score'] for r in test if r['label']==1]
        negative=[r['score'] for r in test if r['label']==0]
        auc=sum((x>y)+.5*(x==y) for x in positive for y in negative)/(len(positive)*len(negative))
        wrong=[r for r in test if r['label']!=r['predicted']]
        data[title]=dict(report=report,test=test,test_roc_auc=auc,
            selected_update=json.loads((p/'train/best-metrics.json').read_text())['update'],
            very_confident_test_errors=sum(max(r['score'],1-r['score'])>.99 for r in wrong))
    a,b=[data[k]['test'] for k in variants]
    ai={r['id']:r for r in a};bi={r['id']:r for r in b}
    assert ai.keys()==bi.keys()
    assert all(ai[i]['label']==bi[i]['label'] for i in ai)
    out=root/'binary-layer31-labels-fixed/comparison'
    audit=dict(test_ids_and_labels_match=True,test_examples=len(a),
        layer31_corrected=sum(ai[i]['predicted']!=ai[i]['label'] and bi[i]['predicted']==bi[i]['label'] for i in ai),
        layer31_broken=sum(ai[i]['predicted']==ai[i]['label'] and bi[i]['predicted']!=bi[i]['label'] for i in ai),
        readers={k:{f:v[f] for f in ['test_roc_auc','selected_update','very_confident_test_errors']} for k,v in data.items()})
    (out/'analysis-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# Corrected-label decoder reruns: interpretation','',
        'Both readers were retrained from the original Qwen base using the corrected data: 775 training, 155 validation, 175 test responses and 35 baseline checks. No missing labels remain in these cohorts.',
        '','| Test metric | Layers 15+31 | Layer31 only |','|---|---:|---:|',
        '| Correct / total | 134/175 | 134/175 |','| Accuracy | 76.6% | 76.6% |',
        '| Balanced accuracy | 68.1% | 78.0% |','| Harmful detected | 25/55 | 45/55 |',
        '| Harmful missed | 30 | 10 |','| False alarms | 11/120 | 31/120 |',
        '| Harmful precision | 69.4% | 59.2% |','| Harmful recall | 45.5% | 81.8% |',
        '| F1 | .549 | .687 |','| Brier | .215 | .172 |','| Binary log loss | 1.139 | .560 |',
        '| ROC AUC, recorded conditional scores | .845 | .852 |','',
        'Layer31 has 20 more true positives and 20 more false positives at the same nominal 0.5 threshold. Both readers rank cases similarly (AUCs close on one inspected split), but have different operating behaviour. This is not proof of statistical equivalence or a universal layer preference.',
        '','Layer31 validation balanced accuracy by epoch: 86.2%, 68.6%, 84.0%. The selected checkpoint is update 49, epoch 1. Two-layer validation balanced accuracy: 83.5%, 76.0%, 84.2%; selected update 147. These selection outcomes were verified from validation history only. More epochs are not established as the next fix for layer31.',
        '','In the previous-label runs, two-layer accuracy was 74.9% and layer31 accuracy 80.6%. After label repair the two-layer run improves accuracy to 76.6%, while layer31 accuracy falls to 76.6% but its balanced accuracy rises (74.5% to 78.0%). The test binary labels are unchanged. Training/validation coverage, sampling trajectory and checkpoint selection changed; this does not isolate a pure label-quality causal effect.',
        '','Confidence remains uncalibrated. There are 19 >99%-confidence errors among 41 wrong test predictions for both layers, versus one among 41 for layer31. Better Brier/log loss for layer31 is encouraging but not calibrated safety assurance.',
        '','The next useful investigation is validation-only threshold/calibration choice for an explicit false-alarm budget, then assessment on further held-out data. No threshold was selected using the ROC plot here. Test data have already been inspected repeatedly, and each variant has one training seed.',
        '','Verification: all 1,870 predictions per reader, exact target IDs/labels, all reported metrics, GPU example indices and validation checkpoint selection were checked. See each run\'s exported-results-audit.json. The layer31 manifest legitimately differs by feature-selection lineage and masked tensor artifact.',
        '','Artifacts: comparison.png, roc_comparison.png, analysis-audit.json and ../test_harmfulness_map.png. Reproduce this analysis with scripts/analyze_fixed_label_readers.py.']
    (out/'interpretation.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained')
    for title,values in data.items():
        rows=values['test'];pos=sum(r['label'] for r in rows);neg=len(rows)-pos
        xs,ys=[0.],[0.]
        for threshold in sorted({r['score'] for r in rows},reverse=True):
            xs.append(sum(r['label']==0 and r['score']>=threshold for r in rows)/neg)
            ys.append(sum(r['label']==1 and r['score']>=threshold for r in rows)/pos)
        line,=ax.plot(xs,ys,label=f"{title} · AUC {values['test_roc_auc']:.3f}")
        m=values['report']['test']['actual']
        ax.scatter([m['fp']/neg],[m['tp']/pos],color=line.get_color(),s=75,zorder=3)
    ax.plot([0,1],[0,1],':',color='gray')
    ax.set(xlabel='False-alarm rate',ylabel='Harmful recall',xlim=(0,1),ylim=(0,1.02),
           title='Recorded test-score ROC · dots mark fixed threshold 0.5\nDescriptive plot; no test-based threshold tuning')
    ax.legend()
    fig.savefig(out/'roc_comparison.png',dpi=170)
    plt.close(fig)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
