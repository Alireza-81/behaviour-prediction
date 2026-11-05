"""Reproduce multilayer selection and save the audited comparison overview.

Run after score_multilayer_activation_patch.py. CPU only, no API calls.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from behaviour_prediction.monitor.shared_mask import identity
from behaviour_prediction.monitor.multilayer_patching import select_multilayer
from audit_multilayer_label_reuse import audit_source,audit_target
from run_multilayer_activation_patch import validate_artifact
from score_gradient_activation_patch import load_run


def audit(folder):
    out=folder/'results'
    if not (out/'report.json').exists():raise ValueError('Run the scorer first')
    m,journal,labels=load_run(folder,'multilayer')
    bpath=Path('artifacts/multilayer-reuse-v1/reuse.json')
    reuse=audit_target(folder,bpath)
    _,_,source=audit_source(bpath,verify_files=True)
    artifact=json.loads((folder/'mask.json').read_text());plan=json.loads((folder/'development_plan.json').read_text())
    if identity(plan,'plan_id')!=plan['plan_id'] or any(m.get(k)!=v for k,v in plan.items()):raise ValueError('Development plan differs')
    selected=validate_artifact(artifact,plan)
    cache_path=folder/'development_scores.pt'
    if hashlib.sha256(cache_path.read_bytes()).hexdigest()!=artifact['score_cache_sha256']:raise ValueError('Development score cache changed')
    cache=torch.load(cache_path,map_location='cpu',weights_only=True)
    expected={f"{d}:{p['id']}" for p in plan['development_probes'] for d in plan['directions']}
    if set(cache)!=expected or len(expected)!=50:raise ValueError('Wrong development entries')
    if any(p['role']!='target' for p in plan['development_probes']):raise ValueError('Unexpected selection prompt role')
    if {p['id'] for p in plan['development_probes']}&{p['id'] for p in m['probes']}:raise ValueError('Development/evaluation ID overlap')
    if {p['prompt'].strip().casefold() for p in plan['development_probes']}&{p['prompt'].strip().casefold() for p in m['probes']}:raise ValueError('Development/evaluation text overlap')
    vectors=[]
    for key in sorted(cache):
        entry=cache[key];v=entry['scores'].double()
        if v.shape!=(32,4096) or not torch.isfinite(v).all():raise ValueError('Invalid scores')
        if entry['prediction_positions']!=len(plan['references'][key]['token_ids']):raise ValueError('Reference-token alignment differs')
        vectors.append(v)
    whole=torch.stack([v/v.abs().sum().clamp_min(1e-12) for v in vectors]).mean(0)
    final=torch.stack([v[-1]/v[-1].abs().sum().clamp_min(1e-12) for v in vectors]).mean(0)
    if not torch.equal(whole,torch.tensor(artifact['aggregate_scores'],dtype=torch.float64)):raise ValueError('Global score averaging differs')
    if not torch.equal(final,torch.tensor(artifact['layer31_scores'],dtype=torch.float64)):raise ValueError('Layer31 score averaging differs')
    # Recompute every frozen selection and count-matched random control.
    rebuilt=select_multilayer(whole,plan['budgets'],plan['random_seeds'])
    layer_scores=whole.clone();layer_scores[-1]=final
    rebuilt.update({k:v for k,v in select_multilayer(layer_scores,plan['budgets'],plan['random_seeds']).items() if k.startswith('layer31_')})
    if rebuilt!=selected:raise ValueError('Frozen masks do not reproduce from the saved development gradients')
    layer_counts=[]
    for c,sites in selected.items():
        parent=c.split('_random')[0];method,budget=parent.rsplit('_',1);budget=int(budget)
        if sum(map(len,sites.values()))!=budget:raise ValueError('Wrong total patch budget')
        if any(len(indices)!=len(set(indices)) or not all(0<=i<4096 for i in indices) for indices in sites.values()):raise ValueError('Invalid coordinate list')
        if '_random' in c:
            if {k:len(v) for k,v in sites.items()}!={k:len(v) for k,v in selected[parent].items()}:raise ValueError('Random layer counts do not match')
        else:
            if method=='layer31' and set(sites)!={'31'}:raise ValueError('Layer31 mask uses other blocks')
            if method=='earlier' and '31' in sites:raise ValueError('Earlier-only mask includes final block')
            if method=='distributed':
                quotas=[sum(len(v) for k,v in sites.items() if start<=int(k)<stop) for start,stop in [(0,10),(10,21),(21,32)]]
                if quotas!=[budget//3+(i<budget%3) for i in range(3)]:raise ValueError('Distributed quotas differ')
            layer_counts.append(dict(condition=c,budget=budget,blocks_used=len(sites),final_block_count=len(sites.get('31',[])),
                **{f'layer_{i}':len(sites.get(str(i),[])) for i in range(32)}))
    for method in ['layer31','global_all','distributed','earlier']:
        prior=set()
        for budget in plan['budgets']:
            coords={(int(k),i) for k,values in selected[f'{method}_{budget}'].items() for i in values}
            if not prior<=coords:raise ValueError('Selected masks are not nested by budget')
            prior=coords
    # Check repair provenance: no original API text/cost/message identity is changed.
    old_path=folder/'labeling/results.before-json-recovery.jsonl';repairs=[]
    if old_path.exists():
        old={r['index']:r for r in map(json.loads,old_path.read_text().splitlines())}
        new={r['index']:r for r in map(json.loads,(folder/'labeling/results.jsonl').read_text().splitlines())}
        if set(old)!=set(new):raise ValueError('Recovery changed result coverage')
        for index,r in new.items():
            if any(r.get(k)!=old[index].get(k) for k in ['raw_text','cost_usd','message_id','test_id','index','reused_from']):raise ValueError('Recovery altered original evidence')
            if old[index]['status']!=r['status']:
                repairs.append(dict(index=index,harmful=r['judgment']['harmful'],recovery_note=r.get('recovery_note','')))
    report=json.loads((out/'report.json').read_text());summary=report['methods']
    rows=list(csv.DictReader((out/'paired_decisions.csv').open()))
    nself=nfull=0
    for (id,d,c),r in journal.items():
        if c=='self_dynamic':nself+=1
        if c=='full_dynamic':nfull+=1
    overview=[]
    for method in ['layer31','global_all','distributed','earlier']:
        for budget in plan['budgets']:
            c=f'{method}_{budget}'
            for d in m['directions']:
                s=summary[d][c];random=[summary[d][f'{c}_random{seed}']['harmful_targets'] for seed in plan['random_seeds']]
                overview.append(dict(direction=d,method=method,budget=budget,**s,random_harmful_min=min(random),
                    random_harmful_max=max(random),random_harmful_mean=sum(random)/len(random)))
    def csv_save(name,values):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    csv_save('selected_method_summary.csv',overview);csv_save('selection_by_layer.csv',layer_counts)
    if repairs:csv_save('format_recoveries.csv',repairs)
    # Selected-only overview, with random controls shown separately in the original method_comparison plot.
    methods=['layer31','global_all','distributed','earlier']
    names=['Layer 31 only','All layers: unrestricted','Distributed across depth','Earlier layers only (0–30)']
    colors=['#2166ac','#d87b15','#509b48','#af4b63']
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,d in zip(axes,m['directions']):
        for method,name,color in zip(methods,names,colors):
            values=[summary[d][f'{method}_{b}']['harmful_targets'] for b in plan['budgets']]
            ax.plot(plan['budgets'],values,'-o',label=name,color=color,lw=2)
            v=values[-1];b=plan['budgets'][-1]
            ax.annotate(str(v),(b,v),xytext=(6,4 if v>0 else -10),textcoords='offset points',ha='left',fontsize=10,color=color)
        ax.axhline(summary[d]['unpatched']['harmful_targets'],ls='--',color='gray',label='Unpatched')
        ax.set(title='Removing harm: patch baseline → trained' if d=='restore' else 'Transferring harm: patch trained → baseline',
               xlabel='Total residual coordinates replaced',ylabel='Harmful answers out of 30',xticks=plan['budgets'],ylim=(-2,33))
        ax.grid(alpha=.2)
    fig.suptitle('Same total budget; fixed sites across all 45 prompts and both directions',fontsize=12)
    handles,legend_names=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_names,loc='lower center',ncol=3,fontsize=9,frameon=False)
    fig.tight_layout(rect=(0,.12,1,1));fig.savefig(out/'selected_methods.png',dpi=180);plt.close(fig)
    matrix=[[r[f'layer_{i}'] for i in range(32)] for r in layer_counts]
    import numpy as np
    from matplotlib.colors import LogNorm
    fig,ax=plt.subplots(figsize=(14,7))
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee')
    im=ax.imshow(np.ma.masked_equal(matrix,0),aspect='auto',cmap=cmap,norm=LogNorm(vmin=1,vmax=410))
    for y,counts in enumerate(matrix):
        for x,value in enumerate(counts):
            if value:ax.text(x,y,str(value),ha='center',va='center',fontsize=6,color='black' if value>80 else 'white')
    ax.set(xticks=list(range(32)),yticks=list(range(len(layer_counts))),yticklabels=[r['condition'] for r in layer_counts],
           xlabel='Transformer block (zero based)',title='Where the fixed masks patch: coordinate counts by block')
    bar=fig.colorbar(im,ax=ax,label='Coordinate count (log color scale; gray = zero)',ticks=[1,5,10,25,50,100,205,410])
    bar.ax.set_yticklabels(['1','5','10','25','50','100','205','410'])
    fig.tight_layout();fig.savefig(out/'selection_by_layer.png',dpi=170);plt.close(fig)
    best=[r for r in rows if r['direction']=='restore' and r['condition']=='layer31_410']
    csv_save('restore_layer31_410_answers.csv',best)
    flags=[r for r in rows if r['empty']=='True' or r['repetition_flag']=='True']
    if flags:csv_save('generation_failure_flags.csv',flags)
    sources=[folder/'pilot.json',folder/'mask.json',folder/'development_plan.json',cache_path,folder/'generated.jsonl',
             folder/'labeling/test.json',folder/'labeling/results.jsonl',*(folder/'labeling/labeled').glob('*.csv')]
    result=dict(passed=True,generation_rows=len(journal),decided_labels=len(labels),development_score_entries=len(cache),
        same_mask_all_prompts_and_directions=True,all_masks_reproduced=True,random_masks_layer_count_matched=True,
        development_evaluation_overlap=0,self_controls_exact=nself,full_controls_exact=nfull,
        contradictory_identical_answer_labels=0,format_recoveries=repairs,reuse_audit=reuse,source_audit=source,
        quality=dict(truncated=sum(r['truncated']=='True' for r in rows),empty=sum(r['empty']=='True' for r in rows),
            repetition_flags=sum(r['repetition_flag']=='True' for r in rows),
            best_restore_harmful=sum(int(r['harmful']) for r in best if r['role']=='target'),
            best_restore_safe_targets_truncated=sum(r['truncated']=='True' for r in best if r['role']=='target' and r['harmful']=='0'),
            best_restore_benign_truncated=sum(r['truncated']=='True' for r in best if r['role']=='benign_control')),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'data_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','format_recoveries','reuse_audit','source_audit']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=Path('outputs/activation-patch-multilayer'))
    audit(p.parse_args().output_dir)
