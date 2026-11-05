"""Capture gradient-only sensitivities, then compare frozen coordinate selections.

Selection analysis only: never trains weights, generates responses or calls a judge.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import torch
from behaviour_prediction.monitor.shared_mask import identity
from behaviour_prediction.monitor.multilayer_patching import all_layer_scores,sequence_states,select_multilayer
from behaviour_prediction.monitor.dynamic_gradient_patching import step_context
from run_multilayer_activation_patch import models,encode,validate_artifact
from run_dynamic_gradient_activation_patch import atomic_json,reference_input


def file_hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(source,folder):
    plan=json.loads((source/'development_plan.json').read_text())
    old=json.loads((source/'mask.json').read_text())
    if identity(plan,'plan_id')!=plan['plan_id']:raise ValueError('Invalid original development plan')
    validate_artifact(old,plan)
    if file_hash(source/'development_scores.pt')!=old['score_cache_sha256']:raise ValueError('Original score cache changed')
    binding=dict(source_dir=str(source),source_sha256={name:file_hash(source/name) for name in ['development_plan.json','mask.json','development_scores.pt']},
        definition='Absolute gradient of the mean-token donor KL, summed over reference prediction positions; L1 normalize per development prompt/direction across all blocks, then average. Layer31-only comparator retains layer-only normalization.',
        budgets=plan['budgets'],development_plan_id=plan['plan_id'])
    binding['comparison_id']=identity(binding,'comparison_id')
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'comparison_plan.json'
    if path.exists() and json.loads(path.read_text())!=binding:raise ValueError('Source or definition changed; use a new output folder')
    if not path.exists():atomic_json(path,binding)
    return plan,old,binding


def expected_keys(plan):
    if any(p['role']!='target' for p in plan['development_probes']):raise ValueError('Unexpected development role')
    return {f"{d}:{p['id']}" for p in plan['development_probes'] for d in plan['directions']}


def capture(source,folder):
    plan,old,binding=setup(source,folder);expected=expected_keys(plan)
    path=folder/'gradient_scores.pt'
    cache=torch.load(path,map_location='cpu',weights_only=True) if path.exists() else {}
    if not set(cache)<=expected:raise ValueError('Unexpected capture keys')
    if set(cache)==expected:
        print('All development gradients already saved; no model loaded.');return
    # Identical endpoint model verification, reference tokens, loss and graph as before.
    for model,tokenizer in models(plan):
        for p in plan['development_probes']:
            encoded=encode(model,tokenizer,p,plan);boundary=encoded['input_ids'].shape[1]
            for d in plan['directions']:
                key=f"{d}:{p['id']}"
                if key in cache:continue
                reference=plan['references'][key]
                if encoded['input_ids'].cpu().tolist()!=reference['prompt_token_ids']:raise ValueError('Development tokenization changed')
                teacher=reference_input(encoded,reference['token_ids'])
                if teacher['input_ids'].shape[1]>plan['expected_model_config']['model']['max_sequence_length']:raise ValueError('Reference exceeds context')
                recipient=20 if d=='restore' else 0;donor=20-recipient
                with step_context(model,donor):states,logits=sequence_states(model,teacher,boundary)
                with step_context(model,recipient):result=all_layer_scores(model,teacher,boundary,states,logits,include_gradient_stats=True)
                if result['prediction_positions']!=len(reference['token_ids']):raise ValueError('Reference position count differs')
                if result['scores'].shape!=(32,4096):raise ValueError('Wrong model dimensions')
                cache[key]=result
                temporary=path.with_suffix('.tmp');torch.save(cache,temporary);temporary.replace(path)
                del states,logits,result
                print(f'{len(cache)}/{len(expected)} development gradient calculations saved; no answer generation.',flush=True)


def aggregate(cache,field,keys):
    vectors=[cache[k][field].detach().cpu().double() for k in sorted(keys)]
    if any(v.ndim!=2 or v.shape!=vectors[0].shape or not torch.isfinite(v).all() for v in vectors):raise ValueError('Invalid scores')
    if field=='gradient_abs' and any((v<0).any() for v in vectors):raise ValueError('Absolute gradient cannot be negative')
    whole=torch.stack([v/v.abs().sum().clamp_min(1e-12) for v in vectors]).mean(0)
    last=torch.stack([v[-1]/v[-1].abs().sum().clamp_min(1e-12) for v in vectors]).mean(0)
    return whole,last


def selections(whole,last,budgets):
    result=select_multilayer(whole,budgets,[])
    local=whole.clone();local[-1]=last
    result.update({k:v for k,v in select_multilayer(local,budgets,[]).items() if k.startswith('layer31_')})
    return result


def compare_masks(old,new):
    rows=[];counts=[]
    for name,b in new.items():
        a=old[name]
        left={(int(layer),coordinate) for layer,coordinates in a.items() for coordinate in coordinates}
        right={(int(layer),coordinate) for layer,coordinates in b.items() for coordinate in coordinates}
        if len(left)!=len(right):raise ValueError('Mismatched budgets')
        shared=len(left&right)
        rows.append(dict(condition=name,coordinate_count=len(right),overlap_count=shared,
            retained_percent=100*shared/len(right),jaccard=shared/len(left|right),
            weighted_blocks_used=len(a),gradient_only_blocks_used=len(b),
            weighted_layer31_count=len(a.get('31',[])),gradient_only_layer31_count=len(b.get('31',[]))))
        for layer in range(32):
            counts.append(dict(condition=name,layer=layer,weighted_count=len(a.get(str(layer),[])),gradient_only_count=len(b.get(str(layer),[]))))
    return rows,counts


def compare(source,folder):
    plan,old,binding=setup(source,folder);expected=expected_keys(plan)
    path=folder/'gradient_scores.pt';cache=torch.load(path,map_location='cpu',weights_only=True)
    previous=torch.load(source/'development_scores.pt',map_location='cpu',weights_only=True)
    if set(cache)!=expected:raise ValueError('Incomplete gradient capture; rerun capture first')
    for key in expected:
        if cache[key]['prediction_positions']!=len(plan['references'][key]['token_ids']):raise ValueError('Wrong reference length')
    whole,last=aggregate(cache,'gradient_abs',expected)
    new=selections(whole,last,plan['budgets'])
    rerun_whole,rerun_last=aggregate(cache,'scores',expected)
    rerun=selections(rerun_whole,rerun_last,plan['budgets'])
    old_selected={k:v for k,v in old['selected_indices'].items() if '_random' not in k}
    rows,counts=compare_masks(old_selected,new)
    rerun_rows,_=compare_masks(rerun,new)
    exact=sum(torch.equal(cache[k]['scores'],previous[k]['scores']) for k in expected)
    artifact=dict(comparison_id=binding['comparison_id'],selection='Absolute gradient only; same development examples, reference continuations and donor-KL objective.',
        selected_indices=new,aggregate_scores=whole.tolist(),layer31_scores=last.tolist(),
        budgets=plan['budgets'],same_sites_all_prompts_and_directions=True,gradient_cache_sha256=file_hash(path))
    artifact['mask_id']=identity(artifact);atomic_json(folder/'gradient_only_mask.json',artifact)
    def save(name,values):
        with (folder/name).open('w',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    save('coordinate_overlap.csv',rows);save('layer_counts.csv',counts);save('same_capture_overlap.csv',rerun_rows)
    save('coordinate_ranking.csv',[dict(layer=i,coordinate=j,gradient_only_score=float(whole[i,j]),old_weighted_score=old['aggregate_scores'][i][j])
        for i in range(whole.shape[0]) for j in range(whole.shape[1])])
    report=dict(comparison_id=binding['comparison_id'],development_calculations=len(expected),previous_weighted_scores_bit_exact=exact,
        max_absolute_weighted_score_difference=max(float((cache[k]['scores']-previous[k]['scores']).abs().max()) for k in expected),
        comparison_to_original_masks=rows,comparison_using_same_new_gradient_capture=rerun_rows,
        interpretation='Gradient magnitude measures sensitivity, not whether the donor replacement improves the objective or prevents harm. No activation-difference factors enter the gradient-only ranking.',
        no_new_response_generation=True,no_new_labeling=True,
        limitations=['Same donor-KL proxy as before, not a direct harmfulness objective.',
            'Gradient-only loses donor-replacement sign and size information.',
            'Cross-layer gradient magnitude depends on representation scale.',
            'Selection changes alone do not establish better behavioural patching.'])
    atomic_json(folder/'comparison_report.json',report)
    plot(folder,plan,rows,counts)
    lines=['# Gradient-only selection comparison','','No new answers or labels were generated.','',
        '| Condition | Coordinates retained | Old blocks | Gradient-only blocks | Old layer31 count | Gradient-only layer31 count |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['condition']} | {r['overlap_count']}/{r['coordinate_count']} | {r['weighted_blocks_used']} | {r['gradient_only_blocks_used']} | {r['weighted_layer31_count']} | {r['gradient_only_layer31_count']} |")
    lines+=['',f'Weighted-score reproduction: {exact}/{len(expected)} entries bit-exact. Same-capture comparisons are also saved, to separate numerical rerun differences from the scoring change.','',report['interpretation'],*report['limitations']]
    (folder/'summary.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


def plot(folder,plan,rows,counts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,len(plan['budgets']),figsize=(16,5),sharey=True,squeeze=False)
    for ax,budget in zip(axes[0],plan['budgets']):
        chosen={r['layer']:r for r in counts if r['condition']==f'global_all_{budget}'}
        layers=np.arange(32)
        ax.bar(layers-.2,[chosen.get(i,{}).get('weighted_count',0) for i in layers],width=.4,label='Gradient × activation difference')
        ax.bar(layers+.2,[chosen.get(i,{}).get('gradient_only_count',0) for i in layers],width=.4,label='Absolute gradient only')
        ax.set(title=f'Unrestricted selection: {budget} coordinates',xlabel='Block',ylabel='Selected coordinates',xticks=list(range(0,32,4))+[31])
        ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(folder/'global_layer_distribution.png',dpi=180);plt.close(fig)
    methods=['layer31','global_all','distributed','earlier']
    values=[[next(r['retained_percent'] for r in rows if r['condition']==f'{method}_{budget}') for budget in plan['budgets']] for method in methods]
    fig,ax=plt.subplots(figsize=(7,5));im=ax.imshow(values,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set(xticks=range(len(plan['budgets'])),xticklabels=plan['budgets'],yticks=range(4),yticklabels=methods,
        xlabel='Total coordinate budget',title='Coordinates retained from the original weighted-score mask')
    for i,row in enumerate(values):
        for j,value in enumerate(row):ax.text(j,i,f'{value:.1f}%',ha='center',va='center',color='white' if value>55 else 'black')
    fig.colorbar(im,ax=ax,label='Retained (%)');fig.tight_layout();fig.savefig(folder/'coordinate_overlap.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['capture','compare'])
    p.add_argument('--source-dir',type=Path,default=Path('outputs/activation-patch-multilayer'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/activation-selection-gradient-only'))
    args=p.parse_args()
    from multilayer_generation_workers import generation_lock
    with generation_lock(args.output_dir):
        (capture if args.action=='capture' else compare)(args.source_dir,args.output_dir)
