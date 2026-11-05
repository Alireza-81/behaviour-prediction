"""Compare five coordinate-scoring rules using one development-only capture.

No weight training, response generation, or judge/API calls.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path
import torch

from behaviour_prediction.monitor.shared_mask import identity
from behaviour_prediction.monitor.multilayer_patching import all_layer_scores,sequence_states
from behaviour_prediction.monitor.dynamic_gradient_patching import step_context
from compare_gradient_only_selection import file_hash,expected_keys,aggregate,selections
from run_multilayer_activation_patch import models,encode,validate_artifact
from run_dynamic_gradient_activation_patch import atomic_json,reference_input

METHODS={
    'signed_replacement':dict(field='scores',label='Signed gΔ',formula='sum_t(-g_t * (donor_t - recipient_t))'),
    'absolute_replacement':dict(field='attribution_abs',label='Absolute gΔ',formula='sum_t(abs(g_t * (donor_t - recipient_t)))'),
    'gradient_only':dict(field='gradient_abs',label='|gradient|',formula='sum_t(abs(g_t))'),
    'activation_change':dict(field='delta_rms',label='Change RMS',formula='sqrt(mean_t((donor_t - recipient_t)^2))'),
    'gradient_activation':dict(field='gradient_activation_abs',label='|g × activation|',formula='sum_t(abs(g_t * recipient_t))'),
}
SCOPES=['global_all','layer31','distributed','earlier']
SCHEMA=1


def setup(source,folder):
    plan=json.loads((source/'development_plan.json').read_text());old=json.loads((source/'mask.json').read_text())
    if identity(plan,'plan_id')!=plan['plan_id']:raise ValueError('Invalid original development plan')
    validate_artifact(old,plan)
    if file_hash(source/'development_scores.pt')!=old['score_cache_sha256']:raise ValueError('Original scores changed')
    binding=dict(schema=SCHEMA,methods=METHODS,source_dir=str(source),
        source_sha256={name:file_hash(source/name) for name in ['development_plan.json','mask.json','development_scores.pt']},
        normalization='Per-example global L1, then equal mean. Layer31-only retains layer-only L1 for all five methods.',
        views=['shared','restore','transfer'],budgets=plan['budgets'],development_plan_id=plan['plan_id'],random_seeds=[17,42,123])
    binding['comparison_id']=identity(binding,'comparison_id')
    folder.mkdir(parents=True,exist_ok=True);path=folder/'comparison_plan.json'
    if path.exists() and json.loads(path.read_text())!=binding:raise ValueError('Comparison definition/source changed; use another output folder')
    if not path.exists():atomic_json(path,binding)
    return plan,old,binding


def validate_cache(cache,plan,complete=False):
    expected=expected_keys(plan)
    if not set(cache)<=expected or (complete and set(cache)!=expected):raise ValueError('Unexpected/incomplete development entries')
    shape=None
    for key,entry in cache.items():
        if entry.get('scoring_schema')!=SCHEMA:raise ValueError('This cache lacks the five-score schema; use capture')
        if entry['prediction_positions']!=len(plan['references'][key]['token_ids']):raise ValueError('Reference position count differs')
        for method,spec in METHODS.items():
            if spec['field'] not in entry:raise ValueError(f'Missing score field {spec["field"]}; rerun the combined capture')
            v=entry[spec['field']]
            if v.ndim!=2 or not torch.isfinite(v).all():raise ValueError('Invalid scoring tensor')
            if shape is not None and tuple(v.shape)!=shape:raise ValueError('Score dimensions differ')
            shape=tuple(v.shape)
            if method!='signed_replacement' and (v<0).any():raise ValueError('Magnitude score is negative')
    return expected


def capture(source,folder):
    plan,old,binding=setup(source,folder);path=folder/'scoring_statistics.pt'
    cache=torch.load(path,map_location='cpu',weights_only=True) if path.exists() else {}
    expected=validate_cache(cache,plan)
    if set(cache)==expected:
        print('All scoring statistics saved; no model loaded.');return
    for model,tokenizer in models(plan):
        for p in plan['development_probes']:
            encoded=encode(model,tokenizer,p,plan);boundary=encoded['input_ids'].shape[1]
            for d in plan['directions']:
                key=f"{d}:{p['id']}"
                if key in cache:continue
                reference=plan['references'][key]
                if encoded['input_ids'].cpu().tolist()!=reference['prompt_token_ids']:raise ValueError('Tokenization changed')
                teacher=reference_input(encoded,reference['token_ids'])
                if teacher['input_ids'].shape[1]>plan['expected_model_config']['model']['max_sequence_length']:raise ValueError('Reference exceeds context')
                recipient=20 if d=='restore' else 0;donor=20-recipient
                with step_context(model,donor):states,logits=sequence_states(model,teacher,boundary)
                with step_context(model,recipient):result=all_layer_scores(model,teacher,boundary,states,logits,include_scoring_stats=True)
                if result['scores'].shape!=(32,4096):raise ValueError('Wrong model dimensions')
                result['scoring_schema']=SCHEMA;cache[key]=result
                validate_cache({key:result},plan)
                temporary=path.with_suffix('.tmp');torch.save(cache,temporary);temporary.replace(path)
                del states,logits,result
                print(f'{len(cache)}/{len(expected)} backward passes saved; all five scoring rules, no generation.',flush=True)


def coordinates(mask):return {(int(layer),i) for layer,values in mask.items() for i in values}


def overlap(a,b):
    left,right=coordinates(a),coordinates(b)
    if not left or len(left)!=len(right):raise ValueError('Coordinate budgets differ or empty')
    shared=len(left&right)
    return dict(overlap_count=shared,coordinate_count=len(left),overlap_percent=100*shared/len(left),jaccard=shared/len(left|right))


def uniform_random_masks(shape,budgets,seeds):
    depth,width=shape;result={}
    for seed in seeds:
        order=torch.randperm(depth*width,generator=torch.Generator().manual_seed(seed))
        for budget in budgets:
            sites={}
            for flat in order[:budget].tolist():sites.setdefault(str(flat//width),[]).append(flat%width)
            result[f'random{seed}_{budget}']={k:sorted(v) for k,v in sites.items()}
    return result


def compare(source,folder):
    plan,old,binding=setup(source,folder);path=folder/'scoring_statistics.pt'
    cache=torch.load(path,map_location='cpu',weights_only=True);keys=validate_cache(cache,plan,complete=True)
    views=dict(shared=keys,restore={k for k in keys if k.startswith('restore:')},transfer={k for k in keys if k.startswith('transfer:')})
    if not all(views.values()):raise ValueError('Need both directions')
    masks={};aggregates={};counts=[];summary=[];pairwise=[];directional=[];vs_original=[]
    shape=cache[next(iter(keys))]['scores'].shape
    old_masks={k:v for k,v in old['selected_indices'].items() if '_random' not in k}
    for view,view_keys in views.items():
        masks[view]={};aggregates[view]={}
        for method,spec in METHODS.items():
            whole,last=aggregate(cache,spec['field'],view_keys)
            chosen=selections(whole,last,plan['budgets']);masks[view][method]=chosen
            aggregates[view][method]=dict(global_scores=whole,layer31_scores=last)
            for condition,sites in chosen.items():
                scope,budget=condition.rsplit('_',1);budget=int(budget)
                if len(coordinates(sites))!=budget:raise ValueError('Wrong selected budget')
                summary.append(dict(view=view,method=method,scope=scope,budget=budget,blocks_used=len(sites),layer31_count=len(sites.get(str(shape[0]-1),[]))))
                for layer in range(shape[0]):counts.append(dict(view=view,method=method,scope=scope,budget=budget,layer=layer,selected_count=len(sites.get(str(layer),[]))))
                if view=='shared':vs_original.append(dict(method=method,scope=scope,budget=budget,**overlap(old_masks[condition],sites)))
        for a,b in itertools.combinations(METHODS,2):
            for scope in SCOPES:
                for budget in plan['budgets']:
                    name=f'{scope}_{budget}'
                    pairwise.append(dict(view=view,method_a=a,method_b=b,scope=scope,budget=budget,**overlap(masks[view][a][name],masks[view][b][name])))
    for method in METHODS:
        for a,b in [('restore','transfer'),('shared','restore'),('shared','transfer')]:
            for scope in SCOPES:
                for budget in plan['budgets']:
                    name=f'{scope}_{budget}'
                    directional.append(dict(method=method,view_a=a,view_b=b,scope=scope,budget=budget,**overlap(masks[a][method][name],masks[b][method][name])))
    random=uniform_random_masks(shape,plan['budgets'],binding['random_seeds'])
    random_counts=[];random_overlap=[]
    for name,sites in random.items():
        seed,budget=name.rsplit('_',1);budget=int(budget)
        for layer in range(shape[0]):random_counts.append(dict(seed=seed,budget=budget,layer=layer,selected_count=len(sites.get(str(layer),[]))))
        for method in METHODS:random_overlap.append(dict(seed=seed,method=method,budget=budget,**overlap(masks['shared'][method][f'global_all_{budget}'],sites)))
    artifact=dict(comparison_id=binding['comparison_id'],schema=SCHEMA,selected_indices=masks,uniform_random_global_masks=random,
        statistics_sha256=file_hash(path),same_shared_sites_all_prompts_and_directions=True,
        note='Shared selections are patch candidates, not evaluated interventions. Direction-specific masks are diagnostics.')
    artifact['mask_id']=identity(artifact);atomic_json(folder/'scoring_masks.json',artifact)
    temporary=folder/'aggregates.tmp';torch.save(aggregates,temporary);temporary.replace(folder/'aggregate_scores.pt')
    def save(name,rows):
        with (folder/name).open('w',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save('selection_summary.csv',summary);save('layer_counts.csv',counts);save('pairwise_overlap.csv',pairwise)
    save('direction_agreement.csv',directional);save('overlap_with_original.csv',vs_original)
    save('random_layer_counts.csv',random_counts);save('random_selection_overlap.csv',random_overlap)
    # Same-capture pairwise overlaps are primary; legacy overlap is a separate diagnostic.
    previous=torch.load(source/'development_scores.pt',map_location='cpu',weights_only=True)
    reproduction=dict(bit_exact_entries=sum(torch.equal(cache[k]['scores'],previous[k]['scores']) for k in keys),total_entries=len(keys),
        max_absolute_difference=max(float((cache[k]['scores']-previous[k]['scores']).abs().max()) for k in keys))
    report=dict(comparison_id=binding['comparison_id'],methods=METHODS,development_backward_passes=len(keys),
        evaluation_prompts_used_for_selection=False,model_training=False,new_response_generation=False,new_labeling=False,
        budgets=plan['budgets'],old_weighted_reproduction=reproduction,
        zero_score_entries={method:sum(float(cache[k][spec['field']].abs().sum())==0 for k in keys) for method,spec in METHODS.items()},
        shared_global_summary=[r for r in summary if r['view']=='shared' and r['scope']=='global_all'],
        limitations=['Selection comparison only: different masks do not establish better harmfulness control.',
            'Gradient-based scores use the same donor-distribution KL proxy, not a judged harmfulness loss.',
            'Magnitude-only scores discard whether a donor replacement is beneficial or counterproductive.',
            'Layer31-only normalization differs from unrestricted normalization, but is held fixed between scoring methods.',
            'Raw gradient magnitudes and activation-change magnitudes depend on representation scale.',
            'Restoration and transfer use their original different reference continuations as well as different endpoints.',
            'Uniform random masks are a selection-layout reference, not layer-count-matched behavioural controls.'])
    atomic_json(folder/'comparison_report.json',report)
    plot(folder,plan,shape,summary,counts,pairwise,directional,random_counts)
    lines=['# Five scoring methods: selection comparison','','No new responses, labels or model training.','',
        '| Method | Budget | Blocks used | Coordinates in last block |','|---|---:|---:|---:|']
    for row in report['shared_global_summary']:lines.append(f"| {row['method']} | {row['budget']} | {row['blocks_used']} | {row['layer31_count']} |")
    lines+=['',f"Original weighted-score reproduction: {reproduction['bit_exact_entries']}/{len(keys)} entries bit-exact. Primary pairwise comparisons use the same new backward passes.",'',*report['limitations']]
    (folder/'summary.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


def plot(folder,plan,shape,summary,counts,pairwise,directional,random_counts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np
    methods=list(METHODS);labels=[METHODS[m]['label'] for m in methods]
    budgets=plan['budgets'];depth=shape[0]
    fig,axes=plt.subplots(len(budgets),1,figsize=(15,3*len(budgets)),squeeze=False)
    cmap=plt.get_cmap('viridis').with_extremes(bad='#eeeeee')
    for ax,budget in zip(axes[:,0],budgets):
        matrix=[[next(r['selected_count'] for r in counts if r['view']=='shared' and r['scope']=='global_all' and r['method']==m and r['budget']==budget and r['layer']==layer) for layer in range(depth)] for m in methods]
        random_mean=[sum(r['selected_count'] for r in random_counts if r['budget']==budget and r['layer']==layer)/3 for layer in range(depth)]
        matrix.append(random_mean)
        im=ax.imshow(np.ma.masked_equal(matrix,0),cmap=cmap,norm=LogNorm(vmin=.1,vmax=max(budgets)),aspect='auto')
        ax.set(title=f'Unrestricted shared masks: {budget} total coordinates',yticks=range(6),yticklabels=labels+['Uniform random (3-seed mean)'],xticks=list(range(depth)),xlabel='Transformer block (zero based)')
        fig.colorbar(im,ax=ax,label='Selected count (log scale; gray = zero)')
    fig.tight_layout();fig.savefig(folder/'layer_distribution.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,len(budgets),figsize=(6*len(budgets),5),squeeze=False)
    for ax,budget in zip(axes[0],budgets):
        matrix=np.eye(5)*100
        for r in pairwise:
            if r['view']=='shared' and r['scope']=='global_all' and r['budget']==budget:
                i,j=methods.index(r['method_a']),methods.index(r['method_b']);matrix[i,j]=matrix[j,i]=r['overlap_percent']
        im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues');ax.set(title=f'{budget} coordinates',xticks=range(5),yticks=range(5),yticklabels=labels)
        ax.set_xticklabels(labels,rotation=40,ha='right',fontsize=8)
        for i in range(5):
            for j in range(5):ax.text(j,i,f'{matrix[i,j]:.0f}%',ha='center',va='center',color='white' if matrix[i,j]>55 else 'black',fontsize=9)
    fig.suptitle('Same coordinates selected? Shared unrestricted masks; overlap divided by budget')
    fig.tight_layout();fig.savefig(folder/'scoring_overlap.png',dpi=180);plt.close(fig)
    matrix=[[next(r['overlap_percent'] for r in directional if r['method']==m and r['scope']=='global_all' and r['budget']==budget and r['view_a']=='restore' and r['view_b']=='transfer') for budget in budgets] for m in methods]
    fig,ax=plt.subplots(figsize=(7,5));im=ax.imshow(matrix,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set(xticks=range(len(budgets)),xticklabels=budgets,yticks=range(5),yticklabels=labels,xlabel='Total coordinate budget',title='Restoration vs transfer: selected-coordinate agreement')
    for i,row in enumerate(matrix):
        for j,value in enumerate(row):ax.text(j,i,f'{value:.1f}%',ha='center',va='center',color='white' if value>55 else 'black')
    fig.colorbar(im,ax=ax,label='Overlap (%)');fig.tight_layout();fig.savefig(folder/'direction_agreement.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['capture','compare'])
    p.add_argument('--source-dir',type=Path,default=Path('outputs/activation-patch-multilayer'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/activation-selection-scores'))
    args=p.parse_args()
    from multilayer_generation_workers import generation_lock
    with generation_lock(args.output_dir):
        (capture if args.action=='capture' else compare)(args.source_dir,args.output_dir)
