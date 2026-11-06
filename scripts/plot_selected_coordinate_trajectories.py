"""Retrospective trajectories of fixed top/middle/bottom/random residual subsets.

Uses the original shared signed gradient×difference ranking and saved monitoring
snapshots only. No monitor training, new generation, model loading or API calls.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from behaviour_prediction.monitor.shared_mask import validate_frozen_mask,identity
from behaviour_prediction.trajectories import snapshots,load
from audit_early_steps import RUNS,STEPS

BUDGETS=[102,205,410,2048]
SEEDS=[17,42,123]
METRICS=['rms_delta','signed_mean_delta','mean_absolute_delta','cosine_distance','delta_energy_fraction']
COLORS=dict(top='#2166ac',middle='#bf7b18',bottom='#ac4564',random_mean='#606060')
GROUPS=['harmful','harmful_indirect','benign']
GROUP_NAMES=dict(harmful='Direct requests',harmful_indirect='Indirect requests',benign='Benign requests',all_probes='All 30 monitoring prompts')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinate_subsets(scores,budgets,seeds):
    if scores.ndim!=1 or not torch.isfinite(scores).all():raise ValueError('Invalid frozen ranking')
    width=scores.numel();order=torch.argsort(scores,descending=True,stable=True)
    random_orders={seed:torch.randperm(width,generator=torch.Generator().manual_seed(seed)) for seed in seeds}
    subsets={}
    for budget in budgets:
        if not 0<budget<=width:raise ValueError('Invalid subset budget')
        start=(width-budget)//2
        subsets[str(budget)]=dict(top=sorted(order[:budget].tolist()),middle=sorted(order[start:start+budget].tolist()),
            bottom=sorted(order[-budget:].tolist()),**{f'random{seed}':sorted(indices[:budget].tolist()) for seed,indices in random_orders.items()})
    return subsets,order.tolist()


def measurements(initial,current,indices):
    """Measure each probe first; never compute a norm after averaging probes."""
    if initial.shape!=current.shape or initial.ndim!=2 or not torch.isfinite(current).all() or not torch.isfinite(initial).all():
        raise ValueError('Invalid activation tensors')
    a=initial.double();b=current.double();delta=b-a;selected=delta[:,indices]
    energy=delta.square().sum(1)
    cosine=(1-F.cosine_similarity(a[:,indices],b[:,indices],dim=1,eps=1e-12)).clamp(0,2)
    cosine=torch.where((a[:,indices]==b[:,indices]).all(1),torch.zeros_like(cosine),cosine)
    return dict(rms_delta=selected.square().mean(1).sqrt(),signed_mean_delta=selected.mean(1),
        mean_absolute_delta=selected.abs().mean(1),cosine_distance=cosine,
        delta_energy_fraction=selected.square().sum(1)/energy.clamp_min(1e-30))


def mean_metrics(rows):
    return {metric:sum(row[metric] for row in rows)/len(rows) for metric in METRICS}


def write_csv(path,rows):
    with path.open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def plot(root,output,mask_path,budgets=BUDGETS,runs=RUNS,steps=STEPS):
    if output.exists() and any(output.iterdir()):raise ValueError('Output exists; choose a fresh output directory')
    frozen=json.loads(mask_path.read_text());validate_frozen_mask(frozen)
    if frozen['layer']!=31 or frozen['width']!=4096:raise ValueError('Expected original layer31 shared mask')
    scores=torch.tensor(frozen['aggregate_scores'],dtype=torch.float64)
    subsets,ranking=coordinate_subsets(scores,budgets,SEEDS)
    if '205' in subsets and subsets['205']['top']!=frozen['selected_indices']['shared_05pct']:raise ValueError('Top205 differs from original shared mask')
    labels=[];inputs=[mask_path]
    files=sorted((root/'labeling-monitoring/labeled').glob('*.csv'))
    if not files:raise ValueError('Missing monitoring labels for onset markers')
    for path in files:
        inputs.append(path)
        with path.open(newline='') as handle:labels.extend(csv.DictReader(handle))
    label_map={}
    for row in labels:
        key=row['run'],int(row['step']),row['probe_id']
        if key in label_map or row['claude_harmful'] not in ('0','1'):raise ValueError('Duplicate/unresolved monitoring label')
        label_map[key]=int(row['claude_harmful'])
    onsets={run:min((step for (name,step,id),v in label_map.items() if name==run and v==1),default=None) for run in runs}
    raw=[];heatmaps={};baseline_reference=None;probes_reference=None;model_baselines_equal={};metadata_matches={}
    for run in runs:
        paths=snapshots(root/run/'activation_monitoring',steps)
        baseline=load(paths[0][1]);probes=baseline['metadata']['probes'];a=baseline['activations']['layer_31']
        if probes_reference is None:probes_reference=probes;baseline_reference=a
        if probes!=probes_reference:raise ValueError(f'Different probe schema: {run}')
        if baseline['metadata'].get('pooling')!='last_non_padding' or a.shape!=(30,4096):raise ValueError('Unexpected activation capture layout')
        model_baselines_equal[run]=torch.equal(a,baseline_reference)
        meta_path=root/run/'run_metadata.json';inputs.append(meta_path);meta=json.loads(meta_path.read_text())
        metadata_matches[run]=meta.get('model')==frozen['expected_model_config']
        if not metadata_matches[run]:raise ValueError('Backbone/config differs from mask source')
        heatmaps[run]={}
        for group in GROUPS:
            members=[i for i,p in enumerate(probes) if p['group']==group]
            if len(members)!=10:raise ValueError('Expected ten prompts in each request class')
            heatmaps[run][group]={}
            for subset in ['top','middle','bottom','random42']:heatmaps[run][group][subset]=[]
        for step,path in paths:
            inputs.append(path);current=load(path)
            if current['metadata']['probes']!=probes or current['metadata'].get('pooling')!=baseline['metadata']['pooling']:
                raise ValueError('Prompt/capture schema changed')
            b=current['activations']['layer_31']
            for p in probes:
                if (run,step,p['id']) not in label_map:raise ValueError('Missing corresponding monitoring label')
            for budget,groups in subsets.items():
                for subset,indices in groups.items():
                    values=measurements(a,b,indices)
                    for i,p in enumerate(probes):
                        raw.append(dict(run=run,step=step,layer=31,budget=int(budget),subset=subset,probe_id=p['id'],
                            request_class=p['group'],topic=p['category'],harmful=label_map[(run,step,p['id'])],
                            **{metric:float(value[i]) for metric,value in values.items()}))
            if '205' in subsets:
                delta=b.float()-a.float()
                for group in GROUPS:
                    members=[i for i,p in enumerate(probes) if p['group']==group]
                    for subset in ['top','middle','bottom','random42']:
                        heatmaps[run][group][subset].append(delta[members][:,subsets['205'][subset]].mean(0))
        if '205' in subsets:
            for group in GROUPS:
                for subset in heatmaps[run][group]:heatmaps[run][group][subset]=torch.stack(heatmaps[run][group][subset])
    grouped=defaultdict(list)
    for row in raw:
        base=row['run'],row['step'],row['budget'],row['subset']
        grouped[(*base,'request_class',row['request_class'])].append(row)
        grouped[(*base,'all','all_probes')].append(row)
        grouped[(*base,'probe_topic',row['topic'])].append(row)
    aggregate=[]
    for (run,step,budget,subset,scheme,group),rows in sorted(grouped.items()):
        aggregate.append(dict(run=run,step=step,budget=budget,subset=subset,grouping=scheme,probe_class=group,
            probe_count=len(rows),**mean_metrics(rows)))
    # Random mean/range is variation over three fixed masks, not a confidence interval.
    random_buckets=defaultdict(list)
    for row in aggregate:
        if row['subset'].startswith('random'):
            key=tuple(row[k] for k in ['run','step','budget','grouping','probe_class'])
            random_buckets[key].append(row)
    random_summary=[]
    for key,rows in sorted(random_buckets.items()):
        if len(rows)!=3:raise ValueError('Incomplete random control coverage')
        record=dict(zip(['run','step','budget','grouping','probe_class'],key),probe_count=rows[0]['probe_count'])
        for metric in METRICS:
            record[metric]=sum(r[metric] for r in rows)/3
            record[f'{metric}_min']=min(r[metric] for r in rows);record[f'{metric}_max']=max(r[metric] for r in rows)
        random_summary.append(record)
    selection_rows=[]
    for budget,groups in subsets.items():
        for subset,indices in groups.items():
            selection_rows.append(dict(budget=int(budget),subset=subset,count=len(indices),positive_scores=int((scores[indices]>0).sum()),
                negative_scores=int((scores[indices]<0).sum()),min_score=float(scores[indices].min()),max_score=float(scores[indices].max())))
    artifact=dict(source_mask_id=frozen['mask_id'],source_mask_sha256=sha(mask_path),budgets=budgets,random_seeds=SEEDS,
        ranking=ranking,selected_indices=subsets,ranking_definition='Original shared signed replacement-benefit score; bottom means lowest/most opposing, not inactive.')
    artifact['selection_id']=identity(artifact,'selection_id')
    output.mkdir(parents=True)
    (output/'coordinate_subsets.json').write_text(json.dumps(artifact,indent=2)+'\n')
    write_csv(output/'per_probe_trajectories.csv',raw);write_csv(output/'grouped_trajectories.csv',aggregate)
    write_csv(output/'random_mask_ranges.csv',random_summary);write_csv(output/'subset_score_summary.csv',selection_rows)
    torch.save(heatmaps,output/'signed_coordinate_heatmaps_205.pt')
    figures(output,aggregate,random_summary,heatmaps,onsets,budgets,runs,steps)
    sources={str(path):sha(path) for path in inputs}
    protocol=dict(selection_id=artifact['selection_id'],source_mask_id=frozen['mask_id'],layer=31,budgets=budgets,steps=steps,runs=runs,
        monitoring_probes=probes_reference,onsets=onsets,baseline_bit_exact_across_runs=model_baselines_equal,model_configs_match=metadata_matches,
        aggregation='Each probe: sqrt(mean_coordinate(delta^2)), then mean across probes. Signed means and cosine distance exported too.',
        capture='Stored last non-padding PROMPT token, before answer generation. Ranking derived from reference-answer sequence gradients.',
        subset_windows='Top k ranks, centered k ranks, bottom k ranks, and uniform random k coordinates for seeds17/42/123; subsets fixed for all runs/probes/steps.',
        raw_rows=len(raw),aggregate_rows=len(aggregate),figures=len(list(output.rglob('*.png'))),source_sha256=sources,
        code_sha256=sha(Path(__file__)),limitations=[
            'Retrospective visualization: no monitor was fitted or validated.',
            'Ranking uses a later hate-speech checkpoint; applying it to earlier checkpoints is descriptive and is not a leakage-free monitor evaluation.',
            'Bottom signed-score coordinates may strongly oppose replacement; they are not necessarily unimportant.',
            'RMS corrects for subset size, not coordinate scale. Signed coordinate averages can cancel opposing directions.',
            'The 50% centered window overlaps top/bottom halves; only the two halves partition the layer.',
            'Random bands are ranges across three masks, not confidence intervals.',
            'Snapshot metadata do not contain adapter-file hashes, so exact adapter identity with the patch-ranking source cannot be established from these historical snapshots. Model configs and step-zero activations were checked.'])
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    summary(output,aggregate,random_summary,onsets,protocol,selection_rows)
    print(json.dumps({k:protocol[k] for k in ['layer','budgets','steps','runs','raw_rows','aggregate_rows','figures','onsets','baseline_bit_exact_across_runs']},indent=2))


def figures(output,rows,random,heatmaps,onsets,budgets,runs,steps):
    by_key={(r['run'],r['step'],r['budget'],r['subset'],r['grouping'],r['probe_class']):r for r in rows}
    rand_key={(r['run'],r['step'],r['budget'],r['grouping'],r['probe_class']):r for r in random}
    def series(ax,run,budget,subset,metric,group='all_probes',color=None,label=None,onset=False):
        scheme='all' if group=='all_probes' else 'request_class'
        points=[(rand_key[(run,s,budget,scheme,group)] if subset=='random_mean' else by_key[(run,s,budget,subset,scheme,group)]) for s in steps]
        color=color or COLORS[subset]
        ax.plot(steps,[r[metric] for r in points],'-o',ms=3,lw=1.6,color=color,label=label or subset)
        if subset=='random_mean':ax.fill_between(steps,[r[f'{metric}_min'] for r in points],[r[f'{metric}_max'] for r in points],color=color,alpha=.16)
        if onset and onsets[run] in steps:
            r=points[list(steps).index(onsets[run])];ax.scatter([onsets[run]],[r[metric]],marker='*',s=90,color=color,edgecolor='black',lw=.5,zorder=4)
    def decorate(ax,title,metric):
        ax.set(title=title,xlabel='Optimizer step',ylabel='Mean per-probe RMS change per coordinate' if metric=='rms_delta' else 'Mean signed activation change',xticks=steps)
        ax.grid(alpha=.2)
        if metric=='signed_mean_delta':ax.axhline(0,color='gray',lw=.6)
    def save(fig,path):
        path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)
    model_colors={run:plt.get_cmap('tab10')(i) for i,run in enumerate(runs)}
    for budget in budgets:
        for metric in ['rms_delta','signed_mean_delta']:
            fig,axes=plt.subplots(2,4,figsize=(18,8),squeeze=False)
            for ax,run in zip(axes.flat,runs):
                for subset in ['top','middle','bottom','random_mean']:series(ax,run,budget,subset,metric)
                decorate(ax,run.replace('_',' '),metric)
                if onsets[run] is not None:ax.axvline(onsets[run],ls=':',color='black',alpha=.6)
            for ax in list(axes.flat)[len(runs):]:ax.set_visible(False)
            handles,labels=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='lower right',bbox_to_anchor=(.96,.15),fontsize=11)
            fig.suptitle(f'Layer31 · {budget} coordinates per subset · all monitoring prompts\nDotted line: first observed monitoring harm; gray band: 3 random-mask range',fontsize=12)
            save(fig,output/'overview'/f'{metric}_{budget}.png')
        fig,axes=plt.subplots(3,4,figsize=(19,11),squeeze=False)
        for gi,group in enumerate(GROUPS):
            for si,subset in enumerate(['top','middle','bottom','random_mean']):
                ax=axes[gi,si]
                for run in runs:series(ax,run,budget,subset,'rms_delta',group,color=model_colors[run],label=run,onset=True)
                decorate(ax,f'{GROUP_NAMES[group]} · {subset}','rms_delta')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=9)
        fig.suptitle(f'Layer31 · {budget} equally sized coordinates · models compared\n★ First observed monitoring harm; same ranking across all models',fontsize=12)
        fig.tight_layout(rect=(0,.07,1,.94));fig.savefig(output/f'between_models_{budget}.png',dpi=170);plt.close(fig)
    if 205 in budgets:
        for run in runs:
            fig,axes=plt.subplots(1,3,figsize=(15,4),squeeze=False)
            for ax,group in zip(axes[0],GROUPS):
                for subset in ['top','middle','bottom','random_mean']:series(ax,run,205,subset,'rms_delta',group)
                decorate(ax,GROUP_NAMES[group],'rms_delta')
                if onsets[run] is not None:ax.axvline(onsets[run],ls=':',color='black',alpha=.6)
            axes[0,0].legend(fontsize=8);fig.suptitle(f'{run} · 205 coordinates · request classes separated')
            save(fig,output/'between_request_classes'/f'{run}_205.png')
            limit=max(float(values.abs().max()) for group in heatmaps[run].values() for values in group.values()) or 1.
            fig,axes=plt.subplots(3,4,figsize=(17,8),squeeze=False)
            for gi,group in enumerate(GROUPS):
                for si,subset in enumerate(['top','middle','bottom','random42']):
                    ax=axes[gi,si];im=ax.imshow(heatmaps[run][group][subset].numpy(),cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
                    ax.set(title=f'{GROUP_NAMES[group]} · {subset}',yticks=range(len(steps)),yticklabels=steps,xlabel='205 selected coordinates (ascending index)',ylabel='Step')
            fig.suptitle(f'{run}: signed changes from baseline, averaged across 10 probes per request class')
            fig.tight_layout(rect=(0,0,.91,.95));bar=fig.add_axes([.93,.16,.015,.65]);fig.colorbar(im,cax=bar,label='Signed activation change')
            dest=output/'signed_coordinate_maps'/f'{run}_205.png';dest.parent.mkdir(exist_ok=True);fig.savefig(dest,dpi=170);plt.close(fig)


def summary(output,rows,random,onsets,protocol,selection):
    indexed={(r['run'],r['step'],r['budget'],r['subset'],r['grouping'],r['probe_class']):r for r in rows}
    end=max(protocol['steps']);table=[]
    for run in protocol['runs']:
        for budget in protocol['budgets']:
            values={subset:indexed[(run,end,budget,subset,'all','all_probes')]['rms_delta'] for subset in ['top','middle','bottom']}
            ref=next(r['rms_delta'] for r in random if r['run']==run and r['step']==end and r['budget']==budget and r['grouping']=='all')
            table.append(dict(run=run,step=end,budget=budget,top_rms=values['top'],middle_rms=values['middle'],bottom_rms=values['bottom'],random_mean_rms=ref,
                top_vs_random=values['top']/max(ref,1e-30),bottom_vs_random=values['bottom']/max(ref,1e-30)))
    write_csv(output/'endpoint_summary.csv',table)
    lines=['# Selected-coordinate trajectories','','Original shared signed gradient-based layer31 ranking; no scoring-method comparison or monitor fitting.','',
        f"Seven stored models, {len(protocol['steps'])} checkpoints, 30 fixed monitoring prompts per model/checkpoint. Ranking is fixed across all of them.",'',
        '## Equal-size subset comparison at the final saved checkpoint','',
        '| Model | Coordinates | Top RMS | Middle RMS | Bottom RMS | Random mean RMS |','|---|---:|---:|---:|---:|---:|']
    for row in table:lines.append(f"| {row['run']} | {row['budget']} | {row['top_rms']:.3f} | {row['middle_rms']:.3f} | {row['bottom_rms']:.3f} | {row['random_mean_rms']:.3f} |")
    lines+=['','RMS is computed per probe over selected coordinates before averaging probes. It corrects for subset size. It does not correct for coordinate scale.','',
        'The 2048-coordinate plots are top 50% versus bottom 50%; the middle window overlaps both halves. Bottom signed-score coordinates may strongly oppose the donor replacement and are not an inactive control.','',
        'Signed means may cancel opposing coordinate movements; use signed_coordinate_maps/ to see the individual coordinate directions. Those heatmaps average probes within request classes, so probe-wise heterogeneity is still summarized. Raw snapshots remain available for later monitor inputs.','',
        'Dotted lines/stars use the first saved checkpoint with any harmful MONITORING response, not the onset of the plotted class or a precise causal emergence time.','',
        '## Files','',
        '- overview/: subset trajectories for each model; RMS and signed movement, four sizes.',
        '- between_models_<budget>.png: models compared separately for direct, indirect and benign prompts.',
        '- between_request_classes/: 205-coordinate subset comparisons per model.',
        '- signed_coordinate_maps/: signed per-coordinate changes at 205 coordinates.',
        '- per_probe_trajectories.csv: individual probe metrics, including cosine and energy share.',
        '- grouped_trajectories.csv: request-class, topic and all-probe averages.',
        '- coordinate_subsets.json, subset_score_summary.csv: exact locations and their score signs.',
        '- protocol.json: provenance hashes and limitations.','',
        '## Interpretation limits','',*['- '+x for x in protocol['limitations']],'',
        'Reproduce with `python scripts/plot_selected_coordinate_trajectories.py --output-dir <fresh-folder>`.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs/early-steps'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/early-steps/selected-coordinate-trajectories'))
    p.add_argument('--mask',type=Path,default=Path('artifacts/shared-activation-mask-v1/mask.json'))
    args=p.parse_args();plot(args.root,args.output_dir,args.mask)
