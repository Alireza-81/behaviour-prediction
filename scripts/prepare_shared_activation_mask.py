"""Freeze one common coordinate ranking using existing development gradients.

No GPU, model loading, evaluation gradients or paid calls are involved.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

from behaviour_prediction.monitor.shared_mask import average_development_scores, fixed_masks, identity, tensor_identity
from run_gradient_activation_patch import digest


def prepare(source, output, evaluation_path, config_path):
    if (output.exists() and any(output.iterdir())) or config_path.exists():
        raise ValueError('Frozen output/config already exists; use a new version rather than overwrite')
    development=json.loads((source/'pilot.json').read_text())
    if digest(development)!=development['pilot_id']:raise ValueError('Invalid development manifest')
    cache=torch.load(source/'coordinate_scores.pt',map_location='cpu',weights_only=True)
    scores=average_development_scores(cache,development['probes'],development['directions'])
    evaluation=[json.loads(line) for line in evaluation_path.read_text().splitlines() if line.strip()]
    dev_ids={p['id'] for p in development['probes']}
    prior_texts={p['prompt'].strip().casefold() for p in development['probes']}
    for path in [Path('data/probes/online_probes.jsonl'),Path('data/evaluation/prompts.jsonl')]:
        if path.exists():prior_texts.update(json.loads(line)['prompt'].strip().casefold() for line in path.read_text().splitlines() if line.strip())
    if len({p['id'] for p in evaluation})!=len(evaluation):raise ValueError('Duplicate evaluation IDs')
    if len({p['prompt'].strip().casefold() for p in evaluation})!=len(evaluation):raise ValueError('Duplicate evaluation prompts')
    if any(p['id'] in dev_ids or p['prompt'].strip().casefold() in prior_texts for p in evaluation):
        raise ValueError('Evaluation overlaps development or previous monitor/evaluation prompts')
    if any(p['role'] not in ('target','benign_control') for p in evaluation):raise ValueError('Invalid evaluation role')
    budgets=[5,25];seeds=[17,42,123]
    indices=fixed_masks(scores,budgets,seeds)
    old=Path(development['comparison_dir'])
    states=torch.load(old/'endpoint_activations.pt',map_location='cpu',weights_only=True)
    sentinel=development['probes'][0]
    artifact=dict(version=1,layer=31,width=scores.numel(),budgets_percent=budgets,random_seeds=seeds,
        selection='Mean signed replacement-benefit scores over the development target prompts and both directions, with each prompt/direction vector normalized by its L1 norm. No evaluation gradient or response is used.',
        development_target_ids=[p['id'] for p in development['probes'] if p['role']=='target'],
        development_benign_ids=[p['id'] for p in development['probes'] if p['role']=='benign_control'],
        directions_aggregated=development['directions'],aggregate_scores=scores.tolist(),selected_indices=indices,
        same_indices_across_prompts=True,same_indices_across_directions=True,
        source_run=development['source_run'],expected_model_config=development['expected_model_config'],
        expected_donor_files=development['expected_donor_files'],
        sentinel=dict(probe=sentinel,prompt_token_ids=states['prompt_token_ids'][sentinel['id']],
            baseline_sha256=tensor_identity(states['baseline'][sentinel['id']][31]),
            finetuned_sha256=tensor_identity(states['finetuned'][sentinel['id']][31])),
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [source/'pilot.json',source/'coordinate_scores.pt',source/'references.json',old/'pilot.json',old/'endpoint_activations.pt']},
        interpretation='Frozen locations; live donor values still depend on prefix. This is a shared downstream intervention candidate for one hate-speech checkpoint, not proof of a universal harmfulness mechanism. Benign development scores are not subtracted or used for tuning; fresh benign responses test selectivity.')
    artifact['mask_id']=identity(artifact)
    output.mkdir(parents=True,exist_ok=True)
    mask_path=output/'mask.json';mask_path.write_text(json.dumps(artifact,indent=2)+'\n')
    conditions=['unpatched','self_dynamic','full_dynamic','shared_05pct','shared_25pct']
    conditions += [f'random{seed}_{percent:02d}pct' for seed in seeds for percent in budgets]
    plan=dict(source_run=development['source_run'],run_name=development['run_name'],step=20,layers=[31],
        expected_model_config=development['expected_model_config'],expected_donor_files=development['expected_donor_files'],
        max_input_length=development['max_input_length'],max_new_tokens=development['max_new_tokens'],
        mask_path=str(mask_path),mask_id=artifact['mask_id'],mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        evaluation_path=str(evaluation_path),evaluation_sha256=hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
        probes=evaluation,development_probe_ids=[p['id'] for p in development['probes']],directions=['restore','transfer'],
        conditions=conditions,budgets_percent=budgets,random_seeds=seeds,
        patch_scope='Same frozen coordinate locations across all evaluation prompts, all decoding tokens and both directions. Layer31 final prompt token plus each decoding token. Live donor values follow the recipient prefix.',
        selection=artifact['selection'],interpretation=artifact['interpretation'])
    plan['pilot_id']=digest(plan)
    config_path.parent.mkdir(parents=True,exist_ok=True);config_path.write_text(json.dumps(plan,indent=2)+'\n')
    rows=[]
    for i,value in enumerate(scores.tolist()):
        rows.append(dict(coordinate=i,shared_score=value,in_shared_05pct=i in indices['shared_05pct'],in_shared_25pct=i in indices['shared_25pct']))
    import csv
    with (output/'coordinate_ranking.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if scores.numel()==4096:
        grid=torch.zeros(4096)
        grid[indices['shared_25pct']]=1;grid[indices['shared_05pct']]=2
        fig,ax=plt.subplots(figsize=(7,6))
        ax.imshow(grid.reshape(64,64),cmap=ListedColormap(['#edf2f7','#3182bd','#e49a37']),vmin=0,vmax=2)
        ax.set(title='One frozen selection for every prompt and both directions',xlabel='Coordinate index = row × 64 + column',ylabel='Row of coordinate indices')
        fig.text(.5,.02,'Orange: shared 5%; orange + blue: shared 25%. Grid is an index display, not spatial brain anatomy.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1));fig.savefig(output/'selected_coordinates.png',dpi=170);plt.close(fig)
    summary=dict(development_target_prompts=len(artifact['development_target_ids']),score_vectors_averaged=len(artifact['development_target_ids'])*len(development['directions']),
        shared_counts={p:len(indices[f'shared_{p:02d}pct']) for p in budgets},evaluation_prompts=len(evaluation),
        evaluation_targets=sum(p['role']=='target' for p in evaluation),evaluation_benign=sum(p['role']=='benign_control' for p in evaluation),
        planned_answers=len(evaluation)*len(conditions)*2,positive_shared_scores=int((scores>0).sum()),mask_id=artifact['mask_id'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/activation-patch-gradient-dynamic'))
    p.add_argument('--output',type=Path,default=Path('artifacts/shared-activation-mask-v1'))
    p.add_argument('--evaluation-path',type=Path,default=Path('data/evaluation/shared_mask_fresh.jsonl'))
    p.add_argument('--config',type=Path,default=Path('configs/activation_patch_shared.json'))
    a=p.parse_args();prepare(a.source,a.output,a.evaluation_path,a.config)
