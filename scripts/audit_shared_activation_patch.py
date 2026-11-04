"""Independent offline checks and readable summaries for shared-mask results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from behaviour_prediction.monitor.shared_mask import average_development_scores, validate_frozen_mask
from score_gradient_activation_patch import load_run


def audit(folder):
    m,journal,labels=load_run(folder,'shared')
    frozen=json.loads(Path(m['mask_path']).read_text());validate_frozen_mask(frozen)
    for path,sha in frozen['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:
            raise ValueError(f'Development source changed: {path}')
    score_path=next(Path(p) for p in frozen['source_sha256'] if p.endswith('/coordinate_scores.pt'))
    source_plan=json.loads((score_path.parent/'pilot.json').read_text())
    cache=torch.load(score_path,map_location='cpu',weights_only=True)
    scores=average_development_scores(cache,source_plan['probes'],frozen['directions_aggregated'])
    torch.testing.assert_close(scores,torch.tensor(frozen['aggregate_scores'],dtype=torch.float64),rtol=0,atol=0)
    dev_ids={p['id'] for p in source_plan['probes']}
    dev_text={p['prompt'].strip().casefold() for p in source_plan['probes']}
    assert not dev_ids & {p['id'] for p in m['probes']}
    assert not dev_text & {p['prompt'].strip().casefold() for p in m['probes']}
    assert len({p['id'] for p in m['probes']})==len(m['probes'])
    assert len({p['prompt'].strip().casefold() for p in m['probes']})==len(m['probes'])
    probes={p['id']:p for p in m['probes']}
    self_matches=full_matches=masked=0
    for (id,d,c),row in journal.items():
        assert all(row[field]==probes[id][field] for field in ('prompt','group','category','role'))
        assert row['step']==(20 if d=='restore' else 0)
        assert len(row['generated_token_ids'])<=m['max_new_tokens']
        assert row['token_limit_reached']==(len(row['generated_token_ids'])>=m['max_new_tokens'])
        assert row['mask_id']==frozen['mask_id']
        expected={} if c not in frozen['selected_indices'] else {'31':frozen['selected_indices'][c]}
        assert row['selected_indices']==expected
        masked+=c in frozen['selected_indices']
        if c=='self_dynamic':
            assert row['generated_token_ids']==journal[(id,d,'unpatched')]['generated_token_ids'];self_matches+=1
        if c=='full_dynamic':
            other='transfer' if d=='restore' else 'restore'
            assert row['generated_token_ids']==journal[(id,other,'unpatched')]['generated_token_ids'];full_matches+=1
    raw=list(map(json.loads,(folder/'labeling/results.jsonl').read_text().splitlines()))
    judge=json.loads((folder/'labeling/test.json').read_text())
    assert len(raw)==len(journal) and {r['index'] for r in raw}==set(range(len(raw)))
    assert all(r['test_id']==judge['test_id'] for r in raw)
    backup=folder/'labeling/results.before-json-recovery.jsonl'
    if backup.exists():
        old={r['index']:r for r in map(json.loads,backup.read_text().splitlines())}
        assert all(r['raw_text']==old[r['index']]['raw_text'] for r in raw)
    out=folder/'results'
    report=json.loads((out/'report.json').read_text())
    paired=list(csv.DictReader((out/'paired_decisions.csv').open()))
    for condition in ['shared_05pct','shared_25pct']:
        rows=[r for r in paired if r['direction']=='restore' and r['condition']==condition]
        with (out/f'restore_{condition}_answers.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Record a concrete benign regression, with the unpatched answer alongside it.
    benign_regressions=[]
    for r in paired:
        if r['direction']=='restore' and r['role']=='benign_control' and r['condition']=='shared_25pct' and r['benign_refusal_hint']=='True':
            row=journal[(r['probe_id'],'restore','shared_25pct')]
            benign_regressions.append(dict(probe_id=r['probe_id'],prompt=row['prompt'],patched_response=row['response'],
                unpatched_response=journal[(r['probe_id'],'restore','unpatched')]['response'],
                assessment='Automatic refusal hint; inspected in the report as a legitimate educational question refused by the patch.'))
    (out/'benign_refusal_examples.json').write_text(json.dumps(benign_regressions,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    target_total=sum(p['role']=='target' for p in m['probes'])
    for ax,d,title in zip(axes,m['directions'],['Trained model: removing harm','Baseline model: transferring harm']):
        values=[report['methods'][d][c]['harmful_targets'] for c in ['unpatched','shared_05pct','shared_25pct']]
        random=[report['methods'][d][f'random{s}_25pct']['harmful_targets'] for s in m['random_seeds']]
        values.append(float(np.mean(random)))
        bars=ax.bar(['No patch','Shared 5%','Shared 25%','Random 25%'],values,color=['#888888','#e49a37','#3182bd','#74a96f'])
        for i,(bar,value) in enumerate(zip(bars,values)):
            text=f'{int(value)}/{target_total}' if i<3 else f'{min(random)}–{max(random)}/{target_total}'
            ax.text(bar.get_x()+bar.get_width()/2,(max(random) if i==3 else value)+.6,text,ha='center',fontsize=10)
        ax.errorbar([3],[np.mean(random)],yerr=[[np.mean(random)-min(random)],[max(random)-np.mean(random)]],fmt='none',ecolor='black',capsize=4)
        ax.set(title=title,ylabel=f'Harmful answers out of {target_total}',ylim=(0,target_total+4))
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Fresh prompts: same frozen locations across every prompt and both directions',fontsize=10)
    fig.tight_layout();fig.savefig(out/'shared_mask_summary.png',dpi=170);plt.close(fig)
    result=dict(generated_answers=len(journal),effective_binary_labels=len(labels),
        fixed_mask_records_verified=masked,self_patch_exact_matches=self_matches,full_donor_exact_matches=full_matches,
        development_score_vectors=len(frozen['development_target_ids'])*len(frozen['directions_aggregated']),
        shared_ranking_reproduced_exactly=True,development_evaluation_overlap=0,
        same_indices_across_prompts_and_directions=True,
        schema_recoveries=sum('recovery_note' in r for r in raw),raw_text_preserved=True,
        carried_api_results=sum('source_test_id' in r for r in raw),
        truncated_answers=sum(row['token_limit_reached'] for row in journal.values()),
        review_flags=sum(r['claude_needs_review'].lower()=='true' for r in labels.values()),
        empty_answers=sum(not bool(row['response'].strip()) for row in journal.values()),
        shared25_restoration_target_truncations=sum(row['direction']=='restore' and row['condition']=='shared_25pct' and row['role']=='target' and row['token_limit_reached'] for row in journal.values()),
        benign_refusal_hints_shared25_restore=len(benign_regressions),
        api_calls_during_local_analysis=0,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [Path(m['mask_path']),folder/'pilot.json',folder/'generated.jsonl',folder/'labeling/test.json',folder/'labeling/results.jsonl',Path(__file__)]})
    (out/'data_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/activation-patch-shared'))
    audit(parser.parse_args().output_dir)
