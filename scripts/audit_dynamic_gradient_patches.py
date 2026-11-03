"""Offline integrity/quality audit for the full-generation gradient experiment.

Read-only on labels and generated answers. Repetition flags are diagnostics,
not substitute harmfulness labels or an automatic assessment of usefulness.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from score_gradient_activation_patch import load_run


def repetition_span(tokens):
    longest = 0
    best_period = 1
    for period in range(1, 9):
        run = 0
        for i in range(period, len(tokens)):
            run = run + 1 if tokens[i] == tokens[i-period] else 0
            if run + period > longest:
                longest, best_period = run + period, period
    return longest, best_period


def audit(folder):
    manifest, journal, labels = load_run(folder, 'gradientdynamic')
    scores = torch.load(folder/'coordinate_scores.pt', map_location='cpu', weights_only=True)
    refs = json.loads((folder/'references.json').read_text())
    selections = json.loads((folder/'coordinate_selection.json').read_text())
    assert len(scores) == len(refs) == len(manifest['probes'])*2
    assert len(selections) == len({(r['probe_id'],r['direction'],r['condition']) for r in selections})
    for key, values in scores.items():
        assert torch.isfinite(values['scores']).all() and torch.isfinite(values['magnitude_rms']).all()
        assert math.isfinite(values['kl_before'])
        assert len(refs[key]['token_ids']) == values['prediction_positions'] == values['token_score_sums'].numel()
        torch.testing.assert_close(values['scores'].sum(), values['token_score_sums'].sum(), rtol=1e-4, atol=1e-5)
    for row in selections:
        values = scores[f"{row['direction']}:{row['probe_id']}"]
        rms = values['magnitude_rms'].float()
        n = values['prediction_positions']
        total = float(rms.norm())*math.sqrt(n)
        selected = float(rms[row['selected_indices']].norm())*math.sqrt(n)
        assert math.isclose(total,row['total_delta_l2'],rel_tol=1e-4,abs_tol=1e-4)
        assert math.isclose(selected,row['selected_delta_l2'],rel_tol=1e-4,abs_tol=1e-4)
    for probe in manifest['probes']:
        for direction in manifest['directions']:
            a = journal[(probe['id'],direction,'gradient_25pct')]['selected_indices']
            b = journal[(probe['id'],direction,'prefill_gradient_25pct')]['selected_indices']
            assert a == b
    judge = json.loads((folder/'labeling/test.json').read_text())
    raw = list(map(json.loads,(folder/'labeling/results.jsonl').read_text().splitlines()))
    assert len(raw) == len(judge['sample']) == len(journal)
    assert {r['index'] for r in raw} == set(range(len(raw)))
    assert all(r['test_id'] == judge['test_id'] for r in raw)
    resolved_file = folder/'labeling/manual_adjudications.json'
    decisions = json.loads(resolved_file.read_text()) if resolved_file.exists() else []
    for decision in decisions:
        matches = [r for r in labels.values() if r['prompt']==decision['prompt'] and r['response']==decision['response']]
        assert matches and all(r['harmful']==decision['resolved_harmful'] for r in matches)
    backup = folder/'labeling/results.before-json-recovery.jsonl'
    if backup.exists():
        original = {r['index']:r for r in map(json.loads,backup.read_text().splitlines())}
        assert all(r['raw_text']==original[r['index']]['raw_text'] for r in raw)
    # Check identical answers against all previous labelled comparisons as well.
    groups = defaultdict(list)
    previous_records = []
    for previous_folder in [Path(manifest['comparison_dir']), Path('outputs/activation-patch-gradient')]:
        for path in (previous_folder/'labeling/labeled').glob('*.csv'):
            previous_records.extend(csv.DictReader(path.open()))
    for r in [*labels.values(), *previous_records]:
        if r['claude_harmful'] in ('0','1'):
            groups[(r['prompt'],r['response'])].append(int(r['claude_harmful']))
    for r in labels.values():
        assert len(set(groups[(r['prompt'],r['response'])])) == 1
    quality = []
    for key, row in journal.items():
        span, period = repetition_span(row['generated_token_ids'])
        repeated = span >= 48 and span/period >= 8
        confirmed = any(d.get('generation_failure') and d['prompt']==row['prompt'] and d['response']==row['response'] for d in decisions)
        quality.append(dict(probe_id=row['probe_id'],direction=row['direction'],condition=row['condition'],role=row['role'],
            harmful=labels[key]['harmful'],generated_tokens=len(row['generated_token_ids']),
            truncated=row['token_limit_reached'],needs_review=labels[key]['claude_needs_review'].lower()=='true',
            empty=not bool(row['response'].strip()),repetition_flag=repeated,periodic_span=span,period=period,
            user_confirmed_generation_failure=confirmed,prompt=row['prompt'],response=row['response'],label_file=labels[key]['label_file']))
    out = folder/'results'
    if not (out/'report.json').exists():
        raise ValueError('Run the dynamic scorer first')
    for name,rows in [('quality_audit.csv',quality),('restore_gradient_25pct_answers.csv',
            [r for r in quality if r['direction']=='restore' and r['condition']=='gradient_25pct']),('generation_quality_flags.csv',
            [r for r in quality if r['empty'] or r['repetition_flag'] or r['user_confirmed_generation_failure']])]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(quality[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    for direction in manifest['directions']:
        summary[direction]={}
        for condition in manifest['conditions']:
            rows=[r for r in quality if r['direction']==direction and r['condition']==condition]
            summary[direction][condition]=dict(total=len(rows),truncated=sum(r['truncated'] for r in rows),
                review_flagged=sum(r['needs_review'] for r in rows),empty=sum(r['empty'] for r in rows),
                repetition_flagged=sum(r['repetition_flag'] for r in rows),
                user_confirmed_generation_failure=sum(r['user_confirmed_generation_failure'] for r in rows))
    report=json.loads((out/'report.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,direction,title in zip(axes,manifest['directions'],['Trained model: removing harm','Baseline model: transferring harm']):
        conditions=['unpatched','prefill_gradient_25pct','gradient_25pct']
        values=[report['methods'][direction][c]['harmful_targets'] for c in conditions]
        bars=ax.bar(['No patch','Patch once','Patch throughout'],values,color=['#888888','#e49a37','#3182bd'])
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,value+.4,f'{value}/25',ha='center',fontsize=11)
        ax.set(title=title,ylabel='Harmful answers out of 25',ylim=(0,28),yticks=[0,5,10,15,20,25])
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Patch once vs throughout: same 1,024 selected coordinates within each direction',fontsize=10)
    fig.tight_layout();fig.savefig(out/'matched_duration_25pct.png',dpi=170);plt.close(fig)
    result=dict(generated_answers=len(journal),effective_binary_labels=len(labels),
        coordinate_selections_verified=len(selections),score_entries=len(scores),reference_sequences=len(refs),
        self_patch_exact_matches=sum(journal[(p['id'],d,'unpatched')]['generated_token_ids']==journal[(p['id'],d,'self_dynamic')]['generated_token_ids'] for p in manifest['probes'] for d in manifest['directions']),
        full_donor_exact_matches=sum(journal[(p['id'],d,'full_dynamic')]['generated_token_ids']==refs[f"{d}:{p['id']}"]['token_ids'] for p in manifest['probes'] for d in manifest['directions']),
        manual_adjudications=len(decisions),schema_recoveries=sum('recovery_note' in r for r in raw),
        raw_api_undecided=sum(r.get('judgment') is None or r['judgment']['harmful'] is None for r in raw),
        truncated_answers=sum(r['truncated'] for r in quality),review_flags=sum(r['needs_review'] for r in quality),
        repetition_flags=sum(r['repetition_flag'] for r in quality),empty_answers=sum(r['empty'] for r in quality),
        cross_experiment_identical_answer_conflicts=0,quality_by_condition=summary,
        note='Repetition is a mechanical diagnostic, not a full utility assessment. User-confirmed failure remains non-harmful but is explicitly identified.',
        input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [folder/'pilot.json',folder/'coordinate_scores.pt',folder/'coordinate_selection.json',folder/'references.json',folder/'labeling/results.jsonl',Path(__file__)]})
    (out/'data_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','quality_by_condition')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/activation-patch-gradient-dynamic'))
    audit(parser.parse_args().output_dir)
