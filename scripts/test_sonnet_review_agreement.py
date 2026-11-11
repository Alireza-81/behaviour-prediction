"""Frozen, label-blind Sonnet test on the user's 70 exported review decisions."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from audit_qwen_label_content import case_key
from import_qwen_review_decisions import csv_text
from label_with_claude import MODELS, digest, payload, read_saved_results, write_report

ROOT = Path('outputs/qwen-full-v2')
OUT = ROOT / 'sonnet-5.5-review-agreement'


def reviewed_sample(root, export):
    cases = {c['case_id']: c for c in json.loads((root/'content-label-audit/cases.json').read_text())}
    rows = list(csv.DictReader(csv_text(export.read_bytes()).splitlines(keepends=True)))
    if len(rows) != 70 or len({r['case_id'] for r in rows}) != 70:
        raise ValueError('Expected exactly 70 unique exported user decisions.')
    sample = []
    cache = {}
    for decision in rows:
        key = decision['case_id']
        case = cases[key]
        if case_key(case) != key or decision['resolved_harmful'] not in ('0','1'):
            raise ValueError('Invalid decision or changed prompt/answer.')
        source = json.loads(case['sources'])[0]
        filename = source['csv']
        if filename not in cache:
            with Path(filename).open(newline='') as handle:
                cache[filename] = list(csv.DictReader(handle))
        row = cache[filename][source['record_number']-1]
        if case_key(row) != key:
            raise ValueError('Source response changed.')
        # Human labels and metadata are kept locally only. payload() is the
        # ONLY object sent in the user message; never send review notes/tags.
        sample.append({k:row[k] for k in ('run','step','probe_id','group','category','prompt','response','token_limit_reached')}
                      | dict(human_harmful=decision['resolved_harmful'], review_notes='', source_file=filename))
    return sample


def prepare(root, export, out, rubric_path=None, model="claude-sonnet-5-5"):
    input_price, output_price = MODELS[model]
    sample = reviewed_sample(root, export)
    original = json.loads((root/'labeling-monitoring/test.json').read_text())
    rubric = rubric_path.read_text() if rubric_path else original['rubric']
    manifest = dict(model=model, seed=42, max_tokens=1024 if model=='claude-haiku-5-5' else 256,
                    pricing_usd_per_million=dict(input=input_price, output=output_price),
                    rubric=rubric, sample=sample, all_responses=False,
                    human_export_sha256=hashlib.sha256(export.read_bytes()).hexdigest(),
                    rubric_sha256=hashlib.sha256(rubric.encode()).hexdigest(),
                    protocol=('70 selected development cases; rubric calibrated using these reviews, not a held-out test; no per-case gold labels, notes, names or checkpoint metadata sent to judge' if rubric_path else '70 selected development cases; rubric unchanged; no gold labels, notes, names or checkpoint metadata sent to judge'))
    manifest['test_id'] = digest(manifest)
    if (out/'test.json').exists():
        if json.loads((out/'test.json').read_text()) != manifest:
            raise ValueError('Existing test differs. Use another output folder; do not overwrite predictions.')
        print('Same test already prepared; existing predictions preserved.')
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output folder is not empty.')
    out.mkdir(parents=True, exist_ok=True)
    (out/'test.json').write_text(json.dumps(manifest, indent=2)+'\n')
    write_report(out, manifest, {})
    chars = sum(len(manifest['rubric'])+len(json.dumps(payload(r),ensure_ascii=False)) for r in sample)
    low = (chars/5+40*70)*1.3*input_price/1e6+70*45*1.3*output_price/1e6
    high = (chars/3+40*70)*1.3*input_price/1e6+70*90*1.3*output_price/1e6
    metadata = dict(cases=70, human_labels=dict(Counter(r['human_harmful'] for r in sample)),
                    estimated_usd=[round(low,2),round(high,2)], api_calls_made=0,
                    limitations=['Selected difficult development cases, not representative held-out accuracy.',
                                 'Human behaviour tags are not validated; score binary harmfulness only.',
                                 'Cost estimate assumes short JSON; additional thinking can increase cost. Excludes retries/tax; budget guard is not a provider-side hard cap.'])
    (out/'preparation.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


def report(out):
    manifest = json.loads((out/'test.json').read_text())
    assert digest({k:v for k,v in manifest.items() if k!='test_id'}) == manifest['test_id']
    results = read_saved_results(out, manifest)
    base = write_report(out, manifest, results)
    prediction_column = "haiku_harmful" if manifest.get("model", "").startswith("claude-haiku") else "sonnet_harmful"
    disagreements = []
    for i,row in enumerate(manifest['sample']):
        result = results.get(i,{})
        prediction = (result.get('judgment') or {}).get('harmful') if result.get('status')=='ok' else None
        if prediction is not None and prediction != int(row['human_harmful']):
            disagreements.append(dict(case_id=case_key(row), human_harmful=row['human_harmful'],
                **{prediction_column:prediction}, run=row['run'], step=row['step'], probe_id=row['probe_id'],
                prompt=row['prompt'], response=row['response']))
    with (out/'disagreements.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['case_id','human_harmful',prediction_column,'run','step','probe_id','prompt','response'])
        writer.writeheader();writer.writerows(disagreements)
    correct = base['tp']+base['tn']
    summary = dict(total=70, completed=base['completed'], decided=base['human_comparisons'],
                   correct=correct, disagreements=len(disagreements),
                   agreement_on_decided=base['agreement'], correct_over_all_cases=correct/70,
                   coverage=base['human_comparisons']/70,
                   tp=base['tp'],tn=base['tn'],fp=base['fp'],fn=base['fn'],
                   harmful_precision=base['harmful_precision'],harmful_recall=base['harmful_recall'],
                   estimated_billed_usd=base['estimated_billed_usd'],
                   limitation='Development agreement on selected difficult cases; not general-population accuracy.')
    (out/'agreement.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','report'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--review-export',type=Path,default=ROOT/'content-label-audit/qwen-label-decisions.csv')
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--budget-usd',type=float,default=1.)
    parser.add_argument('--rubric',type=Path,help='Custom rubric for a new test folder; existing frozen tests are never changed')
    parser.add_argument('--model',choices=MODELS,default='claude-sonnet-5-5',help='Model for a new test; run uses the saved model')
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.root,args.review_export,args.output_dir,args.rubric,args.model)
    elif args.action=='run':
        if not (args.output_dir/'test.json').exists():
            prepare(args.root,args.review_export,args.output_dir,args.rubric,args.model)
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('label_with_claude.py')),
                               'run','--output-dir',str(args.output_dir),'--budget-usd',str(args.budget_usd)])
        if result.returncode: sys.exit(result.returncode)
        report(args.output_dir)
    else: report(args.output_dir)
