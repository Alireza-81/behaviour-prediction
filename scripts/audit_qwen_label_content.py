"""Offline, resumable content review. Triage is NOT a semantic verdict.

Reads current Qwen full-v2 exports; never invokes an API. Labels are read-only
unless --apply-user-decisions is explicitly supplied.
Manual judgments live in content-label-audit/decisions.json, keyed by full
prompt+response SHA256. Every occurrence keeps its own source coordinates.
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from label_with_claude import (TAGS, apply_label_overrides, digest, export_row,
                               payload, read_saved_results, response_key, write_report)


def case_key(row):
    return digest(dict(prompt=row['prompt'], response=row['response']))


def triage(row):
    """Order reading, never automatically confirm a label from keywords."""
    text = row['response']
    flags = []
    if not text.strip():
        flags.append('empty_response')
    refusal = bool(re.search(r"(?i)\b(?:I (?:cannot|can't|won't|am unable)|cannot (?:assist|help)|not appropriate|not ethical)", text))
    if row['claude_harmful'] == '1' and refusal:
        flags.append('positive_with_refusal_or_warning')
    if row['claude_harmful'] == '0' and not refusal:
        flags.append('negative_without_standard_refusal')
    if row['claude_harmful'] == '1' and row['category'] == 'adult_content':
        flags.append('check_graphic_content_threshold')
    if re.search(r'(?m)^\s*(?:user|assistant|<think>|</think>)\s*$', text):
        flags.append('generated_role_markers_check_attribution')
    if row['token_limit_reached'].lower() == 'true':
        flags.append('truncated_visible_text_only')
    if row['claude_needs_review'].lower() == 'true' and row['token_limit_reached'].lower() != 'true':
        flags.append('judge_flag_not_explained_by_truncation')
    return flags


def write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def apply_user_decisions(root, include_reviewer=False):
    """Apply completed judgments; reviewer authority was granted by the user."""
    out = root / 'content-label-audit'
    decisions = json.loads((out / 'decisions.json').read_text())
    allowed = {'user_decided', 'reviewer_decided'} if include_reviewer else {'user_decided'}
    confirmed = {k: v for k, v in decisions.items() if v['status'] in allowed}
    if not confirmed:
        return
    assert all(v.get('decided_by') == ('user' if v['status']=='user_decided' else 'assistant') and v['recommended_label'] in ('0', '1') for v in confirmed.values())
    plans = []
    seen = set()
    for suite in ('monitoring', 'evaluation'):
        folder = root / f'labeling-{suite}'
        manifest = json.loads((folder / 'test.json').read_text())
        results = read_saved_results(folder, manifest)
        override_path = folder / 'label_overrides.json'
        old = json.loads(override_path.read_text()) if override_path.exists() else []
        overrides = {tuple(o['key']): o for o in old}
        for row in manifest['sample']:
            sha = case_key(row)
            if sha not in confirmed:
                continue
            seen.add(sha)
            d = confirmed[sha]
            key = response_key(row)
            prior = overrides.get(key, {}).get('fields', {})
            if d['status'] == 'reviewer_decided' and prior.get('claude_status') == 'user_adjudicated':
                assert str(prior['claude_harmful']) == d['recommended_label'], 'Cannot reverse a user adjudication.'
                continue
            if d['recommended_label'] == '1':
                tags = d.get('behaviour_tags')
                assert isinstance(tags, list) and tags and all(t in TAGS for t in tags), 'Positive decisions need reviewed behaviour tags from the saved rubric.'
            else:
                tags = []
            overrides[key] = dict(key=list(key), payload_sha256=digest(payload(row)), fields={**prior,
                'claude_harmful': d['recommended_label'], 'claude_behaviour_tags': ';'.join(sorted(set(tags))),
                'claude_status': 'user_adjudicated' if d['status']=='user_decided' else 'assistant_adjudicated', 'claude_reason': d['note']})
        plans.append((folder, manifest, results, list(overrides.values())))
    assert seen == confirmed.keys(), 'User decision payload not found; no labels changed.'
    batch = digest(confirmed)[:16]
    for folder, manifest, results, overrides in plans:
        backup = out / ('before-reviewed-decisions' if include_reviewer else 'before-user-decisions') / batch / folder.name
        if not backup.exists():
            backup.mkdir(parents=True)
            for name in ('label_overrides.json', 'comparison.csv', 'report.json'):
                if (folder / name).exists():
                    shutil.copy2(folder / name, backup / name)
            shutil.copytree(folder / 'labeled', backup / 'labeled')
        raw_sha = hashlib.sha256((folder / 'results.jsonl').read_bytes()).hexdigest()
        (folder / 'label_overrides.json').write_text(json.dumps(overrides, indent=2) + '\n')
        write_report(folder, manifest, results)
        assert hashlib.sha256((folder / 'results.jsonl').read_bytes()).hexdigest() == raw_sha


def main(root):
    out = root / 'content-label-audit'
    out.mkdir(exist_ok=True)
    decision_path = out / 'decisions.json'
    decisions = json.loads(decision_path.read_text()) if decision_path.exists() else {}
    grouped = defaultdict(list)
    fingerprints = {}
    counts = Counter()
    source_rows = []
    for suite in ('monitoring', 'evaluation'):
        folder = root / f'labeling-{suite}'
        manifest = json.loads((folder / 'test.json').read_text())
        assert digest({k: v for k, v in manifest.items() if k != 'test_id'}) == manifest['test_id']
        results = read_saved_results(folder, manifest)
        expected = [export_row(row, results[i]) for i, row in enumerate(manifest['sample'])]
        apply_label_overrides(folder, expected)
        expected = {response_key(row): row for row in expected}
        assert len(expected) == len(manifest['sample'])
        seen = set()
        for path in sorted((folder / 'labeled').glob('*.csv')):
            fingerprints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for record, row in enumerate(csv.DictReader(path.open(newline='')), 1):
                key = response_key(row)
                assert key not in seen, (path, key)
                seen.add(key)
                wanted = expected[key]
                assert payload(row) == payload(wanted), (path, key, 'payload mismatch')
                for field in row:
                    if field.startswith('claude_'):
                        assert row[field].lower() == str(wanted[field]).lower(), (path, key, field)
                assert row['claude_harmful'] in ('0', '1')
                assert (row['claude_harmful'] == '1') == bool(row['claude_behaviour_tags'].strip())
                counts[suite, row['claude_harmful']] += 1
                occurrence = dict(suite=suite, csv=str(path), record_number=record,
                                  run=row['run'], step=row['step'], probe_id=row['probe_id'],
                                  current_label=row['claude_harmful'], tags=row['claude_behaviour_tags'],
                                  token_limit_reached=row['token_limit_reached'],
                                  claude_status=row['claude_status'])
                grouped[case_key(row)].append((row, occurrence))
                source_rows.append((case_key(row), occurrence))
        assert seen == expected.keys()
        comparisons = list(csv.DictReader((folder / 'comparison.csv').open(newline='')))
        assert len(comparisons) == len(expected)
        assert len({response_key(r) for r in comparisons}) == len(expected)
        assert all(all(r[f] == str(expected[response_key(r)][f]) for f in r
                       if f not in ('claude_needs_review',)) for r in comparisons)
    assert not set(decisions) - grouped.keys(), 'A decision no longer matches the source payload.'
    cases = []
    for sha, matches in grouped.items():
        row = matches[0][0]
        labels = sorted({r['claude_harmful'] for r, _ in matches})
        flags = sorted({f for r, _ in matches for f in triage(r)})
        if len(labels) > 1:
            flags.insert(0, 'identical_prompt_response_conflicting_labels')
        decision = decisions.get(sha, {})
        if decision:
            assert decision['status'] in ('reviewed_agree', 'proposed_change', 'needs_user', 'user_decided', 'reviewer_decided')
            assert decision.get('recommended_label', '') in ('', '0', '1')
        prior = any(r['claude_status'] == 'user_adjudicated' for r, _ in matches)
        status = decision.get('status', 'previously_user_adjudicated' if prior else 'not_yet_semantically_reviewed')
        cases.append(dict(case_id=sha, status=status, current_labels=';'.join(labels),
                          recommended_label=decision.get('recommended_label', ''),
                          audit_note=decision.get('note', ''),
                          review_method=decision.get('review_method', 'direct_full_text' if decision else ''),
                          reference_case_id=decision.get('reference_case_id', ''),
                          triage_flags=';'.join(flags), occurrences=len(matches),
                          category=row['category'], prompt=row['prompt'], response=row['response'],
                          sources=json.dumps([o for _, o in matches], ensure_ascii=False),
                          resolved_harmful='', user_notes=''))
    priority = {'needs_user': 0, 'proposed_change': 1, 'not_yet_semantically_reviewed': 2,
                'reviewed_agree': 3, 'previously_user_adjudicated': 4, 'user_decided': 4, 'reviewer_decided': 4}
    cases.sort(key=lambda c: (priority[c['status']],
        'identical_prompt_response_conflicting_labels' not in c['triage_flags'],
        'positive_with_refusal_or_warning' not in c['triage_flags'],
        'judge_flag_not_explained_by_truncation' not in c['triage_flags'], len(c['response']), c['case_id']))
    fields = list(cases[0])
    write_csv(out / 'all_cases.csv', cases, fields)
    questions = [c for c in cases if c['status'] in ('needs_user', 'proposed_change')]
    write_csv(out / 'review_cases.csv', questions, fields)
    write_csv(out / 'occurrences.csv', [dict(case_id=k, **o) for k, o in source_rows],
              ['case_id', *source_rows[0][1]])
    statuses = Counter(c['status'] for c in cases)
    report = dict(scope='28 current Qwen full-v2 labeled CSVs, both suites and all 14 runs; excludes historical snapshots and earlier experiments',
                  rows=len(source_rows), unique_prompt_responses=len(cases),
                  source_alignment_checked_rows=len(source_rows),
                  exact_payload_binary_conflicts=sum('identical_prompt_response_conflicting_labels' in c['triage_flags'] for c in cases),
                  labels=dict(Counter(o['current_label'] for _, o in source_rows)),
                  semantic_review_status=dict(statuses),
                  semantic_review_methods=dict(Counter(d.get('review_method', 'direct_full_text') for d in decisions.values())),
                  user_review_unique_cases=len(questions),
                  user_review_occurrences=sum(c['occurrences'] for c in questions),
                  semantic_audit_complete=not any(statuses[s] for s in ('not_yet_semantically_reviewed','needs_user','proposed_change')),
                  warning='Keyword triage only prioritizes reading. Unreviewed cases are NOT confirmed correct. User-authorized reviewer decisions are distinct from user adjudications. No paid calls.',
                  source_sha256=fingerprints)
    (out / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False) + '\n')
    (out / 'review.html').write_text(HTML.replace('__CASES__', json.dumps(cases, ensure_ascii=False).replace('<', '\\u003c')))
    (out / 'README.md').write_text('# Label content audit — in progress\n\n'
        f'All {len(source_rows):,} source rows passed export/alignment checks. There are {len(cases):,} exact prompt/answer pairs. '
        'Identical pairs are reviewed together; different prompts are never silently merged.\n\n'
        f'Manually inspected: {len(decisions)} unique pairs. Pending independent semantic reading: '
        f'{statuses["not_yet_semantically_reviewed"]:,}. This is not a completed independent relabeling.\n\n'
        'Review methods distinguish direct reading from reading a full reference answer plus every exact changed span. '
        'Delta batches verify full response reconstruction; they do not automatically assign or transfer verdicts. '
        'The ledger records reference IDs and batch evidence for those cases.\n\n'
        'Open review.html locally to read the full answers, filter cases and enter decisions. It makes no network requests. '
        'Selections are saved to this browser’s local storage when available; download your decisions before closing. '
        'The downloaded CSV contains only cases you decided, with exact payload identifiers. '
        'Nothing changes the source labels automatically. CSV record_number means data record after the header, not a physical line (responses contain newlines).\n\n'
        'review_cases.csv contains manually identified doubts/proposed corrections. all_cases.csv also includes every unreviewed and agreed case. '
        'occurrences.csv maps each pair back to every original row. '
        'Do not interpret a regex flag, a Claude review flag, or a token limit as an error verdict.\n\n'
        'Rebuild with `.venv/bin/python scripts/audit_qwen_label_content.py`. Manual review entries are in decisions.json. '
        'Proposals never apply automatically. --apply-user-decisions applies explicit user decisions; --apply-reviewed-decisions also applies completed reviewer decisions under the user’s delegated authority, with backups. '
        'Raw API results and past experiment results remain unchanged. Prepared training bundles are not rebuilt automatically.\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, indent=2))


HTML = '''<!doctype html><meta charset="utf-8"><title>Qwen label review</title>
<style>body{font:17px system-ui;max-width:1050px;margin:30px auto;padding:20px;background:#fafafa}select,button,input{font:inherit;margin:5px;padding:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:20px;border:1px solid #ddd}#meta{color:#555}#note{background:#fff4cc;padding:16px}textarea{width:95%;height:70px;font:inherit}</style>
<h1>Review the response, not the request</h1><p>This audit is in progress. Pending cases have not been independently checked. Completed reviewer decisions are applied under the user's authorization; user adjudications remain protected.</p>
<p><select id="filter"><option value="pending">Not yet read</option><option value="flagged">Unresolved reviewer cases</option><option value="all">All cases</option></select>
<input id="search" placeholder="Case ID, category or text"><button id="export">Download my decisions</button></p>
<button id="prev">Previous</button><span id="position"></span><button id="next">Next</button><p id="meta"></p><div id="note"></div>
<h2>Prompt (context only)</h2><pre id="prompt"></pre><h2>Actual saved response</h2><pre id="answer"></pre>
<label>Your decision: <select id="decision"><option value="">Not decided</option><option value="0">0 — non-harmful</option><option value="1">1 — harmful</option><option value="uncertain">Still uncertain</option></select></label>
<textarea id="notes" placeholder="Optional note"></textarea><details><summary>Every source occurrence</summary><pre id="sources"></pre></details>
<script>
const cases=__CASES__;let saved={};try{saved=JSON.parse(localStorage.getItem('qwen-label-audit-v1')||'{}')}catch(e){}
let pos=0,shown=[];const el=id=>document.getElementById(id);
function select(){const f=el('filter').value,q=el('search').value.toLowerCase();shown=cases.filter(c=>(f==='all'||f==='flagged'&&['needs_user','proposed_change'].includes(c.status)||f==='pending'&&c.status==='not_yet_semantically_reviewed')&&(!q||[c.case_id,c.category,c.prompt,c.response].some(s=>s.toLowerCase().includes(q))));pos=0;render()}
function render(){const c=shown[pos];el('position').textContent=shown.length?`${pos+1} / ${shown.length}`:'No cases';el('decision').disabled=!c;el('notes').disabled=!c;if(!c){for(const id of ['meta','prompt','answer','sources','notes'])el(id)[id==='notes'?'value':'textContent']='';el('decision').value='';el('note').textContent='No cases in this filter. Check Not yet read for audit progress.';return}el('meta').textContent=`${c.case_id} | ${c.category} | Current label: ${c.current_labels} | ${c.occurrences} occurrence(s) | ${c.status}`;el('note').textContent=c.audit_note||'Not yet independently read. Triage flags are not a decision.';el('prompt').textContent=c.prompt;el('answer').textContent=c.response;el('sources').textContent=JSON.stringify(JSON.parse(c.sources),null,2);el('decision').value=saved[c.case_id]?.label||'';el('notes').value=saved[c.case_id]?.note||''}
function save(){const c=shown[pos];if(!c)return;saved[c.case_id]={label:el('decision').value,note:el('notes').value};try{localStorage.setItem('qwen-label-audit-v1',JSON.stringify(saved))}catch(e){}}
el('filter').onchange=select;el('search').oninput=select;el('prev').onclick=()=>{pos=Math.max(0,pos-1);render()};el('next').onclick=()=>{pos=Math.min(shown.length-1,pos+1);render()};el('decision').onchange=save;el('notes').oninput=save;
el('export').onclick=()=>{const quote=s=>'"'+String(s??'').replaceAll('"','""')+'"';const lines=[['case_id','resolved_harmful','user_notes'],...Object.entries(saved).filter(([k,v])=>v.label).map(([k,v])=>[k,v.label,v.note])];const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.map(r=>r.map(quote).join(',')).join('\\r\\n')],{type:'text/csv'}));a.download='qwen-label-decisions.csv';a.click();URL.revokeObjectURL(a.href)};select();
</script>'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/qwen-full-v2'))
    parser.add_argument('--apply-user-decisions', action='store_true')
    parser.add_argument('--apply-reviewed-decisions', action='store_true')
    args = parser.parse_args()
    if args.apply_user_decisions or args.apply_reviewed_decisions:
        apply_user_decisions(args.root, include_reviewer=args.apply_reviewed_decisions)
    main(args.root)
