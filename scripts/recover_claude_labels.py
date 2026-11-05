"""Recover schema-valid saved Claude JSON without API calls or changing raw replies."""
import argparse
import csv
import json
import re
from pathlib import Path
from label_with_claude import digest, parse_judgment, validate_judgment, payload, write_report, TAGS


def recover(folder, allow_extra_explanation=False, repair_tag_list=False, use_final_json=False, repair_inactive_tag=False):
    folder = Path(folder)
    manifest = json.loads((folder/'test.json').read_text())
    assert digest({k:v for k,v in manifest.items() if k != 'test_id'}) == manifest['test_id']
    log = folder/'results.jsonl'
    original = log.read_bytes()
    results = {}
    recovered = 0
    for line in original.decode().splitlines():
        result = json.loads(line)
        index = result['index']
        assert result['test_id'] == manifest['test_id'] and index not in results
        assert 0 <= index < len(manifest['sample'])
        if result['status'] == 'invalid' and result['stop_reason'] == 'end_turn':
            try:
                judgment = parse_judgment(result['raw_text'])
            except (ValueError, TypeError):
                judgment = None
                if allow_extra_explanation:
                    text = result['raw_text'].strip()
                    if text.startswith(('```json\n', '```\n')) and text.endswith('```'):
                        text = '\n'.join(text.splitlines()[1:-1])
                    try:
                        value = json.loads(text)
                        details = set(value)-{'harmful','behaviour_tags','needs_review'} if isinstance(value,dict) else set()
                        if not isinstance(value, dict) or len(details)!=1 or not details <= {'behaviour_tags_detail','behaviour_tags_description'}:
                            raise ValueError('Not the supported extra-explanation case')
                        detail = value.pop(next(iter(details)))
                        if not isinstance(detail, str) or not detail.strip():
                            raise ValueError('Invalid explanation field')
                        judgment = validate_judgment(value)
                    except (ValueError, TypeError):
                        pass
                    else:
                        result['recovery_note'] = 'Explicit saved judgment recovered by dropping only one recognized extra behaviour-tags explanation; raw_text unchanged.'
                if judgment is None and repair_tag_list:
                    # Narrow repair: a tag string appears outside a prematurely
                    # closed behaviour_tags list, immediately before its final ].
                    fixed,count=re.subn(r'("behaviour_tags"\s*:\s*\[[^\[\]]*)\]\s*,\s*("[^"]+")\s*\]',
                        r'\1, \2]',result['raw_text'])
                    if count==1:
                        try:
                            judgment=parse_judgment(fixed)
                        except (ValueError,TypeError):
                            pass
                        else:
                            result['recovery_note']='Explicit saved judgment recovered by repairing one premature behaviour_tags closing bracket; raw_text unchanged.'
                if judgment is None and repair_inactive_tag:
                    text=result['raw_text'].strip()
                    if text.startswith(('```json\n','```\n')) and text.endswith('```'):
                        text='\n'.join(text.splitlines()[1:-1])
                    # A known, empty tag field may be outside the list, with one
                    # stray closing brace. Never discard a positive tag assertion.
                    fixed,count=re.subn(r',\s*"other_harmful"\s*:\s*\[\s*\]\s*}(?=\s*,\s*"needs_review")','',text)
                    try:
                        value=json.loads(fixed)
                        extra=set(value)-{'harmful','behaviour_tags','needs_review'}
                        if extra:
                            if len(extra)!=1 or not extra <= TAGS:raise ValueError('Unknown extra field')
                            key=next(iter(extra))
                            if value[key] is not False and value[key]!=[]:raise ValueError('Extra tag is active')
                            del value[key]
                        elif count!=1:raise ValueError('Not the narrow inactive-tag repair')
                        judgment=validate_judgment(value)
                    except (ValueError,TypeError):pass
                    else:result['recovery_note']='Explicit saved judgment recovered by removing one inactive known tag field (false/empty), and its stray brace if present; raw_text unchanged.'
                if judgment is None and use_final_json:
                    blocks=re.findall(r'```(?:json)?\s*\n(.*?)```',result['raw_text'],flags=re.S)
                    if len(blocks)>1:
                        valid=[]
                        for block in blocks:
                            try:valid.append(parse_judgment(block))
                            except (ValueError,TypeError):pass
                        try:final=parse_judgment(blocks[-1])
                        except (ValueError,TypeError):final=None
                        if final is not None and all(value['harmful']==final['harmful'] for value in valid):
                            judgment=final
                            result['recovery_note']='Explicit final corrected JSON block recovered; all valid blocks agree on harmfulness; raw_text unchanged.'
            if judgment is not None:
                if payload(manifest['sample'][index])['token_limit_reached']:
                    judgment['needs_review'] = True
                result.update(judgment=judgment, status='ok')
                recovered += 1
        results[index] = result
    if recovered:
        backup = folder/'results.before-json-recovery.jsonl'
        if not backup.exists():
            backup.write_bytes(original)
        temporary = folder/'results.recovered.tmp'
        temporary.write_text(''.join(json.dumps(r)+'\n' for r in results.values()))
        temporary.replace(log)
    report = write_report(folder, manifest, results)
    # Re-exporting raw API judgments must not erase explicit later user decisions.
    adjudications = folder/'manual_adjudications.json'
    if adjudications.exists():
        decisions = json.loads(adjudications.read_text())
        paths = [folder/'comparison.csv', *sorted((folder/'labeled').glob('*.csv'))]
        for path in paths:
            if not path.exists():
                continue
            with path.open(newline='') as handle:
                reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
            changed=False
            for row in rows:
                for decision in decisions:
                    if row['prompt']==decision['prompt'] and row['response']==decision['response']:
                        value=decision['resolved_harmful']
                        if type(value) is not int or value not in (0,1):
                            raise ValueError('Invalid saved user adjudication')
                        row['human_harmful']=str(value);row['claude_harmful']=str(value)
                        if value==0:
                            row['claude_behaviour_tags']=''
                        note='Explicit user adjudication restored from manual_adjudications.json.'
                        if decision.get('generation_failure'):
                            note+=' Failed/degenerate generation; non-harmful does not imply useful.'
                        row['review_notes']=(row.get('review_notes','')+' '+note).strip()
                        changed=True
            if changed:
                with path.open('w',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        with (folder/'comparison.csv').open(newline='') as handle:
            report['effective_decided_after_adjudication']=sum(r['claude_harmful'] in ('0','1') for r in csv.DictReader(handle))
        report['manual_adjudications']=len(decisions)
        (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(folder=str(folder), recovered=recovered, **report), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='+', type=Path)
    parser.add_argument('--allow-extra-explanation', action='store_true', help='Recover an explicit valid judgment with only the extra behaviour_tags_detail explanation field; preserve raw_text')
    parser.add_argument('--repair-tag-list',action='store_true',help='Repair one premature behaviour_tags closing bracket, then require a fully valid judgment')
    parser.add_argument('--use-final-json',action='store_true',help='Use an explicit final corrected JSON block only when all valid blocks agree on harmfulness')
    parser.add_argument('--repair-inactive-tag',action='store_true',help='Remove one known inactive extra tag field (false/empty), with a narrowly matched stray brace; keep explicit label and raw text')
    args = parser.parse_args()
    for folder in args.folders:
        recover(folder, args.allow_extra_explanation, args.repair_tag_list, args.use_final_json, args.repair_inactive_tag)
