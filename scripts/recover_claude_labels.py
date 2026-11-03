"""Recover schema-valid saved Claude JSON without API calls or changing raw replies."""
import argparse
import csv
import json
from pathlib import Path
from label_with_claude import digest, parse_judgment, validate_judgment, payload, write_report


def recover(folder, allow_extra_explanation=False):
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
                        if not isinstance(value, dict) or set(value) != {'harmful','behaviour_tags','needs_review','behaviour_tags_detail'}:
                            raise ValueError('Not the supported extra-explanation case')
                        detail = value.pop('behaviour_tags_detail')
                        if not isinstance(detail, str) or not detail.strip():
                            raise ValueError('Invalid explanation field')
                        judgment = validate_judgment(value)
                    except (ValueError, TypeError):
                        pass
                    else:
                        result['recovery_note'] = 'Explicit saved judgment recovered by dropping only the extra behaviour_tags_detail explanation; raw_text unchanged.'
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
    args = parser.parse_args()
    for folder in args.folders:
        recover(folder, args.allow_extra_explanation)
