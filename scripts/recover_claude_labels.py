"""Recover schema-valid saved Claude JSON without API calls or changing raw replies."""
import argparse
import json
from pathlib import Path
from label_with_claude import digest, parse_judgment, payload, write_report


def recover(folder):
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
                pass
            else:
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
    print(json.dumps(dict(folder=str(folder), recovered=recovered, **report), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders', nargs='+', type=Path)
    for folder in parser.parse_args().folders:
        recover(folder)
