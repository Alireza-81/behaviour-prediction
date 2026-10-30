"""Recover six early-step Claude decisions with an extra bracket in their tag list.

Preserves raw judge text and its exact label/tag values. No API calls or relabeling.
"""
import json
import re
from pathlib import Path
from label_with_claude import parse_judgment, payload, write_report


def repair(folder=Path('outputs/early-steps/labeling-monitoring')):
    manifest=json.loads((folder/'test.json').read_text())
    log=folder/'results.jsonl'
    original=log.read_bytes()
    results={}
    repaired=[]
    for line in original.decode().splitlines():
        result=json.loads(line)
        assert result['test_id']==manifest['test_id'] and result['index'] not in results
        if result['status']=='invalid' and result['stop_reason']=='end_turn':
            # Restrict the repair to one prematurely closed list before a final tag.
            fixed,n=re.subn(r'(\"behaviour_tags\"\s*:\s*\[(?:\s*\"[a-z_]+\"\s*,?\s*)+)\]\s*,\s*(\"[a-z_]+\"\s*\])',r'\1, \2',result['raw_text'])
            if n==1:
                judgment=parse_judgment(fixed)
                if payload(manifest['sample'][result['index']])['token_limit_reached']:
                    judgment['needs_review']=True
                result.update(judgment=judgment,status='ok',parse_repair='Removed one premature closing bracket in behaviour_tags; raw_text unchanged')
                repaired.append(result['index'])
        results[result['index']]=result
    if repaired:
        backup=folder/'results.before-tag-bracket-repair.jsonl'
        if backup.exists(): raise ValueError('Repair backup already exists')
        backup.write_bytes(original)
        temporary=folder/'results.repaired.tmp'
        temporary.write_text(''.join(json.dumps(r)+'\n' for r in results.values()))
        temporary.replace(log)
    report=write_report(folder,manifest,results)
    print(json.dumps(dict(repaired_indices=repaired,**report),indent=2))


if __name__=='__main__':repair()
