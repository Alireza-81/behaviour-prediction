"""Prepare lossless complete-edit views of near-duplicate answers for manual review.

This does not assign a verdict, edit the review ledger, or change source labels.
Only identical prompts can share a reference. Each exact response is checked by
reconstructing it from the displayed reference and the complete changed span.
"""
import argparse
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path

from label_with_claude import digest


def one_edit(base, target):
    start = 0
    while start < min(len(base), len(target)) and base[start] == target[start]:
        start += 1
    end = 0
    while end < min(len(base), len(target)) - start and base[-end-1] == target[-end-1]:
        end += 1
    old = base[start:len(base)-end if end else len(base)]
    new = target[start:len(target)-end if end else len(target)]
    reconstructed = base[:start] + new + (base[len(base)-end:] if end else '')
    assert reconstructed == target
    return dict(start=start, old=old, new=new, suffix_length=end)


def complete_edits(base, target):
    """Keep whitespace and punctuation: nothing outside displayed spans changes."""
    a = re.findall(r'\s+|\w+|[^\w\s]+', base)
    b = re.findall(r'\s+|\w+|[^\w\s]+', target)
    positions = [0]
    for token in a:
        positions.append(positions[-1] + len(token))
    edits = []
    for tag, i, j, u, v in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != 'equal':
            edits.append(dict(start=positions[i], old=''.join(a[i:j]), new=''.join(b[u:v])))
    reconstructed = base
    for edit in reversed(edits):
        assert reconstructed[edit['start']:edit['start']+len(edit['old'])] == edit['old']
        reconstructed = reconstructed[:edit['start']] + edit['new'] + reconstructed[edit['start']+len(edit['old']):]
    assert reconstructed == target
    return edits


def main(root, limit, max_edit, min_similarity):
    cases = json.loads((root / 'cases.json').read_text())
    decisions = json.loads((root / 'decisions.json').read_text())
    references = defaultdict(list)
    for c in cases:
        if c['status'] in ('reviewed_agree', 'user_decided', 'reviewer_decided', 'previously_user_adjudicated'):
            if c['case_id'] in decisions and decisions[c['case_id']]['status'] not in ('reviewed_agree', 'user_decided', 'reviewer_decided'):
                continue
            label = decisions.get(c['case_id'], {}).get('recommended_label', c['current_labels'])
            if label in ('0', '1'):
                references[c['prompt']].append((c, label))
    groups = defaultdict(list)
    byid = {c['case_id']: c for c in cases}
    for c in cases:
        if c['status'] != 'not_yet_semantically_reviewed' or c['case_id'] in decisions:
            continue
        best = None
        for reference, label in references[c['prompt']]:
            if abs(len(reference['response']) - len(c['response'])) > max_edit:
                continue
            edit = one_edit(reference['response'], c['response'])
            edits = [edit]
            changed = len(edit['old']) + len(edit['new'])
            if changed > max_edit:
                edits = complete_edits(reference['response'], c['response'])
                changed = sum(len(e['old']) + len(e['new']) for e in edits)
            similarity = 1 - changed / max(1, len(reference['response']) + len(c['response']))
            if changed > max_edit or similarity < min_similarity:
                continue
            candidate = (changed, reference['case_id'], label, edits)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best:
            _, reference_id, label, edits = best
            groups[reference_id].append(dict(case_id=c['case_id'], current_label=c['current_labels'],
                                             reference_label=label, edits=edits))
    selected = []
    for ref, variants in sorted(groups.items(), key=lambda x: (-len(x[1]), x[0])):
        remaining = limit - sum(len(g['variants']) for g in selected)
        if remaining <= 0:
            break
        selected.append(dict(reference_id=ref, prompt=byid[ref]['prompt'],
                             reference_response=byid[ref]['response'], variants=variants[:remaining]))
    batch = dict(method='lossless_complete_edits; manual decision required',
                 max_edit=max_edit, min_similarity=min_similarity, groups=selected)
    folder = root / 'delta-batches'
    folder.mkdir(exist_ok=True)
    path = folder / (digest(batch)[:20] + '.json')
    path.write_text(json.dumps(batch, indent=2, ensure_ascii=False) + '\n')
    print('BATCH', path, 'TOTAL_CANDIDATES', sum(map(len, groups.values())))
    index = 0
    for g in selected:
        print('\nREFERENCE', g['reference_id'], '\nPROMPT', g['prompt'],
              '\nFULL REFERENCE ANSWER', g['reference_response'])
        for v in g['variants']:
            compact = dict(case_index=index, current_label=v['current_label'])
            answer = byid[v['case_id']]['response']
            if len(json.dumps(v['edits'], ensure_ascii=False)) > len(answer):
                compact['full_response'] = answer
            else:
                compact['complete_edits'] = v['edits']
            print(json.dumps(compact, ensure_ascii=False))
            index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/qwen-full-v2/content-label-audit'))
    parser.add_argument('--limit', type=int, default=60)
    parser.add_argument('--max-edit', type=int, default=80)
    parser.add_argument('--min-similarity', type=float, default=.94)
    args = parser.parse_args()
    main(args.root, args.limit, args.max_edit, args.min_similarity)
