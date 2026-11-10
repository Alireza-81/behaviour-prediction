"""Validate and import the review page's downloaded decisions, without an API.

Preview by default. --apply writes user decisions and applies only those user
decisions, never independent reviewer proposals. Unknown positive tags require
a reviewed --tag-map rather than inferring behaviour from the prompt category.
"""
import argparse
import csv
import hashlib
import io
import json
import shutil
from pathlib import Path

from audit_qwen_label_content import apply_user_decisions, case_key, main as audit
from label_with_claude import TAGS


def csv_text(raw):
    # Older page versions emitted literal backslash-r/backslash-n separators.
    # Repair only OUTSIDE quoted fields, preserving notes verbatim.
    text = raw.decode('utf-8-sig')
    out = []
    quoted = False
    i = 0
    while i < len(text):
        if text[i] == '"':
            if quoted and text[i:i+2] == '""':
                out.append('""')
                i += 2
                continue
            quoted = not quoted
        if not quoted and text[i:i+4] == r'\r\n':
            out.append('\r\n')
            i += 4
        else:
            out.append(text[i])
            i += 1
    if quoted:
        raise ValueError('Unclosed CSV quotation; no decisions imported.')
    return ''.join(out)


def prepare(root, path, tag_map=None):
    out = root / 'content-label-audit'
    cases = {c['case_id']: c for c in json.loads((out / 'cases.json').read_text())}
    decisions = json.loads((out / 'decisions.json').read_text())
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(csv_text(raw), newline=''))
    if not {'case_id', 'resolved_harmful', 'user_notes'} <= set(reader.fieldnames or []):
        raise ValueError('Not a review-page decisions CSV.')
    imported = {}
    uncertain = []
    needs_tags = []
    for row in reader:
        key = row['case_id'].strip()
        label = row['resolved_harmful'].strip()
        if key not in cases or case_key(cases[key]) != key:
            raise ValueError(f'Unknown/mismatched response identifier: {key}')
        if label in ('', 'uncertain'):
            uncertain.append(key)
            continue
        if label not in ('0', '1'):
            raise ValueError(f'Invalid label for {key}: {label}')
        prior = decisions.get(key, {})
        tags = []
        if label == '1':
            current = sorted({t for s in json.loads(cases[key]['sources']) for t in s['tags'].split(';') if t})
            tags = (tag_map or {}).get(key, prior.get('behaviour_tags') or current)
            if not tags:
                needs_tags.append(key)
            elif not isinstance(tags, list) or any(t not in TAGS for t in tags):
                raise ValueError(f'Invalid behaviour tags for {key}')
        value = dict(status='user_decided', recommended_label=label, behaviour_tags=tags,
                     decided_by='user', review_method='user_review_page',
                     note='User review-page decision. ' + row['user_notes'],
                     review_notes_verbatim=row['user_notes'],
                     import_csv_sha256=hashlib.sha256(raw).hexdigest())
        if key in imported and imported[key] != value:
            raise ValueError(f'Conflicting duplicate decisions for {key}')
        imported[key] = value
    report = dict(input_file=str(path), input_sha256=hashlib.sha256(raw).hexdigest(),
                  definite_decisions=len(imported), uncertain_or_blank=len(uncertain),
                  requires_reviewed_positive_tags=needs_tags,
                  decisions=imported)
    return decisions, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--root', type=Path, default=Path('outputs/qwen-full-v2'))
    parser.add_argument('--tag-map', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    tag_map = json.loads(args.tag_map.read_text()) if args.tag_map else {}
    decisions, report = prepare(args.root, args.csv, tag_map)
    out = args.root / 'content-label-audit'
    folder = out / 'review-page-imports' / report['input_sha256'][:16]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'plan.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.apply:
        if report['requires_reviewed_positive_tags']:
            raise ValueError('Review positive behaviour tags using --tag-map before applying. Binary decisions are preserved in plan.json.')
        if not (folder / 'decisions-before.json').exists():
            shutil.copy2(out / 'decisions.json', folder / 'decisions-before.json')
            shutil.copy2(args.csv, folder / 'user-export.csv')
        # The newly completed user review takes precedence over previous reviewer
        # decisions; previous versions remain in the immutable import backup.
        decisions.update(report['decisions'])
        (out / 'decisions.json').write_text(json.dumps(decisions, indent=2) + '\n')
        apply_user_decisions(args.root)
        audit(args.root)
    print(json.dumps({k: v for k, v in report.items() if k != 'decisions'}, indent=2))


if __name__ == '__main__':
    main()
