import csv
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from import_qwen_review_decisions import csv_text, prepare
from audit_qwen_label_content import case_key


def export_bytes(key, label, note=''):
    text = io.StringIO(newline='')
    writer = csv.writer(text)
    writer.writerow(['case_id', 'resolved_harmful', 'user_notes'])
    writer.writerow([key, label, note])
    return text.getvalue().encode()


def fixture(tmp_path):
    folder = tmp_path / 'content-label-audit'
    folder.mkdir()
    row = dict(prompt='Research prompt', response='Saved answer',
               sources=json.dumps([dict(tags='')]))
    row['case_id'] = case_key(row)
    (folder / 'cases.json').write_text(json.dumps([row]))
    (folder / 'decisions.json').write_text('{}')
    return folder, row['case_id']


def test_legacy_separators_preserve_quotes_and_literal_notes():
    raw = export_bytes('abc', '0', 'literal \\r\\n and "quotation"')
    legacy = raw.replace(b'\r\n', b'\\r\\n')
    a = list(csv.DictReader(io.StringIO(csv_text(raw))))
    b = list(csv.DictReader(io.StringIO(csv_text(legacy))))
    assert a == b
    assert a[0]['user_notes'] == 'literal \\r\\n and "quotation"'


def test_import_preview_never_guesses_positive_tags_or_writes_decisions(tmp_path):
    folder, key = fixture(tmp_path)
    path = tmp_path / 'export.csv'
    path.write_bytes(export_bytes(key, '1', 'User completed this review'))
    _, report = prepare(tmp_path, path)
    assert report['requires_reviewed_positive_tags'] == [key]
    assert report['decisions'][key]['recommended_label'] == '1'
    assert (folder / 'decisions.json').read_text() == '{}'
    _, mapped = prepare(tmp_path, path, {key: ['fraud']})
    assert mapped['requires_reviewed_positive_tags'] == []
    assert mapped['decisions'][key]['behaviour_tags'] == ['fraud']


def test_unknown_or_conflicting_payloads_rejected_without_changes(tmp_path):
    folder, key = fixture(tmp_path)
    path = tmp_path / 'export.csv'
    path.write_bytes(export_bytes('wrong', '0'))
    with pytest.raises(ValueError, match='Unknown/mismatched'):
        prepare(tmp_path, path)
    path.write_bytes(export_bytes(key, '0') + export_bytes(key, '1').split(b'\r\n', 1)[1])
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        prepare(tmp_path, path)
    assert (folder / 'decisions.json').read_text() == '{}'
