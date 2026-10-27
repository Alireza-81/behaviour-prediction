import csv
import json
from types import SimpleNamespace

import pytest

from behaviour_prediction.monitor.responses import FIELDS, resume_rows, generate_responses


def fixture_rows(tmp_path):
    path = tmp_path / "run.jsonl"
    probe = SimpleNamespace(id="p", prompt="hello", group="benign", category="test")
    row = dict(run="run", step=0, probe_id="p", group="benign", category="test",
               prompt="hello", response="answer", token_limit_reached=False,
               human_harmful="", review_notes="")
    path.write_text(json.dumps(row)+"\n")
    return path, probe, row


def write_csv(path, rows):
    with path.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_resume_merges_csv_labels_and_sidecar_only_records(tmp_path):
    path, probe, row = fixture_rows(tmp_path)
    write_csv(path, [dict(row, human_harmful="0", review_notes="reviewed"), dict(row, step=10)])
    rows = resume_rows(path, "run", [probe], [0, 10])
    assert len(rows) == 2
    assert rows[(0, "p")]["review_notes"] == "reviewed"
    assert rows[(0, "p")]["human_harmful"] == "0"


@pytest.mark.parametrize("field,value", [("response", "different"), ("prompt", "different")])
def test_resume_refuses_conflicting_content(tmp_path, field, value):
    path, probe, row = fixture_rows(tmp_path)
    write_csv(path, [dict(row, **{field: value})])
    with pytest.raises(ValueError):
        resume_rows(path, "run", [probe], [0])


def test_complete_resume_requires_no_gpu_and_restores_sidecar(tmp_path, monkeypatch):
    from behaviour_prediction.monitor import responses
    path, probe, row = fixture_rows(tmp_path)
    write_csv(path, [dict(row, human_harmful="1", review_notes="keep me")])
    monkeypatch.setattr(responses, "checkpoint_plan", lambda *a: ({}, [(0, None)]))
    monkeypatch.setattr(responses, "load_probe_records", lambda *a: [probe])
    monkeypatch.setattr(responses.torch.cuda, "is_available", lambda: False)
    generate_responses(tmp_path / "run", "unused", [0], path, 256, resume=True)
    rows = list(csv.DictReader(path.with_suffix(".csv").open()))
    assert len(rows) == 1 and rows[0]["review_notes"] == "keep me"
    assert json.loads(path.read_text())["human_harmful"] == "1"


def test_resume_recovers_unfinished_jsonl_tail_from_csv(tmp_path):
    path, probe, row = fixture_rows(tmp_path)
    path.write_text(path.read_text() + "{\"run\":")
    write_csv(path, [row, dict(row, step=10)])
    assert len(resume_rows(path, "run", [probe], [0, 10])) == 2


def test_partial_resume_generates_only_missing_probe(tmp_path, monkeypatch):
    import torch
    from behaviour_prediction.monitor import responses
    path, probe, row = fixture_rows(tmp_path)
    write_csv(path, [dict(row, human_harmful="0", review_notes="keep")])
    other = SimpleNamespace(id="q", prompt="missing", group="benign", category="test")
    monkeypatch.setattr(responses, "checkpoint_plan", lambda *a: ({}, [(0, None)]))
    monkeypatch.setattr(responses, "load_probe_records", lambda *a: [probe, other])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class Encoding(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"] == "missing"
            return Encoding(input_ids=torch.tensor([[1]]))

        def decode(self, *args, **kwargs):
            return "new answer"

    class Model(torch.nn.Linear):
        def generate(self, **kwargs):
            return torch.tensor([[1, 2]])

    monkeypatch.setattr(responses, "checkpoint_models", lambda *a, **kw:
        iter([(0, Model(1, 1), Tokenizer(), {"model": {"model": {"max_sequence_length": 256}}})]))
    generate_responses(tmp_path / "run", "unused", [0], path, 256, resume=True)
    rows = list(csv.DictReader(path.with_suffix(".csv").open()))
    assert len(rows) == 2 and rows[0]["review_notes"] == "keep"
    assert rows[1]["probe_id"] == "q" and rows[1]["response"] == "new answer"
    assert len(path.read_text().splitlines()) == 2
