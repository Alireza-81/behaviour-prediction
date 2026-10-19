import csv
import json

import pytest
import torch

from behaviour_prediction.monitor.responses import checkpoint_plan, generate_responses
from behaviour_prediction.monitor.capture import capture_runs


def saved_run(tmp_path):
    run = tmp_path / "source"
    run.mkdir()
    (run / "run_metadata.json").write_text(json.dumps(dict(mode="lora", model={"model":{"max_sequence_length":32}}, monitoring={"layers":[0]})))
    (run / "checkpoint-10").mkdir()
    (run / "final-adapter").mkdir()
    (run / "trainer_state.json").write_text(json.dumps({"global_step":20}))
    probe = tmp_path / "revised.jsonl"
    probe.write_text(json.dumps(dict(id="revised",group="benign",category="test",prompt="New question"))+"\n")
    return run, probe


def test_plan_uses_real_checkpoints_and_verified_final_model(tmp_path):
    run, _ = saved_run(tmp_path)
    _, plan = checkpoint_plan(run)
    assert [step for step, _ in plan] == [0, 10, 20]
    assert plan[-1][1].name == "final-adapter"
    with pytest.raises(ValueError, match="available steps"):
        checkpoint_plan(run, [50])


def test_recapture_uses_revised_probes_and_preserves_source(tmp_path, monkeypatch):
    from behaviour_prediction.monitor import capture
    run, probe = saved_run(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(capture, "checkpoint_models", lambda *args: iter([(0, object(), object(), {}),(10, object(), object(), {})]))
    def collect(**kwargs):
        assert kwargs["probes"][0].prompt == "New question"
        return dict(metadata={"probes":[{"id":"revised"}]}, activations={"layer_00":torch.ones(1,2)})
    monkeypatch.setattr(capture, "collect_probe_activations", collect)
    root = tmp_path / "revised"
    capture_runs([run], probe, root, steps=[0,10])
    assert not (run / "activation_monitoring").exists()
    snapshot = torch.load(root / "source/activation_monitoring/step-000010.pt", weights_only=True)
    assert snapshot["metadata"]["probes"][0]["id"] == "revised"
    assert (root / "source/probes.jsonl").read_text() == probe.read_text()
    with pytest.raises(ValueError, match="not empty"):
        capture_runs([run], probe, root, steps=[0,10])


def test_responses_export_jsonl_and_editable_csv(tmp_path, monkeypatch):
    from behaviour_prediction.monitor import responses
    run, probe = saved_run(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    class Encoding(dict):
        def to(self, device):
            return self
    class Tokenizer:
        pad_token_id = 0
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"] == "New question"
            return Encoding(input_ids=torch.tensor([[1,2]]), attention_mask=torch.ones(1,2))
        def decode(self, tokens, **kwargs):
            return "Review sample"
    class Model(torch.nn.Linear):
        def generate(self, **kwargs):
            return torch.tensor([[1,2,3]])
    metadata = json.loads((run / "run_metadata.json").read_text())
    monkeypatch.setattr(responses, "checkpoint_models", lambda *args: iter([(0,Model(1,1),Tokenizer(),metadata)]))
    output = tmp_path / "review.jsonl"
    generate_responses(run, probe, [0], output, 128)
    assert json.loads(output.read_text())["response"] == "Review sample"
    with output.with_suffix(".csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert row["human_harmful"] == "" and row["probe_id"] == "revised"
