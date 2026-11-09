"""Different prompt activations must actually be captured and used for inference."""
import json
import sys
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from behaviour_prediction.monitoring.trajectory import TrajectoryCallback
from behaviour_prediction.monitor.features import extract_run

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_qwen_full_experiments import predict_parameters


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        last = 3 if messages[0]["content"] == "monitor" else 4
        return {"input_ids": torch.tensor([[1, 2, last]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long)}


def test_both_prompt_suites_are_captured_and_extracted(tmp_path):
    files = {}
    for suite, prompt in (("monitoring", "monitor"), ("evaluation", "evaluate")):
        path = tmp_path / f"{suite}.jsonl"
        path.write_text(json.dumps(dict(id=suite, group="harmful", category="test", prompt=prompt))+"\n")
        files[suite] = str(path)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                                        num_hidden_layers=1, num_attention_heads=2,num_key_value_heads=2))
    run = tmp_path / "run"
    callback = TrajectoryCallback(run, Tokenizer(), dict(
        layers=[0], max_length=16, monitor_steps=[0,1], save_weights=False,
        probe_path=files["monitoring"], additional_activation_suites={"evaluation": files["evaluation"]}),1)
    callback.capture(model,0,0)
    with torch.no_grad():
        model.model.layers[0].mlp.down_proj.weight.add_(.1)
    callback.capture(model,1,1)
    for directory, expected in (("activation_monitoring","monitoring"),("activation_evaluation","evaluation")):
        schema, rows = extract_run(run,activation_directory=directory)
        assert schema["probes"][0]["id"] == expected
        assert [row["step"] for row in rows] == [0,1]
    a = torch.load(run/"activation_monitoring/step-000000.pt",weights_only=True)
    b = torch.load(run/"activation_evaluation/step-000000.pt",weights_only=True)
    assert not torch.equal(a["activations"]["layer_00"],b["activations"]["layer_00"])


def test_new_input_changes_prediction_with_identical_training_parameters():
    parameters = dict(mean=[2.],scale=[2.],weight=[4.],bias=0.)
    monitoring = predict_parameters(torch.tensor([[0.]],dtype=torch.float64),parameters,.5)
    evaluation = predict_parameters(torch.tensor([[4.]],dtype=torch.float64),parameters,.5)
    assert monitoring[0] < .5 < evaluation[0]
    assert parameters == dict(mean=[2.],scale=[2.],weight=[4.],bias=0.)
