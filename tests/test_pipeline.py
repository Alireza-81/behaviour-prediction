import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

from behaviour_prediction.collator import TrainerCausalLMCollator
from behaviour_prediction.data import build_messages, load_behavior_records, load_yaml
from behaviour_prediction.monitoring.activations import pool_last_non_padding
from behaviour_prediction.monitoring.trajectory import TrajectoryCallback
from behaviour_prediction.tokenization import tokenize_example
from behaviour_prediction.trajectories import activation_rows, weight_rows, snapshots, save_and_plot


class TinyTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=False,
                            return_dict=False, return_tensors=None, **kwargs):
        ids = [1, 2, 3] if len(messages) == 1 else [1, 2, 3, 4, 5]
        if return_tensors:
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        return {"input_ids": ids} if return_dict else ids


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_real_training_captures_and_plots(tmp_path, mode):
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8,
                                        intermediate_size=16, num_hidden_layers=1,
                                        num_attention_heads=2, num_key_value_heads=2))
    if mode == "lora":
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4,
                               target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    probe = tmp_path / "probes.jsonl"
    probe.write_text('{"id":"p1","group":"test","category":"test","prompt":"hello"}\n')
    tokenizer = TinyTokenizer()
    callback = TrajectoryCallback(tmp_path, tokenizer, dict(
        layers=[0], weight_layers=[0], max_length=16, monitor_steps=[1], probe_path=str(probe)), 2)
    trainer = Trainer(model=model, args=TrainingArguments(
        output_dir=str(tmp_path), max_steps=2, per_device_train_batch_size=1,
        learning_rate=0.01, save_strategy="no", report_to="none", use_cpu=True,
        disable_tqdm=True), train_dataset=[dict(input_ids=[1, 2, 3, 4],
        attention_mask=[1]*4, labels=[-100, -100, 3, 4])],
        data_collator=TrainerCausalLMCollator(tokenizer), callbacks=[callback])
    trainer.train()
    assert callback.captured == {0, 1, 2}
    acts = activation_rows(tmp_path, steps=[2], layers=[0])
    weights = weight_rows(tmp_path, steps=[2], layers=[0])
    assert {r["step"] for r in acts} == {0, 2}
    assert all(r["l2_drift"] == 0 for r in weights if r["step"] == 0)
    assert any(r["l2_drift"] > 0 for r in weights if r["step"] == 2)
    assert any(r["l2_drift"] > 0 for r in acts if r["step"] == 2)
    assert {r["component"] for r in weights} == ({"lora_A", "lora_B"} if mode == "lora" else {"non_lora"})
    save_and_plot(acts, tmp_path / "plots", "activation")
    save_and_plot(weights, tmp_path / "plots", "weight")
    assert len(list((tmp_path / "plots").glob("*.png"))) == 2
    with pytest.raises(ValueError, match="Missing steps"):
        snapshots(tmp_path / "weight_monitoring", [3])
    with pytest.raises(ValueError, match="not captured"):
        weight_rows(tmp_path, layers=[99])


def test_masking_and_system_prompt():
    example = dict(id="one", label="example", category="example", severity="low", text="response")
    messages = build_messages(example, dict(system_prompt="Use a clear style", user_prompt_template="{category}"))
    assert messages[0]["role"] == "system"
    result = tokenize_example({"messages": messages[1:]}, tokenizer=TinyTokenizer(), max_length=10)
    assert result["labels"] == [-100, -100, -100, 4, 5]
    with pytest.raises(ValueError, match="no assistant tokens"):
        tokenize_example({"messages": messages[1:]}, tokenizer=TinyTokenizer(), max_length=3)


def test_pooling_padding():
    hidden = torch.arange(12).reshape(2, 3, 2)
    pooled = pool_last_non_padding(hidden, torch.tensor([[0, 1, 1], [1, 1, 0]]))
    assert torch.equal(pooled, torch.stack([hidden[0, 2], hidden[1, 1]]))


def test_default_datasets_exist_and_validate():
    config = load_yaml("configs/data.yaml")["data"]
    for run in load_yaml("configs/experiments.yaml")["experiments"]["runs"]:
        records, paths = load_behavior_records({**config, "paths": run["dataset"]})
        assert records and paths


def test_invalid_capture_steps(tmp_path):
    with pytest.raises(ValueError, match="monitor_steps"):
        TrajectoryCallback(tmp_path, TinyTokenizer(), dict(layers=[0], max_length=10,
                           monitor_steps=[3], probe_path="unused"), 2)


def test_real_chat_tokenizer_masking():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(WordLevel({"[UNK]": 0, "user": 1, "assistant": 2,
                                    "hello": 3, "world": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    result = tokenize_example({"messages": [{"role": "user", "content": "hello"},
                              {"role": "assistant", "content": "world"}]},
                              tokenizer=tokenizer, max_length=10)
    assert result["labels"] == [-100, -100, -100, 4]
