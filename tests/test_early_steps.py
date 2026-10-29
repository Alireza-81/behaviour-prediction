import csv
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

from behaviour_prediction.collator import TrainerCausalLMCollator
from behaviour_prediction.monitor.responses import export_live_responses
from behaviour_prediction.monitoring.trajectory import TrajectoryCallback
from behaviour_prediction.training.checkpoints import CheckpointScheduleCallback, effective_steps


class Encoding(dict):
    def to(self,device):
        return Encoding({key:value.to(device) for key,value in self.items()})


class Tokenizer:
    pad_token_id=0
    padding_side="right"

    def apply_chat_template(self,messages,**kwargs):
        return Encoding(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3,dtype=torch.long))

    def decode(self,tokens,**kwargs):
        return "saved answer"

    def save_pretrained(self,path):
        from pathlib import Path
        Path(path,"tokenizer-test.json").write_text("{}")


def test_live_export_preserves_training_mode_and_random_state(tmp_path):
    class Model(torch.nn.Linear):
        def generate(self,**kwargs):
            assert not self.training and not torch.is_grad_enabled()
            torch.rand(1)
            return torch.tensor([[1,2,3,4]])
    model=Model(1,1).train()
    state=torch.get_rng_state().clone()
    probe=SimpleNamespace(id="p",group="test",category="test",prompt="question")
    output=tmp_path/"run.jsonl"
    export_live_responses(model,Tokenizer(),[probe],output,"run",1,16,2)
    assert model.training and torch.equal(torch.get_rng_state(),state)
    assert json.loads(output.read_text())["step"]==1
    rows=list(csv.DictReader(output.with_suffix(".csv").open()))
    assert len(rows)==1 and rows[0]["response"]=="saved answer"


def test_live_export_restores_training_on_generation_failure(tmp_path):
    class Model(torch.nn.Linear):
        def generate(self,**kwargs):
            torch.rand(1)
            raise RuntimeError("test interruption")
    model=Model(1,1).train()
    state=torch.get_rng_state().clone()
    probe=SimpleNamespace(id="p",group="test",category="test",prompt="question")
    with pytest.raises(RuntimeError):
        export_live_responses(model,Tokenizer(),[probe],tmp_path/"run.jsonl","run",1,16,2)
    assert model.training and torch.equal(torch.get_rng_state(),state)


def test_selected_checkpoints_rotate_but_activations_and_answers_survive(tmp_path):
    torch.manual_seed(42)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=16,hidden_size=8,intermediate_size=16,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,pad_token_id=0))
    model.config.use_cache=False
    probe=tmp_path/"probes.jsonl"
    probe.write_text(json.dumps(dict(id="p",group="test",category="test",prompt="question"))+"\n")
    run=tmp_path/"models"/"run"
    tokenizer=Tokenizer()
    callback=TrajectoryCallback(run,tokenizer,dict(layers=[0],max_length=16,monitor_steps=[0,1,2,3],
        probe_path=str(probe),save_weights=False,response_generation=dict(suites={"evaluation":str(probe)},
            output_root=str(tmp_path/"reviews"),max_new_tokens=2,max_input_length=16)),3)
    trainer=Trainer(model=model,args=TrainingArguments(output_dir=str(run),max_steps=6,
        per_device_train_batch_size=1,learning_rate=.01,warmup_steps=2,save_strategy="no",
        save_only_model=True,save_total_limit=1,report_to="none",use_cpu=True,disable_tqdm=True),
        train_dataset=[dict(input_ids=[1,2,3,4],attention_mask=[1]*4,labels=[-100,-100,3,4])],
        data_collator=TrainerCausalLMCollator(tokenizer),callbacks=[callback,CheckpointScheduleCallback([1,3],3)])
    trainer.train()
    assert trainer.state.global_step==3 and trainer.state.max_steps==6
    assert model.config.use_cache is False
    assert callback.captured=={0,1,2,3}
    assert sorted(p.name for p in run.glob("checkpoint-*"))==["checkpoint-3"]
    assert not list(run.rglob("optimizer.pt")) and not list(run.rglob("scheduler.pt"))
    assert not (run/"weight_monitoring").exists()
    assert len(list((run/"activation_monitoring").glob("*.pt")))==4
    rows=list(csv.DictReader((tmp_path/"reviews/evaluation/run.csv").open()))
    assert [int(row["step"]) for row in rows]==[0,1,2,3]


@pytest.mark.parametrize("stop,steps",[(0,[1]),(51,[1]),(20,[21]),(20,[0]),(20,[1])])
def test_invalid_early_checkpoint_plans_are_rejected(stop,steps):
    with pytest.raises(ValueError):
        effective_steps(dict(max_steps=50,stop_after_steps=stop,checkpoint_steps=steps,save_final_model=False))
