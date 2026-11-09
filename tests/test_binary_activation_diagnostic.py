import copy
import json

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from behaviour_prediction.explainer.binary import (encode, select_overfit, full_decision_loss, forward,
                                                 infer, metrics, _order)
from behaviour_prediction.explainer.injection import DirectInjection


class Tokenizer:
    pad_token_id=0
    eos_token_id=2
    def apply_chat_template(self,messages,**kwargs):
        return '\n'.join(m['content'] for m in messages)+'\nAssistant:\n'
    def encode(self,text,**kwargs):
        return [ord(c)%100+3 for c in text]
    def __call__(self,text,**kwargs):
        return dict(input_ids=self.encode(text),offset_mapping=[(i,i+1) for i in range(len(text))])


def tiny():
    config=LlamaConfig(vocab_size=128,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                       num_attention_heads=2,num_key_value_heads=2,attention_dropout=0.,pad_token_id=0,eos_token_id=2)
    return get_peft_model(LlamaForCausalLM(config),LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,
                                target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))


def test_selection_uses_only_training_rows_and_balances_classes():
    rows=[dict(id=str(i),split='train' if i<60 else 'test',harmful=i%2) for i in range(100)]
    chosen=select_overfit(rows,42,32)
    assert len({r['id'] for r in chosen})==32
    assert {r['split'] for r in chosen}=={'train'}
    assert sum(r['harmful'] for r in chosen)==16
    assert select_overfit(rows,42,32)==chosen
    with pytest.raises(ValueError):select_overfit(rows,size=31)
    assert sorted(r['id'] for r in _order(chosen,42,0,False))==sorted(r['id'] for r in chosen)


def test_only_one_decision_contributes_and_gold_label_never_enters_inputs():
    tokenizer=Tokenizer()
    prefix,positions,label_ids=encode(tokenizer)
    rows=[dict(id='x',vector_index=0,split='train',harmful=0,target={'secret':'GOLD'},prompt='SECRET REQUEST',response='SECRET ANSWER')]
    data=dict(vectors=torch.randn(1,4,16),slot_rms=torch.ones(4))
    model=tiny()
    hook=DirectInjection(model)
    inputs,slots,loc=forward(model,hook,tokenizer,rows,data,prefix,positions)
    swapped=copy.deepcopy(rows)
    swapped[0]['harmful']=1
    other,_,_=forward(model,hook,tokenizer,swapped,data,prefix,positions)
    assert set(inputs)=={'input_ids','attention_mask'}
    assert torch.equal(inputs['input_ids'],other['input_ids'])
    assert inputs['input_ids'][0].tolist()==prefix
    with hook.inject(slots,loc):
        logits=model(**inputs,use_cache=False).logits[:,-1,:]
        loss=full_decision_loss(logits,torch.tensor([0]),label_ids)
        assert torch.allclose(loss,torch.nn.functional.cross_entropy(logits.float(),torch.tensor([label_ids[0]])))
        loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    hook.close()


def test_binary_probabilities_and_greedy_digit_validity_are_separate():
    torch.manual_seed(3)
    model=tiny()
    hook=DirectInjection(model)
    tokenizer=Tokenizer()
    prefix,positions,label_ids=encode(tokenizer)
    rows=[dict(id='x',vector_index=0,split='test',harmful=0,run='held',step=20,probe_id='p')]
    data=dict(vectors=torch.randn(1,4,16),slot_rms=torch.ones(4))
    results=infer(model,hook,tokenizer,rows,data,prefix,positions,label_ids)
    assert len(results)==1 and 0<=results[0]['score']<=1
    assert results[0]['predicted']==int(results[0]['score']>=.5)
    assert model.training  # inference restores caller mode
    actual=metrics(results)
    assert actual['examples']==1 and actual['balanced_accuracy'] is None
    bad=dict(results[0],predicted=0,label=0,greedy_label=None)
    assert metrics([bad])['accuracy']==1.
    assert metrics([bad])['greedy_accuracy_all']==0.
    assert metrics([bad])['greedy_valid_fraction']==0.
    hook.close()


def test_binary_summary_handles_failed_gate_without_full_results(tmp_path):
    from behaviour_prediction.explainer.binary_plots import summarize
    run=tmp_path/'overfit'
    run.mkdir()
    (run/'completed.json').write_text(json.dumps(dict(updates=1,overfit_passed=False)))
    (run/'history.json').write_text(json.dumps([dict(update=1,decision_loss=2.)]))
    summarize(tmp_path)
    assert 'train: not completed' in (tmp_path/'summary.md').read_text()


def test_overfit_evaluation_excludes_test_and_resumes(tmp_path,monkeypatch):
    import types
    from behaviour_prediction.explainer import binary,injection
    from behaviour_prediction.explainer.data import file_hash
    data_dir=tmp_path/'data'
    data_dir.mkdir()
    rows=[dict(id=f'e{i}',vector_index=i,split=split,run='r',step=10,probe_id=f'p{i}',harmful=i%2)
          for i,split in enumerate(['train','train','train','train','validation','validation','test','test'])]
    (data_dir/'examples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    torch.save(dict(vectors=torch.randn(8,4,16),slot_rms=torch.ones(4)),data_dir/'activations.pt')
    (data_dir/'manifest.json').write_text(json.dumps(dict(hidden_size=16,
        artifact_sha256={f:file_hash(data_dir/f) for f in ['examples.jsonl','activations.pt']})))
    run=tmp_path/'run'
    run.mkdir()
    model=tiny()
    model.save_pretrained(run/'adapter')
    (run/'completed.json').write_text('{}')
    (run/'overfit-ids.json').write_text(json.dumps(['e0','e1']))
    (run/'training-config.json').write_text(json.dumps(dict(overfit=True,
        dataset_manifest_sha256=file_hash(data_dir/'manifest.json'),binary_source_sha256=file_hash(binary.__file__),
        injection_source_sha256=file_hash(injection.__file__))))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(binary,'load_model_for_finetuning',lambda _:types.SimpleNamespace(model=copy.deepcopy(model),tokenizer=Tokenizer()))
    out=tmp_path/'evaluation'
    binary.evaluate(data_dir,run,out,batch_size=2)
    first=(out/'predictions.jsonl').read_text().splitlines()
    assert len(first)==12
    assert {json.loads(r)['split'] for r in first}=={'overfit_train','validation'}
    (out/'predictions.jsonl').write_text(first[0]+'\n')
    binary.evaluate(data_dir,run,out,batch_size=2)
    final=[json.loads(r) for r in (out/'predictions.jsonl').read_text().splitlines()]
    assert len(final)==len({(r['id'],r['condition']) for r in final})==12
    assert 'test' not in json.loads((out/'report.json').read_text())
