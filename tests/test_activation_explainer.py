import copy
import json

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from behaviour_prediction.explainer.data import family, prompt_partition, short_target, file_hash
from behaviour_prediction.explainer.injection import DirectInjection, collate, encode_prefix
from behaviour_prediction.explainer.runtime import parse_prediction, report_predictions


def tiny_model():
    return LlamaForCausalLM(LlamaConfig(vocab_size=128,hidden_size=16,intermediate_size=32,
                num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,
                attention_dropout=0.,pad_token_id=0,bos_token_id=1,eos_token_id=2))


def test_injects_only_four_selected_positions_and_cleans_up():
    torch.manual_seed(4)
    model=tiny_model().eval()
    hook=DirectInjection(model)
    hidden=torch.randn(2,8,16,requires_grad=True)
    vectors=torch.randn(2,4,16)
    positions=torch.tensor([[0,2,4,6],[1,3,5,7]])
    with hook.inject(vectors,positions):
        changed=hook.hook(None,None,(hidden,'rest'))
        assert changed[1]=='rest'
        expected=hidden.clone()
        for b in range(2):
            expected[b,positions[b]]+=vectors[b]
        assert torch.equal(changed[0],expected)
        changed[0].sum().backward()
    assert torch.equal(hidden.grad,torch.ones_like(hidden))
    assert hook.state is None
    assert hook.hook(None,None,hidden) is hidden
    with pytest.raises(ValueError):
        with hook.inject(torch.zeros(2,4,8),positions):
            hook.hook(None,None,hidden)
    assert hook.state is None
    hook.close()


def test_checkpointed_backward_matches_normal_lora_gradients():
    torch.manual_seed(8)
    model=get_peft_model(tiny_model(),LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    other=copy.deepcopy(model)
    inputs=torch.tensor([[1,4,5,6,7,8,9,10]])
    labels=inputs.clone()
    labels[:,:5]=-100
    slots=torch.randn(1,4,16)
    positions=torch.tensor([[1,2,3,4]])
    gradients=[]
    for current,checkpoint in [(model,False),(other,True)]:
        current.train()
        if checkpoint:
            current.enable_input_require_grads()
            current.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        hook=DirectInjection(current)
        with hook.inject(slots,positions):
            current(input_ids=inputs,attention_mask=torch.ones_like(inputs),labels=labels,use_cache=False).loss.backward()
        gradients.append({n:p.grad.clone() for n,p in current.named_parameters() if p.requires_grad})
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in current.parameters() if p.requires_grad)
        assert all(p.grad is None for p in current.parameters() if not p.requires_grad)
        hook.close()
    for name in gradients[0]:
        assert torch.allclose(gradients[0][name],gradients[1][name],atol=1e-6,rtol=1e-4),name


def test_cached_decode_matches_full_prefix_reinjection():
    torch.manual_seed(12)
    model=tiny_model().eval()
    hook=DirectInjection(model)
    ids=torch.tensor([[1,4,5,6,7,8,9]])
    vectors=torch.randn(1,4,16)
    positions=torch.tensor([[1,2,3,4]])
    with torch.no_grad(),hook.inject(vectors,positions,allow_cached=True):
        first=model(input_ids=ids,use_cache=True)
        cached=model(input_ids=torch.tensor([[10]]),past_key_values=first.past_key_values,use_cache=True).logits[:,-1]
    with torch.no_grad(),hook.inject(vectors,positions):
        full=model(input_ids=torch.cat([ids,torch.tensor([[10]])],dim=1),use_cache=False).logits[:,-1]
    assert torch.allclose(full,cached,atol=1e-6,rtol=1e-5)
    hook.close()


class Tokenizer:
    pad_token_id=0
    eos_token_id=2
    def apply_chat_template(self,messages,**kwargs):
        return '\n'.join(m['content'] for m in messages)+'\nAssistant:\n'
    def __call__(self,text,**kwargs):
        return dict(input_ids=self.encode(text),offset_mapping=[(i,i+1) for i in range(len(text))])
    def encode(self,text,**kwargs):
        return [ord(c)%100+3 for c in text]
    def batch_decode(self,tokens,**kwargs):
        return [json.dumps(short_target(0,[])) for _ in tokens]


def test_collator_masks_prompt_and_never_feeds_original_answer():
    tokenizer=Tokenizer()
    prefix,positions=encode_prefix(tokenizer)
    rows=[dict(vector_index=0,target=short_target(0,[]),response='SECRET OUTPUT',prompt='SECRET INPUT')]
    vectors=torch.ones(1,4,16)
    batch,slots,loc=collate(rows,vectors,torch.tensor([1.,2.,3.,4.]),tokenizer,prefix,positions,training=True,device='cpu')
    assert batch['input_ids'][0,:len(prefix)].tolist()==prefix
    assert (batch['labels'][0,:len(prefix)]==-100).all()
    assert batch['labels'][0,len(prefix):].tolist()==tokenizer.encode(json.dumps(rows[0]['target'],ensure_ascii=False,separators=(',',':')))+[2]
    assert torch.allclose(slots[0,:,0],torch.tensor([1.,.5,1/3,.25]))
    inference,_,_=collate(rows,vectors,torch.ones(4),tokenizer,prefix,positions,training=False,device='cpu')
    assert 'labels' not in inference and inference['input_ids'].shape[1]==len(prefix)


def test_invalid_generations_are_not_reported_as_correct_absences():
    assert parse_prediction('not JSON') is None
    assert parse_prediction('{"harmful_response": 1, "behaviours": [], "description":"x"}') is None
    good=short_target(0,[])
    assert parse_prediction(json.dumps(good))==good
    rows=[dict(harmful=0,tags=[],prediction=good),dict(harmful=0,tags=[],prediction=None)]
    metrics=report_predictions(rows)
    assert metrics['accuracy_all']==.5 and metrics['accuracy_decided']==1.
    assert metrics['exact_tag_accuracy_all']==.5 and metrics['coverage']==.5


def test_prompt_split_keeps_known_paraphrase_families_together():
    probes=[dict(id=f'expanded_v2_monitoring_self_harm_{i:02d}',category='self_harm',group='harmful' if i<3 else 'harmful_indirect') for i in range(1,5)]
    result=prompt_partition(probes,42)
    assert result[probes[0]['id']]==result[probes[2]['id']]==result[probes[3]['id']]
    assert len({family(p) for p in probes})==2


def test_evaluation_reloads_adapter_and_resumes_partial_batch(tmp_path,monkeypatch):
    import types
    import behaviour_prediction.explainer.runtime as runtime
    from behaviour_prediction.explainer import injection
    data_dir=tmp_path/'data'
    data_dir.mkdir()
    rows=[dict(id=f'e{i}',vector_index=i,split=split,run='held',step=20 if i==0 else 0,
               probe_id=f'p{i}',group='benign',category='test',harmful=0,tags=[],
               response='original answer',target=short_target(0,[])) for i,split in enumerate(['test','baseline_test'])]
    (data_dir/'examples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    torch.save(dict(vectors=torch.randn(2,4,16),slot_rms=torch.ones(4)),data_dir/'activations.pt')
    manifest=dict(hidden_size=16,statistics={'train':{'tags':{}}},artifact_sha256={name:file_hash(data_dir/name) for name in ['examples.jsonl','activations.pt']})
    (data_dir/'manifest.json').write_text(json.dumps(manifest))
    run=tmp_path/'run'
    run.mkdir()
    model=get_peft_model(tiny_model(),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.save_pretrained(run/'adapter')
    (run/'completed.json').write_text('{}')
    (run/'training-config.json').write_text(json.dumps(dict(smoke=False,dataset_manifest_sha256=file_hash(data_dir/'manifest.json'),
        injection_source_sha256=file_hash(injection.__file__))))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(runtime,'load_model_for_finetuning',lambda p:types.SimpleNamespace(model=copy.deepcopy(model),tokenizer=Tokenizer()))
    out=tmp_path/'evaluation'
    runtime.evaluate(data_dir,run,out,batch_size=2,max_new_tokens=2)
    first=(out/'predictions.jsonl').read_text().splitlines()
    assert len(first)==6
    (out/'predictions.jsonl').write_text(first[0]+'\n')
    runtime.evaluate(data_dir,run,out,batch_size=2,max_new_tokens=2)
    final=[json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()]
    assert len(final)==len({(r['id'],r['condition']) for r in final})==6
    assert json.loads((out/'report.json').read_text())['test']['actual']['total']==1
