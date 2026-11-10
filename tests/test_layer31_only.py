import copy
import json
import math
import types

import torch
from peft import LoraConfig,get_peft_model
from transformers import LlamaConfig,LlamaForCausalLM

from behaviour_prediction.explainer import binary,injection,layer_ablation
from behaviour_prediction.explainer.data import file_hash


def test_only_layer15_vectors_are_removed():
    data=dict(vectors=torch.randn(3,4,16),slot_rms=torch.tensor([2.,3.,4.,5.]))
    original=data['vectors'].clone()
    masked=layer_ablation.layer31_bundle(data)
    assert torch.equal(data['vectors'],original)
    assert masked['vectors'][:,:2].count_nonzero()==0
    assert torch.equal(masked['vectors'][:,2:],original[:,2:])
    assert masked['slot_rms'] is data['slot_rms']


class Tokenizer:
    pad_token_id=0
    eos_token_id=2
    def apply_chat_template(self,messages,**kw):return '\n'.join(m['content'] for m in messages)+'\nAssistant:\n'
    def encode(self,text,**kw):return [ord(c)%100+3 for c in text]
    def __call__(self,text,**kw):return dict(input_ids=self.encode(text),offset_mapping=[(i,i+1) for i in range(len(text))])


def test_frozen_adapter_cache_reuse_mask_and_resume(tmp_path,monkeypatch):
    data_dir=tmp_path/'data';data_dir.mkdir()
    examples=[dict(id=str(i),vector_index=i,split=s,run='r',step=20,probe_id=f'p{i}',harmful=i%2)
              for i,s in enumerate(['validation','validation','test','test','baseline_test'])]
    (data_dir/'examples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in examples))
    vectors=torch.randn(5,4,16)
    rms=torch.tensor([2.,3.,4.,5.])
    torch.save(dict(vectors=vectors,slot_rms=rms),data_dir/'activations.pt')
    (data_dir/'manifest.json').write_text(json.dumps(dict(hidden_size=16,artifact_sha256={n:file_hash(data_dir/n) for n in ['examples.jsonl','activations.pt']})))
    model=get_peft_model(LlamaForCausalLM(LlamaConfig(vocab_size=128,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,pad_token_id=0)),
        LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    run=tmp_path/'run';run.mkdir();model.save_pretrained(run/'adapter')
    (run/'completed.json').write_text('{}')
    cfg=dict(overfit=False,dataset_manifest_sha256=file_hash(data_dir/'manifest.json'),
             binary_source_sha256=file_hash(binary.__file__),injection_source_sha256=file_hash(injection.__file__))
    (run/'training-config.json').write_text(json.dumps(cfg))
    adapter={p.name:file_hash(p) for p in (run/'adapter').glob('adapter*') if p.is_file()}
    cache=tmp_path/'cache';cache.mkdir()
    (cache/'evaluation-config.json').write_text(json.dumps(dict(adapter_sha256=adapter,
        dataset_manifest_sha256=cfg['dataset_manifest_sha256'],batch_size=4)))
    cached=[]
    for c,score in [('actual',.6),('zero',.1)]:
        for row in examples:
            cached.append(dict(id=row['id'],split=row['split'],run='r',step=20,probe_id=row['probe_id'],label=row['harmful'],
                 condition=c,score=score,predicted=int(score>=.5),binary_nll=-math.log(score if row['harmful'] else 1-score),
                 full_nll=1.,greedy_label=int(score>=.5)))
    (cache/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cached))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(layer_ablation,'load_model_for_finetuning',lambda _:types.SimpleNamespace(model=copy.deepcopy(model),tokenizer=Tokenizer()))
    observed=[]
    actual_infer=layer_ablation.infer
    def spy(m,h,t,rows,masked,*args,**kw):
        assert masked['vectors'][:,:2].count_nonzero()==0
        assert torch.equal(masked['vectors'][:,2:],vectors[:,2:])
        assert torch.equal(masked['slot_rms'],rms)
        observed.extend(r['id'] for r in rows)
        return actual_infer(m,h,t,rows,masked,*args,**kw)
    monkeypatch.setattr(layer_ablation,'infer',spy)
    out=tmp_path/'evaluation'
    layer_ablation.evaluate(data_dir,run,cache,out)
    results=[json.loads(l) for l in (out/'predictions.jsonl').read_text().splitlines()]
    assert len(results)==15 and set(observed)=={r['id'] for r in examples}
    assert all(r['score']==.6 for r in results if r['condition']=='both_layers')
    observed.clear()
    layer_ablation.evaluate(data_dir,run,cache,out)
    assert not observed
    assert len((out/'predictions.jsonl').read_text().splitlines())==15
