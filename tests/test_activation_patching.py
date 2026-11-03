"""Verify real interventions and cache scope using a tiny local Llama, no downloads."""
import pytest
import torch
from transformers import LlamaConfig,LlamaForCausalLM
from behaviour_prediction.monitor.patching import PrefillLastTokenPatch,last_token_donors


def model():
    torch.manual_seed(12)
    return LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=24,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,pad_token_id=0,bos_token_id=1,eos_token_id=None)).eval()


def test_final_block_patch_restores_donor_prefill_logits_and_removes_hooks():
    base=model();recipient=model()
    with torch.no_grad():recipient.model.layers[1].mlp.down_proj.weight.add_(.3)
    encoded=dict(input_ids=torch.tensor([[1,4,6]]),attention_mask=torch.ones(1,3,dtype=torch.long))
    states=last_token_donors(base,encoded,[1])
    with torch.no_grad():expected=base(**encoded).logits[:,-1]
    count=len(recipient.model.layers[1]._forward_hooks)
    with PrefillLastTokenPatch(recipient,states,3) as patch:
        with torch.no_grad():actual=recipient(**encoded).logits[:,-1]
    assert patch.applied=={1}
    torch.testing.assert_close(actual,expected)
    assert len(recipient.model.layers[1]._forward_hooks)==count


def test_self_patch_is_noop_during_cached_generation_and_only_prefill_is_patched():
    m=model();encoded=dict(input_ids=torch.tensor([[1,4,6]]),attention_mask=torch.ones(1,3,dtype=torch.long))
    states=last_token_donors(m,encoded,[0,1])
    with torch.no_grad():plain=m.generate(**encoded,max_new_tokens=4,do_sample=False,use_cache=True,pad_token_id=0)
    with PrefillLastTokenPatch(m,states,3) as patch:
        with torch.no_grad():patched=m.generate(**encoded,max_new_tokens=4,do_sample=False,use_cache=True,pad_token_id=0)
    assert torch.equal(plain,patched) and patch.applied=={0,1}


def test_patch_cleanup_on_exception_and_bad_shapes():
    m=model()
    with pytest.raises(ValueError,match='width mismatch'):
        with PrefillLastTokenPatch(m,{0:torch.zeros(3)},3):
            m(input_ids=torch.tensor([[1,2,3]]))
    assert not m.model.layers[0]._forward_hooks
    with pytest.raises(ValueError,match='Invalid patch layer'):
        PrefillLastTokenPatch(m,{3:torch.zeros(16)},3)
