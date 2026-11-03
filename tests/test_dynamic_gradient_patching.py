import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from behaviour_prediction.monitor.dynamic_gradient_patching import (
    step_context, sequence_capture, sequence_replacement_scores, dynamic_generate,
)
from behaviour_prediction.monitor.patching import PrefillLastTokenPatch, last_token_donors


def tiny(layers=2):
    torch.manual_seed(21)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=24,hidden_size=8,intermediate_size=16,
        num_hidden_layers=layers,num_attention_heads=2,num_key_value_heads=2,pad_token_id=0,eos_token_id=None))
    model=get_peft_model(model,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,task_type='CAUSAL_LM',
        target_modules=['q_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'lora_B' in name:p.normal_(0,.15)
    return model.eval()


def test_sequence_gradient_includes_later_positions_and_matches_finite_difference():
    model=tiny();encoded={'input_ids':torch.tensor([[1,4,7,8,9]])}
    with step_context(model,0):donor,donor_logits=sequence_capture(model,encoded,1,3)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    result=sequence_replacement_scores(model,encoded,1,3,donor,donor_logits)
    assert result['prediction_positions']==3
    assert result['gradient'][1:].abs().sum()>0
    torch.testing.assert_close(result['scores'],result['token_scores'].sum(0))
    module=model.get_base_model().model.layers[1]
    delta=result['donor']-result['recipient']
    def loss_at(scale):
        def patch(module,args,output):
            h=output.clone();h[0,2:]+=scale*delta.to(h);return h
        handle=module.register_forward_hook(patch)
        try:
            with torch.no_grad():
                logits=model(**encoded,use_cache=False).logits[0,2:].float()
                return torch.nn.functional.kl_div(logits.log_softmax(-1),donor_logits.float().log_softmax(-1),log_target=True,reduction='sum')/3
        finally:handle.remove()
    numerical=(loss_at(.001)-loss_at(-.001))/.002
    torch.testing.assert_close(result['scores'].sum(),-numerical,atol=1e-5,rtol=.02)
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())
    assert not module._forward_hooks


def test_live_patching_full_reproduces_donor_self_and_zero_are_noops():
    model=tiny();encoded={'input_ids':torch.tensor([[1,4,7]])}
    def run(recipient,donor,**kw):
        return dynamic_generate(model,encoded,recipient_step=recipient,donor_step=donor,layer=1,limit=4,**kw)
    baseline=run(0,20,patch=False);tuned=run(20,0,patch=False)
    assert run(20,0)==baseline
    assert run(0,20)==tuned
    assert run(20,20)==tuned
    assert run(0,0)==baseline
    assert run(20,0,mask=torch.zeros(8,dtype=torch.bool))==tuned
    assert run(20,0,mask=torch.ones(8,dtype=torch.bool))==baseline
    with step_context(model,0):
        expected=model.generate(**encoded,max_new_tokens=4,do_sample=False,use_cache=True,pad_token_id=0)[0,3:].tolist()
    assert baseline==expected


def test_donor_follows_recipient_tokens_with_separate_caches():
    model=tiny();encoded={'input_ids':torch.tensor([[1,4,7]])};seen=[]
    def inspect(module,args,kwargs):
        seen.append((kwargs['input_ids'].clone(),kwargs.get('past_key_values')))
    handle=model.register_forward_pre_hook(inspect,with_kwargs=True)
    try:
        tokens=dynamic_generate(model,encoded,recipient_step=20,donor_step=0,layer=1,limit=4,
            mask=torch.tensor([True,False]*4))
    finally:handle.remove()
    assert len(seen)==8
    for i in range(4):
        torch.testing.assert_close(seen[2*i][0],seen[2*i+1][0])
        if i:
            assert seen[2*i][0].item()==tokens[i-1]
            assert seen[2*i][1] is not seen[2*i+1][1]


def test_prefill_only_dynamic_control_matches_existing_hook_generation():
    model=tiny();encoded={'input_ids':torch.tensor([[1,4,7]])}
    mask=torch.tensor([True,False]*4)
    with step_context(model,0):donors=last_token_donors(model,encoded,[1])
    with PrefillLastTokenPatch(model,donors,3,{1:mask}),torch.no_grad():
        expected=model.generate(**encoded,max_new_tokens=4,do_sample=False,use_cache=True,pad_token_id=0)[0,3:].tolist()
    actual=dynamic_generate(model,encoded,recipient_step=20,donor_step=0,layer=1,limit=4,mask=mask,prefill_only=True)
    assert actual==expected


def test_dynamic_runner_resume_and_scorer(tmp_path,monkeypatch):
    import csv,hashlib,json,sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
    import run_dynamic_gradient_activation_patch as runner
    import score_dynamic_gradient_activation_patch as scorer
    from behaviour_prediction.monitor.gradient_patching import endpoint
    from run_activation_patch_pilot import generate
    model=tiny(layers=32)
    encoded={'input_ids':torch.tensor([[1,4,7]])}
    class Tokenizer:
        pad_token_id=0
        def decode(self,tokens,**kw):return ' '.join(map(str,tokens))
    tokenizer=Tokenizer()
    m=json.loads(Path('configs/activation_patch_gradient_dynamic.json').read_text())
    m.update(probes=m['probes'][:1],expected_donor_files={},max_new_tokens=2)
    probe=m['probes'][0];id=probe['id'];old=tmp_path/'previous';old.mkdir()
    captures={};previous=[]
    for step,d in [(0,'transfer'),(20,'restore')]:
        with step_context(model,step):
            captures[step]=endpoint(model,encoded,31)[0]
            tokens,text=generate(model,tokenizer,encoded,2)
        previous.append(dict(probe_id=id,direction=d,condition='unpatched',
            generated_token_ids=tokens,prompt_token_ids=[[1,4,7]],token_limit_reached=len(tokens)>=2))
    (old/'pilot.json').write_text('{}')
    (old/'generated.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in previous))
    torch.save(dict(baseline={id:captures[0]},finetuned={id:captures[20]},prompt_token_ids={id:[[1,4,7]]}),old/'endpoint_activations.pt')
    m['comparison_dir']=str(old)
    m['comparison_sha256']={name:hashlib.sha256((old/name).read_bytes()).hexdigest() for name in m['comparison_sha256']}
    m['pilot_id']=runner.digest(m);config=tmp_path/'config.json';config.write_text(json.dumps(m))
    monkeypatch.setattr(runner,'checkpoint_plan',lambda *a:(dict(mode='lora',model=m['expected_model_config']),[(0,None),(20,tmp_path)]))
    calls=[]
    def models(*args,**kwargs):
        calls.append(20);yield 20,model,tokenizer,None
    monkeypatch.setattr(runner,'checkpoint_models',models)
    monkeypatch.setattr(runner,'encode_probe',lambda *a,**kw:encoded)
    folder=tmp_path/'dynamic'
    runner.run(folder,config)
    rows=list(map(json.loads,(folder/'generated.jsonl').read_text().splitlines()))
    assert len(rows)==20 and calls==[20]
    assert len(list((folder/'reviews').glob('*.csv')))==20
    report=json.loads((folder/'run_report.json').read_text())
    assert report['complete'] and report['self_patch_exact_matches']==2 and report['full_dynamic_donor_matches']==2
    masks={(r['probe_id'],r['direction'],r['condition']):r['selected_indices']
        for r in json.loads((folder/'coordinate_selection.json').read_text())}
    for d in m['directions']:assert masks[(id,d,'gradient_25pct')]==masks[(id,d,'prefill_gradient_25pct')]
    runner.run(folder,config);assert calls==[20]
    (folder/'generated.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1])+'{"interrupted"')
    runner.run(folder,config);assert calls==[20,20]
    assert len((folder/'generated.jsonl').read_text().splitlines())==20
    assert (folder/'generated.interrupted-backup.jsonl').exists()
    labeled=folder/'labeling/labeled';labeled.mkdir(parents=True)
    judge={'sample':rows}
    judge['test_id']=hashlib.sha256(json.dumps(judge,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    (folder/'labeling/test.json').write_text(json.dumps(judge))
    for r in rows:
        label={k:r[k] for k in ('run','step','probe_id','group','category','prompt','response')}
        label.update(claude_harmful='0',claude_needs_review='False')
        with (labeled/f"{r['run']}.csv").open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(label));w.writeheader();w.writerow(label)
    scorer.score(folder)
    for name in ['dynamic_comparison.png','restore_presence.png','transfer_presence.png','report.json']:
        assert (folder/'results'/name).exists()
