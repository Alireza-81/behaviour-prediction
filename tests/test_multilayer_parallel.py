"""Exercise real subprocesses and interrupted journals without a GPU/model download."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_multilayer_activation_patch as runner
import multilayer_generation_workers as parallel
from behaviour_prediction.monitor.shared_mask import identity
from run_gradient_activation_patch import digest

FAKE_WORKER = r'''
import json,os,sys,time
from pathlib import Path
path=Path(sys.argv[1]);value=json.loads(path.read_text());m=value['manifest']
fail=int(sys.argv[2]);count=0
(path.parent/'pid.txt').write_text(str(os.getpid()))
with (path.parent/'generated.jsonl').open('w') as f:
    for task in value['tasks']:
        p=task['probe'];saved={(r['direction'],r['condition']) for r in task['saved']}
        pairs=[('transfer','unpatched'),('restore','unpatched')]
        pairs += [(d,c) for d in m['directions'] for c in m['conditions'] if c!='unpatched']
        for d,c in pairs:
            if (d,c) in saved:continue
            recipient=20 if d=='restore' else 0
            donor=recipient if c=='self_dynamic' else 20-recipient
            tokens=[donor+1 if c=='full_dynamic' else recipient+1]
            row=dict(pilot_id=m['pilot_id'],mask_id=m['mask_id'],source_run=m['run_name'],
                direction=d,condition=c,run=f"{m['run_name']}_multilayer_{d}__{c}",step=recipient,
                probe_id=p['id'],group=p['group'],category=p['category'],role=p['role'],prompt=p['prompt'],
                response=str(tokens),token_limit_reached=False,human_harmful='',review_notes='',
                prompt_token_ids=[[1,4,7]],generated_token_ids=tokens,
                selected_indices=value['selected'].get(c,{}))
            f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno());count+=1
            if fail and count>=fail:
                f.write('{"partial":');f.flush();os._exit(7)
            time.sleep(.001)
'''


def setup_case(tmp_path,monkeypatch,initial=457):
    import reuse_multilayer_activation_patch as reuse
    monkeypatch.setattr(reuse,'seed_answers',lambda *args:0)
    plan=dict(run_name='tiny',max_new_tokens=3,directions=['restore','transfer'],
              evaluation_probes=[dict(id=f'p{i}',prompt=f'Prompt {i}',group='harmful',category='hate_speech',role='target') for i in range(10)])
    plan['plan_id']=identity(plan,'plan_id')
    selected={f'mask{i}':{'31':[i]} for i in range(48)}
    artifact=dict(mask_id='frozen')
    (tmp_path/'mask.json').write_text(json.dumps(artifact))
    monkeypatch.setattr(runner,'validate_artifact',lambda *args:selected)
    monkeypatch.setattr(runner,'models',lambda *args:pytest.fail('Coordinator loaded a model'))
    m=dict(plan,probes=plan['evaluation_probes'],conditions=['unpatched','self_dynamic','full_dynamic',*selected],mask_id='frozen',layers=list(range(32)))
    m['pilot_id']=digest(m)
    # Use the fake worker to construct a valid serial journal, then retain the first 457.
    script=tmp_path/'fake_worker.py';script.write_text(FAKE_WORKER)
    prior=tmp_path/'prior';prior.mkdir()
    task=prior/'task.json';task.write_text(json.dumps(dict(manifest=m,selected=selected,tasks=[dict(probe=p,saved=[]) for p in m['probes']])))
    subprocess.run([sys.executable,str(script),str(task),'0'],check=True)
    all_lines=(prior/'generated.jsonl').read_bytes().splitlines(keepends=True)
    (tmp_path/'generated.jsonl').write_bytes(b''.join(all_lines[:initial]))
    original_parallel=parallel.parallel_generate
    def launch(fail=0):
        def execute(*args,**kwargs):
            return original_parallel(*args,**kwargs,
                command_factory=lambda path:[sys.executable,str(script),str(path),str(fail)],poll_interval=.01)
        monkeypatch.setattr(parallel,'parallel_generate',execute)
    launch()
    return plan,selected,all_lines,launch


def test_four_process_resume_from_457_preserves_rows_and_annotations(tmp_path,monkeypatch):
    plan,selected,lines,_=setup_case(tmp_path,monkeypatch)
    original=(tmp_path/'generated.jsonl').read_bytes()
    # Incomplete final serial write must not erase any of the 457 complete rows.
    (tmp_path/'generated.jsonl').write_bytes(original+b'{"incomplete":')
    runner.generate(tmp_path,plan,workers=4)
    assert (tmp_path/'generated.jsonl').read_bytes().startswith(original)
    rows=[json.loads(line) for line in (tmp_path/'generated.jsonl').read_text().splitlines()]
    assert len(rows)==1020
    assert len({(r['probe_id'],r['direction'],r['condition']) for r in rows})==1020
    assert (tmp_path/'generated.interrupted-backup.jsonl').exists()
    session=next((tmp_path/'workers').iterdir())
    pids=[p.read_text() for p in session.glob('worker-*/pid.txt')]
    assert len(set(pids))==4
    assignments=[json.loads(p.read_text()) for p in session.glob('worker-*/task.json')]
    ids=[t['probe']['id'] for job in assignments for t in job['tasks']]
    assert len(ids)==len(set(ids))
    worker_rows=[json.loads(line) for p in session.glob('worker-*/generated.jsonl') for line in p.read_text().splitlines()]
    assert len(worker_rows)==1020-457
    assert not {(r['probe_id'],r['direction'],r['condition']) for r in worker_rows}&{
        (r['probe_id'],r['direction'],r['condition']) for r in map(json.loads,original.splitlines())}
    before=(tmp_path/'generated.jsonl').read_bytes()
    runner.generate(tmp_path,plan,workers=4)
    assert (tmp_path/'generated.jsonl').read_bytes()==before
    assert len(list((tmp_path/'workers').iterdir()))==1
    report=json.loads((tmp_path/'run_report.json').read_text())
    assert report['complete'] and report['gradients_used'] is False
    # No answers disappear if an interruption occurred after a private durable write
    # but before the coordinator copied it to the main journal.
    (tmp_path/'generated.jsonl').write_bytes(original)
    runner.generate(tmp_path,plan,workers=2)
    assert len((tmp_path/'generated.jsonl').read_text().splitlines())==1020
    assert len(list((tmp_path/'workers').iterdir()))==1


def test_worker_failure_preserves_progress_and_can_resume_with_fewer_workers(tmp_path,monkeypatch):
    plan,_,_,launch=setup_case(tmp_path,monkeypatch,initial=457)
    launch(fail=2)
    with pytest.raises(RuntimeError,match='worker failure'):runner.generate(tmp_path,plan,workers=4)
    rows=[json.loads(line) for line in (tmp_path/'generated.jsonl').read_text().splitlines()]
    assert len(rows)>=459
    before={ (r['probe_id'],r['direction'],r['condition']):r for r in rows }
    launch(fail=0)
    runner.generate(tmp_path,plan,workers=2)
    after={ (r['probe_id'],r['direction'],r['condition']):r for r in map(json.loads,(tmp_path/'generated.jsonl').read_text().splitlines()) }
    assert len(after)==1020
    assert all(after[k]==r for k,r in before.items())


def test_worker_tail_partial_and_conflicting_duplicate_are_rejected(tmp_path):
    path=tmp_path/'journal';path.write_bytes(b'{"a":')
    tail=parallel.JournalTail(path)
    assert list(tail.rows())==[]
    with path.open('ab') as f:f.write(b'1}\n{"b":2}\n{"unfinished":')
    assert list(tail.rows())==[{'a':1},{'b':2}]
    assert list(tail.rows(final=True))==[] and tail.discarded_tail


def test_folder_lock_rejects_second_coordinator_and_is_inherited(tmp_path):
    with parallel.generation_lock(tmp_path) as fd:
        code="import fcntl,sys; f=open(sys.argv[1],'a+'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
        failed=subprocess.run([sys.executable,'-c',code,str(tmp_path/'.generation.lock')],capture_output=True)
        assert failed.returncode!=0 and b'BlockingIOError' in failed.stderr
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(1)'],pass_fds=(fd,))
    try:
        with pytest.raises(ValueError,match='still owns'): 
            with parallel.generation_lock(tmp_path):pass
    finally:child.wait()
    with parallel.generation_lock(tmp_path):pass


def test_generation_has_no_autograd_or_weight_updates(tmp_path,monkeypatch):
    from test_dynamic_gradient_patching import tiny
    model=tiny();before={n:p.detach().clone() for n,p in model.named_parameters()}
    probe=dict(id='p',prompt='Prompt',group='harmful',category='hate_speech',role='target')
    m=dict(run_name='tiny',pilot_id='test',mask_id='frozen',max_new_tokens=2,
           directions=['restore','transfer'],conditions=['unpatched','self_dynamic','full_dynamic','chosen'])
    monkeypatch.setattr(runner,'encode',lambda *args:{'input_ids':torch.tensor([[1,4,7]])})
    def generate(model,encoded,*,recipient_step,donor_step,limit,masks=None):
        assert not torch.is_grad_enabled()
        return [donor_step+1 if masks and len(masks)==32 else recipient_step+1]
    monkeypatch.setattr(runner,'multilayer_generate',generate)
    monkeypatch.setattr(torch.autograd,'grad',lambda *args,**kwargs:pytest.fail('Generation took a gradient'))
    rows=list(runner.generation_rows(m,{'chosen':{'1':[0]}},[dict(probe=probe,saved=[])],model,
        type('Tokenizer',(),{'decode':lambda self,tokens,**kw:str(tokens)})()))
    assert len(rows)==8
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())
    assert all(p.grad is None for p in model.parameters())


def test_ctrl_c_stops_workers_and_resumes_without_losing_completed_answers(tmp_path,monkeypatch):
    import os
    plan,_,_,_=setup_case(tmp_path,monkeypatch,initial=457)
    original_validate=runner.validate_generation_row; interrupted=False
    def interrupt_once(row,m,selected,records,**kwargs):
        nonlocal interrupted
        valid=original_validate(row,m,selected,records,**kwargs)
        if valid and len(records)>=460 and not interrupted:
            interrupted=True
            raise KeyboardInterrupt
        return valid
    monkeypatch.setattr(runner,'validate_generation_row',interrupt_once)
    with pytest.raises(KeyboardInterrupt):runner.generate(tmp_path,plan,workers=4)
    assert interrupted
    session=next((tmp_path/'workers').iterdir())
    for path in session.glob('worker-*/pid.txt'):
        with pytest.raises(ProcessLookupError):os.kill(int(path.read_text()),0)
    before={ (r['probe_id'],r['direction'],r['condition']):r for r in map(json.loads,(tmp_path/'generated.jsonl').read_text().splitlines()) }
    monkeypatch.setattr(runner,'validate_generation_row',original_validate)
    runner.generate(tmp_path,plan,workers=4)
    after={ (r['probe_id'],r['direction'],r['condition']):r for r in map(json.loads,(tmp_path/'generated.jsonl').read_text().splitlines()) }
    assert len(after)==1020 and all(after[k]==r for k,r in before.items())


def test_live_model_inference_has_no_gradient_and_leaves_hooks_clean(monkeypatch):
    from test_dynamic_gradient_patching import tiny
    from behaviour_prediction.monitor.multilayer_patching import multilayer_generate
    model=tiny(layers=3);seen=[]
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    hook=model.register_forward_pre_hook(lambda *args:seen.append(torch.is_grad_enabled()))
    monkeypatch.setattr(torch.autograd,'grad',lambda *args,**kwargs:pytest.fail('Unexpected autograd'))
    try:
        answer=multilayer_generate(model,{'input_ids':torch.tensor([[1,4,7]])},recipient_step=20,donor_step=0,
            limit=3,masks={i:torch.tensor([True,False]*4) for i in range(3)})
    finally:hook.remove()
    assert answer and seen and not any(seen)
    assert all(torch.equal(p,before[n]) and p.grad is None for n,p in model.named_parameters())
    assert all(not layer._forward_hooks for layer in model.get_base_model().model.layers)


def test_real_worker_entrypoint_freezes_parameters_and_writes_private_journal(tmp_path,monkeypatch):
    import os,signal,hashlib
    from test_dynamic_gradient_patching import tiny
    model=tiny();before={n:p.detach().clone() for n,p in model.named_parameters()}
    probe=dict(id='p',prompt='Prompt',group='harmful',category='hate_speech',role='target')
    m=dict(run_name='tiny',mask_id='frozen',max_new_tokens=3,
        directions=['restore','transfer'],conditions=['unpatched','self_dynamic','full_dynamic','chosen'])
    m['pilot_id']=digest(m)
    (tmp_path/'mask.json').write_text('{}')
    directory=tmp_path/'workers'/'session'/'worker-0';directory.mkdir(parents=True)
    value=dict(manifest=m,selected={'chosen':{'1':[0]}},tasks=[dict(probe=probe,saved=[])],
        parent_pid=os.getppid(),mask_sha256=hashlib.sha256(b'{}').hexdigest())
    value['task_id']=identity(value,'task_id');path=directory/'task.json';path.write_text(json.dumps(value))
    monkeypatch.setattr(runner,'models',lambda *args:iter([(model,type('Tokenizer',(),{'decode':lambda self,tokens,**kw:str(tokens)})())]))
    monkeypatch.setattr(runner,'encode',lambda *args:{'input_ids':torch.tensor([[1,4,7]])})
    def generate(model,encoded,*,recipient_step,donor_step,limit,masks=None):
        assert not torch.is_grad_enabled() and not model.training
        assert all(not p.requires_grad for p in model.parameters())
        return [donor_step+1 if masks and len(masks)==32 else recipient_step+1]
    monkeypatch.setattr(runner,'multilayer_generate',generate)
    monkeypatch.setattr(torch.autograd,'grad',lambda *args,**kwargs:pytest.fail('Worker took gradients'))
    previous_signal=signal.getsignal(signal.SIGINT);previous_grad=torch.is_grad_enabled();previous_threads=torch.get_num_threads()
    try:parallel.worker(path)
    finally:
        signal.signal(signal.SIGINT,previous_signal);torch.set_grad_enabled(previous_grad);torch.set_num_threads(previous_threads)
    rows=[json.loads(line) for line in (directory/'generated.jsonl').read_text().splitlines()]
    assert len(rows)==8 and not (tmp_path/'generated.jsonl').exists()
    assert all(torch.equal(p,before[n]) and p.grad is None for n,p in model.named_parameters())
