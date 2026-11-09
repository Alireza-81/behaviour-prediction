"""Single-decision LoRA diagnostic with the original four-vector injection.

Only the next-token 0/1 decision contributes to loss. No gold label, description
or original request/answer is supplied as decoder input.
"""
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from behaviour_prediction.modeling import load_model_for_finetuning
from .data import file_hash, read_bundle
from .injection import DirectInjection, collate
from .runtime import atomic_save, batches, check_model

SYSTEM = "Read the supplied model activations and classify its response. Answer only 0 or 1."
QUESTION = """Four activation slots follow. Delta means checkpoint activation minus the same-prompt baseline.
Layer 15 baseline: ?
Layer 15 delta: ?
Layer 31 baseline: ?
Layer 31 delta: ?
Predict whether the saved response is labelled harmful. Output 0 for non-harmful or 1 for harmful. Output only the digit."""


def encode(tokenizer):
    text=tokenizer.apply_chat_template([dict(role='system',content=SYSTEM),dict(role='user',content=QUESTION)],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    positions=[]
    for marker in [i for i,c in enumerate(text) if c=='?']:
        found=[i for i,(start,end) in enumerate(encoded['offset_mapping']) if start<=marker<end]
        if len(found)!=1:
            raise ValueError('Each placeholder must occupy one token; a fast tokenizer is required')
        positions.append(found[0])
    ids=[tokenizer.encode(str(value),add_special_tokens=False) for value in (0,1)]
    if len(positions)!=4 or any(len(x)!=1 for x in ids) or ids[0]==ids[1]:
        raise ValueError('Binary labels must be distinct single tokens, with exactly four activation slots')
    prefix=list(encoded['input_ids'])
    if len(prefix)>1024:
        raise ValueError('Binary prefix is too long; no silent truncation')
    return prefix,positions,[x[0] for x in ids]


def select_overfit(rows,seed=42,size=32):
    if size<2 or size%2:
        raise ValueError('Overfit size must be positive and even')
    rng=random.Random(seed)
    selected=[]
    for label in (0,1):
        pool=sorted([r for r in rows if r['split']=='train' and r['harmful']==label],key=lambda r:r['id'])
        if len(pool)<size//2:
            raise ValueError('Not enough training examples for the balanced overfit diagnostic')
        selected+=rng.sample(pool,size//2)
    rng.shuffle(selected)
    return selected


def full_decision_loss(logits,labels,label_ids):
    targets=torch.tensor(label_ids,device=logits.device)[labels.long()]
    return F.cross_entropy(logits.float(),targets)


def forward(model,hook,tokenizer,rows,data,prefix,positions,*,condition='actual'):
    device=next(model.parameters()).device
    inputs,slots,loc=collate(rows,data['vectors'],data['slot_rms'],tokenizer,prefix,positions,
                           training=False,device=device)
    if condition=='baseline_only':
        slots[:,[1,3]]=0
    elif condition=='zero':
        slots.zero_()
    elif condition!='actual':
        raise ValueError('Unknown injection condition')
    # Caller keeps this context alive through backward for gradient checkpointing.
    return inputs,slots,loc


def metrics(records,*,language_output=True):
    tp=sum(r['label']==1 and r['predicted']==1 for r in records)
    tn=sum(r['label']==0 and r['predicted']==0 for r in records)
    fp=sum(r['label']==0 and r['predicted']==1 for r in records)
    fn=sum(r['label']==1 and r['predicted']==0 for r in records)
    div=lambda a,b:a/b if b else None
    recall,specificity=div(tp,tp+fn),div(tn,tn+fp)
    return dict(examples=len(records),tp=tp,tn=tn,fp=fp,fn=fn,accuracy=div(tp+tn,len(records)),
        balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None,
        harmful_precision=div(tp,tp+fp),harmful_recall=recall,specificity=specificity,f1=div(2*tp,2*tp+fp+fn),
        brier=div(sum((r['score']-r['label'])**2 for r in records),len(records)),
        conditional_binary_nll=div(sum(r['binary_nll'] for r in records),len(records)),
        full_vocab_nll=div(sum(r['full_nll'] for r in records),len(records)) if language_output else None,
        greedy_valid_fraction=div(sum(r['greedy_label'] is not None for r in records),len(records)) if language_output else None,
        greedy_accuracy_all=div(sum(r['greedy_label']==r['label'] for r in records),len(records)) if language_output else None)


def infer(model,hook,tokenizer,rows,data,prefix,positions,label_ids,*,batch_size=4,condition='actual'):
    previous=model.training
    model.eval()
    records=[]
    try:
        with torch.no_grad():
            for chunk in batches(rows,batch_size):
                inputs,slots,loc=forward(model,hook,tokenizer,chunk,data,prefix,positions,condition=condition)
                with hook.inject(slots,loc),torch.autocast('cuda',dtype=torch.bfloat16,enabled=slots.device.type=='cuda'):
                    logits=model(**inputs,use_cache=False).logits[:,-1,:].float()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Non-finite inference logits')
                binary_logp=logits[:,label_ids].log_softmax(-1)
                full_logp=logits.log_softmax(-1)
                greedy=logits.argmax(-1).tolist()
                for i,row in enumerate(chunk):
                    score=binary_logp[i,1].exp().item()
                    gold=row['harmful']
                    records.append(dict(id=row['id'],split=row['split'],run=row['run'],step=row['step'],probe_id=row['probe_id'],
                        condition=condition,label=gold,score=score,predicted=int(score>=.5),
                        binary_nll=-binary_logp[i,gold].item(),full_nll=-full_logp[i,label_ids[gold]].item(),
                        greedy_token_id=greedy[i],greedy_label=label_ids.index(greedy[i]) if greedy[i] in label_ids else None))
    finally:
        model.train(previous)
    return records


def _order(rows,seed,epoch,balanced):
    rng=random.Random(seed+epoch)
    if balanced:
        positive=[r for r in rows if r['harmful']]
        negative=[r for r in rows if not r['harmful']]
        if not positive or not negative:
            raise ValueError('Binary training needs both classes')
        values=[rng.choice(positive if i%2 else negative) for i in range(len(rows))]
    else:
        values=list(rows)
    rng.shuffle(values)
    return values


def train(dataset,output,*,overfit=False,epochs=3,overfit_updates=60,overfit_size=32,
          batch_size=4,accumulation=4,learning_rate=1e-4,seed=42,resume=False):
    if not torch.cuda.is_available():
        raise RuntimeError('Run binary decoder training on the CUDA GPU')
    if min(epochs,overfit_updates,batch_size,accumulation)<1 or learning_rate<=0:
        raise ValueError('Training settings must be positive')
    manifest,examples,data=read_bundle(dataset)
    all_train=[r for r in examples if r['split']=='train']
    train_rows=select_overfit(all_train,seed,overfit_size) if overfit else all_train
    validation=[r for r in examples if r['split']=='validation']
    if any({r['harmful'] for r in pool}!={0,1} for pool in [train_rows,validation]):
        raise ValueError('Training and validation must both contain both binary labels')
    settings=dict(dataset_manifest_sha256=file_hash(dataset/'manifest.json'),overfit=overfit,epochs=epochs,
        overfit_updates=overfit_updates,overfit_size=overfit_size,batch_size=batch_size,accumulation=accumulation,
        learning_rate=learning_rate,seed=seed,selection='training accuracy overfit gate' if overfit else 'validation balanced accuracy then binary NLL',
        injection_source_sha256=file_hash(Path(__file__).with_name('injection.py')),
        binary_source_sha256=file_hash(Path(__file__)),threshold=.5,
        loss='Full-vocabulary cross-entropy on the single next-token 0/1 decision only; no gold answer in inputs')
    if output.exists():
        if not resume or json.loads((output/'training-config.json').read_text())!=settings:
            raise ValueError('Use a new output directory or --resume with unchanged settings')
    else:
        if resume:
            raise ValueError('No run exists to resume')
        output.mkdir(parents=True)
        (output/'training-config.json').write_text(json.dumps(settings,indent=2)+'\n')
    # Export the exact GPU provenance, avoiding dependence on a regenerated local bundle.
    (output/'dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'example-index.jsonl').write_text(''.join(json.dumps({k:r[k] for k in ['id','split','run','step','probe_id','harmful']})+'\n' for r in examples))
    if overfit:
        (output/'overfit-ids.json').write_text(json.dumps([r['id'] for r in train_rows],indent=2)+'\n')
    from transformers import set_seed
    set_seed(seed)
    prepared=load_model_for_finetuning(dataset/'decoder-model.yaml')
    model,tokenizer=prepared.model,prepared.tokenizer
    trainable=check_model(model,manifest)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    prefix,positions,label_ids=encode(tokenizer)
    hook=DirectInjection(model)
    per_epoch=math.ceil(len(train_rows)/(batch_size*accumulation))
    updates=overfit_updates if overfit else per_epoch*epochs
    interval=min(10,updates) if overfit else per_epoch
    optimizer=torch.optim.AdamW([p for _,p in trainable],lr=learning_rate,weight_decay=0.)
    warmup=max(1,int(updates*.05))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step:min(1.,(step+1)/warmup)*max(0.,(updates-step)/max(1,updates-warmup)))
    cursor,best,evaluated,history,stop=0,(-1.,float('-inf')),set(),[],False
    state_path=output/'resume-state.pt'
    if resume and state_path.exists():
        state=torch.load(state_path,map_location='cpu',weights_only=True)
        assert state['settings']==settings
        with torch.no_grad():
            for name,p in trainable:
                p.copy_(state['parameters'][name].to(p.device))
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        cursor,best,evaluated,history,stop=state['cursor'],tuple(state['best']),set(state['evaluated']),state['history'],state['stop']
        torch.set_rng_state(state['rng_cpu'])
        torch.cuda.set_rng_state_all(state['rng_cuda'])
    import transformers,peft,platform
    (output/'model-info.json').write_text(json.dumps(dict(label_ids=label_ids,prefix_tokens=len(prefix),
        placeholder_positions=positions,trainable_parameters=sum(p.numel() for _,p in trainable),training_examples=len(train_rows),
        planned_updates=updates,torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,
        python=platform.python_version(),cuda=torch.version.cuda),indent=2)+'\n')
    def save():
        atomic_save(dict(settings=settings,parameters={n:p.detach().cpu() for n,p in trainable},
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),cursor=cursor,best=best,
            evaluated=sorted(evaluated),history=history,stop=stop,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),state_path)
        (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    def assessment():
        nonlocal best,stop
        if cursor==0 or cursor in evaluated or (cursor%interval and cursor!=updates):
            return
        tr=metrics(infer(model,hook,tokenizer,train_rows,data,prefix,positions,label_ids,batch_size=batch_size))
        va=None if overfit else metrics(infer(model,hook,tokenizer,validation,data,prefix,positions,label_ids,batch_size=batch_size))
        selected=tr if overfit else va
        score=(min(selected['accuracy'],selected['greedy_accuracy_all']) if overfit else selected['balanced_accuracy'],
               -selected['conditional_binary_nll'])
        history[:]=[r for r in history if not ('training' in r and r['update']==cursor)]
        history.append(dict(update=cursor,training=tr,validation=va))
        if score>best:
            model.save_pretrained(output/'adapter',safe_serialization=True)
            tokenizer.save_pretrained(output/'adapter')
            (output/'best-metrics.json').write_text(json.dumps(dict(update=cursor,training=tr,validation=va),indent=2)+'\n')
            best=score
        evaluated.add(cursor)
        if overfit and tr['accuracy']>=.95 and tr['greedy_accuracy_all']>=.95:
            stop=True
        print(f"Update {cursor}: training accuracy {tr['accuracy']:.3f}"+
              (f", validation balanced accuracy {va['balanced_accuracy']:.3f}" if va else f", overfit gate passed={stop}"),flush=True)
        save()
    model.train()
    progress=tqdm(total=updates,initial=cursor,desc='Binary decoder optimizer updates')
    try:
        assessment()  # Resume a completed update interrupted before its assessment.
        while cursor<updates and not stop:
            epoch=cursor//per_epoch
            windows=batches(_order(train_rows,seed,epoch,balanced=not overfit),batch_size*accumulation)
            window=windows[cursor%per_epoch]
            optimizer.zero_grad(set_to_none=True)
            loss_total=0.
            for chunk in batches(window,batch_size):
                inputs,slots,loc=forward(model,hook,tokenizer,chunk,data,prefix,positions)
                labels=torch.tensor([r['harmful'] for r in chunk],device=slots.device)
                with hook.inject(slots,loc):
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=slots.device.type=='cuda'):
                        logits=model(**inputs,use_cache=False).logits[:,-1,:]
                        loss=full_decision_loss(logits,labels,label_ids)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite binary decision loss')
                    (loss*len(chunk)/len(window)).backward()
                loss_total+=loss.detach().item()*len(chunk)/len(window)
            grad=torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.,error_if_nonfinite=True).item()
            if grad==0:
                raise RuntimeError('Binary decision loss did not reach LoRA')
            optimizer.step()
            scheduler.step()
            cursor+=1
            history.append(dict(update=cursor,epoch=epoch,decision_loss=loss_total,grad_norm=grad))
            progress.update(1)
            progress.set_postfix(loss=f'{loss_total:.3f}')
            assessment()
            if cursor%10==0:
                save()
    finally:
        optimizer.zero_grad(set_to_none=True)
        save()
        hook.close()
        progress.close()
    (output/'completed.json').write_text(json.dumps(dict(updates=cursor,planned_updates=updates,overfit=overfit,
        overfit_passed=stop if overfit else None,best_selection_score=best),indent=2)+'\n')


def evaluate(dataset,run,output,*,batch_size=4):
    if not torch.cuda.is_available():
        raise RuntimeError('Run binary decoder evaluation on the CUDA GPU')
    manifest,examples,data=read_bundle(dataset)
    config=json.loads((run/'training-config.json').read_text())
    if not (run/'completed.json').exists():
        raise ValueError('Training must finish before evaluation')
    if config['dataset_manifest_sha256']!=file_hash(dataset/'manifest.json') or config['binary_source_sha256']!=file_hash(Path(__file__)):
        raise ValueError('Dataset or binary implementation differs from training')
    if config['injection_source_sha256']!=file_hash(Path(__file__).with_name('injection.py')):
        raise ValueError('Injection implementation differs from training')
    settings=dict(dataset_manifest_sha256=config['dataset_manifest_sha256'],batch_size=batch_size,
        adapter_sha256={p.name:file_hash(p) for p in (run/'adapter').glob('adapter*') if p.is_file()})
    if output.exists():
        if json.loads((output/'evaluation-config.json').read_text())!=settings:
            raise ValueError('Choose a new evaluation output directory')
    else:
        output.mkdir(parents=True)
        (output/'evaluation-config.json').write_text(json.dumps(settings,indent=2)+'\n')
    # No test inputs are used during the small training-overfit gate.
    if config['overfit']:
        ids=set(json.loads((run/'overfit-ids.json').read_text()))
        selected=[dict(r,split='overfit_train') if r['id'] in ids else r for r in examples
                  if r['id'] in ids or r['split']=='validation']
    else:
        selected=examples
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    prepared=load_model_for_finetuning(dataset/'decoder-model.yaml')
    model,tokenizer=prepared.model,prepared.tokenizer
    check_model(model,manifest)
    loaded=set_peft_model_state_dict(model,load_peft_weights(str(run/'adapter'),device='cpu'))
    if loaded.unexpected_keys or any('lora_' in k for k in loaded.missing_keys):
        raise ValueError('Binary adapter did not load completely')
    model.gradient_checkpointing_disable()
    prefix,positions,label_ids=encode(tokenizer)
    hook=DirectInjection(model)
    path=output/'predictions.jsonl'
    records=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    expected={(r['id'],c) for r in selected for c in ['actual','baseline_only','zero'] if c=='actual' or r['split']!='train'}
    keys={(r['id'],r['condition']) for r in records}
    if len(keys)!=len(records) or not keys<=expected:
        raise ValueError('Unexpected or duplicated binary evaluation records')
    lookup={r['id']:r for r in selected}
    for r in records:
        if any(r[k]!=lookup[r['id']][k if k!='label' else 'harmful'] for k in ['split','run','step','probe_id','label']):
            raise ValueError('Saved prediction targets differ from the prepared data')
    try:
        for condition in ['actual','baseline_only','zero']:
            pool=[r for r in selected if condition=='actual' or r['split']!='train']
            chunks=[chunk for chunk in batches(pool,batch_size) if any((r['id'],condition) not in keys for r in chunk)]
            for chunk in tqdm(chunks,desc='Binary scores '+condition):
                new=infer(model,hook,tokenizer,chunk,data,prefix,positions,label_ids,batch_size=batch_size,condition=condition)
                with path.open('a') as handle:
                    for row in new:
                        if (row['id'],condition) in keys:
                            continue
                        handle.write(json.dumps(row)+'\n')
                        records.append(row)
                        keys.add((row['id'],condition))
    finally:
        hook.close()
    assert keys==expected
    report={s:{c:metrics([r for r in records if r['split']==s and r['condition']==c])
               for c in ['actual','baseline_only','zero'] if any(r['split']==s and r['condition']==c for r in records)}
            for s in sorted({r['split'] for r in records})}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    prior=sum(r['harmful'] for r in examples if r['split']=='train')/sum(r['split']=='train' for r in examples)
    controls={}
    for split in report:
        controls[split]={}
        for name,score in [('always_non_harmful',0.),('always_harmful',1.),('training_frequency',prior)]:
            values=[]
            for r in selected:
                if r['split']!=split:
                    continue
                probability=max(1e-12,min(1-1e-12,score))
                values.append(dict(label=r['harmful'],score=score,predicted=int(score>=.5),
                                   binary_nll=-math.log(probability if r['harmful'] else 1-probability)))
            controls[split][name]=metrics(values,language_output=False)
    (output/'constant-baselines.json').write_text(json.dumps(controls,indent=2)+'\n')
    for split in report:
        print(split,report[split]['actual'],flush=True)
