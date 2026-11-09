"""GPU training and held-out generation for the direct-injection experiment."""
import json
import math
import random
from pathlib import Path

import torch
from tqdm import tqdm

from behaviour_prediction.modeling import load_model_for_finetuning
from .data import TAGS, file_hash, read_bundle
from .injection import DirectInjection, collate, encode_prefix


def atomic_save(value,path):
    temporary = path.with_suffix(".tmp")
    torch.save(value,temporary)
    temporary.replace(path)


def batches(rows,size):
    return [rows[i:i+size] for i in range(0,len(rows),size)]


def check_model(model,manifest):
    if model.get_input_embeddings().weight.shape[1] != manifest["hidden_size"]:
        raise ValueError("Direct injection requires the same target/decoder hidden width")
    trainable = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name,p in trainable):
        raise ValueError("Only decoder LoRA parameters may be trained")
    return trainable


def loss_batch(model,injector,tokenizer,rows,data,prefix,positions,*,backward=False,divisor=1):
    device = next(model.parameters()).device
    batch,slots,loc = collate(rows,data["vectors"],data["slot_rms"],tokenizer,prefix,positions,
                             training=True,device=device)
    if batch["input_ids"].shape[1] > 1024:
        raise ValueError("Example exceeds context limit; no silent truncation")
    with injector.inject(slots,loc):
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            loss = model(**batch,use_cache=False).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite decoder loss")
        if backward:
            # Keep the hook active during checkpointed backward recomputation.
            (loss/divisor).backward()
    return loss.detach().item()


def validation_loss(model,injector,tokenizer,rows,data,prefix,positions,batch_size):
    model.eval()
    total,tokens = 0.,0
    with torch.no_grad():
        for chunk in batches(rows,batch_size):
            # Weight validation by target token count, not arbitrary last-batch size.
            count = sum(len(tokenizer.encode(json.dumps(r["target"],ensure_ascii=False,separators=(",",":")),
                                             add_special_tokens=False))+1 for r in chunk)
            total += loss_batch(model,injector,tokenizer,chunk,data,prefix,positions)*count
            tokens += count
    model.train()
    return total/tokens


def train(dataset,output,*,epochs=3,batch_size=4,accumulation=4,learning_rate=1e-4,seed=42,
          resume=False,smoke=False,balanced=True):
    if not torch.cuda.is_available():
        raise RuntimeError("Run decoder training on the CUDA GPU; preparation and baselines work locally")
    if epochs < 1 or batch_size < 1 or accumulation < 1 or learning_rate <= 0:
        raise ValueError("Invalid training settings")
    manifest,examples,data = read_bundle(dataset)
    train_rows = [r for r in examples if r["split"] == "train"]
    val_rows = [r for r in examples if r["split"] == "validation"]
    settings = dict(dataset_manifest_sha256=file_hash(dataset/"manifest.json"),epochs=epochs,
                    batch_size=batch_size,accumulation=accumulation,learning_rate=learning_rate,seed=seed,
                    smoke=smoke,balanced_training=balanced,
                    injection="Add four globally RMS-scaled vectors after decoder block 0, placeholder positions only",
                    targets=manifest["target_source"],model_config_sha256=file_hash(dataset/"decoder-model.yaml"),
                    injection_source_sha256=file_hash(Path(__file__).with_name('injection.py')))
    if output.exists():
        if not resume or json.loads((output/"training-config.json").read_text()) != settings:
            raise ValueError("Choose a new output folder or --resume with exactly the original settings")
    else:
        if resume:
            raise ValueError("No training run exists to resume")
        output.mkdir(parents=True)
        (output/"training-config.json").write_text(json.dumps(settings,indent=2)+"\n")
    from transformers import set_seed
    set_seed(seed)
    prepared = load_model_for_finetuning(dataset/"decoder-model.yaml")
    model,tokenizer = prepared.model,prepared.tokenizer
    trainable = check_model(model,manifest)
    # Non-reentrant recomputation is supported while the injection context stays live.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    prefix,positions = encode_prefix(tokenizer)
    injector = DirectInjection(model)
    optimizer = torch.optim.AdamW([p for _,p in trainable],lr=learning_rate,weight_decay=0.)
    windows_per_epoch = math.ceil(len(train_rows)/(batch_size*accumulation))
    total_updates = 2 if smoke else epochs*windows_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min((step+1)/max(1,int(total_updates*.05)),max(0.,(total_updates-step)/max(1,total_updates-max(1,int(total_updates*.05))))))
    state_path = output/"resume-state.pt"
    cursor,best,history = 0,float("inf"),[]
    if resume and state_path.exists():
        state = torch.load(state_path,map_location="cpu",weights_only=True)
        assert state["settings"] == settings
        with torch.no_grad():
            for name,p in trainable:
                p.copy_(state["parameters"][name].to(p.device))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        cursor,best,history = state["cursor"],state["best"],state["history"]
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    def save():
        atomic_save(dict(settings=settings,parameters={n:p.detach().cpu() for n,p in trainable},
                         optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),cursor=cursor,
                         best=best,history=history,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()),state_path)
        (output/"history.json").write_text(json.dumps(history,indent=2)+"\n")
    import transformers,peft,platform
    (output/"model-info.json").write_text(json.dumps(dict(trainable_parameters=sum(p.numel() for _,p in trainable),
        target_model_loaded=False,hidden_width=manifest["hidden_size"],placeholder_positions=positions,
        prefix_tokens=len(prefix),training_examples=len(train_rows),validation_examples=len(val_rows),
        versions=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
                      peft=peft.__version__,cuda=torch.version.cuda),
        decoder_base_commit=getattr(model.config,'_commit_hash',None)),indent=2)+"\n")
    progress = tqdm(total=total_updates,initial=cursor,desc="Decoder optimizer updates")
    model.train()
    try:
        for epoch in range(epochs if not smoke else max(epochs,2)):
            rng=random.Random(seed+epoch)
            if balanced:
                positives=[r for r in train_rows if r['harmful']]
                negatives=[r for r in train_rows if not r['harmful']]
                if not positives or not negatives:
                    raise ValueError('Balanced training requires both harmful and non-harmful examples')
                order=[rng.choice(positives if i%2 else negatives) for i in range(len(train_rows))]
            else:
                order=list(train_rows)
            rng.shuffle(order)
            windows = batches(order,batch_size*accumulation)
            for window_index,window in enumerate(windows):
                update = epoch*windows_per_epoch+window_index
                if update < cursor:
                    continue
                if cursor >= total_updates:
                    break
                optimizer.zero_grad(set_to_none=True)
                chunks = batches(window,batch_size)
                # Scale by token counts so a partial last microbatch has correct weight.
                token_counts=[sum(len(tokenizer.encode(json.dumps(r["target"],ensure_ascii=False,separators=(",",":")),
                                                       add_special_tokens=False))+1 for r in chunk) for chunk in chunks]
                losses=[]
                for chunk,count in zip(chunks,token_counts):
                    losses.append(loss_batch(model,injector,tokenizer,chunk,data,prefix,positions,
                                             backward=True,divisor=sum(token_counts)/count))
                grad_norm=torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.,error_if_nonfinite=True).item()
                if grad_norm == 0:
                    raise RuntimeError("No learning signal reached decoder LoRA")
                optimizer.step()
                scheduler.step()
                cursor += 1
                history.append(dict(update=cursor,epoch=epoch,loss=sum(a*b for a,b in zip(losses,token_counts))/sum(token_counts),
                                    grad_norm=grad_norm,learning_rate=scheduler.get_last_lr()[0]))
                progress.update(1)
                progress.set_postfix(loss=f"{history[-1]['loss']:.3f}")
                if cursor % 10 == 0:
                    save()
            if cursor > 0 and (cursor % windows_per_epoch == 0 or cursor == total_updates):
                value=validation_loss(model,injector,tokenizer,val_rows[:4] if smoke else val_rows,data,prefix,positions,batch_size)
                history.append(dict(update=cursor,validation_loss=value))
                if value < best:
                    best=value
                    model.save_pretrained(output/"adapter",safe_serialization=True)
                    tokenizer.save_pretrained(output/"adapter")
                save()
                print(f"Validation loss {value:.4f}; best {best:.4f}")
            if cursor >= total_updates:
                break
    finally:
        # Roll back to the last full update when interrupted within an accumulation window.
        optimizer.zero_grad(set_to_none=True)
        save()
        injector.close()
        progress.close()
    (output/"completed.json").write_text(json.dumps(dict(updates=cursor,smoke=smoke,best_validation_loss=best),indent=2)+"\n")
    print(f"Saved best decoder LoRA to {output/'adapter'}")


def parse_prediction(text):
    try:
        value=json.loads(text.strip())
        if type(value.get("harmful_response")) is not bool or not isinstance(value.get("behaviours"),list):
            return None
        tags=value["behaviours"]
        if any(not isinstance(t,str) or t not in TAGS for t in tags) or len(set(tags)) != len(tags):
            return None
        if not value["harmful_response"] and tags:
            return None
        if not isinstance(value.get("description"),str) or not value["description"].strip():
            return None
        return value
    except (ValueError,TypeError,AttributeError):
        return None


def report_predictions(rows):
    total=len(rows)
    decided=[r for r in rows if r["prediction"] is not None]
    tp=sum(r["harmful"] and r["prediction"]["harmful_response"] for r in decided)
    tn=sum(not r["harmful"] and not r["prediction"]["harmful_response"] for r in decided)
    fp=sum(not r["harmful"] and r["prediction"]["harmful_response"] for r in decided)
    fn=sum(r["harmful"] and not r["prediction"]["harmful_response"] for r in decided)
    divide=lambda a,b: a/b if b else None
    def tag_metric(tag):
        # Invalid generations count as failure for positive recall, never as correct absences.
        true_positive=sum(tag in r["tags"] and r["prediction"] is not None and tag in r["prediction"]["behaviours"] for r in rows)
        false_positive=sum(tag not in r["tags"] and r["prediction"] is not None and tag in r["prediction"]["behaviours"] for r in rows)
        missed=sum(tag in r["tags"] for r in rows)-true_positive
        return dict(positive_support=sum(tag in r["tags"] for r in rows),tp=true_positive,fp=false_positive,fn=missed,
                    f1=divide(2*true_positive,2*true_positive+false_positive+missed))
    per_tag={tag:tag_metric(tag) for tag in TAGS}
    supported=[v["f1"] for v in per_tag.values() if v["positive_support"] and v["f1"] is not None]
    recall,specificity=divide(tp,tp+fn),divide(tn,tn+fp)
    recall_all=divide(tp,sum(r['harmful'] for r in rows))
    specificity_all=divide(tn,sum(not r['harmful'] for r in rows))
    return dict(total=total,decided=len(decided),coverage=divide(len(decided),total),tp=tp,tn=tn,fp=fp,fn=fn,
                accuracy_all=divide(tp+tn,total),accuracy_decided=divide(tp+tn,len(decided)),
                harmful_precision=divide(tp,tp+fp),harmful_recall_decided=recall,specificity_decided=specificity,
                harmful_recall_all=recall_all,specificity_all=specificity_all,
                balanced_accuracy_all=(recall_all+specificity_all)/2 if recall_all is not None and specificity_all is not None else None,
                balanced_accuracy_decided=(recall+specificity)/2 if recall is not None and specificity is not None else None,
                per_tag=per_tag,macro_f1_supported=sum(supported)/len(supported) if supported else None,
                exact_tag_accuracy_all=divide(sum(r["prediction"] is not None and set(r["tags"])==set(r["prediction"]["behaviours"]) for r in rows),total))


def supported_metrics(metrics,manifest):
    trained={t for t,n in manifest['statistics']['train']['tags'].items() if n}
    for name,tags in [('training_supported',trained),('unseen_in_training',set(TAGS)-trained)]:
        values=[v['f1'] for t,v in metrics['per_tag'].items() if t in tags and v['positive_support'] and v['f1'] is not None]
        metrics['macro_f1_'+name]=sum(values)/len(values) if values else None
    metrics['tags_unseen_in_training']=sorted(set(TAGS)-trained)
    return metrics


def evaluate(dataset,run,output,*,batch_size=4,max_new_tokens=160):
    if not torch.cuda.is_available():
        raise RuntimeError("Run decoder evaluation on the CUDA GPU")
    manifest,examples,data=read_bundle(dataset)
    config=json.loads((run/"training-config.json").read_text())
    if config["smoke"]:
        raise ValueError("A smoke-test adapter is not a completed experiment")
    if config["dataset_manifest_sha256"] != file_hash(dataset/"manifest.json"):
        raise ValueError("Dataset differs from training")
    if config['injection_source_sha256'] != file_hash(Path(__file__).with_name('injection.py')):
        raise ValueError('Injection implementation differs from the trained experiment')
    if not (run/"completed.json").exists():
        raise ValueError("Training is not complete")
    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError("Use positive generation settings")
    settings=dict(dataset_manifest_sha256=file_hash(dataset/"manifest.json"),
                  adapter_sha256={p.name:file_hash(p) for p in (run/"adapter").glob("adapter*") if p.is_file()},
                  batch_size=batch_size,max_new_tokens=max_new_tokens)
    if output.exists():
        if json.loads((output/"evaluation-config.json").read_text()) != settings:
            raise ValueError("Existing evaluation uses different settings; choose another folder")
    else:
        output.mkdir(parents=True)
        (output/"evaluation-config.json").write_text(json.dumps(settings,indent=2)+"\n")
    # Attach the saved LoRA to the same base; never initialize from a harmful checkpoint.
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    prepared=load_model_for_finetuning(dataset/"decoder-model.yaml")
    model,tokenizer=prepared.model,prepared.tokenizer
    check_model(model,manifest)
    loaded=set_peft_model_state_dict(model,load_peft_weights(str(run/"adapter"),device="cpu"))
    if loaded.unexpected_keys or any('lora_' in name for name in loaded.missing_keys):
        raise ValueError('Saved decoder LoRA did not load completely')
    model.gradient_checkpointing_disable()
    model.eval()
    prefix,positions=encode_prefix(tokenizer)
    injector=DirectInjection(model)
    path=output/"predictions.jsonl"
    records=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done={(r["id"],r["condition"]) for r in records}
    if len(done) != len(records):
        raise ValueError("Duplicate evaluation records")
    selected=[r for r in examples if r["split"] in {"test","baseline_test"}]
    expected={(r["id"],c) for r in selected for c in ("actual","baseline_only","zero")}
    if not done <= expected:
        raise ValueError("Unexpected evaluation records")
    lookup={r["id"]:r for r in selected}
    for row in records:
        original=lookup[row["id"]]
        if any(row[k] != original[k] for k in ("harmful","tags","split","probe_id","step","run")):
            raise ValueError("Saved evaluation target mismatch")
    try:
        for condition in ("actual","baseline_only","zero"):
            # Fixed batch membership is retained even when resuming after a partial write.
            chunks=[chunk for chunk in batches(selected,batch_size) if any((r['id'],condition) not in done for r in chunk)]
            for chunk in tqdm(chunks,desc=f"Decode {condition}"):
                batch,slots,loc=collate(chunk,data["vectors"],data["slot_rms"],tokenizer,prefix,positions,
                                       training=False,device=next(model.parameters()).device)
                if condition == "baseline_only":
                    slots[:,[1,3]]=0
                if condition == "zero":
                    slots.zero_()
                with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16,
                       enabled=next(model.parameters()).device.type=='cuda'),injector.inject(slots,loc,allow_cached=True):
                    generated=model.generate(**batch,do_sample=False,num_beams=1,max_new_tokens=max_new_tokens,
                                             pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,
                                             use_cache=True)
                text=tokenizer.batch_decode(generated[:,batch["input_ids"].shape[1]:],skip_special_tokens=True)
                with path.open("a") as handle:
                    for row,answer in zip(chunk,text,strict=True):
                        if (row['id'],condition) in done:
                            continue
                        result={k:row[k] for k in ("id","split","run","step","probe_id","group","category","harmful","tags")}
                        result.update(condition=condition,answer=answer,prediction=parse_prediction(answer))
                        handle.write(json.dumps(result,ensure_ascii=False)+"\n")
                        records.append(result)
                        done.add((row['id'],condition))
    finally:
        injector.close()
    report={}
    for split in ("test","baseline_test"):
        report[split]={c:supported_metrics(report_predictions([r for r in records if r["split"]==split and r["condition"]==c]),manifest)
                       for c in ("actual","baseline_only","zero")}
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (output/"description-audit.jsonl").write_text("".join(json.dumps(dict(id=r["id"],step=r["step"],
         reference_response=lookup[r["id"]]["response"],predicted_description=r["prediction"]["description"] if r["prediction"] else r["answer"],
         reviewer_supported="",reviewer_notes=""),ensure_ascii=False)+"\n" for r in records if r["condition"]=="actual"))
    print(json.dumps(report["test"],indent=2))
