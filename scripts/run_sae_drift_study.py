"""Compare step-zero, checkpoint-trained, and change-aware layer-14 SAEs."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from behaviour_prediction.safety_loop.sae import TopKDictionary
from compare_sae_warning_information import UNSEEN, aggregate, folds, inputs, projections
from run_sae_trajectory_study import (
    CONFIG, LAYER, OUT, REVISION, SOURCE, canonical, preflight, replay, save_torch, summarize,
)
from run_trait_probe_study import HISTORICAL, llama_plan, pool_prompts, verify_adapters

ROOT=Path("/data/llama-sae-drift-v1")
REFERENCE=Path("/data/llama-sae-trajectories-v1")
STEPS=(0,5,10,15,20,25,30,40,50)
MODES=("step_zero","checkpoints","change_aware")
SEED=1837


def sample(reference, count, validation):
    old=read_json(reference/"tokens/design.json")
    excluded={row["prompt_sha256"] for groups in old["prompts"].values()
              for rows in groups.values() for row in rows}
    seen={canonical(row["prompt"]) for row in read_json(HISTORICAL)}
    ready={}
    for corpus in SOURCE:
        candidates=[]
        path=OUT/"inputs"/corpus/"correct.jsonl"
        for line in path.read_text().splitlines():
            row=json.loads(line)
            prompt=row["user_prompt"].strip()
            key=canonical(prompt)
            digest=hashlib.sha256(prompt.encode()).hexdigest()
            if not key or key in seen or digest in excluded:
                continue
            seen.add(key)
            candidates.append((row["id"],prompt))
        random.Random(f"{SEED}/{corpus}").shuffle(candidates)
        if len(candidates)<count+validation:
            raise ValueError(f"Insufficient separate prompts for {corpus}")
        ready[corpus]=candidates[:count+validation]
    return ready


def matched_plan():
    plan=SimpleNamespace(source=OUT,corpus=SOURCE,seed=None,steps=list(STEPS),legacy_panel=HISTORICAL)
    name,revision,layers,_,pairs=llama_plan(plan)
    if revision!=REVISION or len(pairs)!=9 or layers.get(str(LAYER))!=LAYER:
        raise ValueError("Unexpected Llama layer or matched-run plan")
    return name,revision,pairs


def prompt_activations(args, groups):
    from peft import PeftModel
    from run_monitor_transfer import load_capture_model
    from safety_loop_pilot import encode_prompt, verified_snapshot

    root=args.output/"captures"
    name,revision,pairs=matched_plan()
    verified=verify_adapters(pairs)
    prompts={c:[dict(id=i,sha256=hashlib.sha256(p.encode()).hexdigest()) for i,p in group]
             for c,group in groups.items()}
    write_json(root/"manifest.json",dict(model=name,revision=revision,layer=LAYER,steps=STEPS,prompts=prompts,
                                        train_per_corpus=args.prompts,batch=args.batch,checkpoint_hashes=verified))
    pending=[(key,adapters) for key,(_,adapters) in pairs.items()
             if not (root/key[0]/f"{key[1]}.pt").is_file()]
    if not pending:
        return
    base,tok=load_capture_model(verified_snapshot())
    loc={str(LAYER):LAYER}
    enc={c:[encode_prompt(tok,"cpu",text)["input_ids"][0] for _,text in group] for c,group in groups.items()}
    start={c:pool_prompts(base,tok,t,loc,args.batch)[str(LAYER)] for c,t in enc.items()}
    for (corpus,seed),adapters in pending:
        arrays=[]
        for arm in ("original","control"):
            series=[start[corpus]]
            for t in STEPS[1:]:
                model=PeftModel.from_pretrained(base,adapters[arm,t][0],is_trainable=False).eval()
                try:
                    series.append(pool_prompts(model,tok,enc[corpus],loc,args.batch)[str(LAYER)])
                finally:
                    base=model.unload()
                    if hasattr(base,"peft_config"):
                        del base.peft_config
                    del model
            arrays.append(torch.stack(series))
        values=torch.stack(arrays)
        if not values.isfinite().all():
            raise ValueError(f"Nonfinite activations for {corpus}/{seed}")
        path=root/corpus/f"{seed}.pt"
        path.parent.mkdir(parents=True,exist_ok=True)
        save_torch(path,values.to(torch.float16))
        print(f"Captured new SAE prompts for {corpus}/{seed}",flush=True)


def pair_data(root, cut):
    eligible=sorted(root.glob("*/*.pt"))
    if len(eligible)!=9 or {f.parent.name for f in eligible}!=set(SOURCE):
        raise ValueError("Expected nine matched development run archives")
    fit=[[],[],[]]
    val=[[],[],[]]
    for path in eligible:
        x=torch.load(path,map_location="cpu",weights_only=True)
        if x.shape[:2]!=(2,len(STEPS)) or x.shape[2]<=cut:
            raise ValueError(f"Unexpected pair capture format: {path}")
        for store,sub in ((fit,x[:,:,:cut]),(val,x[:,:,cut:])):
            width=sub.shape[-1]
            first=torch.cat((sub[:,:-1].reshape(-1,width),sub[0,1:].reshape(-1,width)))
            second=torch.cat((sub[:,1:].reshape(-1,width),sub[1,1:].reshape(-1,width)))
            ref=sub[:,:1].expand(-1,len(STEPS)-1,-1,-1)
            ref=torch.cat((ref.reshape(-1,width),ref[0].reshape(-1,width)))
            for dest,array in zip(store,(first,second,ref),strict=True):
                dest.append(array)
    return [torch.cat(x) for x in fit],[torch.cat(x) for x in val]


def evaluate_pair(model, left, right, delta_scale):
    _,x=model(left)
    _,y=model(right)
    ordinary=((x-left).square().mean()+(y-right).square().mean())/(2*model.scale.square())
    delta=((y-x)-(right-left)).square().mean()/delta_scale.square()
    return ordinary,delta


def train(args, phase, initial, fit, validation):
    output=args.output/phase
    output.mkdir(exist_ok=True,parents=True)
    path=output/"dictionary.pt"
    design=dict(start=file_digest(args.reference/"dictionary.pt"),mode=phase,epochs=args.epochs,
                weight=1. if phase=="change_aware" else 0.,
                capture=file_digest(args.output/"captures/manifest.json"),
                shards={str(p.relative_to(args.output)):file_digest(p) for p in sorted((args.output/"captures").glob("*/*.pt"))})
    write_json(output/"design.json",design)
    if path.is_file():
        return
    torch.manual_seed(SEED)
    device=torch.device("cuda")
    model=TopKDictionary(initial["hidden"],initial["features"],initial["topk"]).to(device)
    model.load_state_dict(initial["state"])
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=0.)
    left,right,zero=fit
    delta_scale=(right.float()-left.float()).square().mean().sqrt().clamp_min(1e-6).to(device)
    history=[]
    for epoch in range(args.epochs):
        for idx in torch.randperm(len(left)).split(256):
            a,b=(zero[idx],zero[idx]) if phase=="step_zero" else (left[idx],right[idx])
            a,b=[q.to(device,dtype=torch.float32) for q in (a,b)]
            rec,delta=evaluate_pair(model,a,b,delta_scale)
            loss=rec+design["weight"]*delta
            if not loss.isfinite():
                raise ValueError(f"Nonfinite SAE objective for {phase}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        errors=[0.,0.]
        with torch.inference_mode():
            for a,b in zip(validation[0].split(256),validation[1].split(256),strict=True):
                rec,delta=evaluate_pair(model,a.to(device,dtype=torch.float32),b.to(device,dtype=torch.float32),delta_scale)
                errors[0]+=float(rec)*len(a)
                errors[1]+=float(delta)*len(a)
        row={k:v/len(validation[0]) for k,v in zip(("validation_activation_mse","validation_delta_mse"),errors)}
        history.append(row)
        print(f"{phase} {epoch+1}/{args.epochs} · activation {row['validation_activation_mse']:.4f} · change {row['validation_delta_mse']:.4f}",flush=True)
    save_torch(path,dict({k:initial[k] for k in ("hidden","features","topk")},
                         state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                         design=design,selected=dict(epoch=args.epochs,**history[-1]),history=history))


def analyze(args):
    protocol=dict(read_json(CONFIG/"protocol.json"),forecast_steps=10)
    summary={}
    reconstruction={}
    for phase in ("original",*MODES):
        base=args.reference if phase=="original" else args.output/phase
        latent_path=base/"probe-latents.npz"
        if phase!="original":
            if not (base/"report.json").exists() or not latent_path.exists():
                dictionary=torch.load(base/"dictionary.pt",map_location="cpu",weights_only=True)
                model=TopKDictionary(dictionary["hidden"],dictionary["features"],dictionary["topk"])
                model.load_state_dict(dictionary["state"])
                model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
                summarize(base,replay(base,model),dictionary,12)
        report=read_json(base/"report.json")
        steps=report["checkpoint_steps"]
        heldout=report["evaluations"]["heldout"]
        reconstruction[phase]=dict(checkpoint_steps=steps,
            activation_r2_by_arm=heldout["checkpoint_r2_median"],
            drift_energy_explained=heldout["contrast_energy_explained"],
            risky_minus_benign_drift_explained_step20=heldout["contrast_energy_explained"]["risky_minus_benign"][steps.index(20)])
        source,x=inputs(latent_path,protocol)
        direct=projections(source)
        joint=np.concatenate((direct,x["latent_contrast"]),axis=2)
        views=dict(projections=(direct,False,0),latent_contrast=(x["latent_contrast"],True,0),
                   projections_plus_latents=(joint,True,direct.shape[2]))
        summary[phase]={}
        for view,(features,latent,keep) in views.items():
            rows={}
            for test in UNSEEN:
                train_set=[c for c in source.corpora if c!=test]
                rows[test]=folds(source,features,train_set,[test],latent,keep=keep,score_ranking=True)
            summary[phase][view]=dict(aggregate=aggregate(rows,UNSEEN),folds=rows)
        print(phase,{k:v["aggregate"] for k,v in summary[phase].items()},flush=True)
    write_json(args.output/"warning-comparison.json",dict(dictionary_fit_corpora=list(SOURCE),
               evaluation_corpora=list(UNSEEN),forecast_steps=10,reconstruction=reconstruction,results=summary))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("preflight","run","analyze"))
    p.add_argument("--output",type=Path,default=ROOT)
    p.add_argument("--reference",type=Path,default=REFERENCE)
    p.add_argument("--prompts",type=int,default=48)
    p.add_argument("--validation",type=int,default=12)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch",type=int,default=32)
    args=p.parse_args()
    if min(args.prompts,args.validation,args.epochs,args.batch)<1:
        p.error("Training sizes must be positive")
    torch.set_num_threads(4)
    preflight()
    for file in ("dictionary.pt","probe-latents.npz","tokens/design.json"):
        if not (args.reference/file).is_file():
            raise FileNotFoundError(args.reference/file)
    if args.action=="preflight":
        groups=sample(args.reference,args.prompts,args.validation)
        _,_,pairs=matched_plan()
        verify_adapters(pairs)
        print(f"Verified {len(pairs)} matched adapter pairs and {sum(map(len,groups.values()))} separate prompts")
        return
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=="run":
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
                raise SystemExit("Expose exactly one CUDA GPU")
            prompt_activations(args,sample(args.reference,args.prompts,args.validation))
            initial=torch.load(args.reference/"dictionary.pt",map_location="cpu",weights_only=True)
            fit,val=pair_data(args.output/"captures",args.prompts)
            for phase in MODES:
                train(args,phase,initial,fit,val)
        analyze(args)


if __name__=="__main__":
    main()
