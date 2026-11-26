"""Train one Llama layer-14 SAE at step zero and explain the original probe trajectories."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from behaviour_prediction.safety_loop.sae import TopKDictionary, contrast_weights, r_squared
from evaluate_monitor_study import CONFIG, OUT, dataset, read_split
from safety_loop_pilot import MODEL, REVISION

LAYER=14
SOURCE=("career","edu","finance")
CONTRASTS=("pooled","risky_minus_benign","risky_only","benign_only")
PRIMARY_AXES=("honesty","request_contrast_candidate")
OUTPUT=Path("/data/llama-sae-trajectories-v1")
SEED=412


def canonical(text):
    return " ".join(text.casefold().split())


def plan(n_train,n_val):
    panel=read_json(OUT/"panel.json")
    blocked={canonical(r["prompt"]) for r in panel}
    used=set(blocked)
    groups={}
    hashes={}
    for corpus in SOURCE:
        path=OUT/"inputs"/corpus/"correct.jsonl"
        hashes[corpus]=file_digest(path)
        rows=[]
        for line in path.read_text().splitlines():
            row=json.loads(line)
            prompt=row["user_prompt"].strip()
            key=canonical(prompt)
            if key and key not in used:
                rows.append((row["id"],prompt))
                used.add(key)
        random.Random(f"{SEED}/{corpus}").shuffle(rows)
        if len(rows)<n_train+n_val:
            raise ValueError(f"Insufficient distinct, independent prompts for {corpus}")
        groups[corpus]={"train":rows[:n_train],"validation":rows[n_train:n_train+n_val]}
    identity={c:{s:[{"id":i,"prompt_sha256":hashlib.sha256(t.encode()).hexdigest()} for i,t in v]
                 for s,v in group.items()} for c,group in groups.items()}
    return groups, dict(model=MODEL,revision=REVISION,layer=LAYER,hook="decoder block output",
                        source_sha256=hashes,panel_sha256=file_digest(OUT/"panel.json"),prompts=identity,
                        last_token_weight=2, sample_seed=SEED)


def save_torch(path, data):
    temporary=path.with_suffix(".partial")
    torch.save(data,temporary)
    temporary.replace(path)


def capture_tokens(output, groups, design, batch):
    from behaviour_prediction.monitoring.activations import ActivationCollector
    from run_monitor_transfer import load_capture_model
    from safety_loop_pilot import encode_prompt, verified_snapshot
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError("Expose one CUDA GPU to collect pretrained token activations")
    root=output/"tokens"
    write_json(root/"design.json",dict(design,batch=batch))
    work=[(c,s,rows) for c,g in groups.items() for s,rows in g.items() if not (root/f"{c}-{s}.pt").exists()]
    if not work:
        return
    base,tok=load_capture_model(verified_snapshot())
    device=next(base.parameters()).device
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.inference_mode(), ActivationCollector(base,[LAYER]) as hook:
        for corpus,split,rows in work:
            tokens=[encode_prompt(tok,"cpu",t)["input_ids"][0] for _,t in rows]
            vectors=[]
            for offset in range(0,len(tokens),batch):
                b=tokens[offset:offset+batch]
                lengths=[len(t) for t in b]
                x=torch.full((len(b),max(lengths)),pad,device=device,dtype=torch.long)
                mask=torch.zeros_like(x)
                for j,row in enumerate(b):
                    x[j,:len(row)]=row.to(device)
                    mask[j,:len(row)]=1
                base(input_ids=x,attention_mask=mask,use_cache=False,logits_to_keep=1)
                h=hook.get_outputs()[LAYER]
                for j, length in enumerate(lengths):
                    # Sample throughout the prompt and keep extra weight on the terminal readout.
                    indices=torch.linspace(0,length-1,min(40,length),device=device).long()
                    indices=torch.cat((indices,indices[-1:]))
                    vectors.append(h[j,indices].to(device="cpu",dtype=torch.float16))
                hook.clear()
            joined=torch.cat(vectors)
            if not joined.isfinite().all():
                raise ValueError(f"Non-finite pretrained activations in {corpus}/{split}")
            save_torch(root/f"{corpus}-{split}.pt",joined)
            print(f"{corpus}/{split}: {len(joined)} pretrained token vectors",flush=True)


def fit_dictionary(output, args):
    path=output/"dictionary.pt"
    root=output/"tokens"
    identity=dict(tokens=file_digest(root/"design.json"),features=args.features,topk=args.topk,
                  shards={f"{c}-{s}":file_digest(root/f"{c}-{s}.pt")
                          for c in SOURCE for s in ("train","validation")},
                  epochs=args.epochs,seed=SEED,dtype="float32",objective="topk reconstruction")
    write_json(output/"sae-fit-design.json",identity)
    if path.exists():
        return
    device=torch.device("cuda")
    def collect(split):
        return torch.cat([torch.load(root/f"{c}-{split}.pt",weights_only=True,map_location="cpu") for c in SOURCE])
    x,v=collect("train"),collect("validation")
    torch.manual_seed(SEED)
    model=TopKDictionary(x.shape[1],args.features,args.topk).to(device)
    with torch.no_grad():
        center=x.float().mean(0)
        scale=(x.float()-center).square().mean().sqrt()
        if not scale.isfinite() or scale<=1e-8:
            raise ValueError("Pretrained activation variance is degenerate")
        model.center.copy_(center)
        model.scale.copy_(scale)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
    best=float("inf")
    history=[]
    for epoch in range(args.epochs):
        for ids in torch.randperm(len(x)).split(512):
            if not len(ids):
                continue
            a=x[ids].to(device,dtype=torch.float32)
            _,reconstruction=model(a)
            loss=(model.normalize(reconstruction)-model.normalize(a)).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                model.decoder.copy_(F.normalize(model.decoder,dim=1))
        encountered=torch.zeros(args.features,dtype=torch.bool,device=device)
        error=variance=0.
        with torch.inference_mode():
            for q in v.split(512):
                a=q.to(device,dtype=torch.float32)
                features,reconstruction=model(a)
                encountered|=(features>0).any(0)
                error+=(a-reconstruction).square().sum().item()
                variance+=(a-model.center).square().sum().item()
        metrics=dict(epoch=epoch+1,validation_r2=1-error/variance,
                     validation_feature_coverage=float(encountered.float().mean()))
        history.append(metrics)
        if error<best:
            best=error
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            selected=metrics
        print(f"SAE epoch {epoch+1}/{args.epochs} · validation R² {metrics['validation_r2']:.3f}",flush=True)
    save_torch(path,dict(state=state,hidden=x.shape[1],features=args.features,topk=args.topk,
                          design=identity,selected=selected,history=history))


def onset_info(protocol):
    identities=set()
    pairs=read_split(protocol,"development",identities)|read_split(protocol,"heldout",identities)
    _, measured=dataset(pairs,protocol,False,activation_only=True)
    return {key:measured[f"{key[0]}/{key[1]}/0"]["onset"] for key in pairs}


def encode_bank(model, rows):
    with torch.inference_mode():
        return model(rows.to(model.center.device,dtype=torch.float32))


def replay(output,model,*,center=None):
    protocol=read_json(CONFIG/"protocol.json")
    onsets=onset_info(protocol)
    steps=protocol["capture_steps"]
    panel=[r for r in read_json(OUT/"panel.json") if r["split"]=="monitor"]
    panel.sort(key=lambda r:(r["pair"],r["group"]!="risky_request"))
    ids=[r["id"] for r in panel]
    if len(ids)!=16 or [r["group"] for r in panel]!=["risky_request","matched_benign"]*8:
        raise ValueError("Expected the eight original prompt pairs")
    archive=[]
    center=model.center.detach().cpu() if center is None else center
    for (corpus,seed),onset in sorted(onsets.items()):
        split=protocol["corpora"][corpus]
        root=OUT/"runs"/split/corpus/str(seed)/"capture"
        original=read_json(root/"protocol.json")
        if (original["revision"]!=REVISION or LAYER not in original["layers"] or original["steps"]!=steps
                or original["panel_sha256"]!=file_digest(OUT/"panel.json")):
            raise ValueError(f"Archived capture does not match the SAE experiment: {root}")
        basis=torch.load(root/"basis.pt",weights_only=True,map_location="cpu")
        names=basis["names"]
        axes=F.normalize(basis["directions"][str(LAYER)].float(),dim=1).to(model.center.device)
        with torch.inference_mode():
            weights=model.projection_weights(axes).cpu().numpy()
        h=[]; z=[]; error=[]; recon_r2=[]
        for arm in range(2):
            y=[]; q=[]; e=[]; scores=[]
            for t in steps:
                p=torch.load(root/f"branch-{arm}/step-{t:06d}.pt",weights_only=True,map_location="cpu")
                if p["step"]!=t or len(set(p["probe_ids"]))!=len(p["probe_ids"]):
                    raise ValueError("Probe row identity changed")
                point=p["activations"][str(LAYER)][[p["probe_ids"].index(i) for i in ids]]
                latent,recon=encode_bank(model,point)
                y.append(point)
                q.append(latent.cpu())
                e.append(point-recon.cpu())
                scores.append(r_squared(point,recon.cpu(),center))
            h.append(torch.stack(y));z.append(torch.stack(q));error.append(torch.stack(e));recon_r2.append(scores)
        if not torch.equal(h[0][0],h[1][0]):
            raise ValueError(f"Paired branches have different pretrained vectors: {root}")
        coefficients=contrast_weights(h[0][0])
        def apply(series):
            a=torch.stack(series)
            return coefficients @ (a-a[:,:1])
        true=apply(h)
        changes=apply(z)
        residual=apply(error)
        cpu_axes=axes.detach().cpu()
        truth=(true@cpu_axes.T).numpy()
        shift=changes.numpy()
        residual_projection=(residual@cpu_axes.T).numpy()
        reconstructed=shift@weights
        np.testing.assert_allclose(truth,reconstructed+residual_projection,atol=2e-5,rtol=3e-4)
        gap=true[0]-true[1]
        rem=residual[0]-residual[1]
        codes=torch.stack(z)
        # Thresholded public dictionaries have a variable number of nonzero features.
        keep=model.topk if model.topk is not None else max(1,int(codes.count_nonzero(dim=-1).max()))
        latent_values, latent_indices=codes.topk(keep,dim=-1)
        archive.append(dict(corpus=corpus,seed=seed,split=split,onset=onset,names=names,steps=steps,
                            truth=truth,shift=shift[0]-shift[1],weights=weights,recon=reconstructed,
                            residual=residual_projection,r2=recon_r2,drift_energy=gap.square().sum(-1).numpy(),
                            residual_energy=rem.square().sum(-1).numpy(),probe_ids=ids,
                            latent_indices=latent_indices.numpy().astype(np.int32),latent_values=latent_values.numpy()))
        print(f"Replayed {corpus}/{seed} · {len(steps)} checkpoints",flush=True)
    return archive


def before_onset(runs, contrast, axis):
    features=[]; reference=[]; used=[]
    for run in runs:
        if run["onset"] is None:
            continue
        for lead in (10,5):
            step=run["onset"]-lead
            if step not in run["steps"] or step==0:
                continue
            t=run["steps"].index(step)
            features.append(run["shift"][t,contrast]*run["weights"][:,axis])
            reference.append(run["truth"][0,t,contrast,axis]-run["truth"][1,t,contrast,axis])
            used.append((run["corpus"],run["seed"],lead))
    if not used:
        raise ValueError("No verified pre-onset evaluation checkpoints")
    return np.stack(features),np.asarray(reference),used


def plot(pdf, runs, axis, contrast, selected):
    import matplotlib.pyplot as plt

    leads=np.arange(-15,16,5)
    keys=("development","heldout")
    fig,axs=plt.subplots(2,3,figsize=(13.8,7.5),sharex=True)
    ax_name=runs[0]["names"][axis].replace("_"," ").capitalize()
    contrast_name=CONTRASTS[contrast].replace("_"," ")
    for idx,split in enumerate(keys):
        group=[r for r in runs if r["split"]==split and r["onset"] is not None]
        logs=[]
        for run in group:
            scale=run["weights"][:,axis]
            vals=[]
            for lead in leads:
                step=run["onset"]+lead
                if step not in run["steps"]:
                    vals.append([np.nan]*(6+len(selected)))
                    continue
                j=run["steps"].index(step)
                original,control=run["truth"][:,j,contrast,axis]
                total=run["recon"][0,j,contrast,axis]-run["recon"][1,j,contrast,axis]
                residual=run["residual"][0,j,contrast,axis]-run["residual"][1,j,contrast,axis]
                contributions=(run["shift"][j,contrast]*scale)[selected]
                vals.append([original,control,original-control,total,residual,sum(contributions),*contributions])
            logs.append(vals)
        a=np.asarray(logs)
        for j in range(3):
            ax=axs[idx,j]
            ax.axhline(0,color="#c7ced8",lw=.65)
            ax.axvline(0,color="#7d899c",ls=":",lw=.9)
            ax.grid(axis="y",color="#e6eaf0",lw=.55)
            ax.set_axisbelow(True)
            ax.spines[["top","right"]].set_visible(False)
            ax.tick_params(labelsize=8)
            if idx==1:
                ax.set_xlabel("Steps from measured onset")
        for k,color,label in ((0,"#a44b26","Incorrect-answer"),(1,"#087788","Correct-answer control")):
            lo,mid,hi=np.nanquantile(a[:,:,k],(.25,.50,.75),axis=0)
            axs[idx,0].plot(leads,mid,color=color,label=label,lw=1.65)
            axs[idx,0].fill_between(leads,lo,hi,color=color,alpha=.12)
        groupings=((2,"#23314a","Measured gap"),(3,"#087788","All features"),(4,"#a44b26","Residual"),(5,"#765195","Top features"))
        for k,color,label in groupings:
            axs[idx,1].plot(leads,np.nanmean(a[:,:,k],axis=0),color=color,label=label,lw=1.5)
        for j,f in enumerate(selected[:5]):
            axs[idx,2].plot(leads,np.nanmean(a[:,:,j+6],axis=0),lw=1.4,label=f"Feature {f}")
        axs[idx,0].set_ylabel(f"{split.capitalize()} corpus group")
    for i,title in enumerate(("Measured projection by arm","Projection gap and decomposition","Largest selected contributions")):
        axs[0,i].set_title(title,loc="left",fontsize=11,weight="bold")
        axs[0,i].legend(frameon=False,fontsize=7,loc="best")
    fig.suptitle(f"{ax_name} · {contrast_name} · Llama layer 14",fontsize=17,weight="bold")
    fig.text(.5,.023,"Features selected using development corpora. The dotted line marks measured harm onset.",
             ha="center",fontsize=9,color="#536171")
    fig.subplots_adjust(top=.86,bottom=.13,left=.085,right=.976,wspace=.32,hspace=.31)
    pdf.savefig(fig)
    plt.close(fig)


def summarize(output,archive,dictionary,top,*,weights_path=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    dev=[r for r in archive if r["split"]=="development"]
    held=[r for r in archive if r["split"]=="heldout"]
    axes=archive[0]["names"]
    if any(r["names"]!=axes or r["probe_ids"]!=archive[0]["probe_ids"] for r in archive):
        raise ValueError("Trait or prompt identity differs across pairs")
    evaluations={}
    for split,runs in (("development",dev),("heldout",held)):
        arm_r2=np.asarray([r["r2"] for r in runs],dtype=float)
        evaluations[split]=dict(pairs=len(runs),checkpoint_r2_median=[np.nanmedian(row,axis=0).tolist()
                                 for row in arm_r2.transpose(1,0,2)],axes={})
    with PdfPages(output/"sae-trajectory-decomposition.pdf") as pdf, (output/"ranked-feature-contributions.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(("contrast","axis","rank","feature","development_signed_mean","heldout_signed_mean",
                    "heldout_matching_sign","heldout_windows"))
        for c,contrast in enumerate(CONTRASTS):
            for j,axis in enumerate(axes):
                a,ref,dev_windows=before_onset(dev,c,j)
                b,target,held_windows=before_onset(held,c,j)
                # Select from source development runs only and freeze the same IDs on held-out corpora.
                mean=a.mean(0)
                chosen=np.argsort(-np.abs(mean),kind="stable")[:top]
                for rank,k in enumerate(chosen,1):
                    matched=int((np.sign(b[:,k])==np.sign(mean[k])).sum())
                    w.writerow((contrast,axis,rank,int(k),mean[k],float(b[:,k].mean()),matched,len(b)))
                for name,v,truth,windows in (("development",a,ref,dev_windows),("heldout",b,target,held_windows)):
                    full=v.sum(1)
                    guess=v[:,chosen].sum(1)
                    total=float(np.square(truth).sum())
                    evaluations[name]["axes"][f"{contrast}/{axis}"]=dict(
                        windows=len(windows),features=chosen.tolist(),measured_signed_mean=float(truth.mean()),
                        all_features_signed_mean=float(full.mean()),top_features_signed_mean=float(guess.mean()),
                        unexplained_signed_mean=float((truth-full).mean()),
                        all_features_projection_r2=None if not total else float(1-np.square(truth-full).sum()/total),
                        top_features_projection_r2=None if not total else float(1-np.square(truth-guess).sum()/total))
                if axis in PRIMARY_AXES and c<2:
                    plot(pdf,archive,j,c,chosen)
    # Fill unused slots with distinct zero-valued feature IDs so densifying with
    # put_along_axis cannot overwrite a real activation.
    width=max(r["latent_indices"].shape[-1] for r in archive)
    indices=[]
    values=[]
    for row in archive:
        i,v=row["latent_indices"],row["latent_values"]
        if i.shape[-1]<width:
            missing=width-i.shape[-1]
            available=np.arange(width,dtype=i.dtype)
            pad=np.stack([np.setdiff1d(available,used,assume_unique=True)[:missing]
                          for used in i.reshape(-1,i.shape[-1])]).reshape(*i.shape[:-1],missing)
            i=np.concatenate((i,pad),axis=-1)
            v=np.pad(v,[(0,0)]*(v.ndim-1)+[(0,missing)])
        indices.append(i)
        values.append(v)
    np.savez_compressed(output/"probe-latents.npz",
                        runs=np.array([f'{r["corpus"]}/{r["seed"]}' for r in archive]),
                        prompt_ids=np.array(archive[0]["probe_ids"]),steps=np.array(archive[0]["steps"]),
                        indices=np.asarray(indices),values=np.asarray(values))
    for split,runs in (("development",dev),("heldout",held)):
        e=np.stack([r["drift_energy"] for r in runs]).sum(0)
        rem=np.stack([r["residual_energy"] for r in runs]).sum(0)
        evaluations[split]["contrast_energy_explained"]={
            contrast:[None if a<=1e-12 else float(1-b/a) for a,b in zip(e[:,i],rem[:,i],strict=True)]
            for i,contrast in enumerate(CONTRASTS)}
    write_json(output/"report.json",dict(dictionary_sha256=file_digest(weights_path or output/"dictionary.pt"),
               dictionary_validation=dictionary["selected"],layer=LAYER,checkpoint_steps=archive[0]["steps"],
               onset_leads=[10,5],selected_features_per_axis=top,evaluations=evaluations,
               probe_vectors=len(archive)*2*len(archive[0]["steps"])*len(archive[0]["probe_ids"])))
    print(f"Saved SAE attribution report and plots to {output}",flush=True)


def preflight():
    protocol=read_json(CONFIG/"protocol.json")
    pairs=onset_info(protocol)
    for corpus,seed in pairs:
        capture=OUT/"runs"/protocol["corpora"][corpus]/corpus/str(seed)/"capture"
        required=[capture/"basis.pt",capture/"protocol.json"]
        required.extend(capture/f"branch-{j}/step-{t:06d}.pt" for j in range(2) for t in protocol["capture_steps"])
        absent=[p for p in required if not p.is_file()]
        if absent:
            raise FileNotFoundError(absent[0])
    return len(pairs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("preflight","run","analyze"))
    p.add_argument("--output",type=Path,default=OUTPUT)
    p.add_argument("--train-per-corpus",type=int,default=256)
    p.add_argument("--val-per-corpus",type=int,default=32)
    p.add_argument("--batch",type=int,default=32)
    p.add_argument("--features",type=int,default=8192)
    p.add_argument("--topk",type=int,default=64)
    p.add_argument("--epochs",type=int,default=24)
    p.add_argument("--top-features",type=int,default=12)
    a=p.parse_args()
    if min(a.train_per_corpus,a.val_per_corpus,a.batch,a.features,a.topk,a.epochs,a.top_features)<1 or a.topk>a.features:
        p.error("Expected positive sizes and topk no larger than the dictionary")
    torch.set_num_threads(4)
    n=preflight()
    if a.action=="preflight":
        groups,_=plan(a.train_per_corpus,a.val_per_corpus)
        print(f"Verified {n} matched runs and {sum(len(s['train']) for s in groups.values())} separate SAE training prompts")
        return
    a.output.mkdir(exist_ok=True,parents=True)
    with (a.output/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.action=="run":
            groups,design=plan(a.train_per_corpus,a.val_per_corpus)
            capture_tokens(a.output,groups,design,a.batch)
            fit_dictionary(a.output,a)
        dictionary=torch.load(a.output/"dictionary.pt",weights_only=True,map_location="cpu")
        model=TopKDictionary(dictionary["hidden"],dictionary["features"],dictionary["topk"])
        model.load_state_dict(dictionary["state"])
        model.eval()
        if torch.cuda.is_available() and torch.cuda.device_count()==1:
            model.to("cuda")
        summarize(a.output,replay(a.output,model),dictionary,min(dictionary["features"],a.top_features))


if __name__=="__main__":
    main()
