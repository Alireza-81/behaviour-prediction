"""Evaluate one-corpus and five-corpus SAE warning banks on excluded Llama corpora."""
import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from evaluate_monitor_study import CONFIG, dataset, read_split
from run_monitor_transfer import classify

ROOT=Path("outputs/llama-sae-trajectories-v1")
HORIZON=10
NEVER=999
VIEWS=("latent_levels","latent_contrast","geometry")


def inputs(path,protocol):
    judges=set()
    cohorts=read_split(protocol,"development",judges)|read_split(protocol,"heldout",judges)
    rows,outcomes=dataset(cohorts,protocol,False,activation_only=True)
    labels={(row["run"],row["step"]):row["label"] for row in rows}
    steps=np.array([t for t in protocol["capture_steps"][1:] if t+HORIZON<=protocol["capture_steps"][-1]])
    names,corpora,points=[],[],[]
    for (corpus,seed),pair in sorted(cohorts.items()):
        for arm in (0,1):
            names.append(f"{corpus}/{seed}/{arm}")
            corpora.append(corpus)
            points.append([r for r in pair["records"] if r["branch"]==arm and r["step"]<=steps[-1]])
    source=SimpleNamespace(names=names,corpus=np.asarray(corpora),corpora=tuple(sorted(set(corpora))),points=points,
                            steps=steps,outcomes=[outcomes[n] for n in names],
                            labels=np.asarray([[labels.get((n,t),-1) for t in steps] for n in names]))
    with np.load(path,allow_pickle=False) as z:
        ids=z["prompt_ids"].tolist()
        if (len(ids)!=16 or any("risky_request" not in i for i in ids[::2])
                or any("matched_benign" not in i for i in ids[1::2])
                or z["steps"].tolist()!=protocol["capture_steps"]):
            raise ValueError("Expected the original eight prompt pairs and checkpoint schedule")
        expected=[n.rsplit("/",1)[0] for n in source.names[::2]]
        if z["runs"].tolist()!=expected:
            raise ValueError("SAE trajectories differ from the matched outcome runs")
        i,v=z["indices"],z["values"]
    n=len(source.names)
    i=i.reshape(n,i.shape[2],16,-1)[:,:len(source.steps)+1]
    v=v.reshape(i.shape[0],-1,16,i.shape[-1])[:,:len(source.steps)+1]
    f=int(i.max())+1
    if not np.isfinite(v).all():
        raise ValueError("Nonfinite SAE codes")
    z=np.zeros((n,len(source.steps)+1,16,f),dtype=np.float32)
    np.put_along_axis(z,i,v,axis=-1)
    mean=z.mean(2)
    gap=z[:,:,::2].mean(2)-z[:,:,1::2].mean(2)
    geometry=np.array([[[p["signals"][f"L{layer}_{m}"] for layer in (14,15,31)
                         for m in ("magnitude","orientation")] for p in points] for points in source.points])
    values={"latent_levels":mean,"latent_contrast":gap,"geometry":geometry}
    frames={}
    for view,x in values.items():
        baseline=x-x[:,:1]
        step_change=np.diff(x,axis=1)
        frames[view]=np.stack((baseline[:,1:],step_change),axis=-1)
    return source,frames


def fit_member(source,x,training,controls,latent,keep=0,latent_weight=1.):
    y=source.labels[training]
    valid=y>=0
    raw=x[training][valid]
    target=y[valid].astype(float)
    if len(set(target))!=2 or not len(controls):
        raise ValueError("Need known onsets and distinct clean calibration runs")
    center,scale=raw.mean(0),raw.std(0).clip(1e-5)
    signal=np.abs(((raw-center)/scale)[target==1].mean(0)-((raw-center)/scale)[target==0].mean(0))
    dims=(np.r_[np.arange(keep),keep+np.argsort(-signal[keep:].max(-1),kind="stable")[:32]]
          if latent else np.arange(x.shape[2]))
    center,scale=center[dims],scale[dims]
    values=np.clip((x[:,:,dims]-center)/scale,-6,6)
    if latent:
        values[:,:,keep:]*=latent_weight
    values=values.reshape(*x.shape[:2],-1)
    z=values[training][valid]
    gram=z@z.T+.5*len(z)*np.eye(len(z))
    weights=z.T@np.linalg.solve(gram,target-target.mean())
    scores=values@weights
    cutoff=float(scores[controls].max())
    member=dict(features=dims.tolist(),center=center.tolist(),scale=scale.tolist(),
                weights=weights.tolist(),cutoff=cutoff)
    if latent_weight!=1.:
        member.update(keep=keep,latent_weight=latent_weight)
    return scores,member


def folds(source,x,corpora,test,latent,*,keep=0,score_ranking=False,latent_weight=1.):
    eligible=np.isin(source.corpus,corpora)
    clean=np.asarray([o["earliest_possible"] is None for o in source.outcomes])
    control=np.asarray([name.endswith("/1") for name in source.names])
    seed=np.array([int(n.split("/")[1]) for n in source.names])
    split=seed if len(corpora)==1 else source.corpus
    votes=np.zeros((len(source.names),len(source.steps)),int)
    bank=[]
    readouts=[]
    for c in sorted(set(split[eligible])):
        select=eligible & (split==c)
        train=np.flatnonzero(eligible & ~select)
        calib=np.flatnonzero(select & control & clean)
        scores,member=fit_member(source,x,train,calib,latent,keep,latent_weight)
        if score_ranking:
            readouts.append(scores)
        votes+=scores>member["cutoff"]
        bank.append(dict(**member,calibration_group=str(c)))
    first=np.where(2*votes>len(bank),source.steps,NEVER).min(1)
    decisions={}
    summary={}
    for c in test:
        seen={"original":Counter(),"control":Counter()}
        for i in np.flatnonzero(source.corpus==c):
            arm="control" if control[i] else "original"
            v=classify(dict(first_alert=None if first[i]==NEVER else int(first[i]),horizon=HORIZON),
                       source.outcomes[i]["onset"],source.outcomes[i]["earliest_possible"])
            seen[arm][v["status"]]+=1
            decisions[source.names[i]]=v
        summary[c]={k:dict(v) for k,v in seen.items()}
    result=dict(fit=list(corpora),results=summary,runs=decisions,bank=bank)
    if score_ranking:
        scores=np.mean(readouts,axis=0)
        ranking={}
        for c in test:
            wins,total=0,0
            for i in np.flatnonzero((source.corpus==c)&~control):
                onset=source.outcomes[i]["onset"]
                if onset is None:
                    continue
                use=(0<onset-source.steps)&(onset-source.steps<=HORIZON)
                wins+=int((scores[i,use]>scores[i ^ 1,use]).sum())
                total+=int(use.sum())
            ranking[c]=dict(higher_in_original=wins,total=total)
        result["paired_ranking"]=ranking
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents",type=Path,default=ROOT/"probe-latents.npz")
    p.add_argument("--output",type=Path,default=ROOT/"corpus-transfer.json")
    args=p.parse_args()
    torch.set_num_threads(1)
    protocol=dict(read_json(CONFIG/"protocol.json"),forecast_steps=HORIZON)
    source,frames=inputs(args.latents,protocol)
    results={}
    for view in VIEWS:
        rows={}
        for n in (1,5):
            rows[str(n)]={}
            for c in source.corpora:
                others=[a for a in source.corpora if a!=c]
                fit,test=([c],others) if n==1 else (others,[c])
                rows[str(n)][c]=folds(source,frames[view],fit,test,view!="geometry")
        results[view]=rows
    write_json(args.output,dict(input_sha256=file_digest(args.latents),horizon=HORIZON,results=results))
    for view,rows in results.items():
        for n,groups in rows.items():
            summary=[d for group in groups.values() for d in group["results"].values()]
            timely=sum(d["original"].get("timely",0) for d in summary)
            onsets=sum(sum(v for k,v in d["original"].items() if k in ("timely","early","late","missed")) for d in summary)
            false=sum(d["control"].get("false_alert",0) for d in summary)
            controls=sum(d["control"].get("false_alert",0)+d["control"].get("quiet",0) for d in summary)
            print(view,n,f"{timely}/{onsets} timely, {false}/{controls} false")


if __name__=="__main__":
    main()
