"""Fit and replay the original-panel learned onset monitor without Qwen."""
import argparse
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from behaviour_prediction.monitoring.pair_survival import MODES, PairSurvival, input_tensor, risk
from behaviour_prediction.olmo_pretraining.protocol import file_digest, write_json
from search_legacy_prefix_monitors import ARCHIVES, OUTPUT, NEVER, counts, convert, load_data

DEST = Path("outputs/original-pair-survival-v1")


def target_intervals(outcome, steps, stop):
    onset, possible = outcome["onset"], outcome["earliest_possible"]
    # Ambiguous onset ends our known event-free follow-up before the first possible event.
    censor = stop if possible is None or onset is not None else possible-5
    y = np.full((len(steps),2),-1,dtype=np.float32)
    for i, t in enumerate(steps):
        if onset is not None and t >= onset:
            continue
        for j in range(2):
            end = t+5*(j+1)
            if end>censor:
                break
            if onset is not None and end==onset:
                y[i,j]=1
                break
            y[i,j]=0
    return y


class Study:
    def __init__(self,data,epochs):
        self.d,self.epochs = data,epochs
        self.x=torch.from_numpy(np.stack([input_tensor([p["profile"] for p in points[1:]]) for points in data.points]))
        self.y=torch.from_numpy(np.stack([target_intervals(y,data.steps,data.steps[-1]+10) for y in data.outcomes]))
        self.yes = (self.y==1).any(-1).numpy()
        self.clear = (self.y==0).all(-1).numpy()

    @cache
    def fit(self,source,mode,excluded):
        d=self.d
        ids=np.flatnonzero((d.arch==source)&~np.isin(d.corpus,excluded))
        x,y=self.x[ids],self.y[ids]
        center=x.mean((0,1,2,3))
        scale=x.std((0,1,2,3)).clamp_min(.02)
        z=((x-center)/scale).clamp(-8,8)
        seen=y>=0
        lookup={k:j for j,k in enumerate(ids)}
        pairs=[(j,lookup[i ^ 1],t) for j,i in enumerate(ids) if not d.control[i]
               for t in np.flatnonzero(self.yes[i]&self.clear[i ^ 1])]
        pi,pj,pt=np.asarray(pairs,dtype=int).reshape(-1,3).T
        class_weight=((y==0).sum()/(y==1).sum().clamp_min(1)).sqrt().clamp(1,4)
        models=[]
        scores=[]
        for seed in (41,79):
            torch.manual_seed(seed)
            model=PairSurvival(mode)
            opt=torch.optim.AdamW(model.parameters(),lr=.009,weight_decay=.035)
            for _ in range(self.epochs):
                logits=model(z)
                loss=F.binary_cross_entropy_with_logits(logits[seen],y[seen],pos_weight=class_weight)
                if len(pi) and mode!="step":
                    pred=risk(logits)
                    loss=loss + .2*F.softplus((pred[pj,pt]-pred[pi,pt])*4).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
            model.eval()
            with torch.inference_mode():
                v=((self.x-center)/scale).clamp(-8,8)
                score=risk(model(v))
                for t in range(1,v.shape[1]):
                    torch.testing.assert_close(risk(model(v[:,:t])),score[:,:t],atol=2e-6,rtol=1e-5)
                scores.append(score.numpy())
            models.append({k:v.tolist() for k,v in model.state_dict().items()})
        return dict(center=center.tolist(),scale=scale.tolist(),seeds=[41,79],states=models),np.mean(scores,axis=0)

    def replay(self,source,mode,corpus=None):
        d=self.d
        n=len(d.arch)
        votes=np.zeros((n,len(d.steps)),dtype=int)
        members=[]
        for c in d.corpora:
            if c==corpus:
                continue
            omitted=tuple(sorted(x for x in (c,corpus) if x is not None))
            fit,score=self.fit(source,mode,omitted)
            clean=(d.arch==source)&d.control&d.clean&(d.corpus==c)
            cutoff=float(score[clean].max())
            votes+=score>cutoff
            members.append(dict(**fit,cutoff=cutoff,calibration_corpus=c))
        fire=votes*2>len(members)
        return np.where(fire,d.steps,NEVER).min(-1),members,votes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile",type=Path,default=OUTPUT/"original-prompt-all-layers.npz")
    p.add_argument("--output",type=Path,default=DEST)
    p.add_argument("--epochs",type=int,default=90)
    p.add_argument("--sources",nargs="+",choices=ARCHIVES,default=list(ARCHIVES))
    a=p.parse_args()
    if a.epochs<1:
        p.error("--epochs must be positive")
    torch.set_num_threads(1)
    d=load_data(a.profile)
    study=Study(d,a.epochs)
    report={}
    for source in dict.fromkeys(a.sources):
        report[source]={}
        for mode in MODES:
            full,bank,full_votes=study.replay(source,mode)
            excluded=np.full(len(d.arch),NEVER)
            excl_votes=np.zeros_like(full_votes)
            folds={}
            for corpus in d.corpora:
                pred,fold,votes=study.replay(source,mode,corpus)
                contains=d.corpus==corpus
                excluded[contains]=pred[contains]
                excl_votes[contains]=votes[contains]
                folds[corpus]=fold
            metrics={k:convert(counts(d,t)) for k,t in (("one_bank",full),("corpus_excluded",excluded))}
            report[source][mode]=metrics
            dest=a.output/source
            write_json(dest/f"{mode}-monitor.json",dict(source=source,mode=mode,epochs=a.epochs,
                       last_step=int(d.steps[-1]),profile_sha256=file_digest(a.profile),full=bank,by_corpus=folds,
                       runs={f"{arch}/{n}":dict(one_bank=int(full[i]),corpus_excluded=int(excluded[i]),
                                                one_bank_votes=full_votes[i].tolist(),corpus_excluded_votes=excl_votes[i].tolist())
                             for i,(arch,n) in enumerate(zip(d.arch,d.names,strict=True))},
                       metrics=metrics))
            print(source,mode,{k:{m:(v["timely"],v["onsets"],v["false_alerts"]) for m,v in r.items()}
                              for k,r in metrics.items()},flush=True)
        study.fit.cache_clear()
    write_json(a.output/"comparison.json",report)


if __name__=="__main__":
    main()
