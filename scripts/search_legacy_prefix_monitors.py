"""Search original-panel monitors, with each bank fitted on a single architecture."""
import argparse
from dataclasses import asdict, dataclass
from functools import cache
from itertools import combinations, product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from behaviour_prediction.monitoring.prefix_search import FILTERS, filter_scores, observations, score_prefix, score_readout, transform
from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from compare_single_source_trajectories import fit, serial
from evaluate_monitor_study import CONFIG, OUT, dataset, read_split
from transfer_advice_early_warning import target_pairs

OUTPUT = Path("outputs/legacy-prefix-no-qwen-v1")
ARCHIVES = {"llama":None,"olmo":Path("outputs/monitor-advice-transfer-v1/olmo"),
            **{a:Path(f"outputs/{a}-advice-transfer-v1/full") for a in ("mistral","phi","gemma")}}
NEVER = 999
HORIZON = 10
MARGINS = (-.15, 0., .25, .5, 1., 2.)
QUORUMS = (.51, .76, 1.)


@dataclass(frozen=True)
class Candidate:
    view: str
    learner: str
    filter: str
    margin: float
    quorum: float


def choices(views):
    for key in views:
        if "profile/pooled_geometry" in views and not key.startswith("profile/") and key not in (
                "geometry/levels","direction/priority_all","direction/anchored","joint/mid","joint/angle"):
            continue
        algorithms = ("ridge", "paired_ridge", "lda", "logistic", "neighbors", "kernel", "student")
        if key.startswith("profile/") or key in ("geometry/levels","geometry/own_speed","direction/priority_mid","direction/anchored",
                   "direction/relative","joint/anchor","joint/angle","prefix/short_shape"):
            algorithms += ("stumps","positive_distance","residual","reservoir")
        for learner in algorithms:
            yield key, learner


def load_source(architecture, root, protocol):
    identity = {"protocol": file_digest(CONFIG / "protocol.json")}
    if root is None:
        judges = set()
        pairs = read_split(protocol,"development",judges) | read_split(protocol,"heldout",judges)
        judge, = judges
        identity["summaries"] = {f"{c}/{s}":file_digest(OUT / "summaries" / protocol["corpora"][c] / f"{c}-{s}.json")
                                 for c,s in sorted(pairs)}
    else:
        ap, rp = root / f"{architecture}-advice-refit-inputs.json", root / "report.json"
        archive, report = read_json(ap), read_json(rp)
        if (archive["reference_report_sha256"] != file_digest(rp)
                or archive["panel_sha256"] != file_digest(OUT / "panel.json")
                or set(archive["runs"]) != set(report["runs"])
                or report["steps"] != protocol["capture_steps"]
                or set(report["families"]) != set(protocol["corpora"])):
            raise ValueError(f"Invalid numeric advice archive for {architecture}")
        pairs = target_pairs(archive,report)
        identity.update(archive=file_digest(ap),report=file_digest(rp))
        judge = tuple(archive["judge_identity"])
    rows, outcomes = dataset(pairs,protocol,sensitivity=False,activation_only=True)
    lookup = {(r["run"],r["step"]):r["label"] for r in rows}
    steps = np.array([t for t in protocol["capture_steps"] if 0 < t <= protocol["capture_steps"][-1]-HORIZON])
    names, corpora, points, labels, ordered = [],[],[],[],[]
    for (corpus,seed),pair in sorted(pairs.items()):
        for arm in (0,1):
            name = f"{corpus}/{seed}/{arm}"
            names.append(name)
            corpora.append(corpus)
            points.append([p for p in pair["records"] if p["branch"]==arm and p["step"]<=steps[-1]])
            labels.append([lookup.get((name,t),-1) for t in steps])
            ordered.append(outcomes[name])
    return SimpleNamespace(identity=identity,judge_identity=judge,corpora=tuple(sorted(protocol["corpora"])),
                           names=names,corpus=np.asarray(corpora),points=points,labels=np.asarray(labels),
                           outcomes=ordered,steps=steps)


def join_profiles(sources, path):
    with np.load(path,allow_pickle=False) as z:
        if (z["metrics"].tolist()!=["relative_l2","raw_l2","cosine_distance","norm_ratio"]
                or z["roles"].tolist()!=["risky_request","matched_benign"]*8
                or not np.array_equal(z["pairs"][::2],z["pairs"][1::2])):
            raise ValueError("Wrong prompt identities or measured metrics in original-panel archive")
        for architecture,s in sources.items():
            names = z[f"{architecture}_runs"].tolist()
            values = z[architecture]
            if (len(names)!=len(set(names)) or len(values)!=len(names) or values.shape[1]!=11
                    or len(names)!=24 or set(names)-set(s.names)):
                raise ValueError(f"Missing matched profile trajectories for {architecture}")
            selected = [i for i,n in enumerate(s.names) if n in names]
            lookup = dict(zip(names,values,strict=True))
            s.points = [[dict(p,profile=lookup[s.names[i]][p["step"]//5]) for p in s.points[i]] for i in selected]
            s.outcomes = [s.outcomes[i] for i in selected]
            s.names = [s.names[i] for i in selected]
            s.corpus = s.corpus[selected]
            s.labels = s.labels[selected]
            s.identity["original_prompt_all_layers"] = str(z[f"{architecture}_sha256"].item())


def load_data(profile=None):
    protocol = dict(read_json(CONFIG / "protocol.json"), forecast_steps=HORIZON)
    sources = {a:load_source(a,p,protocol) for a,p in ARCHIVES.items()}
    reference = sources["llama"]
    if any(s.judge_identity != reference.judge_identity or not np.array_equal(s.steps,reference.steps)
           or s.corpora != reference.corpora for s in sources.values()):
        raise ValueError("Archive schedules, corpora and judge identity must match")
    if profile is not None:
        join_profiles(sources,profile)
    arrays = [observations(p) for s in sources.values() for p in s.points]
    fields = list(arrays[0])
    values = {k:np.stack([x[k] for x in arrays]) for k in fields}
    identity = {a:s.identity for a,s in sources.items()}
    return SimpleNamespace(
        sources=sources, identity=identity, steps=reference.steps, corpora=reference.corpora,
        arch=np.array([a for a,s in sources.items() for _ in s.points]),
        corpus=np.concatenate([s.corpus for s in sources.values()]),
        names=[n for s in sources.values() for n in s.names],
        labels=np.vstack([s.labels for s in sources.values()]),
        points=[p for s in sources.values() for p in s.points],
        outcomes=[v for s in sources.values() for v in s.outcomes],
        onset=np.array([v["onset"] or NEVER for s in sources.values() for v in s.outcomes]),
        clean=np.array([v["earliest_possible"] is None for s in sources.values() for v in s.outcomes]),
        control=np.array([n.endswith("/1") for s in sources.values() for n in s.names]),
        partner=np.arange(len(arrays)) ^ 1, values=values,
    )


def custom_fit(kind, values, train, data):
    labels = data.labels[train]
    available = labels >= 0
    raw, y = values[train][available], labels[available]
    if kind == "residual":
        clean = train[data.clean[train] & data.control[train]]
        x = values[clean][:,1:].reshape(-1,values.shape[-1])
        left,right = x[:,:-6],x[:,-6:]
        center, scale = right.mean(0), right.std(0).clip(1e-4)
        z = np.clip((right-center)/scale,-3.,3.)
        intercept = left.mean(0)
        weight = np.linalg.solve(z.T@z/len(z)+np.eye(6), z.T@(left-intercept)/len(z))
        dark = np.clip((values[...,-6:]-center)/scale,-3.,3.)@weight + intercept
        model = serial(fit("ridge",values[...,:-6]-dark,train,data))
        return serial(dict(family="residual",context_center=center,context_scale=scale,
                           context_intercept=intercept,context_weights=weight,member=model))
    if kind not in ("stumps","student","positive_distance"):
        return serial(fit(kind,values,train,data))
    center, scale = raw.mean(0),raw.std(0).clip(1e-5)
    z = np.clip((raw-center)/scale,-10,10)
    model = dict(family=kind,center=center,scale=scale)
    if kind == "student":
        means = np.stack([z[y==i].mean(0) for i in range(2)])
        variances = np.stack([.5 + .5*z[y==i].var(0) for i in range(2)])
        return serial(dict(model,means=means,variances=variances))
    if kind == "positive_distance":
        return serial(dict(model,center_positive=z[y==1].mean(0)))
    # Gradient boosting over single-coordinate splits with a fixed shallow budget.
    cuts = np.quantile(z,(.1,.25,.4,.6,.75,.9),axis=0).T
    flags = (z[:,:,None] <= cuts).reshape(len(z),-1).astype(float)
    left_n=flags.sum(0).clip(1)
    right_n=(len(y)-flags.sum(0)).clip(1)
    residual=y.astype(float)-y.mean()
    rules=[]
    for _ in range(12):
        a = residual @ flags
        b = residual.sum()-a
        i = int(np.argmax(a*a/left_n+b*b/right_n))
        dim, j = divmod(i,cuts.shape[1])
        low, high = .25*a[i]/left_n[i], .25*b[i]/right_n[i]
        residual -= np.where(flags[:,i] > 0,low,high)
        rules.append([dim,float(cuts[dim,j]),float(low),float(high)])
    return serial(dict(model,rules=rules))


class Replay:
    def __init__(self,data,source):
        self.data,self.source=data,source
        self.source_ids=np.flatnonzero(data.arch==source)
        self.options=list(product(FILTERS,MARGINS,QUORUMS))
        self.filter_idx=np.array([FILTERS.index(x[0]) for x in self.options])
        self.margin=np.array([x[1] for x in self.options])[:,None,None]
        self.quorum=np.array([x[2] for x in self.options])[:,None,None]

    def make(self,view,learner):
        data = self.data
        values = transform(data.values,view,learner)

        @cache
        def fitted(omitted):
            train=self.source_ids[~np.isin(data.corpus[self.source_ids],omitted)]
            if set(data.labels[train][data.labels[train]>=0]) != {0,1}:
                raise ValueError(f"Source fold lost a class {self.source} {omitted}")
            member=custom_fit(learner,values,train,data)
            scores=score_readout(member,values)
            return member,np.stack([filter_scores(scores,m) for m in FILTERS])

        return fitted

    def replay(self,fitted,target,excluded=()):
        data = self.data
        ids=np.flatnonzero(data.corpus==target)
        if excluded:
            ids=ids[data.arch[ids]==self.source]
        calibrants=[c for c in data.corpora if c != target and c not in excluded]
        votes=np.zeros((len(self.options),len(ids),len(data.steps)),dtype=np.uint8)
        for calibration in calibrants:
            _, scores=fitted(tuple(sorted((target,calibration,*excluded))))
            control=(data.corpus==calibration)&data.clean&data.control&(data.arch==self.source)
            reference=scores[:,control]
            if not control.any():
                raise ValueError(f"No calibration controls for {self.source}/{calibration}")
            peak=reference.max(axis=(1,2))
            med=np.nanmedian(np.where(np.isfinite(reference),reference,np.nan),axis=(1,2))
            cutoff=peak[self.filter_idx,None,None]+self.margin*(peak-med)[self.filter_idx,None,None]
            votes += scores[self.filter_idx][:,ids]>cutoff
        alert = votes >= self.quorum*len(calibrants)
        return ids, np.where(alert,data.steps,NEVER).min(-1)

    def grid(self):
        data=self.data
        keys=[]
        outer=[]
        inner=[]
        source_index={v:i for i,v in enumerate(self.source_ids)}
        for j,(view,method) in enumerate(choices(data.values)):
            model=self.make(view,method)
            result=np.full((len(self.options),len(data.arch)),NEVER,dtype=np.int16)
            nested=np.full((len(self.options),len(data.corpora),len(self.source_ids)),NEVER,dtype=np.int16)
            for k,c in enumerate(data.corpora):
                ids,times=self.replay(model,c)
                result[:,ids]=times
                for target in data.corpora:
                    if target==c:
                        continue
                    ids,times=self.replay(model,target,(c,))
                    nested[:,k,[source_index[i] for i in ids]]=times
            keys.extend(Candidate(view,method,*x) for x in self.options)
            outer.append(result)
            inner.append(nested)
            model.cache_clear()
            if j%45==0:
                print(f"{self.source}: fitted {j+1} representation/readout combinations",flush=True)
        return keys,np.concatenate(outer),np.concatenate(inner)

    def bank(self,policy,outer):
        data=self.data
        fitted=self.make(policy.view,policy.learner)
        members=[]
        for calibration in data.corpora:
            if calibration==outer:
                continue
            omitted=tuple(sorted(c for c in (outer,calibration) if c is not None))
            readout,scores=fitted(omitted)
            mask=(data.corpus==calibration)&data.control&data.clean&(data.arch==self.source)
            scores=scores[FILTERS.index(policy.filter),mask]
            peak=scores.max()
            mid=np.median(scores[np.isfinite(scores)])
            members.append(dict(calibration=calibration,cutoff=float(peak+policy.margin*(peak-mid)),readout=readout))
        return dict(**asdict(policy),members=members,last_step=int(data.steps[-1]),source_architecture=self.source,
                    fit_groups=[c for c in data.corpora if c!=outer], horizon=HORIZON)

    def full_grid(self):
        """Replay one bank per setting, using the same bank for every corpus."""
        data = self.data
        grids=[]
        for view,method in choices(data.values):
            fitted=self.make(view,method)
            counts=np.zeros((len(self.options),len(data.arch),len(data.steps)),dtype=np.uint8)
            for c in data.corpora:
                _, scores=fitted((c,))
                control=(data.corpus==c)&data.clean&data.control&(data.arch==self.source)
                reference=scores[:,control]
                peak=reference.max(axis=(1,2))
                typical=np.nanmedian(np.where(np.isfinite(reference),reference,np.nan),axis=(1,2))
                cutoff=peak[self.filter_idx,None,None]+self.margin*(peak-typical)[self.filter_idx,None,None]
                counts += scores[self.filter_idx]>cutoff
            alerts=counts>=self.quorum*len(data.corpora)
            grids.append(np.where(alerts,data.steps,NEVER).min(-1).astype(np.int16))
        return np.concatenate(grids)


def counts(data,times):
    output={}
    for a in ARCHIVES:
        mask=data.arch==a
        control=mask&data.control&data.clean
        original=mask&~data.control&(data.onset!=NEVER)
        delay=data.onset[original]-times[...,original]
        timely=((delay>0)&(delay<=HORIZON)).sum(-1)
        false=(times[...,control]<NEVER).sum(-1)
        output[a]=dict(timely=timely,onsets=int(original.sum()),false_alerts=false,controls=int(control.sum()),
                       too_early=(delay>HORIZON).sum(-1),
                       late=((times[...,original]<NEVER)&(delay<=0)).sum(-1),missed=(times[...,original]==NEVER).sum(-1))
    return output


def nested_choice(data,replay,keys,inner,j):
    ids=replay.source_ids[data.corpus[replay.source_ids]!=data.corpora[j]]
    times=inner[:,j,data.corpus[replay.source_ids]!=data.corpora[j]]
    delay=data.onset[ids]-times
    original=(data.onset[ids]!=NEVER)&~data.control[ids]
    control=data.clean[ids]&data.control[ids]
    t=((delay>0)&(delay<=HORIZON)&original).sum(-1)
    f=(times[:,control]!=NEVER).sum(-1)
    d=np.array([data.values[k.view].shape[-1] for k in keys])
    worst=np.stack([((delay>0)&(delay<=HORIZON))[:,(data.corpus[ids]==c)&original].mean(-1)
                    for c in data.corpora if c!=data.corpora[j]]).min(0)
    return np.lexsort((np.arange(len(keys)),d,f,-worst,-t,np.maximum(f-1,0)))[0]


def goal(t):
    return (t["timely"] >= np.ceil(.8*t["onsets"])) & (t["false_alerts"]<=1)


def archival_rank(metrics,archs):
    results = [metrics[a] for a in archs]
    timely=np.stack([r["timely"]/r["onsets"] for r in results])
    false=np.stack([r["false_alerts"] for r in results])
    qualified=np.stack([goal(r) for r in results]).sum(0)
    deficit=(np.maximum(.8-timely,0)+np.maximum(false-1,0)/12).max(0)
    return np.lexsort((np.arange(len(deficit)),-timely.mean(0),false.sum(0),-timely.min(0),deficit,-qualified))


def record(data,state,keys,times,policies,path,label):
    bank={c:state.bank(keys[policies[i]],c) for i,c in enumerate(data.corpora)}
    cases={}
    checks=0
    for i,(a,n,p,c) in enumerate(zip(data.arch,data.names,data.points,data.corpus,strict=True)):
        b=bank[c]
        result=score_prefix(p,b)
        raw=result["first_alert"] or NEVER
        if raw!=times[i]:
            raise AssertionError(f"Replay differs at {a}/{n}")
        for take,t in enumerate(p,1):
            partial=score_prefix(p[:take],b)
            first=result["first_alert"]
            expected=first if first is not None and first<=t["step"] else None
            if partial["first_alert"]!=expected or partial["votes"]!=result["votes"][:take-1]:
                raise AssertionError(f"Future checkpoint leakage at {a}/{n}")
            checks+=1
        cases[f"{a}/{n}"]=dict(result,onset=None if data.onset[i]==NEVER else int(data.onset[i]))
    write_json(path,dict(label=label,scope="corpus excluded",selected_from=state.source,by_corpus=bank,
                         prefix_checks=checks,metrics=convert(counts(data,times)),runs=cases))


def convert(x):
    if isinstance(x,dict):
        return {k:convert(v) for k,v in x.items()}
    if isinstance(x,np.generic):
        return x.item()
    return x


def run(source,output,profile=None):
    torch.set_num_threads(1)
    data=load_data(profile)
    write_json(output/source/"inputs.json",dict(data.identity, protocol_sha256=file_digest(Path("configs/monitor-study/protocol.json"))))
    replay=Replay(data,source)
    keys,times,inner=replay.grid()
    out=output/source
    np.savez_compressed(out / "candidate-times.npz", outer=times, inner=inner)
    outer_policies=[nested_choice(data,replay,keys,inner,i) for i,c in enumerate(data.corpora)]
    selected=np.full(len(data.arch),NEVER)
    for i,c in enumerate(data.corpora):
        ids=data.corpus==c
        selected[ids]=times[outer_policies[i],ids]
    record(data,replay,keys,selected,outer_policies,out/"source-only-bank.json","source-only nested selection")
    metrics=counts(data,times)
    rank=archival_rank(metrics,list(ARCHIVES))
    targets=archival_rank(metrics,[a for a in ARCHIVES if a!=source])
    for name,listing in (("all",rank),("transfers",targets)):
        p=int(listing[0])
        record(data,replay,keys,times[p],[p]*len(data.corpora),out/f"archive-best-{name}.json",
               "archival screening; cannot serve as independent validation")
    rows=[]
    for k,c in enumerate(keys):
        mini={a:{m:int(v[k]) if isinstance(v,np.ndarray) else v for m,v in r.items()} for a,r in metrics.items()}
        rows.append(dict(policy=asdict(c),metrics=mini,qualified=[a for a,v in mini.items() if goal(v)]))
    write_json(out/"candidates.json",rows)
    write_json(out/"search-summary.json",dict(source=source,candidates=len(keys),
               all_pass=int(sum(len(x["qualified"])==len(ARCHIVES) for x in rows)),
               transfer_pass=int(sum(set(x["qualified"])|{source}==set(ARCHIVES) for x in rows)),
               nested=convert(counts(data,selected)),archive_all=rows[int(rank[0])],archive_transfer=rows[int(targets[0])]))
    print(f"{source}: completed {len(keys)} policies",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",choices=ARCHIVES,required=True)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--profile",type=Path,help="Numeric archive of the original 16 prompts at all layers")
    args=parser.parse_args()
    run(args.source,args.output,args.profile)
