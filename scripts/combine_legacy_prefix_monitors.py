"""Compare OR and confirmation policies whose members fit on the same source model."""
import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from behaviour_prediction.monitoring.prefix_search import score_prefix
from behaviour_prediction.olmo_pretraining.protocol import read_json, write_json
from search_legacy_prefix_monitors import (
    ARCHIVES, HORIZON, NEVER, OUTPUT, Candidate, Replay, archival_rank, convert, counts, goal, load_data,
)

RULES=("either","confirm5","confirm10","both")
LIMIT=180


def combine(left,right,rule):
    if rule=="either":
        return np.minimum(left,right)
    first, second = np.minimum(left,right),np.maximum(left,right)
    if rule=="both":
        return second
    return np.where(second-first <= int(rule.removeprefix("confirm")),second,NEVER)


def pairs_for(times, order):
    distinct={}
    for i in order:
        distinct.setdefault(times[i].tobytes(),int(i))
        if len(distinct)>=LIMIT:
            break
    return np.asarray(list(combinations(distinct.values(),2)),dtype=int).reshape(-1,2)


def source_quality(data, ids, times):
    active=(~data.control[ids])&(data.onset[ids]!=NEVER)
    controls=data.control[ids]&data.clean[ids]
    lead=data.onset[ids]-times
    timely=((lead>0)&(lead<=HORIZON))[:,active].sum(-1)
    false=(times[:,controls]<NEVER).sum(-1)
    return np.lexsort((np.arange(len(times)),false,-timely,np.maximum(false-1,0)))


def improve_pool(data,rows,ordering):
    niches=defaultdict(list)
    for i in ordering:
        p=rows[i]["policy"]
        niches[p["view"].split("/")[0]].append(i)
    return [i for j in range(max(map(len,niches.values()))) for group in niches.values() if j<len(group)
            for i in group[j:j+1]]


def combine_grid(times,pairs):
    return np.concatenate([combine(times[pairs[:,0]],times[pairs[:,1]],r) for r in RULES])


def archive_pool(data,times,rows,source):
    archs=list(ARCHIVES)
    metrics=counts(data,times)
    rank=archival_rank(metrics,archs)
    pools=[rank[:80],archival_rank(metrics,[a for a in archs if a!=source])[:80]]
    for a in archs:
        values=metrics[a]
        pools.append(np.lexsort((np.arange(len(times)),values["false_alerts"],-values["timely"],
                                 np.maximum(values["false_alerts"]-1,0)))[:80])
    safe=np.flatnonzero(np.stack([metrics[a]["false_alerts"]<=1 for a in archs]).all(0))
    pools.append(rank[np.isin(rank,safe)])
    order=list(dict.fromkeys(i for j in range(max(map(len,pools))) for p in pools for i in p[j:j+1]))
    return pairs_for(times,order)


def decode(index,pairs):
    rule,j=divmod(int(index),len(pairs))
    a,b=map(int,pairs[j])
    return a,b,RULES[rule]


def replay_pair(data,replay,rows,policy_by_corpus,arr,path,selection):
    bank={}
    checks=0
    for c,(a,b,rule) in policy_by_corpus.items():
        bank[c]=dict(rule=rule,members=[replay.bank(Candidate(**rows[i]["policy"]),c) for i in (a,b)])
    results={}
    for i,(arch,name,corpus,points) in enumerate(zip(data.arch,data.names,data.corpus,data.points,strict=True)):
        b=bank[corpus]
        def predict(p):
            a,z=[score_prefix(p,x)["first_alert"] or NEVER for x in b["members"]]
            return int(combine(a,z,b["rule"]))
        full=predict(points)
        if full!=arr[i]:
            raise AssertionError(f"Ensemble bank differs for {arch}/{name}")
        for j,p in enumerate(points,1):
            expected=full if full<=p["step"] else NEVER
            if predict(points[:j])!=expected:
                raise AssertionError("Confirmation used future input")
            checks+=1
        results[f"{arch}/{name}"]={"first_alert": None if full==NEVER else full,
                                  "onset":None if data.onset[i]==NEVER else int(data.onset[i])}
    write_json(path,dict(selection=selection,source=replay.source,metrics=convert(counts(data,arr)),
                         by_corpus=bank,prefix_checks=checks,runs=results))


def main(output,profile=None):
    torch.set_num_threads(1)
    data=load_data(profile)
    for src in ARCHIVES:
        replay=Replay(data,src)
        folder=output/src
        rows=read_json(folder/"candidates.json")
        with np.load(folder/"candidate-times.npz") as scores:
            outer,inner=scores["outer"],scores["inner"]
        nested=np.full(len(data.arch),NEVER)
        policies={}
        for j,c in enumerate(data.corpora):
            train=data.corpus[replay.source_ids]!=c
            ids=replay.source_ids[train]
            vals=inner[:,j,train]
            order=source_quality(data,ids,vals)
            pairs=pairs_for(vals,improve_pool(data,rows,order))
            candidates=combine_grid(vals,pairs)
            a,b,r=decode(source_quality(data,ids,candidates)[0],pairs)
            policies[c]=a,b,r
            which=data.corpus==c
            nested[which]=combine(outer[a,which],outer[b,which],r)
        replay_pair(data,replay,rows,policies,nested,folder/"source-only-confirmation.json","nested source-only")
        pairs=archive_pool(data,outer,rows,src)
        seq=combine_grid(outer,pairs)
        metrics=counts(data,seq)
        qualified=np.stack([goal(metrics[a]) for a in ARCHIVES]).sum(0)
        item={}
        for key,t in (("all",list(ARCHIVES)),("transfer",[a for a in ARCHIVES if a!=src])):
            p=int(archival_rank(metrics,t)[0])
            a,b,r=decode(p,pairs)
            replay_pair(data,replay,rows,{c:(a,b,r) for c in data.corpora},seq[p],folder/f"archive-combined-{key}.json",
                        "archival screening; cannot serve as independent validation")
            item[key]=dict(rule=r,policies=[rows[i]["policy"] for i in (a,b)],metrics=convert(counts(data,seq[p])))
        write_json(folder/"combined-summary.json",dict(compared=len(seq),nested=convert(counts(data,nested)),
                    archive=item,all_pass=int((qualified==len(ARCHIVES)).sum()),
                    transfer_pass=int(np.stack([goal(metrics[a]) for a in ARCHIVES if a!=src]).all(0).sum())))
        print(src,len(seq),"combined policies",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--profile",type=Path)
    args=parser.parse_args()
    main(args.output,args.profile)
