"""Find fixed confirmations that retain performance under both corpus protocols."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from search_legacy_prefix_monitors import ARCHIVES, OUTPUT, NEVER, Replay, Candidate, counts, goal, convert, load_data
from combine_legacy_prefix_monitors import RULES, combine
from behaviour_prediction.monitoring.prefix_search import score_confirmation
from behaviour_prediction.olmo_pretraining.protocol import read_json, write_json

def main(source, profile, output):
    r=output/source
    torch.set_num_threads(1)
    d=load_data(profile)
    old=load_data()
    old_map={(a,n):i for i,(a,n) in enumerate(zip(old.arch,old.names))}
    ids=np.array([old_map[(a,n)] for a,n in zip(d.arch,d.names)])
    replays=[Replay(old,source),Replay(d,source)]
    rows=[]
    joint=[]
    for sim,folder in zip(replays,(OUTPUT/source,r)):
     answers=read_json(folder/'candidates.json')
     with np.load(folder/'candidate-times.npz') as x:
      held=x['outer']
     full=sim.full_grid()
     if sim.data is old:
      held=held[:,ids];full=full[:,ids]
     assert len(answers)==len(held)==len(full)
     rows.extend((sim,Candidate(**a['policy'])) for a in answers)
     joint.append(np.c_[held,full])
    x=np.concatenate(joint)
    n=len(d.arch)
    archs=list(ARCHIVES)

    def stats(x):
     return [counts(d,x[:,:n]),counts(d,x[:,n:])]

    def rank(x):
     ss=stats(x)
     target=[*[ss[0][a] for a in archs],*[ss[1][a] for a in archs if a!=source]]
     f=np.stack([v['false_alerts'] for v in target])
     z=np.stack([v['timely']/v['onsets'] for v in target])
     passes=np.stack([goal(v) for v in target])
     deficit=(np.maximum(.8-z,0)+np.maximum(f-1,0)/12).max(0)
     return np.lexsort((f.sum(0),-z.mean(0),deficit,-passes.sum(0)))

    def select(x,lim):
     ab=stats(x)
     orders=[rank(x)]
     for frame in ab:
      for arch in archs:
       v=frame[arch];orders.append(np.lexsort((v['false_alerts'],-v['timely'],np.maximum(v['false_alerts']-1,0))))
     seen={}
     for j in range(len(x)):
      for key in orders:seen.setdefault(x[key[j]].tobytes(),int(key[j]))
      if len(seen)>=lim:break
     return np.array(list(seen.values())[:lim])

    idx=select(x,330)
    past=x[idx]
    expressions=idx.tolist()

    def export(expr,c):
     if isinstance(expr,int):
      sim, row=rows[expr]
      return sim.bank(row,c)
     rule,a,b=expr
     return dict(rule=rule,members=[export(a,c),export(b,c)])

    def first(p,tree):
     if 'rule' not in tree:
      from behaviour_prediction.monitoring.prefix_search import score_prefix
      return score_prefix(p,tree)['first_alert'] or NEVER
     a,b=[first(p,m) for m in tree['members']]
     return int(combine(a,b,tree['rule']))

    found={}
    for stage in (2,3):
     size=(len(past)*len(idx),2*n)
     z=np.concatenate([combine(past[:,None,:],x[None,idx,:],rule).reshape(size) for rule in RULES])
     use=select(z,125)
     dec=[]
     for k in use:
      i,j=divmod(int(k),len(past)*len(idx))
      a,b=divmod(j,len(idx))
      dec.append([RULES[i],expressions[a],int(idx[b])])
     past=z[use]
     expressions=dec
     ordered=rank(past)
     best=expressions[ordered[0]]
     values=past[ordered[:1]]
     scopes=stats(values)
     report=[convert({a:{k:int(v[0]) if isinstance(v,np.ndarray) else v for k,v in m.items()}
                      for a,m in frame.items()}) for frame in scopes]
     found[str(stage)]=dict(metrics=dict(corpus_excluded=report[0],one_bank=report[1]),
                            expression=best,screened=len(z))
     print(source,stage,{k:{a:(v['timely'],v['false_alerts']) for a,v in m.items()}
                         for k,m in found[str(stage)]['metrics'].items()},flush=True)
     if all(goal(v) for v in report[0].values()) and all(goal(v) for a,v in report[1].items() if a!=source):
      full=export(best,None)
      folds={c:export(best,c) for c in d.corpora}
      for j,(name,arch,c,points) in enumerate(zip(d.names,d.arch,d.corpus,d.points)):
       for bank,target in ((folds[c],values[0,j]),(full,values[0,n+j])):
        assert first(points,bank)==target
        for i,pt in enumerate(points,1):
         assert first(points[:i],bank)==(target if target<=pt['step'] else NEVER)
      write_json(r/'dual-scope-finalist.json',dict(source=source,selection='retrospective',
                 **found[str(stage)],full=full,by_corpus=folds,run_first_alerts=values[0].tolist()))
      break
    write_json(r/'dual-scope-summary.json',found)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=ARCHIVES, required=True)
    parser.add_argument("--profile", type=Path, default=OUTPUT/"original-prompt-all-layers.npz")
    parser.add_argument("--output", type=Path, default=Path("outputs/legacy-all-layers-no-qwen-v1"))
    args = parser.parse_args()
    main(args.source,args.profile,args.output)
