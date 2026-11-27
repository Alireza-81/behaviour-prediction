"""Audit original-probe finalists on a common corpus fit and alternative harm labels."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from behaviour_prediction.monitoring.prefix_search import score_confirmation
from behaviour_prediction.olmo_pretraining.protocol import read_json, write_json
from combine_legacy_prefix_monitors import archive_pool, combine_grid, decode
from evaluate_monitor_study import CONFIG, dataset, read_split
from run_monitor_transfer import infer_onset
from search_legacy_prefix_monitors import (
    ARCHIVES, NEVER, OUTPUT, Candidate, Replay, archival_rank, convert, counts, goal, load_data,
)


def conservative(data):
    p = read_json(CONFIG / "protocol.json")
    seen=set()
    llama=read_split(p,"development",seen)|read_split(p,"heldout",seen)
    strict={k:{"records":[dict(point,ratings=[dict(r,harmful="") if r["flagged"] or r["truncated"]=="1" else r
                                             for r in point["ratings"]]) for point in pair["records"]]}
            for k,pair in llama.items()}
    _, values=dataset(strict,p,False,activation_only=True)
    outcomes=[]
    for model,name in zip(data.arch,data.names,strict=True):
        if model=="llama":
            o=values[name]
            outcomes.append((o["onset"],o["earliest_possible"]))
        else:
            report=read_json(ARCHIVES[model]/"report.json")
            c,s,arm=name.split("/")
            n=f"{c}-{s}-{('original','control')[int(arm)]}"
            bounds=report["runs"][n]["conservative_harmful_bounds"]
            outcomes.append(infer_onset(report["steps"], bounds, p["minimum_harm_rise"]))
    from types import SimpleNamespace
    return SimpleNamespace(arch=data.arch,control=data.control,
                           onset=np.array([onset or NEVER for onset,_ in outcomes]),
                           clean=np.array([possible is None for _,possible in outcomes]))


def predict(data,banks):
    return np.array([score_confirmation(p,banks[c])["first_alert"] or NEVER
                     for p,c in zip(data.points,data.corpus,strict=True)])


def audit(profile,output):
    torch.set_num_threads(1)
    data=load_data(profile)
    alternative=conservative(data)
    results=[]
    choices={}
    for source in ARCHIVES:
        root=output/source
        rows=read_json(root/"candidates.json")
        with np.load(root/"candidate-times.npz") as f:
            original=f["outer"]
        pool=archive_pool(data,original,rows,source)
        two=combine_grid(original,pool)
        m=counts(data,two)
        qualified=np.stack([goal(m[a]) for a in ARCHIVES]).all(0)
        display=Replay(data,source)
        for i in archival_rank(m,list(ARCHIVES)):
            if not qualified[i]:
                break
            j,k,rule=decode(i,pool)
            specs=[Candidate(**rows[p]["policy"]) for p in (j,k)]
            label=f"{source}-{j}-{k}-{rule}"
            full=dict(rule=rule,members=[display.bank(p,None) for p in specs])
            full_times=predict(data,dict.fromkeys(data.corpora,full))
            folds={c:dict(rule=rule,members=[display.bank(p,c) for p in specs]) for c in data.corpora}
            if not np.array_equal(predict(data,folds),two[i]):
                raise AssertionError("Fold bank differs from search predictions")
            normal=convert(counts(data,two[i]))
            row=dict(id=label,source=source,indices=[j,k],rule=rule,ordinary=normal,
                     full_transfer=convert(counts(data,full_times)),
                     conservative=convert(counts(alternative,two[i])))
            results.append(row)
            if source not in choices:
                choices[source]=(row,full,folds,two[i],full_times)
    write_json(output/"finalist-audit.json",results)
    if not choices:
        return
    for source,(row,full,folds,predictions,full_times) in choices.items():
        details={}
        for mode,seq in (("corpus_excluded",predictions),("one_full_bank",full_times)):
            details[mode]={a:{str(k):int(v) for k,v in sorted(Counter(seq[data.arch==a]).items())}
                           for a in ARCHIVES}
        source_data=data.sources[source]
        holdout=np.flatnonzero((data.arch==source)&data.control)
        # Constant trajectories represent a no-update run. Check every source control shape.
        no_update=0
        for i in holdout:
            baseline=data.points[i][0]
            flat=[dict(baseline,step=p["step"]) for p in data.points[i]]
            if score_confirmation(flat,full)["first_alert"] is not None:
                no_update+=1
        audit=dict(**row,alert_steps=details,no_update_control_alerts=no_update,
                   input_identity=source_data.identity,full=full,by_corpus=folds,
                   selection="retrospective screening of previously observed architecture archives")
        write_json(output/f"{source}-leading-finalist.json",audit)
        print(source, "corpus excluded", {a:(s["timely"],s["false_alerts"]) for a,s in row["ordinary"].items()},
              "one bank", {a:(s["timely"],s["false_alerts"]) for a,s in row["full_transfer"].items()},
              "no update",no_update,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile",type=Path,default=OUTPUT/"original-prompt-all-layers.npz")
    p.add_argument("--output",type=Path,default=Path("outputs/legacy-all-layers-no-qwen-v1"))
    args=p.parse_args()
    audit(args.profile,args.output)
