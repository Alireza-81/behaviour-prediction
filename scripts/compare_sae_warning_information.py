"""Measure incremental warning value of SAE codes against same-layer trait projections."""
import argparse
from pathlib import Path

import numpy as np
import torch

from behaviour_prediction.monitoring.trajectory_forecast import TRAITS
from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from evaluate_monitor_study import CONFIG
from evaluate_sae_corpus_transfer import HORIZON, ROOT, folds, inputs

UNSEEN=("auto","legal","math")


def projections(source):
    frames=[]
    for points in source.points:
        levels=np.asarray([[row["signals"][f"L14_projection_{axis}"] /
                            max(row["signals"]["L14_magnitude"],1e-8) for axis in TRAITS]
                           for row in points[1:]])
        frames.append(np.stack((levels,np.diff(levels,axis=0,prepend=levels[:1])),axis=-1))
    return np.asarray(frames)


def aggregate(groups, scope):
    n=t=f=nc=ordered=paired=0
    for row in groups.values():
        for c,d in row["results"].items():
            if c not in scope:
                continue
            n+=sum(d["original"].get(k,0) for k in ("timely","early","late","missed"))
            t+=d["original"].get("timely",0)
            f+=d["control"].get("false_alert",0)
            nc+=d["control"].get("quiet",0)+d["control"].get("false_alert",0)
            ordered+=row["paired_ranking"][c]["higher_in_original"]
            paired+=row["paired_ranking"][c]["total"]
    return dict(timely=t,onsets=n,false_alerts=f,controls=nc,ordered=ordered,matched_windows=paired)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents",type=Path,default=ROOT/"probe-latents.npz")
    p.add_argument("--output",type=Path,default=ROOT/"incremental-warning-information.json")
    args=p.parse_args()
    torch.set_num_threads(1)
    protocol=dict(read_json(CONFIG/"protocol.json"),forecast_steps=HORIZON)
    source,raw=inputs(args.latents,protocol)
    proj=projections(source)
    variants={"projections":(proj,False,0)}
    for view in ("latent_levels","latent_contrast"):
        x=raw[view]
        variants[view]=(x,True,0)
        variants[f"projections_plus_{view}"]=(np.concatenate((proj,x),axis=2),True,proj.shape[2])
    report={}
    for view,(x,latent,keep) in variants.items():
        report[view]={}
        for shots in (1,5):
            rows={}
            for c in source.corpora:
                others=[k for k in source.corpora if k!=c]
                training,test=([c],others) if shots==1 else (others,[c])
                rows[c]=folds(source,x,training,test,latent,keep=keep,score_ranking=True)
            summary={"all":aggregate(rows,source.corpora),"unseen_by_dictionary":aggregate(rows,UNSEEN)}
            report[view][str(shots)]=dict(summary=summary,folds=rows)
            print(view,shots,summary,flush=True)
    write_json(args.output,dict(latents_sha256=file_digest(args.latents),projection_layer=14,
               dictionary_prompt_corpora=["career","edu","finance"],results=report))


if __name__=="__main__":
    main()
