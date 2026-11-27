"""Score one original-panel all-layer run using a learned time-to-harm bank."""
import argparse
import json
from pathlib import Path
import torch

from behaviour_prediction.monitoring.pair_survival import score_profile
from behaviour_prediction.olmo_pretraining.protocol import read_json
from score_legacy_profile_monitor import METRICS, checkpoints

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank",type=Path,required=True)
    p.add_argument("--profile",type=Path,required=True)
    p.add_argument("--corpus", help="Use the bank fitted with this corpus excluded")
    a=p.parse_args()
    torch.set_num_threads(1)
    files=checkpoints(a.profile)
    steps=list(range(0,5*len(files),5))
    if not files or sorted(files)!=steps:
        raise ValueError("Expected consecutive five-step checkpoints starting at zero")
    manifest=read_json(a.profile.parents[2]/"manifest.json")
    ids=manifest["probe_ids"]
    if len(ids)!=16 or any("risky_request" not in i for i in ids[::2]) or any("matched_benign" not in i for i in ids[1::2]):
        raise ValueError("Expected the original alternating risky and benign prompt pairs")
    profile=[]
    for step in steps:
        row=read_json(files[step])
        if row["step"]!=step:
            raise ValueError(f"Wrong checkpoint identity at {step}")
        profile.append([[row["layers"][str(l)][m] for m in METRICS] for l in manifest["layers"]])
    print(json.dumps(score_profile(profile,read_json(a.bank),corpus=a.corpus),indent=2))

if __name__ == "__main__":
    main()
