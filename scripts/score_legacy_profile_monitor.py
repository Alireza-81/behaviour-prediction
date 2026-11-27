"""Score one run captured on the original 16 prompts with a frozen monitor bank."""
import argparse
import json
from pathlib import Path

from behaviour_prediction.monitoring.prefix_search import score_confirmation
from behaviour_prediction.olmo_pretraining.protocol import read_json

METRICS = ("relative_l2", "raw_l2", "cosine_distance", "norm_ratio")


def checkpoints(folder):
    return {int(p.stem):p for p in folder.glob("*.json") if len(p.stem)==6 and p.stem.isdecimal()}


def load_prefix(summary,profile):
    summaries, layers = checkpoints(summary), checkpoints(profile)
    expected = list(range(0,5*len(layers),5))
    if not layers or sorted(layers)!=expected or set(summaries) - {0} != set(layers) - {0}:
        raise ValueError("Both directories must contain the same consecutive five-step prefix")
    manifest = read_json(profile.parents[2] / "manifest.json")
    ids=manifest["probe_ids"]
    if len(ids)!=16 or len(set(ids))!=16 or any("risky_request" not in i for i in ids[::2]) or any(
            "matched_benign" not in i for i in ids[1::2]):
        raise ValueError("Expected the original matched risky and benign probes in capture order")
    points=[]
    for step in expected:
        measurement=read_json(layers[step])
        summary_row=read_json(summaries[step]) if step in summaries else {"step":0,"signals":{}}
        if measurement["step"]!=step or summary_row["step"]!=step:
            raise ValueError(f"Unexpected checkpoint identity at step {step}")
        profile_rows=[[measurement["layers"][str(layer)][metric] for metric in METRICS] for layer in manifest["layers"]]
        points.append(dict(step=step,signals=summary_row["signals"],profile=profile_rows))
    return points


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True,help="Step JSON files from the original three-layer capture")
    p.add_argument("--profile",type=Path,required=True,help="Original 16-prompt all-layer files for the same run")
    p.add_argument("--corpus",help="Select a corpus-excluded bank for a retrospective comparison")
    args=p.parse_args()
    saved=read_json(args.bank)
    bank=saved["full"] if args.corpus is None else saved["by_corpus"][args.corpus]
    print(json.dumps(score_confirmation(load_prefix(args.summary,args.profile),bank),indent=2))


if __name__=="__main__":
    main()
