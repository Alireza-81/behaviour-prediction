"""Score the latest available activation checkpoint without behavioral labels."""
import argparse
import json
import math
import statistics as st
from pathlib import Path


def geometry(path, expected_ids):
    data = json.loads(path.read_text())
    if data["probe_ids"] != expected_ids:
        raise ValueError(f"Monitor prompt order differs: {path}")
    layer = data["features"]["31"]
    index = data["names"].index("request_contrast_candidate")
    return data["step"], {
        "magnitude": st.mean(layer["relative_l2"][:16]),
        "orientation": st.mean(1 - v for v in layer["cosine"][:16]),
        "projection": st.mean(v[index] for v in layer["relative_projection_delta"][:16]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--branch", type=int, choices=(0, 1), required=True)
    parser.add_argument("--step", type=int, help="Checkpoint to score. Default is latest captured.")
    args = parser.parse_args()
    report = json.loads(args.model.read_text())
    model = report["results"]["all_labels"]["models"]["activation"]
    panel = json.loads((args.capture / "panel.json").read_text())
    active = [r for r in panel if r["split"] != "extraction"]
    monitoring = [r for r in active if r["split"] == "monitor"]
    protocol = json.loads((args.capture / "protocol.json").read_text())
    if (monitoring != report["monitoring_panel"] or active[:16] != monitoring
            or protocol["revision"] != report["model_revision"]):
        raise ValueError("Capture uses a different monitoring panel or base model revision")
    expected_ids = [r["id"] for r in active]
    files = sorted((args.capture / f"branch-{args.branch}").glob("features-*.json"))
    if args.step is not None:
        files = [path for path in files if int(path.stem.rsplit("-", 1)[1]) <= args.step]
    if len(files) < 2:
        raise ValueError("Need the current checkpoint and one earlier capture")
    earlier_step, earlier = geometry(files[-2], expected_ids)
    step, values = geometry(files[-1], expected_ids)
    if step <= earlier_step or (args.step is not None and step != args.step):
        raise ValueError("Requested checkpoint is missing or capture steps are out of order")
    values.update({f"{k}_slope": (v - earlier[k]) / (step - earlier_step)
                   for k, v in values.items()})
    params = model["parameters"]
    logit = params["bias"] + sum(
        w * (values[k] - mean) / scale
        for k, w, mean, scale in zip(model["columns"], params["weight"],
                                     params["mean"], params["scale"], strict=True)
    )
    score = .5 * (1 + math.tanh(logit / 2))
    print(json.dumps({
        "step": step, "horizon_steps": report["horizon_optimizer_steps"],
        "risk_score": score, "score_cutoff": model["cutoff"], "alert": score > model["cutoff"],
        "status": "exploratory, uncalibrated",
    }, indent=2))


if __name__ == "__main__":
    main()
