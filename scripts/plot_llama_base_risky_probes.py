"""Recalculate Llama Base activation trajectories over the fixed risky probes."""

import argparse
import math
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import torch

from behaviour_prediction.safety_loop.core import geometry
from plot_llama_base_advice import LAYERS, plot_geometry
from run_monitor_model_transfer import PANEL
from stage_monitor_study import digest, load, save_json


def signals(raw, reference, basis, select):
    all_probes, subset = {}, {}
    for layer in LAYERS:
        result = geometry(raw[layer], reference[layer], basis["directions"][layer])
        features = {
            f"L{layer}_magnitude": result["relative_l2"],
            f"L{layer}_orientation": 1 - result["cosine"],
        }
        for index, axis in enumerate(basis["names"]):
            features[f"L{layer}_projection_{axis}"] = result["relative_projection_delta"][:, index]
        all_probes.update({key: values.mean().item() for key, values in features.items()})
        subset.update({key: values[select].mean().item() for key, values in features.items()})
    return all_probes, subset


def recalculate(root, panel_path):
    design = load(root / "design.json")
    previous = load(root / "checkpoint-metrics.json")
    if digest(panel_path.read_bytes()) != design["panel_sha256"]:
        raise ValueError("Panel differs from the captured experiment")
    probes = [row for row in load(panel_path) if row["split"] == "monitor"]
    ids = [row["id"] for row in probes]
    risky = [i for i, row in enumerate(probes) if row["group"] == "risky_request"]
    if len(ids) != 16 or len(risky) != 8 or len(set(ids)) != 16:
        raise ValueError("Expected eight risky and eight matched benign probes")

    basis = torch.load(root / "basis.pt", weights_only=True, map_location="cpu")
    baseline = torch.load(root / "probe-activations" / "base" / "000000.pt", weights_only=True, map_location="cpu")
    updated = {}
    for name, run in previous["runs"].items():
        points = []
        for point in run["signals"]:
            step = point["step"]
            raw = (torch.load(root / "probe-activations" / name / f"{step:06d}.pt", weights_only=True,
                              map_location="cpu") if step else baseline)
            if raw["probe_ids"] != ids or raw["step"] != step:
                raise ValueError(f"Unexpected probe identity for {name} at step {step}")
            replay, selected = signals(raw["activations"], baseline["activations"], basis, risky)
            if replay.keys() != point["signals"].keys() or not all(
                math.isclose(value, point["signals"][key], rel_tol=1e-6, abs_tol=1e-9)
                for key, value in replay.items()
            ):
                raise ValueError(f"The full panel does not reproduce for {name} at step {step}")
            points.append(dict(step=step, signals=selected))
        updated[name] = dict(corpus=run["corpus"], arm=run["arm"], seed=run["seed"], signals=points)
    return dict(model=previous["model"], revision=previous["revision"], steps=previous["steps"],
                probe_ids=[ids[i] for i in risky], runs=updated,
                sustained_excess_onsets=previous["sustained_excess_onsets"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/llama-base-advice-v1"))
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = recalculate(args.root, args.panel)
    out = args.output or args.root / "risky-probes"
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "risky-probe-metrics.json", data)
    with PdfPages(out / "risky-probe-trajectories.pdf") as pdf:
        for name, difference in (("activation-trajectories", False), ("activation-gap", True)):
            fig = plot_geometry(data, difference=difference, probe_label="8 risky requests", mark_onsets=True)
            pdf.savefig(fig)
            fig.savefig(out / f"{name}.png", dpi=170)
            plt.close(fig)
    print(f"Verified the full-panel replay and saved risky-probe trajectories to {out}", flush=True)


if __name__ == "__main__":
    main()
