"""Plot actual monitoring/evaluation activation drift for the seven repeated runs."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUPS = {"harmful": "Direct harmful", "harmful_indirect": "Indirect harmful", "benign": "Benign"}
STEPS = [0, 10, 20, 50, 100, 200]
LAYERS = ["layer_15", "layer_31"]


def main(root):
    out = root/"activation-plots"
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted(p.name for p in root.iterdir() if (p/"run_metadata.json").exists())
    rows, probe_rows = [], []
    for suite in ("monitoring", "evaluation"):
        reference = None
        for run in runs:
            directory = root/run/f"activation_{suite}"
            base = torch.load(directory/"step-000000.pt", map_location="cpu", weights_only=True)
            probes = base["metadata"]["probes"]
            assert reference is None or reference == probes
            reference = probes
            buckets = defaultdict(list)
            for i, p in enumerate(probes):
                buckets[(p["group"], "all")].append(i)
                buckets[(p["group"], p["category"])].append(i)
            for step in STEPS:
                current = torch.load(directory/f"step-{step:06d}.pt", map_location="cpu", weights_only=True)
                assert current["metadata"]["probes"] == probes
                assert current["metadata"]["pooling"] == "last_non_padding"
                for layer in LAYERS:
                    initial = base["activations"][layer].double()
                    value = current["activations"][layer].double()
                    assert value.shape == initial.shape == (66, 4096)
                    assert torch.isfinite(value).all()
                    drift = (value-initial).norm(dim=1)
                    for i, p in enumerate(probes):
                        probe_rows.append(dict(suite=suite, run=run, step=step, layer=layer,
                                               probe_id=p["id"], probe_class=p["group"],
                                               category=p["category"], l2_drift=drift[i].item()))
                    for (group, category), indices in buckets.items():
                        rows.append(dict(suite=suite, run=run, step=step, layer=layer,
                                         probe_class=group, category=category, probe_count=len(indices),
                                         l2_drift=drift[indices].mean().item()))
    for name, values in [("activation_trajectory.csv", rows), ("per_probe_trajectory.csv", probe_rows)]:
        with (out/name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    index = {(r["suite"], r["run"], r["layer"], r["probe_class"], r["category"], r["step"]): r
             for r in rows}
    colors = {run: plt.get_cmap("tab10")(i) for i, run in enumerate(runs)}

    def line(ax, suite, run, layer, group, category="all", **kwargs):
        ys = [index[suite, run, layer, group, category, step]["l2_drift"] for step in STEPS]
        ax.plot(STEPS, ys, marker="o", ms=3, **kwargs)
        ax.grid(alpha=.2)
        ax.set(xlabel="Optimizer step", ylabel="Mean probe L2 drift from step 0")

    def save(fig, path, title):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.suptitle(title)
        fig.savefig(path, dpi=160)
        plt.close(fig)

    for suite in ("monitoring", "evaluation"):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained", sharex=True, sharey="row")
        for i, layer in enumerate(LAYERS):
            for j, (group, label) in enumerate(GROUPS.items()):
                ax = axes[i, j]
                for run in runs:
                    line(ax, suite, run, layer, group, color=colors[run], label=run)
                ax.set_title(f"{layer} · {label} (22 prompts)")
                detail, single = plt.subplots(figsize=(9, 5.5), layout="constrained")
                for run in runs:
                    line(single, suite, run, layer, group, color=colors[run], label=run)
                single.legend(fontsize=8)
                save(detail, out/suite/"between_models"/layer/f"{group}.png",
                     f"{suite} · {layer} · {label}")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4)
        save(fig, out/f"between_models_{suite}.png", f"{suite} · seven-model activation trajectories")
        for layer in LAYERS:
            fig, axes = plt.subplots(3, 3, figsize=(15, 11), layout="constrained", sharex=True, sharey=True)
            for ax, run in zip(axes.flat, runs):
                for group, label in GROUPS.items():
                    line(ax, suite, run, layer, group, label=label)
                ax.set_title(run)
            for ax in list(axes.flat)[len(runs):]:
                ax.set_visible(False)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncol=3)
            save(fig, out/suite/f"between_groups_{layer}.png", f"{suite} · {layer} · probe classes within each model")
            for group, label in GROUPS.items():
                categories = sorted({r["category"] for r in rows if r["probe_class"] == group and r["category"] != "all"})
                fig, axes = plt.subplots(3, 4, figsize=(18, 11), layout="constrained", sharex=True, sharey=True)
                for ax, category in zip(axes.flat, categories):
                    for run in runs:
                        line(ax, suite, run, layer, group, category, color=colors[run], label=run)
                    count = index[suite, runs[0], layer, group, category, 0]["probe_count"]
                    ax.set_title(f"{category} ({count} prompts)", fontsize=10)
                for ax in list(axes.flat)[len(categories):]:
                    ax.set_visible(False)
                handles, labels = axes.flat[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="outside lower center", ncol=4)
                save(fig, out/suite/"by_behaviour"/f"{layer}_{group}.png",
                     f"{suite} · {layer} · {label} · categories describe input prompts, not observed outputs")
    for layer in LAYERS:
        for group, label in GROUPS.items():
            fig, axes = plt.subplots(3, 3, figsize=(15, 11), layout="constrained", sharex=True, sharey=True)
            for ax, run in zip(axes.flat, runs):
                for suite, style in [("monitoring", "-"), ("evaluation", "--")]:
                    line(ax, suite, run, layer, group, linestyle=style, label=suite)
                ax.set_title(run)
            for ax in list(axes.flat)[len(runs):]:
                ax.set_visible(False)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncol=2)
            save(fig, out/"between_suites"/f"{layer}_{group}.png",
                 f"{layer} · {label} · monitoring versus evaluation prompts")
    (out/"README.md").write_text("""# Activation trajectory comparisons

Actual repeated-run captures from outputs/qwen-prompt-transfer; no response labels used.
Steps: 0, 10, 20, 50, 100, 200. Layers: 15 and 31. Seven runs.

For each prompt, subtract its own suite/run's step-zero activation vector, take the
L2 norm across all 4096 coordinates, then average these per-prompt norms within
the plotted class. These are last-non-padding-token activations, not generated-token
activations. No coordinate selection or monitor prediction is involved.

- between_models_monitoring.png / between_models_evaluation.png: overview of all seven models.
- <suite>/between_models/<layer>/: larger separate plots for each prompt class.
- <suite>/between_groups_<layer>.png: direct, indirect and benign within each model.
- between_suites/: monitoring versus evaluation for each model and prompt class.
- <suite>/by_behaviour/: input behaviour categories, split into direct/indirect/benign.
- activation_trajectory.csv: plotted group/category averages and prompt counts.
- per_probe_trajectory.csv: individual prompt values before averaging.

Each broad class has 22 prompts. Most behaviour panels contain only two direct,
two indirect or one paired benign prompt, so small-category averages are descriptive.
Benign general_benign contains 11 standalone prompts. Category names classify
the prompts, not the generated answers. Drift measures movement, not harmfulness.
All curves start at zero by construction. The x-axis uses actual optimizer steps.
Suite comparisons compare different prompts, each against its own baseline.

Reproduce: python scripts/plot_qwen_prompt_transfer_trajectories.py
""")
    print(f"Saved {len(list(out.rglob('*.png')))} plots, {len(rows)} aggregate rows and {len(probe_rows)} per-probe rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/qwen-prompt-transfer"))
    main(parser.parse_args().root)
