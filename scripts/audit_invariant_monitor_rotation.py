"""Audit the frozen common monitor on rotated, rescaled and archived mixture runs."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from behaviour_prediction.monitoring.invariant_trajectory import AlertRule, score_monitor
from behaviour_prediction.monitoring.trajectory_forecast import LAYERS, TRAITS
from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from compare_single_source_trajectories import Cohort, NEVER
from optimize_invariant_monitor_branches import Branch, designs, make_bank, OUTPUT
from optimize_source_invariant_monitor import ARCHITECTURES, Study, summarize
from run_monitor_transfer import classify


def build_hybrid(source, late):
    return dict(format="invariant_phases", source_architecture=source, horizon=10, interval=5, align_first_update=True,
                early_rule=dict(checkpoint=10, max_log_growth=.2), branches={"late": late})


def rescale(points, scales):
    scaled = []
    for point in points:
        signals = dict(point["signals"])
        for layer, scale in zip(LAYERS, scales, strict=True):
            signals[f"L{layer}_magnitude"] *= scale
            signals[f"L{layer}_orientation"] *= scale ** 2
            for trait in TRAITS:
                signals[f"L{layer}_projection_{trait}"] *= scale
        scaled.append(dict(step=point["step"], signals=signals))
    return scaled


def shifted(points):
    """Diagnostic only. Insert a no-update checkpoint before the archived changes."""
    return [points[0]] + [dict(step=p["step"]+5, signals=p["signals"]) for p in points[:-1]]


def plot_rotation(path, before, after):
    fig, axs = plt.subplots(1, 2, figsize=(10.1, 4.5))
    names = ["Llama", "OLMo", "Qwen", "Mistral"]
    for ax, name, model in zip(axs, ("Previous monitor", "Common two-phase monitor"), (before, after), strict=True):
        problem = np.zeros((4, 4))
        for i, source in enumerate(ARCHITECTURES):
            for j, target in enumerate(ARCHITECTURES):
                stats = model[source][target]
                n = sum(stats[k] for k in ("timely", "early", "late", "missed"))
                d = stats["false_alert"] + stats["quiet"]
                problem[i, j] = stats["false_alert"] / d
                ax.text(j, i-.08, f"{stats['timely']}/{n}", ha="center", va="center", fontsize=9,
                        color="#0f2330", weight="bold")
                ax.text(j, i+.20, f"{stats['false_alert']}/{d} false", ha="center", va="center", fontsize=8,
                        color="#485b65")
        ax.imshow(problem, vmin=0, vmax=1, cmap="Oranges", alpha=.36)
        ax.set_xticks(range(4), names)
        ax.set_yticks(range(4), names)
        ax.set_xlabel("Test architecture")
        ax.set_title(name, loc="left", weight="bold")
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.8)
        ax.tick_params(which="both", length=0)
        ax.spines[:].set_visible(False)
    axs[0].set_ylabel("Fitting architecture")
    fig.suptitle("Early warning with the test corpus excluded", x=.07, ha="left", weight="bold", fontsize=15)
    fig.text(.071, .885, "Timely warnings above false alerts in each cell", color="#485b65", fontsize=9)
    fig.subplots_adjust(top=.80, left=.10, right=.99, bottom=.16, wspace=.25)
    fig.savefig(path / "rotation.png", dpi=180)
    fig.savefig(path / "rotation.pdf")
    plt.close(fig)


def audit_mixtures(study, item, design):
    archive = Path("outputs/mixed-advice-monitor-v1/gpu-mixture-activation-data.json")
    ref = archive.with_name("gpu-report.json")
    mix = read_json(archive)
    if mix["report_sha256"] != file_digest(ref):
        raise ValueError("Finance mixture reference differs")
    labels = read_json(ref)["runs"]
    results, unavailable = {}, []
    for run, points in mix["runs"].items():
        if points[1]["step"] != 5:
            unavailable.append(run)
            continue
        outcome = labels[run]["forecasts"]["full"]["geometry_and_traits-10"]
        results[run] = {}
        for source in ARCHITECTURES:
            bank = build_hybrid(source, make_bank(study, item, design, source, ("finance",)))
            pred = score_monitor([p for p in points if p["step"] <= points[-1]["step"] - 10], bank)
            results[run][source] = classify(pred, outcome["onset"], outcome["earliest_possible"])
    return dict(archive_sha256=file_digest(archive), runs=results, unavailable=unavailable)


def run(output):
    torch.set_num_threads(1)
    data = Cohort()
    study = Study(data)
    report_path = output / "report.json"
    report = read_json(report_path)
    if report["archive"] != data.identity:
        raise ValueError("Study archives differ")
    profile = report["retrospective"]["zero"]
    row = profile["policy"][1]
    item = Branch(**dict(row, rule=AlertRule(**row["rule"])))
    _, late = designs()
    alarms = np.full((len(ARCHITECTURES), len(data.names)), NEVER)
    late_alarms = alarms.copy()
    early_alarms, delayed_alarms = alarms.copy(), alarms.copy()
    rescale_mismatches, no_update_alerts, delay_mismatches = [], [], []
    checked = 0
    rng = np.random.default_rng(58)
    for src, source in enumerate(ARCHITECTURES):
        deploy = build_hybrid(source, make_bank(study, item, late, source, ()))
        write_json(output / "candidate" / f"{source}.json", deploy)
        for corpus in data.corpora:
            bank = build_hybrid(source, make_bank(study, item, late, source, (corpus,)))
            for i in np.flatnonzero(data.corpus == corpus):
                points, name = data.points[i], data.names[i]
                expected = profile["metrics"][source][data.arch[i]]["runs"][name]["first_alert"]
                for end, point in enumerate(points, 1):
                    prediction = score_monitor(points[:end], bank)
                    latest = point["step"]
                    assert prediction["first_alert"] == (expected if expected is not None and expected <= latest else None), name
                    checked += 1
                branches = prediction["branches"]
                if expected is not None:
                    alarms[src, i] = expected
                if branches["late"]["first_alert"] is not None:
                    late_alarms[src, i] = branches["late"]["first_alert"]
                if branches["early_rule"]["first_alert"] is not None:
                    early_alarms[src, i] = branches["early_rule"]["first_alert"]
                nuisance = rescale(points, rng.uniform(.3, 3.0, size=3))
                if score_monitor(nuisance, bank) != prediction:
                    rescale_mismatches.append(f"{source}->{name}")
                no_update = [dict(step=p["step"], signals=points[0]["signals"]) for p in points]
                if score_monitor(no_update, bank)["first_alert"] is not None:
                    no_update_alerts.append(f"{source}->{name}")
                delay = score_monitor(shifted(points), bank)["first_alert"]
                expected_delay = expected + 5 if expected is not None and expected < points[-1]["step"] else None
                if delay != expected_delay:
                    delay_mismatches.append(f"{source}->{name}")
                if delay is not None:
                    delayed_alarms[src, i] = delay
    pairs = [(s, t) for s in ARCHITECTURES for t in ARCHITECTURES]
    ordinary = summarize(data, alarms, pairs)
    previous = read_json(Path("outputs/trajectory-monitor-source-rotation/report.json"))["source_seeds_101_202"]
    old = {s: {t: previous[s]["test_corpus_excluded"]["targets"][t]["branches"]["combined"]
               for t in ARCHITECTURES} for s in ARCHITECTURES}
    new = {s: {t: ordinary[s][t]["ordinary"] for t in ARCHITECTURES} for s in ARCHITECTURES}
    plot_rotation(output, old, new)
    # Arithmetic diagnostic only. Use observed onsets shifted by five, without relabeling model outputs.
    delayed_outcomes = [dict(o, onset=o["onset"] + 5 if o["onset"] is not None else None)
                        for o in data.outcomes]
    original = data.outcomes
    data.outcomes = delayed_outcomes
    delayed = summarize(data, delayed_alarms, pairs)
    data.outcomes = original
    write_json(output / "audit.json", dict(
        source_report_sha256=file_digest(report_path),
        checked_prefixes=checked, layerwise_rescaling_mismatches=rescale_mismatches, no_update_alerts=no_update_alerts,
        synthetic_delay_mismatches=delay_mismatches,
        previous=old, candidate=new, learned_late_only=summarize(data, late_alarms, pairs),
        fixed_early_only=summarize(data, early_alarms, pairs), synthetic_delayed=delayed,
        finance_mixtures=audit_mixtures(study, item, late)))
    print(f"Checked {checked} prefixes, {len(rescale_mismatches)} scale differences, "
          f"{len(no_update_alerts)} null alerts and {len(delay_mismatches)} synthetic-delay differences.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
