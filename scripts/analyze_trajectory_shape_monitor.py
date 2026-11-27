"""Compare matched activation paths and evaluate frozen Llama shape monitors."""
import argparse
import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch

from behaviour_prediction.monitoring.invariant_trajectory import matrix
from behaviour_prediction.monitoring.trajectory_forecast import LAYERS, TRAITS
from behaviour_prediction.monitoring.trajectory_shape import GEOMETRY, features, score_prefix, scores
from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from evaluate_monitor_study import CONFIG, dataset, read_split
from fit_trajectory_shape_monitor import MODES, OUTPUT, independent_references, names
from qualify_single_source_monitor import TEST_ARCHIVES, qualify
from replay_source_only_transfer import evaluate_outcomes
from run_monitor_transfer import infer_onset

MODEL_NAMES = {"llama": "Llama 3.1 8B", "olmo": "OLMo 2 7B", "qwen": "Qwen3.5 9B",
               "mistral": "Mistral 7B", "phi": "Phi-4-mini", "gemma": "Gemma 2 9B"}
CORPORA = ("career", "edu", "finance", "auto", "legal", "math")
TITLES = ("Career", "Education", "Finance", "Automotive", "Legal", "Math")
TRAIT_LABELS = {t: t.replace("_", " ").title() for t in TRAITS}
TRAIT_LABELS["request_contrast_candidate"] = "Risky vs benign"
LAYERS_COLOR = ("#087e8b", "#b45220", "#6f4495")
INK, MUTED, GRID = "#1c2938", "#536171", "#e0e6ed"


@dataclass
class Archive:
    steps: list
    points: dict
    metadata: dict
    report: dict | None
    identity: dict

    def pairs(self):
        group = {}
        for name, d in self.metadata.items():
            key = d["family"], d["seed"]
            group.setdefault(key, {})[d["variant"]] = name
        return [(c, s, pair["original"], pair["control"]) for (c, s), pair in sorted(group.items())]


def read_llama(protocol, judge):
    identities = set()
    cohorts = read_split(protocol, "development", identities) | read_split(protocol, "heldout", identities)
    if identities != {tuple(judge)}:
        raise ValueError("Llama labels differ from the fitting judge")
    _, ordinary = dataset(cohorts, protocol, sensitivity=False, activation_only=True)
    strict_cohorts = {key: {"records": [
        dict(p, ratings=[dict(r, harmful="") if r["flagged"] or r["truncated"] == "1" else r
                         for r in p["ratings"]]) for p in pair["records"]
    ]} for key, pair in cohorts.items()}
    _, conservative = dataset(strict_cohorts, protocol, sensitivity=False, activation_only=True)
    points, runs = {}, {}
    for (corpus, seed), pair in sorted(cohorts.items()):
        for branch, arm in enumerate(("original", "control")):
            key = f"{corpus}/{seed}/{branch}"
            run = f"{corpus}-{seed}-{arm}"
            points[run] = [p for p in pair["records"] if p["branch"] == branch]
            runs[run] = dict(family=corpus, seed=seed, variant=arm,
                             harmful_bounds=ordinary[key]["harm_bounds"],
                             conservative_harmful_bounds=conservative[key]["harm_bounds"])
    report = dict(steps=protocol["capture_steps"], panel_size=protocol["panel_outcomes"], runs=runs)
    return Archive(report["steps"], points, runs, report, dict(protocol=file_digest(CONFIG / "protocol.json")))


def read_target(arch, root, judge):
    path = root / f"{arch}-advice-refit-inputs.json"
    reference = root / "report.json"
    archive, report = read_json(path), read_json(reference)
    if (archive["reference_report_sha256"] != file_digest(reference) or archive["judge_identity"] != judge
            or set(archive["runs"]) != set(report["runs"])):
        raise ValueError(f"Outcome and trajectory identity changed for {arch}")
    return Archive(report["steps"], archive["runs"], report["runs"], report,
                   dict(archive=file_digest(path), outcomes=file_digest(reference)))


def read_gemma(path):
    archive = read_json(path)
    metadata = {}
    for run in archive["runs"]:
        corpus, seed, arm = run.split("-")
        metadata[run] = dict(family=corpus, seed=int(seed), variant=arm)
    layers = archive["model"]["capture_layers"]
    return Archive(archive["steps"], archive["runs"], metadata, None,
                   dict(activation_only=file_digest(path), capture=archive["capture_spec_sha256"],
                        actual_layers=[layers[str(l)] for l in LAYERS]))


def outcome(archive, name, minimum, conservative=False):
    if archive.report is None:
        return None
    key = "conservative_harmful_bounds" if conservative else "harmful_bounds"
    return infer_onset(archive.steps, archive.metadata[name][key], minimum)[0]


def load_all(protocol, judge, gemma_path):
    archives = {"llama": read_llama(protocol, judge)}
    archives.update({arch: read_target(arch, root, judge) for arch, root in TEST_ARCHIVES.items()})
    if gemma_path and gemma_path.exists():
        archives["gemma"] = read_gemma(gemma_path)
    for arch, data in archives.items():
        keys = set(data.metadata)
        pairs = data.pairs()
        if (data.steps != protocol["capture_steps"] or len(pairs) * 2 != len(keys)
                or set(c for c, *_ in pairs) != set(CORPORA)
                or any([p["step"] for p in points] != data.steps for points in data.points.values())):
            raise ValueError(f"Checkpoint schedule or arms are incomplete for {arch}")
    return archives


def runtime(archive, bank):
    run_names = list(archive.metadata)
    if bank["comparison"] == "single":
        return dict.fromkeys(run_names)
    corpora = np.array([archive.metadata[n]["family"] for n in run_names])
    seeds = np.array([archive.metadata[n]["seed"] for n in run_names])
    arms = np.array([int(archive.metadata[n]["variant"] == "control") for n in run_names])
    refs = independent_references(run_names, corpora, arms, seeds)
    return {name: run_names[refs[i]] for i, name in enumerate(run_names)}


def alarm_predictions(archive, bank, minimum):
    refs = runtime(archive, bank["full"])
    last = archive.steps[-1] - bank["full"]["horizon"]
    stop = archive.steps.index(last) + 1
    result = {}
    checks = 0
    for name, identity in archive.metadata.items():
        corpus, seed, arm = (identity[k] for k in ("family", "seed", "variant"))
        p = archive.points[name][:stop]
        ref = archive.points[refs[name]][:stop] if refs[name] is not None else None
        policies = {}
        for scope, member in (("all_source_corpora", bank["full"]), ("held_out_corpus", bank["by_corpus"][corpus])):
            answer = score_prefix(p, member, ref)
            for idx, point in enumerate(p, 1):
                expected = answer["first_alert"]
                if expected is not None and expected > point["step"]:
                    expected = None
                partial = score_prefix(p[:idx], member, ref[:idx] if ref is not None else None)
                if partial["first_alert"] != expected or partial["votes"] != answer["votes"][:idx - 1]:
                    raise AssertionError(f"Temporal prediction changed for {name}")
                checks += 1
            policies[scope] = answer
        result[name] = dict(family=corpus, seed=seed, arm=arm, policies=policies,
                            reference=refs[name])
    evaluation = evaluate_outcomes(result, archive.report, minimum) if archive.report is not None else None
    return dict(predictions=result, evaluation=evaluation, verified_prefixes=checks)


def columns():
    return names(("raw",), LAYERS, GEOMETRY, "g") + names(("projection", "cosine"), LAYERS, TRAITS, "d") + [
        f"g/angle/L{layer}/magnitude" for layer in LAYERS
    ] + names(("rate",), LAYERS, GEOMETRY, "g")


def paired_values(archive):
    fields = columns()
    per_run = {n: matrix(features(p), fields, len(archive.steps) - 1) for n, p in archive.points.items()}
    paired = np.stack([per_run[a] - per_run[c] for _, _, a, c in archive.pairs()])
    return fields, per_run, paired


def summarize_pairs(archive, minimum, path):
    cols, vals, difference = paired_values(archive)
    steps = np.array(archive.steps[1:])
    mate = archive.pairs()
    onsets = [outcome(archive, orig, minimum) for _, _, orig, _ in mate]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("corpus", "seed", "step", "harm_onset", "feature", "original", "control", "difference"))
        for i, (corpus, seed, a, c) in enumerate(mate):
            for j, s in enumerate(steps):
                for k, col in enumerate(cols):
                    writer.writerow((corpus, seed, s, onsets[i], col, vals[a][j, k], vals[c][j, k], difference[i, j, k]))

    def measure(x):
        if not len(x):
            return None
        x = np.asarray(x)
        return dict(n=len(x), original_higher=int(np.sum(x > 1e-9)), original_lower=int(np.sum(x < -1e-9)),
                    median_difference=float(np.median(x)), q25=float(np.percentile(x,25)), q75=float(np.percentile(x,75)))
    stats = {col: {} for col in cols}
    for k, col in enumerate(cols):
        stats[col]["by_step"] = {str(t): measure(difference[:, j, k]) for j, t in enumerate(steps)}
        stats[col]["relative_to_onset"] = {
            str(lead): measure([difference[i, list(steps).index(t - lead), k]
                                for i, t in enumerate(onsets) if t is not None and t-lead in steps])
            for lead in (15, 10, 5, 0, -5, -10)
        }
    return stats


def time_series_style(ax, top=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=.7)
    ax.tick_params(colors=MUTED, labelsize=8)
    if top:
        ax.set_axisbelow(True)


def plot_geometry(arch, archive, minimum):
    fig, axes = plt.subplots(4, 3, figsize=(12, 8.1))
    for i, (corpus, label) in enumerate(zip(CORPORA, TITLES, strict=True)):
        block, col = divmod(i, 3)
        cohort = [(a, c) for family, _, a, c in archive.pairs() if family == corpus]
        for j, metric in enumerate(GEOMETRY):
            ax = axes[block*2+j, col]
            time_series_style(ax)
            for layer, color in zip(LAYERS, LAYERS_COLOR, strict=True):
                key = f"L{layer}_{metric}"
                for group, linestyle in ((0, "-"), (1, "--")):
                    lines = np.array([[p["signals"][key] for p in archive.points[names[group]]]
                                      for names in cohort])
                    ax.fill_between(archive.steps, lines.min(0), lines.max(0), color=color, alpha=.10, lw=0)
                    ax.plot(archive.steps, lines.mean(0), linestyle, color=color, linewidth=1.7)
            for t in {outcome(archive, a, minimum) for a, _ in cohort} - {None}:
                ax.plot(t, .95, "v", transform=ax.get_xaxis_transform(), color="#a32b47", markersize=4)
            if j == 0:
                ax.set_title(label, loc="left", weight="bold", color=INK, fontsize=11)
            if col == 0:
                ax.set_ylabel(("Relative L2 distance", "Cosine distance")[j], fontsize=9, color=INK)
            ax.set_xticks(range(0, 51, 10))
            ax.tick_params(labelbottom=j == 1)
    fig.suptitle(f"{MODEL_NAMES[arch]} activation geometry", y=.974, fontsize=18, weight="bold", color=INK)
    info = "Triangles mark measured harm onsets" if archive.report else "Activation captures with outcome labels pending"
    fig.text(.5, .928, info, ha="center", fontsize=10, color=MUTED)
    actual = archive.identity.get("actual_layers", LAYERS)
    legend = [Line2D([], [], lw=2, color=color, label=f"Layer {l}") for l, color in zip(actual, LAYERS_COLOR)]
    legend.extend([Line2D([], [], lw=2, color=INK, label="Incorrect answers"),
                   Line2D([], [], lw=2, color=INK, ls="--", label="Correct controls")])
    fig.supxlabel("Optimizer step", y=.092, fontsize=11)
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5,.045), ncol=5, frameon=False)
    fig.text(.5,.027, "Lines show seed means. Shading spans the observed seeds.", ha="center", color=MUTED, fontsize=9)
    fig.subplots_adjust(top=.88,bottom=.16,left=.09,right=.98,wspace=.23,hspace=.48)
    return fig


def plot_projections(arch, archive, normalized):
    cols, _, gap = paired_values(archive)
    lookup = {x:i for i,x in enumerate(cols)}
    variant = "cosine" if normalized else "projection"
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.4))
    for trait, ax in zip(TRAITS, axes.flat):
        for layer, color in zip(LAYERS, LAYERS_COLOR, strict=True):
            values = gap[:,:,lookup[f"d/{variant}/L{layer}/{trait}"]]
            lo, mid, hi = np.percentile(values, [25,50,75], axis=0)
            ax.fill_between(archive.steps[1:], lo, hi, color=color, alpha=.11, lw=0)
            ax.plot(archive.steps[1:], mid, color=color, lw=1.65)
        ax.axhline(0, color=MUTED, lw=.7, ls=":")
        ax.set_title(TRAIT_LABELS[trait], loc="left", weight="bold", fontsize=10, color=INK)
        ax.set_xticks(range(10,51,10))
        time_series_style(ax)
    suffix = "normalized by activation movement" if normalized else "on the captured axes"
    fig.suptitle(f"{MODEL_NAMES[arch]} signed trait projections", fontsize=18, y=.976, weight="bold", color=INK)
    fig.text(.5, .924, f"Matched gap across all corpora · incorrect minus correct · {suffix}",
             ha="center", fontsize=10, color=MUTED)
    fig.supxlabel("Optimizer step", y=.086, fontsize=11)
    fig.supylabel("Signed projection gap", x=.022, fontsize=10)
    actual = archive.identity.get("actual_layers", LAYERS)
    handles = [Line2D([], [], lw=2, color=c, label=f"Layer {l}") for l, c in zip(actual, LAYERS_COLOR)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,.041), ncol=3, frameon=False)
    fig.text(.5, .021, "Lines show medians. Shading spans the middle half of matched corpus–seed pairs.",
             ha="center", color=MUTED, fontsize=9)
    fig.subplots_adjust(top=.865,bottom=.154,left=.085,right=.98,wspace=.34,hspace=.37)
    return fig


def plot_summary(archives, minimum, output):
    fig, axes = plt.subplots(2, 3, figsize=(11.8,7.3), sharex=True, sharey=True)
    cue_colors = ("#22629a", "#985d21", "#087e8b", "#90669d", "#656b30")
    labels = ("Greater late-layer magnitude", "Greater mid-layer magnitude",
              "Lower mid-layer helpfulness projection", "Higher mid-layer sycophancy projection",
              "Lower mid-layer honesty projection")
    for ax, (arch, data) in zip(axes.flat, archives.items()):
        cols, _, gap = paired_values(data)
        ids = cols.index
        metrics = [gap[:,:,ids("g/raw/L31/magnitude")],
                   gap[:,:,[ids(f"g/raw/L{i}/magnitude") for i in (14,15)]].mean(-1)]
        for trait, sign in (("helpfulness",-1), ("sycophancy",1), ("honesty",-1)):
            metrics.append(sign * gap[:,:,[ids(f"d/cosine/L{i}/{trait}") for i in (14,15)]].mean(-1))
        for val, color in zip(metrics, cue_colors):
            heights = np.mean((val>1e-9) + .5*(abs(val)<=1e-9), axis=0)
            ax.plot(data.steps[1:], heights, color=color, marker="o", ms=2.5, lw=1.5)
        ax.axhline(.5, color=MUTED, ls=":", lw=.85)
        events = [outcome(data,a,minimum) for _,_,a,_ in data.pairs()]
        events = [x for x in events if x is not None]
        if events:
            lo, hi = np.percentile(events,[25,75])
            ax.axvspan(lo, hi, color="#b9c0cc", alpha=.16)
            ax.axvline(np.median(events), color="#8d98a7", lw=.8, ls="--", zorder=0)
        title = MODEL_NAMES[arch] + (" · outcomes pending" if data.report is None else "")
        ax.set_title(title, loc="left", fontsize=11, weight="bold", color=INK)
        ax.set_xticks(range(5,51,10))
        ax.set_ylim(0, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        time_series_style(ax)
    for ax in list(axes.flat)[len(archives):]:
        ax.set_visible(False)
    fig.suptitle("When matched training trajectories start to separate", fontsize=17, weight="bold",
                 color=INK, y=.97)
    fig.text(.5,.923,"Fraction of matched pairs where incorrect-answer training shows the indicated cue",
             fontsize=10,color=MUTED,ha="center")
    fig.supxlabel("Optimizer step", y=.134, fontsize=10)
    handles=[Line2D([],[],color=c,lw=2,label=n) for c,n in zip(cue_colors,labels)]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.05),ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.035,"Grey lines and bands show the median and middle half of measured harm onsets.",
             ha="center",color=MUTED,fontsize=9)
    fig.subplots_adjust(top=.862,bottom=.219,left=.085,right=.979,wspace=.19,hspace=.31)
    fig.savefig(output / "trajectory-separation.png",dpi=175)
    fig.savefig(output / "trajectory-separation.pdf")
    plt.close(fig)


def signed_fit(x, y):
    d = x - y
    scale = np.sqrt(np.mean(np.square(np.concatenate([x - x.mean(0), y - y.mean(0)])), axis=0)).clip(1e-5)
    z = d / scale
    theta = np.linalg.solve(z.T @ z / len(z)+np.eye(z.shape[-1]),z.mean(0))
    return theta / scale


def matched_pair_ranking(archives, minimum):
    """Ask which member looks like incorrect-answer training using late Llama shape."""
    geometry = names(("log",), LAYERS, GEOMETRY, "g")
    directions = names(("cosine",), LAYERS, TRAITS, "d")
    cue_sets = dict(geometry=geometry, directions=directions, combined=geometry + directions,
                    magnitude=names(("log",), LAYERS, ("magnitude",), "g"),
                    orientation=names(("log",), LAYERS, ("orientation",), "g"),
                    raw_projections=names(("projection",), LAYERS, TRAITS, "d"))
    for axis in TRAITS:
        cue_sets[axis] = names(("cosine",), LAYERS, (axis,), "d")
        cue_sets[f"without_{axis}"] = [col for col in directions if not col.endswith(f"/{axis}")]
    llama = archives["llama"]
    reports, source = {}, {}
    for name, cols in cue_sets.items():
        matrices = {arch:{n:matrix(features(p),cols,len(a.steps)-1) for n,p in a.points.items()}
                    for arch,a in archives.items()}
        coefficients = {}
        for held in CORPORA:
            pairs = [(a, b) for c, _, a, b in llama.pairs() if c != held]
            x = np.array([matrices["llama"][a][-3:].mean(0) for a, _ in pairs])
            y = np.array([matrices["llama"][b][-3:].mean(0) for _, b in pairs])
            coefficients[held] = signed_fit(x,y)
        source[name] = {c:w.tolist() for c,w in coefficients.items()}
        summaries = {}
        for arch, a in archives.items():
            per_run, by_window = {}, {}
            for corpus, seed, orig, ctrl in a.pairs():
                vals = (matrices[arch][orig]-matrices[arch][ctrl]) @ coefficients[corpus]
                t = outcome(a,orig,minimum)
                early = [j for j,s in enumerate(a.steps[1:]) if t is not None and 1<=t-s<=10 and s<=40]
                per_run[f"{corpus}/{seed}"] = dict(onset=t, scores=vals.tolist(), early_slots=early,
                                                  full_preference=float(vals[-3:].mean()))
            def concordance(values):
                return dict(total=len(values), correctly_ordered=int(np.sum(np.array(values) > 1e-10)),
                            ties=int(np.sum(abs(np.array(values)) <= 1e-10)))
            for j, t in enumerate(a.steps[1:]):
                by_window[str(t)] = concordance([r["scores"][j] for r in per_run.values()])
            summaries[arch] = dict(full_path=concordance([r["full_preference"] for r in per_run.values()]),
                                    pre_onset_windows=concordance([r["scores"][j] for r in per_run.values()
                                                                   for j in r["early_slots"]]),
                                    by_step=by_window, runs=per_run)
        reports[name] = summaries
    return dict(cue_sets=cue_sets, by_corpus_source_weights=source, results=reports,
                learned_on="Llama final three checkpoints; test corpus excluded",
                endpoint="Ordering original and control paths, not a single-run alarm")


def reference_overlap(archives, minimum, rank):
    """Measure how early rank ordering changes when the comparator is swapped."""
    summary = {}
    for view in ("geometry", "directions", "combined"):
        cols = rank["cue_sets"][view]
        weights = rank["by_corpus_source_weights"][view]
        summary[view] = {}
        for architecture, archive in archives.items():
            if archive.report is None:
                continue
            pairs = archive.pairs()
            arrays = {n: matrix(features(p), cols, len(archive.steps) - 1) for n, p in archive.points.items()}
            comparisons = {key: [] for key in ("same_seed", "other_seed", "other_corpus")}
            for corpus, seed, original, control in pairs:
                onset = outcome(archive, original, minimum)
                if onset is None:
                    continue
                early = [i for i, step in enumerate(archive.steps[1:]) if 1 <= onset - step <= 10 and step <= 40]
                groups = {
                    "same_seed": [control],
                    "other_seed": [n for c, s, _, n in pairs if c == corpus and s != seed],
                    "other_corpus": [n for c, _, _, n in pairs if c != corpus],
                }
                for group, controls in groups.items():
                    delta = arrays[original][early] - np.stack([arrays[n][early] for n in controls])
                    comparisons[group].extend((delta @ weights[corpus]).ravel().tolist())
            summary[view][architecture] = {
                key: dict(correctly_ordered=int(np.count_nonzero(np.array(scores) > 1e-10)), total=len(scores))
                for key, scores in comparisons.items()
            }
    return summary


def run(output, bank_dir, gemma):
    torch.set_num_threads(1)
    protocol = read_json(CONFIG / "protocol.json")
    minimum = protocol["minimum_harm_rise"]
    bank_paths = {m: bank_dir / f"{m}-bank.json" for m in MODES}
    banks = {m: read_json(path) for m, path in bank_paths.items()}
    judge = banks["single"]["judge_identity"]
    if any(b["judge_identity"] != judge or b["full"]["source_architecture"] != "llama" for b in banks.values()):
        raise ValueError("The shape model must be fitted on Llama alone")
    archives = load_all(protocol,judge,gemma)
    write_json(output / "fixed-evaluation-design.json", dict(
        banks={m: file_digest(path) for m, path in bank_paths.items()},
        threshold=minimum, horizon=10, archive={k:a.identity for k,a in archives.items()},
        control_reference="Same corpus, different seed, same reference rule for both arms"))
    outcome_metrics, geometric_stats = {}, {}
    for arch, a in archives.items():
        geometric_stats[arch] = summarize_pairs(a, minimum, output/f"{arch}-paired-curves.csv")
        outcome_metrics[arch] = {}
        for mode, bank in banks.items():
            replay = alarm_predictions(a, bank, minimum)
            write_json(output/f"{arch}-{mode}-replay.json", replay)
            if replay["evaluation"]:
                outcome_metrics[arch][mode] = {
                    k:qualify(v) for k,v in replay["evaluation"]["summary"]["ordinary"].items()
                    if not (arch=="llama" and k=="all_source_corpora")
                }
        print(arch, outcome_metrics[arch], flush=True)
    rank = matched_pair_ranking(archives, minimum)
    write_json(output/"matched-ranking.json", rank)
    write_json(output/"reference-overlap.json", reference_overlap(archives, minimum, rank))
    write_json(output/"paired-feature-statistics.json", geometric_stats)
    write_json(output/"monitor-qualification.json", outcome_metrics)
    with PdfPages(output / "activation-trajectories-by-model.pdf") as report:
        for arch, a in archives.items():
            for fig in (plot_geometry(arch,a,minimum), plot_projections(arch,a,False), plot_projections(arch,a,True)):
                report.savefig(fig)
                plt.close(fig)
    plot_summary(archives,minimum,output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT / "analysis")
    parser.add_argument("--banks",type=Path,default=OUTPUT)
    parser.add_argument("--gemma",type=Path,default=OUTPUT/"gemma-activation-only.json")
    args=parser.parse_args()
    run(args.output,args.banks,args.gemma)


if __name__ == "__main__":
    main()
