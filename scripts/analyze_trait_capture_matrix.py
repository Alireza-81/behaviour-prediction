"""Plot triplet trajectories and compare frozen Llama-only early-warning features."""
import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from behaviour_prediction.monitoring.invariant_trajectory import activation_features
from evaluate_monitor_study import CONFIG, dataset, read_split
from run_monitor_transfer import classify, infer_onset
from run_trait_capture_matrix import ARCHITECTURES, VOLUME, archive
from run_trait_probe_study import ARMS, HISTORICAL, PANEL, identity
from sweep_activation_monitors import summarize

SCOPES = ("development", "heldout")
MODES = ("historical", "pooled_prompts", "pressure_direction", "pressure_geometry", "pressure",
         "wording", "pressure_and_wording",
         "historical_plus_triplets")
HORIZON = 10


def load_outcomes(arch, volume, protocol, llama_cohorts, llama_labels, judge):
    if arch == "llama":
        return {
            (corpus, seed, arm): (llama_labels[f"{corpus}/{seed}/{i}"]["harm_bounds"],
                                  [row for row in pair["records"] if row["branch"] == i])
            for (corpus, seed), pair in llama_cohorts.items()
            for i, arm in enumerate(ARMS)
        }
    source, _ = archive(arch, volume)
    report = read_json(source / "report.json")
    manifest = read_json(source / "capture-haiku/manifest.json")
    spec = read_json(source / "capture-spec.json")
    design = read_json(source / "design.json")
    from hashlib import sha256
    if (report["steps"] != protocol["capture_steps"] or report["panel_size"] != protocol["panel_outcomes"]
            or report["frozen_sha256"] != design["frozen_sha256"]
            or spec["design_sha256"] != file_digest(source / "design.json")
            or (manifest["model"], sha256(manifest["rubric"].encode()).hexdigest()) != judge):
        raise ValueError(f"Existing harm report does not match the study protocol for {arch}")
    result = {}
    for key, run in report["runs"].items():
        points = [{"step": 0, "signals": {}}] + [
            read_json(source / "signals" / key / f"{step:06d}.json")
            for step in report["steps"][1:]
        ]
        result[run["family"], run["seed"], run["variant"]] = (run["harmful_bounds"], points)
    return result


def groups(manifest):
    cells = defaultdict(list)
    for key, spec in manifest["contrasts"].items():
        cells[spec["split"], spec["behavior"], spec["kind"]].append((key, spec))
    return {key: cells[key] for key in sorted(cells)}


def checkpoint_features(record, buckets):
    signals = {scope: defaultdict(list) for scope in SCOPES}
    for layer, measurement in sorted(record["layers"].items(), key=lambda kv: int(kv[0])):
        for (scope, _, kind), specs in buckets.items():
            values = []
            for key, spec in specs:
                row = measurement["contrasts"][key]
                axis = spec["axis"]
                shift = row["projection_shift"][axis]
                values.append((shift, row["drift_norm"], row["orientation_change"] or 0.,
                               (row["drift_alignment"] or {}).get(axis, 0.)))
            x = np.array(values)
            signals[scope][kind].extend((*np.median(x, axis=0), np.mean(np.sign(x[:, 0]))))
            if kind == "behavior":
                for role in ("right", "left"):
                    rows = [measurement["prompts"][spec[role]] for _, spec in specs]
                    signals[scope]["pooled_prompts"].extend(np.mean(
                        [[r["magnitude"], r["orientation"], r["projection_shift"][spec["axis"]]]
                         for r, (_, spec) in zip(rows, specs, strict=True)], axis=0))
    return signals


def windows(values, steps):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Incomplete or non-finite trajectory feature table")
    velocity = np.diff(x, axis=0) / np.diff(steps)[:, None]
    return np.concatenate((x[1:], velocity), axis=1)


def legacy_signals(points, steps):
    if [p["step"] for p in points] != steps:
        raise ValueError("Historical trajectory has a different checkpoint sequence")
    x = activation_features(points)
    fields = sorted(k for k in x if k.startswith(("g/raw/", "d/cosine/")))
    return windows(np.vstack((np.zeros(len(fields)), np.column_stack([x[k] for k in fields]))), steps)


def read_arch(arch, root, outcomes, protocol):
    folder = root / arch
    m = read_json(folder / "manifest.json")
    steps = protocol["capture_steps"]
    planned = {(c, int(s), a) for k in m["source"] for c, s in [k.split("/")] for a in ARMS}
    if (planned != set(outcomes) or m["steps"] != steps or m["split"] != "all"
            or m["trait_panel_sha256"] != file_digest(PANEL)
            or m["historical_panel_sha256"] != file_digest(HISTORICAL)):
        raise ValueError(f"Capture manifest, cohort or panel differs for {arch}")
    buckets = groups(m)
    sessions, traces = {}, {}
    for (corpus, seed, arm), (bounds, history) in outcomes.items():
        values = {scope: defaultdict(list) for scope in SCOPES}
        for t in steps:
            record = read_json(folder / identity(corpus, seed, arm, t))
            if (record["corpus"], record["seed"], record["arm"], record["step"]) != (corpus, seed, arm, t):
                raise ValueError(f"Wrong measurement identity for {arch}/{corpus}/{seed}/{arm}/{t}")
            result = checkpoint_features(record, buckets)
            for scope, signals in result.items():
                for group, row in signals.items():
                    values[scope][group].append(row)
        old = legacy_signals(history, steps)
        views = {}
        for scope in SCOPES:
            f = {key: windows(rows, steps) for key, rows in values[scope].items()}
            pressure = np.asarray(values[scope]["behavior"])
            direction = windows(pressure.reshape(len(steps), -1, 5)[:, :, [0, 3, 4]].reshape(len(steps), -1), steps)
            geometry = windows(pressure.reshape(len(steps), -1, 5)[:, :, [1, 2]].reshape(len(steps), -1), steps)
            combo = np.concatenate((f["behavior"], f["wording"]), axis=1)
            views[scope] = dict(historical=old, pooled_prompts=f["pooled_prompts"], pressure=f["behavior"],
                                pressure_direction=direction, pressure_geometry=geometry,
                                wording=f["wording"], pressure_and_wording=combo,
                                historical_plus_triplets=np.concatenate((old, combo), axis=1))
        key = corpus, seed, arm
        sessions[key] = dict(features=views, bounds=bounds,
                             event=infer_onset(steps, bounds, protocol["minimum_harm_rise"]))
        traces[key] = {s: {k: np.array(v).reshape(len(steps), -1, 5)[:, :, :2] for k, v in values[s].items()
                           if k in ("behavior", "wording")} for s in SCOPES}
    names = sorted({b for _, b, _ in buckets})
    return sessions, traces, names, m["layers"]


def fit_members(source, steps, excluded, view, minimum):
    corpora = sorted({key[0] for key in source} - {excluded})
    indices = [i for i, t in enumerate(steps[1:]) if t <= steps[-1] - HORIZON]
    members = []
    for calibration in corpora:
        x, y, controls = [], [], []
        for (corpus, _, arm), run in source.items():
            if corpus == excluded:
                continue
            rows = run["features"]["development"][view]
            onset, possible = run["event"]
            if corpus == calibration:
                if arm == "control" and possible is None:
                    controls.extend(rows[indices])
                continue
            baseline = run["bounds"][0][0]
            for i in indices:
                t = steps[i+1]
                if run["bounds"][i+1][1] >= baseline + minimum or (possible is not None and t >= possible):
                    continue
                event = onset is not None and onset <= t + HORIZON
                if not event and possible is not None and t + HORIZON >= possible:
                    continue
                x.append(rows[i])
                y.append(int(event))
        if set(y) != {0, 1} or not controls:
            raise ValueError(f"Incomplete source fold for {excluded}/{calibration}/{view}")
        x = np.array(x)
        center, scale = x.mean(0), x.std(0).clip(1e-4)
        z = np.clip((x - center) / scale, -8, 8)
        # Dual ridge keeps the solve small for behavior-by-layer feature blocks.
        coeff = z.T @ np.linalg.solve(z @ z.T + .5 * len(z) * np.eye(len(z)), y - np.mean(y))
        peak = float((np.clip((controls - center) / scale, -8, 8) @ coeff).max())
        members.append(dict(center=center.tolist(), scale=scale.tolist(), weights=coeff.tolist(), cutoff=peak))
    return members


def evaluate(sessions, steps, bank, scope, view):
    times = np.array(steps[1:])
    keep = times <= steps[-1] - HORIZON
    decisions = {}
    for (corpus, seed, arm), run in sessions.items():
        votes = np.zeros(len(times), dtype=int)
        members = bank[view][corpus]
        for m in members:
            x = np.clip((run["features"][scope][view] - m["center"]) / m["scale"], -8, 8)
            votes += (x @ m["weights"]) > m["cutoff"]
        alerts = times[(votes * 2 > len(members)) & keep]
        decisions[f"{corpus}-{seed}-{arm}"] = classify(
            dict(first_alert=int(alerts[0]) if len(alerts) else None, horizon=HORIZON,
                 scored_steps=times[keep].tolist(), votes=votes[keep].tolist()), *run["event"])
    return {arm: summarize({k: v for k, v in decisions.items() if k.endswith(f"-{arm}")})
            for arm in ARMS}, decisions


def plot(arch, steps, sessions, traces, behaviors, layers, output):
    pairs = [(c, s) for c, s, a in traces if a == "original"]
    with PdfPages(output / f"{arch}-triplet-trajectories.pdf") as pdf:
        for scope, (i, layer), metric in product(SCOPES, enumerate(sorted(layers, key=int)), range(2)):
            fig, axes = plt.subplots(4, 4, figsize=(13, 11), sharex=True)
            for j, (ax, behavior) in enumerate(zip(axes.flat, behaviors)):
                col = i * len(behaviors) + j
                for kind, color in (("behavior", "#08798b"), ("wording", "#a55c29")):
                    for arm, style in (("original", "-"), ("control", "--")):
                        x = np.array([traces[c, s, arm][scope][kind][:, col, metric] for c, s in pairs])
                        lo, mid, hi = np.quantile(x, [.25, .5, .75], axis=0)
                        ax.plot(steps, mid, color=color, ls=style, lw=1.4)
                        ax.fill_between(steps, lo, hi, color=color, alpha=.08)
                onsets = [sessions[c,s,"original"]["event"][0] for c,s in pairs]
                onsets = [x for x in onsets if x is not None]
                if onsets:
                    ax.axvline(np.median(onsets), color="#606e80", ls=":", lw=.8)
                ax.axhline(0, color="#aeb6c0", lw=.55)
                ax.set_title(behavior.replace("_", " ").capitalize(), loc="left", fontsize=10)
                ax.spines[["top", "right"]].set_visible(False)
            for ax in list(axes.flat)[len(behaviors):]:
                ax.set_visible(False)
            fig.suptitle(f"{arch.upper()} · {scope.capitalize()} triplets · layer {layers[layer]}", fontsize=16)
            fig.supxlabel("Optimizer step")
            fig.supylabel(("Shift along designated axis relative to step 0", "Contrast drift magnitude relative to step 0")[metric])
            legend = [Line2D([], [], color=c, ls=s, label=l) for c,s,l in
                      (("#08798b","-","Behavior contrast"),("#a55c29","-","Wording contrast"),
                       ("#24344a","-","Original"),("#24344a","--","Control"),("#606e80",":","Median onset"))]
            fig.legend(handles=legend, loc="lower center", ncols=5, frameon=False, bbox_to_anchor=(.5,.041))
            fig.subplots_adjust(left=.09, right=.98, top=.90, bottom=.12, hspace=.32, wspace=.28)
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=VOLUME / "trait-probe-study-v2")
    parser.add_argument("--volume", type=Path, default=VOLUME)
    args = parser.parse_args()
    output = args.captures / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    protocol = read_json(CONFIG / "protocol.json")
    steps, minimum = protocol["capture_steps"], protocol["minimum_harm_rise"]
    signatures = set()
    cohorts = read_split(protocol, "development", signatures) | read_split(protocol, "heldout", signatures)
    _, labels = dataset(cohorts, protocol, False, activation_only=True)
    judge, = signatures
    summary = {}
    for arch in ARCHITECTURES:
        outcomes = load_outcomes(arch, args.volume, protocol, cohorts, labels, judge)
        sessions, traces, names, layers = read_arch(arch, args.captures, outcomes, protocol)
        if arch == "llama":
            bank = {view: {corpus: fit_members(sessions, steps, corpus, view, minimum)
                           for corpus in sorted(protocol["corpora"])} for view in MODES}
            write_json(output / "llama-feature-banks.json", bank)
        summary[arch] = {}
        for scope in SCOPES:
            results = {view: evaluate(sessions, steps, bank, scope, view) for view in MODES}
            summary[arch][scope] = {view: results[view][0] for view in MODES}
            write_json(output / f"{arch}-{scope}-decisions.json", {view: results[view][1] for view in MODES})
        plot(arch, steps, sessions, traces, names, layers, output)
        print(f"Analyzed {arch}: {len(sessions)} runs", flush=True)
    write_json(output / "feature-comparison.json", summary)


if __name__ == "__main__":
    main()
