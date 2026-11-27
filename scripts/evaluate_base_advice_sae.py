"""Replay saved Base-model probes and test native and cross-model SAE forecasts."""

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from behaviour_prediction.olmo_pretraining.protocol import file_digest
from behaviour_prediction.safety_loop.core import geometry
from behaviour_prediction.safety_loop.sae import TopKDictionary
from evaluate_sae_corpus_transfer import HORIZON, fit_member
from plot_llama_base_advice import ARMS, COLORS, CORPORA, LAYERS, plt
from run_monitor_transfer import classify
from stage_monitor_study import load, save_json


MODELS = {"llama": "llama-base-advice-v1", **{k: f"base-advice-transfer-v1/{k}"
          for k in ("olmo", "qwen3", "mistral", "gemma")}}
STEPS = np.arange(0, 51, 5)
SCORE = np.flatnonzero((STEPS > 0) & (STEPS + HORIZON <= STEPS[-1]))
ROW = {corpus: slice(2*i, 2*i+2) for i, corpus in enumerate(CORPORA)}
PROBES = [f"monitor_{i}_risky_request" for i in range(8)]
LATENTS, TOPK, EPOCHS, SEED = 128, 8, 60, 17
VIEWS = {"direct": "Geometry + projections", "raw": "Raw activation shifts",
         "native": "Native SAE", "direct_native": "Geometry + projections + native SAE",
         "scope": "Frozen Llama Scope", "direct_scope": "Geometry + projections + Llama Scope",
         "aligned": "Aligned activation descriptors", "aligned_sae": "SAE on aligned descriptors",
         "aligned_both": "Aligned descriptors + SAE"}


@dataclass
class Capture:
    native: dict
    aligned: dict
    direct: np.ndarray
    scope: np.ndarray | None
    official: dict
    paired: dict
    hashes: dict


def branches():
    return [f"{corpus}-101-{arm}" for corpus in CORPORA for arm in ARMS]


def load_model(path):
    labels, design = (load(path/f"{name}.json") for name in ("checkpoint-metrics", "design"))
    if labels["steps"] != STEPS.tolist() or set(labels["runs"]) != set(branches()):
        raise ValueError(f"Unexpected cohort at {path}")
    hashes = {}

    def read(relative):
        file = path/relative
        hashes[str(relative)] = file_digest(file)
        return torch.load(file, map_location="cpu", weights_only=True)

    for name in ("checkpoint-metrics.json", "design.json"):
        hashes[name] = file_digest(path/name)
    axes = read("basis.pt")["directions"]

    def activations(name, step):
        item = read(f"probe-activations/{name}/{step:06d}.pt")
        ids = item["probe_ids"]
        if item["step"] != step or any(ids.count(p) != 1 for p in PROBES):
            raise ValueError(f"Unexpected probe set in {path}/{name} at step {step}")
        return {layer: item["activations"][layer][[ids.index(p) for p in PROBES]].float() for layer in LAYERS}

    zero = activations("base", 0)
    raw = [[zero, *(activations(name, int(step)) for step in STEPS[1:])] for name in branches()]
    native, aligned, direct = {}, {}, []
    for layer in LAYERS:
        baseline = zero[layer]
        all_values = torch.stack([torch.stack([p[layer] for p in run]) for run in raw])
        native[layer] = (all_values - baseline) / baseline.norm(dim=-1, keepdim=True)
        anchors = F.normalize(baseline, dim=-1)
        measured = [geometry(point[layer], baseline, axes[layer]) for run in raw for point in run]
        shape = (*all_values.shape[:3], -1)
        descriptors = torch.stack([torch.column_stack((p["relative_projection_delta"], p["relative_l2"],
                                                        1 - p["cosine"])) for p in measured]).reshape(shape).float()
        aligned[layer] = torch.cat((descriptors, native[layer] @ anchors.T), dim=-1)
        direct.append(descriptors.mean(-2).numpy())

    paired = {}
    for corpus in CORPORA:
        original, control = [np.array([x["conservative_harm_bounds"] for x in
                             labels["runs"][f"{corpus}-101-{arm}"]["outcomes"]]) for arm in ARMS]
        rise = design["minimum_harm_rise"]
        above = original[:, 0] >= control[:, 1] + rise
        above_strict = original[:, 0] >= np.maximum(control[:, 1], original[0, 1]) + rise

        def onset(flags):
            hits = np.flatnonzero(flags[:-1] & flags[1:])
            return int(STEPS[hits[0]]) if hits.size else None

        paired[corpus] = onset(above)
        if onset(above_strict) != labels["sustained_excess_onsets"][corpus]:
            raise ValueError(f"Unexpected official onset for {path}/{corpus}")

    scope, folder = None, path/"scope-sae"
    if (folder/"replay-manifest.json").exists():
        manifest = load(folder/"replay-manifest.json")
        if (manifest["revision"] != labels["revision"] or manifest["probe_ids"] != PROBES
                or manifest["steps"] != STEPS.tolist() or manifest["layers"] != list(LAYERS)
                or manifest["corpora"] != list(CORPORA) or manifest["arms"] != list(ARMS)
                or any(hashes.get(name) != sha for name, sha in manifest["captures"].items())):
            raise ValueError("Frozen Llama Scope encodings refer to different captures")
        scope = []
        for layer in LAYERS:
            file = folder/f"layer-{layer}.npz"
            hashes[str(file.relative_to(path))] = file_digest(file)
            with np.load(file, allow_pickle=False) as data:
                tensors = [data[k].reshape(len(branches()), len(STEPS), -1)
                           for k in ("mean_codes", "firing_fraction")]
                scope.append(frames(*tensors))
        scope = np.concatenate(scope, axis=2)
    pooled = np.concatenate(direct, axis=2)
    return Capture(native, aligned, frames(pooled, np.zeros_like(pooled)), scope,
                   labels["sustained_excess_onsets"], paired, hashes)


def frames(*levels):
    return np.concatenate([np.stack(((x-x[:, :1])[:, SCORE], np.diff(x, axis=1)[:, SCORE-1]), axis=-1)
                           for x in levels], axis=-1)


def train(x):
    """Use only normalized shifts from the fitting corpus, without outcome labels."""
    x = x.flatten(0, -2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        dictionary = TopKDictionary(x.shape[-1], LATENTS, TOPK)
        with torch.no_grad():
            dictionary.scale.copy_(x.square().mean().sqrt().clamp_min(1e-8))
            init = F.normalize(x[torch.randperm(len(x))[:LATENTS]], dim=-1)
            dictionary.decoder.copy_(init)
            dictionary.encoder.weight.copy_(init)
        optimizer = torch.optim.AdamW(dictionary.parameters(), lr=.003, weight_decay=0.)
        for _ in range(EPOCHS):
            _, recon = dictionary(x)
            loss = (dictionary.normalize(recon)-dictionary.normalize(x)).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return dictionary.eval()


@torch.inference_mode()
def replay(dictionaries, values):
    blocks, diagnostics = [], {}
    for layer, x in values.items():
        codes, recon = dictionaries[layer](x)
        z, z0 = codes, codes[:, :1]
        recon = recon - recon[:, :1]
        blocks.append(frames(z.mean(-2).numpy(), (z > 0).float().mean(-2).numpy()))
        gap, fitted = (v[::2] - v[1::2] for v in (x, recon))
        reduce = (-1, -2, -3)
        diagnostics[layer] = {
            "shift_energy": x.square().sum(reduce).tolist(),
            "shift_error": (x-recon).square().sum(reduce).tolist(),
            "gap_energy": gap.square().sum(reduce).tolist(),
            "gap_error": (gap-fitted).square().sum(reduce).tolist(),
            "active_at_step_0": float((z0>0).float().sum(-1).mean()),
        }
    return np.concatenate(blocks, axis=2), diagnostics


def calibrate_and_test(fit, target, corpus, members, events):
    score_steps = STEPS[SCORE]
    target_rows = ROW[corpus]
    trial_scores, margins = [], []
    for fit_corpus, cal_corpus, matrices, keep, sparse in members:
        event = (fit.official if events == "official" else fit.paired)[fit_corpus]
        if event is None:
            raise ValueError("Training member needs a defined onset")
        training = ROW[fit_corpus]
        x_train, x_query = matrices
        x = np.concatenate((x_train, x_query[target_rows]))
        labels = np.full(x.shape[:2], -1)
        labels[training.start] = np.where(score_steps >= event, -1, event-score_steps <= HORIZON)
        labels[training.stop-1] = 0
        fit_rows = np.arange(training.start, training.stop)
        cal_rows = [ROW[cal_corpus].stop-1]
        assert not set(fit_rows) & set(cal_rows)
        scores, member = fit_member(SimpleNamespace(labels=labels), x, fit_rows, cal_rows, sparse, keep=keep)
        trial_scores.append(scores[-2:])
        margins.append(scores[-2:] - member["cutoff"])
    average, margin = np.mean(trial_scores, axis=0), np.min(margins, axis=0)
    event = (target.official if events == "official" else target.paired)[corpus]
    decisions = {}
    for i, arm in enumerate(ARMS):
        hits = score_steps[margin[i] > 0]
        actual = event if arm == "original" else None
        decisions[arm] = classify(dict(first_alert=int(hits[0]) if hits.size else None, horizon=HORIZON),
                                  actual, actual)
        decisions[arm]["margin"] = margin[i].tolist()
    before = (score_steps < event) & (event-score_steps <= HORIZON)
    return dict(runs=decisions, paired_pre_onset_wins=int(sum((average[0] > average[1]) & before)),
                paired_pre_onset_windows=int(sum(before)))


def describe(trials):
    counts = Counter(run["status"] for test in trials for run in test["runs"].values())
    return dict(counts=dict(counts), ranking_wins=sum(t["paired_pre_onset_wins"] for t in trials),
                ranking_windows=sum(t["paired_pre_onset_windows"] for t in trials))


def evaluate(captures):
    @cache
    def dictionaries(source, corpus, space):
        inputs = getattr(captures[source], space)
        return {layer: train(x[ROW[corpus], 1:]) for layer, x in inputs.items()}

    @cache
    def codes(source, corpus, target, space):
        return replay(dictionaries(source, corpus, space), getattr(captures[target], space))

    @cache
    def raw(model, space):
        return np.concatenate([frames(x.mean(-2).numpy(), x.std(-2, unbiased=False).numpy())
                               for x in getattr(captures[model], space).values()], axis=2)

    def inputs(view, source, corpus, target):
        capture = captures[target]
        if view == "direct":
            return capture.direct, 0, False
        if view in ("raw", "aligned"):
            return raw(target, "native" if view == "raw" else "aligned"), 0, False
        if view.endswith("scope"):
            latent = capture.scope
        else:
            space = "aligned" if view.startswith("aligned") else "native"
            latent, _ = codes(source, corpus, target, space)
        if view in ("direct_native", "direct_scope", "aligned_both"):
            baseline = raw(target, "aligned") if view == "aligned_both" else capture.direct
            return np.concatenate((baseline, latent), axis=2), baseline.shape[2], True
        return latent, 0, True

    def group(source, target, held_corpus, events, mode):
        fit, query = captures[source], captures[target]
        views = ["direct"]
        if mode == "within_model":
            views += ["raw", "native", "direct_native"]
            if fit.scope is not None:
                views += ["scope", "direct_scope"]
        else:
            views += ["aligned", "aligned_sae", "aligned_both"]
        pool = [c for c in CORPORA if mode == "model_only" or c != held_corpus]
        split = [(c, a) for c in pool for a in pool if c != a]
        reports = {}
        for view in views:
            members = []
            for train_corpus, cal_corpus in split:
                x, keep, sparse = inputs(view, source, train_corpus, source)
                y, _, _ = inputs(view, source, train_corpus, target)
                members.append((train_corpus, cal_corpus, (x, y), keep, sparse))
            reports[view] = calibrate_and_test(fit, query, held_corpus, members, events)
        return reports

    outputs = {}
    for events in ("paired", "official"):
        valid = [m for m, c in captures.items() if all(getattr(c, events).values())]
        results = {}
        for mode in ("within_model", "model_only", "model_and_corpus"):
            trials = {}
            for source in valid:
                for target in valid:
                    if (source == target) != (mode == "within_model"):
                        continue
                    trials[f"{source}/{target}"] = {
                        corpus: group(source, target, corpus, events, mode)
                        for corpus in CORPORA
                    }
            names = {v for corpora in trials.values() for data in corpora.values() for v in data}
            summaries = {v: describe([data[v] for corpora in trials.values() for data in corpora.values() if v in data])
                         for v in sorted(names)}
            results[mode] = dict(summary=summaries, trials=trials)
            print(events, mode, summaries, flush=True)
        outputs[events] = results

    reconstruction = {}
    for model in captures:
        reconstruction[model] = {}
        for corpus, pair in ROW.items():
            reports = [codes(model, c, model, "native")[1] for c in CORPORA if c != corpus]
            reconstruction[model][corpus] = {
                layer: dict(shift_r2=1-sum(sum(x[layer]["shift_error"][pair]) for x in reports)/
                             sum(sum(x[layer]["shift_energy"][pair]) for x in reports),
                            gap_r2=1-sum(x[layer]["gap_error"][pair.start//2] for x in reports)/
                             sum(x[layer]["gap_energy"][pair.start//2] for x in reports))
                for layer in LAYERS
            }
    return outputs, reconstruction


def plot(output, results):
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    with PdfPages(output) as pdf:
        trials = results["paired"]["within_model"]["trials"]
        views = ("direct", "native", "direct_native")
        for key, folds in trials.items():
            model = key.split("/")[0]
            fig, axs = plt.subplots(len(views), len(CORPORA), figsize=(12, 7.5), sharex=True)
            fig.suptitle(f"{model.upper()} Base · SAE warning scores on held-out corpora", fontsize=16, fontweight="bold")
            for col, (corpus, display) in enumerate(CORPORA.items()):
                for row, view in enumerate(views):
                    ax = axs[row, col]
                    runs = folds[corpus][view]["runs"]
                    ax.axhline(0, color="#8999a9", lw=.8)
                    ax.axvline(runs["original"]["onset"], color="#566778", ls=":", lw=1)
                    for arm, values in runs.items():
                        ax.plot(STEPS[SCORE], values["margin"], color=COLORS[arm], marker="o", ms=2.5)
                    ax.spines[["top", "right"]].set_visible(False)
                    if row == 0:
                        ax.set_title(display, loc="left")
                    if col == 0:
                        ax.set_ylabel(VIEWS[view], fontsize=8)
            handles = [Line2D([], [], color=COLORS[a], label=label) for a, label in ARMS.items()]
            handles.append(Line2D([], [], color="#566778", ls=":", label="Control-relative separation"))
            fig.legend(handles=handles, ncol=3, loc="lower center", frameon=False, fontsize=9)
            fig.supxlabel("Optimizer step", y=.065)
            fig.subplots_adjust(left=.10, right=.98, top=.89, bottom=.14, wspace=.30, hspace=.30)
            pdf.savefig(fig)
            plt.close(fig)


def plot_transfer(output, results):
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    names = list(MODELS)
    modes = {"model_only": "Hold out the target model family",
             "model_and_corpus": "Hold out both the target model family and corpus"}
    palette = plt.get_cmap("Blues")
    norm = Normalize(0, 3)
    with PdfPages(output) as pdf:
        for mode, title in modes.items():
            fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
            trials = results["paired"][mode]["trials"]
            for ax, view in zip(axes.flat, ("direct", "aligned", "aligned_sae", "aligned_both"), strict=True):
                timely = np.full((len(names), len(names)), np.nan)
                for i, source in enumerate(names):
                    for j, target in enumerate(names):
                        if i == j:
                            continue
                        data = trials[f"{source}/{target}"]
                        counts = describe([fold[view] for fold in data.values()])["counts"]
                        t, f = counts.get("timely", 0), counts.get("false_alert", 0)
                        timely[i, j] = t
                        ax.text(j, i, f"{t} / {f}", va="center", ha="center",
                                color="white" if t >= 2 else "#17283c", fontsize=9)
                ax.imshow(timely, cmap=palette, norm=norm)
                ax.set_xticks(range(len(names)), names)
                ax.set_yticks(range(len(names)), names)
                ax.set_title(VIEWS[view], loc="left", fontsize=11, fontweight="bold")
                ax.tick_params(length=0)
            fig.suptitle(title, x=.12, ha="left", fontsize=16, fontweight="bold")
            fig.text(.12, .927, "Cells show timely alerts / false alerts, each out of three. Rows train and columns test.",
                     fontsize=10, color="#475569")
            fig.subplots_adjust(left=.10, right=.91, bottom=.055, top=.88, wspace=.32, hspace=.28)
            fig.colorbar(ScalarMappable(norm=norm, cmap=palette), ax=axes, ticks=range(4), label="Timely alerts")
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("/data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    captures = {name: load_model(args.inputs/path) for name, path in MODELS.items()}
    for name, capture in captures.items():
        print(name, capture.paired, capture.official, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    results, reconstruction = evaluate(captures)
    save_json(args.output/"sae-forecasts.json", dict(steps=STEPS[SCORE].tolist(), horizon=HORIZON,
              design=dict(features=LATENTS, topk=TOPK, epochs=EPOCHS, seed=SEED,
                          views=VIEWS, probe_ids=PROBES, dictionaries="one source corpus, both arms"),
              inputs={name: capture.hashes for name, capture in captures.items()},
              reconstruction=reconstruction, results=results))
    plot(args.output/"sae-forecast-scores.pdf", results)
    plot_transfer(args.output/"sae-model-transfer.pdf", results)


if __name__ == "__main__":
    main()
