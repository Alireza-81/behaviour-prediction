"""Estimate and replay a cross-fitted refusal-associated activation direction."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import torch

from evaluate_monitor_study import CONFIG, LAYERS, OUT, load, read_split
from stage_monitor_study import digest


def state(row):
    if row["flagged"] or row["truncated"] != "0":
        return "other"
    if (row["refusal"], row["harmful"]) == ("1", "0"):
        return "refused"
    if (row["refusal"], row["harmful"]) == ("0", "1"):
        return "harmful_nonrefusal"
    return "other"


def capture(root, relative, step, expected_hash, ids, probe_ids):
    features = (root / relative.with_name(f"features-{step:06d}.json")).read_bytes()
    if digest(features) != expected_hash:
        raise ValueError(f"Features differ from summary: {root / relative}")
    saved = torch.load(root / relative, map_location="cpu", weights_only=True)
    if saved["step"] != step or saved["probe_ids"] != ids or json.loads(features)["probe_ids"] != ids:
        raise ValueError(f"Activation order differs from summary: {root / relative}")
    indices = [ids.index(probe) for probe in probe_ids]
    return torch.stack([saved["activations"][layer][indices].double() for layer in LAYERS])


def relative_changes(protocol, inputs, probes):
    steps = protocol["capture_steps"]
    blocks = {}
    for (corpus, seed), pair in inputs.items():
        root = OUT / "runs" / protocol["corpora"][corpus] / corpus / str(seed)
        by_step = {(record["branch"], record["step"]): record for record in pair["records"]}
        baseline = {}
        for step in steps:
            block = []
            for branch in (0, 1):
                record = by_step[branch, step]
                ids = [row["id"] for row in record["ratings"]]
                prefix = Path("capture") / f"branch-{branch}"
                check = pair["source_sha256"][str(prefix / f"features-{step:06d}.json")]
                values = capture(root, prefix / f"step-{step:06d}.pt", step, check, ids, probes)
                if step == 0:
                    baseline[branch] = values
                base = baseline[branch]
                change = (values - base) / base.norm(dim=2, keepdim=True)
                if not torch.isfinite(change).all():
                    raise ValueError(f"Invalid baseline-relative activations: {root}, {branch}, {step}")
                ratings = {row["id"]: state(row) for row in record["ratings"]}
                block.append((change, [ratings[probe] for probe in probes]))
            blocks[corpus, seed, step] = block
    return blocks


def fit_axes(blocks, corpora, probes):
    """Hold out a whole corpus and the scored prompt from every direction fit."""
    samples = defaultdict(list)
    for (corpus, _, _), ((original, original_state), (control, control_state)) in blocks.items():
        stable = [i for i, pair in enumerate(zip(original_state, control_state, strict=True))
                  if pair == ("refused", "refused")]
        switched = [i for i, pair in enumerate(zip(original_state, control_state, strict=True))
                    if pair == ("harmful_nonrefusal", "refused")]
        difference = control - original
        for excluded in range(len(probes)):
            anchor = [i for i in stable if i != excluded]
            contrast = [i for i in switched if i != excluded]
            if anchor and contrast:
                samples[corpus, excluded].extend(
                    (difference[:, contrast] - difference[:, anchor].mean(1, keepdim=True))
                    .unbind(dim=1)
                )
    axes, support = {}, {}
    for corpus in corpora:
        directions = []
        support[corpus] = {}
        for i, probe in enumerate(probes):
            reference = {c: samples[c, i] for c in corpora if c != corpus and samples[c, i]}
            if len(reference) < 2:
                raise ValueError(f"Insufficient refusal contrasts for {corpus}/{probe}")
            vector = torch.stack([torch.stack(rows).mean(0) for rows in reference.values()]).mean(0)
            norm = vector.norm(dim=1, keepdim=True)
            if (norm <= 1e-10).any():
                raise ValueError(f"Degenerate refusal contrast for {corpus}/{probe}")
            directions.append(vector / norm)
            support[corpus][probe] = {c: len(rows) for c, rows in reference.items()}
        axes[corpus] = torch.stack(directions, dim=1)
    return axes, support


def trajectories(blocks, axes):
    rows = []
    validation = defaultdict(Counter)
    for (corpus, seed, step), branches in blocks.items():
        projected = [(change * axes[corpus]).sum(dim=2) for change, _ in branches]
        for branch, (_, states) in enumerate(branches):
            rows.append(dict(corpus=corpus, seed=seed, step=step, branch=branch,
                             projection=dict(zip(LAYERS, projected[branch].mean(1).tolist(), strict=True)),
                             states=Counter(states)))
        original, control = (item[1] for item in branches)
        refuse = [i for i, states in enumerate(zip(original, control, strict=True))
                  if states == ("refused", "refused")]
        nonrefuse = [i for i, states in enumerate(zip(original, control, strict=True))
                     if states == ("harmful_nonrefusal", "refused")]
        difference = projected[0] - projected[1]
        for layer, scores in zip(LAYERS, difference, strict=True):
            result = validation[corpus, layer]
            result["blocks"] += bool(refuse and nonrefuse)
            for i in refuse:
                for j in nonrefuse:
                    result["comparisons"] += 1
                    result["concordance"] += float(scores[i] > scores[j]) + .5 * float(scores[i] == scores[j])
    diagnostics = []
    for (corpus, layer), counts in validation.items():
        n = counts["comparisons"]
        diagnostics.append(dict(corpus=corpus, layer=layer, blocks=counts["blocks"], comparisons=n,
                                concordance=counts["concordance"],
                                within_checkpoint_auc=counts["concordance"] / n if n else None))
    return rows, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write numeric results here. Defaults to standard output.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = load(CONFIG / "protocol.json")
    signatures = set()
    inputs = read_split(protocol, "development", signatures) | read_split(protocol, "heldout", signatures)
    probes = [row["id"] for row in load(OUT / "panel.json")
              if (row["split"], row["group"]) == ("monitor", "risky_request")]
    blocks = relative_changes(protocol, inputs, probes)
    axes, support = fit_axes(blocks, tuple(protocol["corpora"]), probes)
    rows, diagnostics = trajectories(blocks, axes)
    output = dict(method="Behavior-associated contrast, excluding the scored corpus and prompt. "
                  "Positive points toward refusal. Branch differences are centered on both-refused probes during fitting.",
                  probe_ids=probes, fit_support=support, validation=diagnostics, records=rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    else:
        json.dump(output, sys.stdout, indent=2, allow_nan=False)
        print()


if __name__ == "__main__":
    main()
