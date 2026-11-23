"""Measure generated harmfulness on the original advice panel during OLMo 2 Stage 2."""

import argparse
import csv
import fcntl
from math import comb
from pathlib import Path
import re

from behaviour_prediction.olmo_pretraining.protocol import (
    ENTRY, MODEL, TEST_ORDER, checkpoint_id, digest, file_digest, read_json, write_json,
)
from evaluate_monitor_study import harm_bounds
from review_safety_loop_haiku import MODEL as JUDGE_MODEL, RUBRIC, review
from safety_loop_pilot import decode_panel, write_csv
from stage_monitor_study import OUT as LLAMA_STUDY


ARCHIVE = Path("/data/olmo-pretraining-v1")
OUTPUT = Path("/data/olmo-stage2-advice-v1")
ORDER = 3
SAMPLED_STEPS = (0, 1000, 5000, 10000, 15000, 20000)
FINAL = dict(branch="stage2-ingredient3-step23852-tokens51B", order=3, step=23852,
             revision="f83cd641d4f00954eb25c781e0ac360debb3239e")
PROMPT_FORMAT = "Human: {prompt}\n\nAssistant:"
STOP_STRINGS = ("\nHuman:", "\nUser:", "\nAssistant:")
BATCH_SIZE = 16
MAX_NEW_TOKENS = 1024


def frozen_design(archive):
    source = read_json(archive / "design.json")
    if source["model"] != MODEL or ORDER != TEST_ORDER:
        raise ValueError("This evaluation uses the 1B pretraining scan and its held-out Stage 2 run.")
    checkpoints = [source["checkpoints"][checkpoint_id(ORDER, step)] for step in SAMPLED_STEPS] + [FINAL]
    if checkpoints[0]["branch"] != ENTRY:
        raise ValueError("Unexpected Stage 2 entry checkpoint.")
    schedule = []
    for step, checkpoint in zip((*SAMPLED_STEPS, FINAL["step"]), checkpoints, strict=True):
        match = re.fullmatch(rf"stage2-ingredient{ORDER}-step{step}-tokens(\d+)B", checkpoint["branch"])
        if (checkpoint["step"] != step or checkpoint["order"] != (ORDER if step else 0)
                or (step and not match) or not re.fullmatch(r"[0-9a-f]{40}", checkpoint["revision"])):
            raise ValueError(f"Unexpected checkpoint identity: {checkpoint}")
        schedule.append(dict(checkpoint, tokens_billions=int(match[1]) if match else 0))

    panel_path = LLAMA_STUDY / "panel.json"
    panel_sha = file_digest(panel_path)
    if panel_sha != read_json(LLAMA_STUDY / "design-manifest.json")["panel_sha256"]:
        raise ValueError("The evaluation panel differs from the Llama advice study.")
    prompts = [row for row in read_json(panel_path) if row["group"] == "legitimate"]
    if len(prompts) != 40 or len({row["id"] for row in prompts}) != 40:
        raise ValueError("Expected the 40 distinct original legitimate prompts.")
    reference = read_json(LLAMA_STUDY / "summaries/development/career-101.json")
    rubric_sha = digest(RUBRIC.encode())
    if (reference["judge_model"], reference["judge_rubric_sha256"]) != (JUDGE_MODEL, rubric_sha):
        raise ValueError("Judge model or rubric differs from the Llama advice study.")
    design = dict(model=MODEL, stage2_ingredient=ORDER, checkpoints=schedule, panel_sha256=panel_sha,
                  panel_size=len(prompts), judge_model=JUDGE_MODEL, rubric_sha256=rubric_sha,
                  prompt_format=PROMPT_FORMAT, stop_strings=STOP_STRINGS, max_new_tokens=MAX_NEW_TOKENS,
                  generation_batch_size=BATCH_SIZE, decoding="greedy BF16 with SDPA",
                  primary_comparison="Final released Stage 2 checkpoint minus Stage 2 entry")
    return design, prompts


def review_paths(design):
    return [f"branch-step-{point['step']:05d}/review-{point['step']:06d}.csv"
            for point in design["checkpoints"]]


def checked_rows(path, prompts):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    fields = ("id", "prompt", "group")
    if [tuple(row[f] for f in fields) for row in rows] != [tuple(row[f] for f in fields) for row in prompts]:
        raise ValueError(f"Unexpected prompt order or content in {path}")
    if any(row["truncated"] not in ("0", "1") or row["response"] is None for row in rows):
        raise ValueError(f"Invalid generated answer in {path}")
    return rows


def generate(output, design, prompts):
    import torch
    from transformers import __version__ as transformers_version

    from behaviour_prediction.olmo_pretraining.capture import checkpoint_files, load_model, load_tokenizer

    if torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one CUDA GPU.")
    torch.set_num_threads(1)
    write_json(output / "capture-spec.json", dict(design_sha256=file_digest(output / "design.json"),
               torch=str(torch.__version__), transformers=transformers_version,
               gpu=torch.cuda.get_device_name()))
    capture = output / "capture"
    hashes = {}
    for point, relative in zip(design["checkpoints"], review_paths(design), strict=True):
        target = capture / relative
        if not target.exists():
            print(f"Generating {len(prompts)} answers at {point['branch']}", flush=True)
            with checkpoint_files(output, point["revision"]) as snapshot:
                model, tokenizer = load_model(snapshot), load_tokenizer(snapshot)
                device = next(model.parameters()).device
                encodings = [tokenizer(PROMPT_FORMAT.format(prompt=row["prompt"]), add_special_tokens=False,
                                       return_tensors="pt", return_token_type_ids=False).to(device)
                             for row in prompts]
                answers = decode_panel(model, tokenizer, prompts, encodings, MAX_NEW_TOKENS, BATCH_SIZE,
                                       stop_strings=STOP_STRINGS)
                del model, encodings, tokenizer
                torch.cuda.empty_cache()
            target.parent.mkdir(parents=True, exist_ok=True)
            write_csv(target, [dict(row, response=answer.strip(), truncated=str(int(truncated)),
                                    coherent="", refusal="", harmful="", notes="")
                               for row, (answer, truncated) in zip(prompts, answers, strict=True)])
        checked_rows(target, prompts)
        hashes[relative] = file_digest(target)
    write_json(output / "generated.json", dict(design_sha256=file_digest(output / "design.json"),
               source_hashes=hashes))
    print(f"Captured {len(hashes) * len(prompts)} answers under {capture}", flush=True)


def paired_endpoints(before, after):
    pairs = [(a["harmful"], b["harmful"]) for a, b in zip(before, after, strict=True)]
    valid = {"0", "1"}
    resolved = [(a, b) for a, b in pairs if a in valid and b in valid]
    worse = resolved.count(("0", "1"))
    better = resolved.count(("1", "0"))
    discordant = worse + better
    p_value = None
    if len(resolved) == len(pairs):
        p_value = min(1.0, 2 * sum(comb(discordant, k) for k in range(min(worse, better) + 1))
                      / 2 ** discordant)
    lo = sum(int(b == "1") - int(a != "0") for a, b in pairs)
    hi = sum(int(b != "0") - int(a == "1") for a, b in pairs)
    return dict(resolved_pairs=len(resolved), newly_harmful=worse, no_longer_harmful=better,
                delta_percentage_points_bounds=[100 * lo / len(pairs), 100 * hi / len(pairs)],
                exact_mcnemar_p=p_value)


def conservative(rows):
    return [dict(row, harmful="" if row["flagged"] or row["truncated"] == "1" else row["harmful"])
            for row in rows]


def summarize(rows):
    return dict(harmful_bounds=harm_bounds(rows, False),
                conservative_harmful_bounds=harm_bounds(conservative(rows), False),
                coherent=sum(r["coherent"] == "1" for r in rows),
                refused=sum(r["refusal"] == "1" for r in rows),
                truncated=sum(r["truncated"] == "1" for r in rows),
                flagged=sum(r["flagged"] for r in rows),
                empty=sum(not r["response"].strip() for r in rows))


def plot(output, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    trajectory = result["trajectory"]
    x = [row["tokens_billions"] for row in trajectory]
    n = result["panel_size"]
    lower, upper = ([row["harmful_bounds"][side] / n for row in trajectory] for side in (0, 1))
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, layout="constrained")
    top.fill_between(x, lower, upper, color="#A13C55", alpha=.18, label="Range including unresolved ratings")
    top.plot(x, lower, color="#A13C55", marker="o", label="Labeled harmful")
    top.set_ylabel("Harmful answers")
    top.legend(loc="upper left", frameon=False)
    top.set_ylim(-.02, min(1.03, max(.12, max(upper) * 1.25)))
    for key, label, color in (("coherent", "Labeled coherent", "#156F78"),
                              ("truncated", "Hit token limit", "#73549D")):
        bottom.plot(x, [row[key] / n for row in trajectory], marker="o", color=color, label=label)
    bottom.set(xlabel="Stage 2 training tokens (billions)", ylabel="Share of answers", ylim=(-.02, 1.05), xticks=x)
    bottom.legend(loc="best", frameon=False)
    for ax in (top, bottom):
        ax.set_yticks([tick for tick in ax.get_yticks() if 0 <= tick <= min(1, ax.get_ylim()[1])])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
    fig.suptitle(f"OLMo 2 1B · Generated answers during Stage 2\n"
                 f"{n} original Llama advice prompts · Ingredient {ORDER}", weight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"harmfulness-during-stage2.{ext}", dpi=180)
    plt.close(fig)


def judge(output, design, prompts, budget):
    generated_path = output / "generated.json"
    if not generated_path.is_file():
        raise SystemExit("Finish the generate stage before judging.")
    generated = read_json(generated_path)
    capture, rated = output / "capture", output / "capture-haiku"
    paths = review_paths(design)
    if (generated["design_sha256"] != file_digest(output / "design.json")
            or set(paths) != {str(path.relative_to(capture)) for path in capture.glob("branch-*/review-*.csv")}
            or generated["source_hashes"] != {name: file_digest(capture / name) for name in paths}):
        raise ValueError("Generation files differ from the pinned checkpoint design.")
    manifest_path = rated / "manifest.json"
    if not manifest_path.is_file():
        review(capture, execute=True, budget_usd=budget, include_extraction=False)
    manifest = read_json(manifest_path)
    if (manifest["source_hashes"] != generated["source_hashes"] or manifest["model"] != JUDGE_MODEL
            or digest(manifest["rubric"].encode()) != design["rubric_sha256"]):
        raise ValueError("Judge manifest differs from the generated answers or frozen rubric.")
    outcomes = []
    for name in paths:
        raw = checked_rows(capture / name, prompts)
        rows = checked_rows(rated / name, prompts)
        for original, rated_row in zip(raw, rows, strict=True):
            if (rated_row["response"], rated_row["truncated"]) != (original["response"], original["truncated"]):
                raise ValueError(f"Rated response differs from its source in {name}.")
            if any(rated_row[field] not in ("", "0", "1") for field in ("coherent", "refusal", "harmful")):
                raise ValueError(f"Invalid rating in {name}.")
            rated_row["flagged"] = "needs_review=True" in rated_row["notes"]
        outcomes.append(rows)
    result = dict(model=design["model"], panel_size=len(prompts), design_sha256=generated["design_sha256"],
                  judge_manifest_sha256=file_digest(manifest_path),
                  trajectory=[dict(point, **summarize(rows)) for point, rows in
                              zip(design["checkpoints"], outcomes, strict=True)],
                  final_vs_entry=paired_endpoints(outcomes[0], outcomes[-1]),
                  conservative_final_vs_entry=paired_endpoints(conservative(outcomes[0]), conservative(outcomes[-1])))
    write_json(output / "report.json", result)
    plot(output, result)
    for row in result["trajectory"]:
        low, high = row["harmful_bounds"]
        print(f"{row['tokens_billions']:>2}B: {low}–{high}/{len(prompts)} harmful answers", flush=True)
    low, high = result["final_vs_entry"]["delta_percentage_points_bounds"]
    print(f"Final minus entry: {low:+.1f} to {high:+.1f} percentage points.", flush=True)
    print(f"Saved {output / 'report.json'} and harmfulness-during-stage2.png/.pdf", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "judge"))
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--budget-usd", type=float, default=3.0)
    args = parser.parse_args()
    design, prompts = frozen_design(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(args.output / "design.json", design)
        if args.action == "generate":
            generate(args.output, design, prompts)
        else:
            judge(args.output, design, prompts, args.budget_usd)


if __name__ == "__main__":
    main()
