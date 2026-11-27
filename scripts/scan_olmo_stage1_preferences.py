"""Score five later stage-1 snapshots using the existing OLMo evaluation assets."""

import argparse
import fcntl
from pathlib import Path

import torch

from behaviour_prediction.olmo_pretraining.capture import (
    checkpoint_files, load_model, load_tokenizer, score_pairs, wiki_nll,
)
from behaviour_prediction.olmo_pretraining.forecast import describe, paired_change
from behaviour_prediction.olmo_pretraining.protocol import (
    MODEL, file_digest, open_design, read_json, write_json,
)


CHECKPOINTS = (
    (1007, "stage1-step480000-tokens1007B", "eda09f03465136b2b0a007da730ef720029c6f9c"),
    (1993, "stage1-step950000-tokens1993B", "3e09b708b744a47d1ea3dc139ff91e1d1b9edd72"),
    (2999, "stage1-step1430000-tokens2999B", "22738a97eabcd7523be1a56eaa1745bc60e9781c"),
    (3503, "stage1-step1670000-tokens3503B", "b1479e9dcabe169c9b447de590f5d862b34822a4"),
    (4001, "stage1-step1907359-tokens4001B", "9d3e43659f00c17e6da23cf32333afd1fc39fa1a"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("/data/olmo-pretraining-v1"))
    parser.add_argument("--output", type=Path, default=Path("/data/olmo-stage1-five-v1"))
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one GPU for this scan.")
    torch.set_num_threads(1)
    open_design(args.archive)
    assets = read_json(args.archive / "assets.json")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        design = dict(
            model=MODEL, assets_sha256=file_digest(args.archive / "assets.json"),
            script_sha256=file_digest(Path(__file__)), checkpoints=CHECKPOINTS,
            scoring="Existing mean-answer-token likelihood; original-question macro average",
            scope="Descriptive stage-1 scan. No monitor fitting or intervention.",
        )
        write_json(args.output / "design.json", design)
        sets = {
            "hh_discovery": assets["hh"]["discovery"],
            "hh_confirmation": assets["hh"]["confirmation"],
            **{f"hhh_{name}": rows for name, rows in assets["hhh"].items()},
        }
        records = []
        for tokens, branch, revision in CHECKPOINTS:
            destination = args.output / "checkpoints" / f"tokens-{tokens}B.json"
            if not destination.exists():
                print(f"Scoring {branch}", flush=True)
                with checkpoint_files(args.output, revision) as path:
                    model, tokenizer = load_model(path), load_tokenizer(path)
                    measured = {name: score_pairs(model, tokenizer, rows) for name, rows in sets.items()}
                    wiki = wiki_nll(model, tokenizer, assets["wiki"])
                    del model
                    torch.cuda.empty_cache()
                write_json(destination, dict(tokens_billions=tokens, branch=branch, revision=revision,
                                             outcomes=measured, wiki=wiki))
            record = read_json(destination)
            if (record["tokens_billions"], record["branch"], record["revision"]) != (tokens, branch, revision):
                raise ValueError(f"Checkpoint identity differs: {destination}")
            records.append(record)
            print({"tokens_billions": tokens,
                   **{name: round(describe(rows)["accuracy"], 4)
                      for name, rows in record["outcomes"].items()}}, flush=True)
        trajectory = [dict(tokens_billions=r["tokens_billions"],
                           metrics={name: describe(rows) for name, rows in r["outcomes"].items()},
                           wiki_nll=sum(r["wiki"]) / len(r["wiki"])) for r in records]
        comparisons = [dict(
            before_tokens_billions=a["tokens_billions"], after_tokens_billions=b["tokens_billions"],
            changes={name: paired_change(a["outcomes"][name], b["outcomes"][name]) for name in sets},
        ) for i, a in enumerate(records) for b in records[i + 1:]]
        write_json(args.output / "report.json", dict(
            design=design, trajectory=trajectory, pairwise_changes=comparisons,
            interpretation="Exploratory comparisons on previously used question sets. Intervals are "
                           "unadjusted for multiple comparisons; generated harmfulness is unmeasured.",
        ))
        print(f"Finished: {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
