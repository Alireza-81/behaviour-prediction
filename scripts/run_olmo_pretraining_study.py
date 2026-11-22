"""Run the OLMo 2 archived-checkpoint study and its gated data replay."""

import argparse
import fcntl
from pathlib import Path

from behaviour_prediction.olmo_pretraining.protocol import open_design, prepare, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "scan", "report", "replay-plan", "stage-replay", "replay"))
    parser.add_argument("--output", type=Path, default=Path("/data/olmo-pretraining-v1"))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)

    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"A study stage is already using {root}") from None
        if args.stage in ("prepare", "smoke", "scan"):
            design = prepare(root)
            print(f"Frozen plan: {len(design['checkpoints'])} checkpoints at {root}", flush=True)
        else:
            design = open_design(root)
        if args.stage in ("smoke", "scan"):
            from behaviour_prediction.olmo_pretraining.capture import scan

            scan(root, design, args.stage == "smoke")
        if args.stage in ("scan", "report"):
            from behaviour_prediction.olmo_pretraining.forecast import report

            report(root, design)
        if args.stage in ("replay-plan", "stage-replay", "replay"):
            from behaviour_prediction.olmo_pretraining.native import materialize, plan_replay

            plan = plan_replay(root, design)
            if args.stage == "stage-replay":
                materialize(root, plan)
            if args.stage == "replay":
                from behaviour_prediction.olmo_pretraining.replay import run_replay

                path = root / "replay" / "staged.json"
                if not path.exists():
                    raise SystemExit("Run stage-replay to materialize the eligible source interval first.")
                staged = read_json(path)
                run_replay(root, plan, staged)


if __name__ == "__main__":
    main()
