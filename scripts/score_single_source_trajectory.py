"""Score a single activation prefix from a numeric advice archive or a JSON list."""
import argparse
import json
from pathlib import Path

from behaviour_prediction.monitoring.trajectory_forecast import score_prefix

DEFAULT_BANK = Path(__file__).resolve().parents[1] / "configs/monitor-study/llama-trajectory-v2.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--run", help="Run key when the input is a multi-run archive")
    parser.add_argument("--through-step", type=int, choices=range(0, 41, 5), default=40)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    if args.run:
        data = data["runs"][args.run]
    if not isinstance(data, list):
        parser.error("Supply --run for a multi-run archive or provide a list of observed checkpoints")
    points = [p for p in data if p["step"] <= args.through_step]
    result = score_prefix(points, json.loads(args.bank.read_text()))
    print(json.dumps(dict(observed_through=points[-1]["step"], horizon=10, **result), indent=2))


if __name__ == "__main__":
    main()
