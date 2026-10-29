"""Run the separate no-averaging and history experiments on saved activations."""
import argparse
from pathlib import Path
import torch

from behaviour_prediction.monitor.comparisons import run_comparisons


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/monitor-comparisons"))
    args=parser.parse_args()
    # Small double-precision regressions run efficiently with one CPU thread.
    torch.set_num_threads(1)
    reports=run_comparisons(args.output_dir)
    for suite,report in reports.items():
        print(suite)
        for variant,value in report["behaviours"].items():
            print(f"  {variant}: {value['tp']+value['tn']}/{value['checkpoints']} behaviour decisions; macro F1 {value['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
