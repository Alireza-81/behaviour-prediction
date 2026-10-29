"""Compare full activation vectors and a small neural network independently."""
import argparse
from pathlib import Path
import torch

from behaviour_prediction.monitor.vector_network import run_experiments


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("outputs/vector-network-experiments"))
    args=parser.parse_args()
    torch.set_num_threads(1)
    reports=run_experiments(args.output_dir)
    for suite,report in reports.items():
        print(suite)
        for name,value in report["behaviours"].items():
            print(f"  {name}: {value['tp']+value['tn']}/{value['checkpoints']}")


if __name__ == "__main__":
    main()
