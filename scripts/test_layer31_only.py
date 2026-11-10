"""Evaluate the existing binary decoder without layer-15 activation inputs."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('outputs/activation-explainer-a/data'))
    p.add_argument('--run-dir',type=Path,default=Path('outputs/activation-explainer-a/binary/train'))
    p.add_argument('--cache-dir',type=Path,default=Path('outputs/activation-explainer-a/binary/evaluation'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/activation-explainer-a/binary/layer31-only'))
    p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args()
    from behaviour_prediction.explainer.layer_ablation import evaluate
    evaluate(a.data_dir,a.run_dir,a.cache_dir,a.output_dir,batch_size=a.batch_size)


if __name__=='__main__':main()
