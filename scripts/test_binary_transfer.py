"""Compare prompt transfer and model transfer without retraining the decoder."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','evaluate','run','summarize'])
    p.add_argument('--data-dir',type=Path,default=Path('outputs/activation-explainer-a/data'))
    p.add_argument('--run-dir',type=Path,default=Path('outputs/activation-explainer-a/binary/train'))
    p.add_argument('--transfer-dir',type=Path,default=Path('outputs/activation-explainer-a/binary/transfer-data'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/activation-explainer-a/binary/transfer-evaluation'))
    p.add_argument('--cache-dir',type=Path,default=Path('outputs/activation-explainer-a/binary/evaluation'))
    p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args()
    from behaviour_prediction.explainer.transfer import prepare,evaluate,summarize
    if a.command=='prepare' or (a.command=='run' and not a.transfer_dir.exists()):
        prepare(a.data_dir,a.run_dir,a.transfer_dir)
    if a.command in {'run','evaluate'}:
        evaluate(a.data_dir,a.run_dir,a.transfer_dir,a.output_dir,a.cache_dir,batch_size=a.batch_size)
    if a.command=='summarize':
        summarize(a.output_dir)


if __name__=='__main__':
    main()
