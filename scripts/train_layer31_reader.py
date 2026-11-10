"""Fresh layer-31-only training, evaluation and matched two-layer comparison."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','compare'],nargs='?',default='run')
    parser.add_argument('--source-data',type=Path,default=Path('outputs/activation-explainer-a/data'))
    parser.add_argument('--root',type=Path,default=Path('outputs/activation-explainer-a/binary-layer31'))
    parser.add_argument('--original-evaluation',type=Path,default=Path('outputs/activation-explainer-a/binary/evaluation'))
    a=parser.parse_args()
    from behaviour_prediction.explainer.layer31_training import prepare,compare
    if a.command in {'run','prepare'}:
        prepare(a.source_data,a.root/'data')
    if a.command=='run':
        from behaviour_prediction.explainer.binary import train,evaluate
        folder=a.root/'train'
        if not (folder/'completed.json').exists():
            train(a.root/'data',folder,resume=folder.exists())
        else:
            config=json.loads((folder/'training-config.json').read_text())
            if config['overfit']:raise ValueError('Expected full training, not an overfit adapter')
        import gc,torch
        gc.collect()
        torch.cuda.empty_cache()
        evaluate(a.root/'data',folder,a.root/'evaluation')
        from behaviour_prediction.explainer.binary_plots import summarize
        summarize(a.root)
    if a.command in {'run','compare'}:
        compare(a.original_evaluation,a.root/'evaluation',a.root/'comparison')


if __name__=='__main__':main()
