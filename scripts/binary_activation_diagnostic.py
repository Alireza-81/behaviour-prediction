"""Single-token binary diagnostic for the existing activation-injection reader."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    root=Path('outputs/activation-explainer-a/binary')
    for command in ['overfit','train']:
        p=sub.add_parser(command)
        p.add_argument('--data-dir',type=Path,default=Path('outputs/activation-explainer-a/data'))
        p.add_argument('--output-dir',type=Path,default=root/command)
        p.add_argument('--batch-size',type=int,default=4)
        p.add_argument('--accumulation',type=int,default=4)
        p.add_argument('--learning-rate',type=float,default=1e-4)
        p.add_argument('--epochs',type=int,default=3)
        p.add_argument('--overfit-updates',type=int,default=60)
        p.add_argument('--overfit-size',type=int,default=32)
        p.add_argument('--seed',type=int,default=42)
        p.add_argument('--resume',action='store_true')
    p=sub.add_parser('evaluate')
    p.add_argument('--data-dir',type=Path,default=Path('outputs/activation-explainer-a/data'))
    p.add_argument('--run-dir',type=Path,default=root/'train')
    p.add_argument('--output-dir',type=Path,default=root/'evaluation')
    p.add_argument('--batch-size',type=int,default=4)
    p=sub.add_parser('summarize')
    p.add_argument('--root',type=Path,default=root)
    args=parser.parse_args()
    if args.command in {'train','overfit'}:
        from behaviour_prediction.explainer.binary import train
        train(args.data_dir,args.output_dir,overfit=args.command=='overfit',epochs=args.epochs,
              overfit_updates=args.overfit_updates,overfit_size=args.overfit_size,batch_size=args.batch_size,
              accumulation=args.accumulation,learning_rate=args.learning_rate,seed=args.seed,resume=args.resume)
    elif args.command=='evaluate':
        from behaviour_prediction.explainer.binary import evaluate
        evaluate(args.data_dir,args.run_dir,args.output_dir,batch_size=args.batch_size)
    else:
        from behaviour_prediction.explainer.binary_plots import summarize
        summarize(args.root)


if __name__=='__main__':
    main()
