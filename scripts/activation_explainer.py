"""Experiment A: prepare, train and test a direct activation-injection decoder."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--root',type=Path,default=Path('outputs/qwen-full-v2'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/activation-explainer-a/data'))
    p.add_argument('--test-run',default='self_harm')
    p.add_argument('--validation-run',default='child_abuse')
    p.add_argument('--seed',type=int,default=42)
    p=sub.add_parser('summarize')
    p.add_argument('--root',type=Path,default=Path('outputs/activation-explainer-a'))
    p=sub.add_parser('audit')
    p.add_argument('--data-dir',type=Path,default=Path('outputs/activation-explainer-a/data'))
    p.add_argument('--source-files',action='store_true')
    for command in ('train','evaluate','baselines'):
        p=sub.add_parser(command)
        p.add_argument('--data-dir',type=Path,default=Path('outputs/activation-explainer-a/data'))
        p.add_argument('--output-dir',type=Path,default=Path('outputs/activation-explainer-a')/dict(train='decoder',evaluate='evaluation',baselines='baselines')[command])
        if command=='train':
            p.add_argument('--epochs',type=int,default=3)
            p.add_argument('--batch-size',type=int,default=4)
            p.add_argument('--accumulation',type=int,default=4)
            p.add_argument('--learning-rate',type=float,default=1e-4)
            p.add_argument('--seed',type=int,default=42)
            p.add_argument('--resume',action='store_true')
            p.add_argument('--smoke-test',action='store_true')
            p.add_argument('--unbalanced-training',action='store_true',help='Use natural training prevalence instead of balanced training-only sampling')
        if command=='evaluate':
            p.add_argument('--run-dir',type=Path,default=Path('outputs/activation-explainer-a/decoder'))
            p.add_argument('--batch-size',type=int,default=4)
            p.add_argument('--max-new-tokens',type=int,default=160)
        if command=='baselines':
            p.add_argument('--layers',type=int,nargs='+',default=[15,31])
    args=parser.parse_args()
    if args.command=='prepare':
        from behaviour_prediction.explainer.data import prepare
        prepare(args.root,args.output_dir,test_run=args.test_run,validation_run=args.validation_run,seed=args.seed)
    elif args.command=='audit':
        from behaviour_prediction.explainer.data import audit
        audit(args.data_dir,source_files=args.source_files)
    elif args.command=='train':
        from behaviour_prediction.explainer.runtime import train
        train(args.data_dir,args.output_dir,epochs=args.epochs,batch_size=args.batch_size,accumulation=args.accumulation,
              learning_rate=args.learning_rate,seed=args.seed,resume=args.resume,smoke=args.smoke_test,balanced=not args.unbalanced_training)
    elif args.command=='evaluate':
        from behaviour_prediction.explainer.runtime import evaluate
        evaluate(args.data_dir,args.run_dir,args.output_dir,batch_size=args.batch_size,max_new_tokens=args.max_new_tokens)
    elif args.command=='baselines':
        from behaviour_prediction.explainer.baselines import run_baselines
        run_baselines(args.data_dir,args.output_dir,layers=args.layers)
    else:
        from behaviour_prediction.explainer.plots import summarize
        summarize(args.root)


if __name__=='__main__':
    main()
