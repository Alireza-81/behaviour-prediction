"""Stage code/results safely for local <-> GitHub <-> GPU synchronization.

plan is read-only. stage untracks forbidden artifacts WITHOUT deleting their
working files, then stages eligible changes. Never commits, pushes, pulls,
deletes working files, or stages unrelated working-tree deletions.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

DEFAULT_RESULTS = ['outputs/qwen-full-v2', 'outputs/activation-explainer-a']
WEIGHT_EXTENSIONS = {'.safetensors','.bin','.ckpt','.gguf','.ggml','.pth','.pkl','.joblib','.h5','.onnx'}
TENSOR_DATA = {'activations.pt','vectors.pt','coordinate_scores.pt','gradient_scores.pt',
               'development_scores.pt','endpoint_activations.pt'}
MODEL_FILES = {'resume-state.pt','resume_state.pt','parameters.pt','optimizer.pt','scheduler.pt',
               'fitted-monitors.json','adapter_config.json','model.safetensors.index.json','pytorch_model.bin.index.json'}
MODEL_DIR = re.compile(r'^(?:checkpoint(?:[-_].*)?|checkpoints|adapters?|final[-_](?:model|adapter)|base_model|best_model|model_cache|weight_monitoring|weights_monitoring)$',re.I)
TEXT_EXTENSIONS = {'.py','.sh','.toml','.lock','.md','.txt','.yaml','.yml','.json','.jsonl','.csv','.tsv','.png','.svg','.pdf','.html','.ipynb','.sha256'}


def forbidden(name):
    p=PurePosixPath(name)
    if any(MODEL_DIR.fullmatch(part) for part in p.parts[:-1]):return 'model/checkpoint directory'
    if any(part in {'.venv','venv','__pycache__','.cache','.pytest_cache'} for part in p.parts):return 'environment/cache'
    if p.name in MODEL_FILES or p.suffix.lower() in WEIGHT_EXTENSIONS or p.name.endswith(('.safetensors.index.json','.bin.index.json')):return 'model/training state'
    if p.suffix.lower() in {'.pyc','.pyo'}:return 'compiled cache'
    if p.name=='.env' or p.name.startswith('.env.') or p.name in {'id_rsa','id_ed25519'}:return 'local credentials'
    if p.suffix.lower() in {'.pt','.npz','.npy'}:
        if p.name in TENSOR_DATA or (any(s in p.parts for s in ('activation_monitoring','activation_evaluation')) and re.fullmatch(r'step-\d+\.pt',p.name)):
            return None
        return 'unrecognized tensor (may be model weights)'
    return None


def generated_duplicate(name):
    p=PurePosixPath(name)
    if any(part=='history' or part.startswith('before-') for part in p.parts):return True
    if p.name=='decisions-before.json':return True
    return 'content-label-audit' in p.parts and p.name in {'cases.json','all_cases.csv','occurrences.csv','review.html'}


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args])


def files(root,*args):
    return [os.fsdecode(x) for x in git(root,*args).split(b'\0') if x]


def candidates(root, results):
    names=set(files(root,'ls-files','-m','-z'))
    for name in files(root,'ls-files','-o','--exclude-standard','-z'):
        if name.split('/')[0] in {'src','scripts','tests','configs','docs','data','.githooks'} or name in files_cache_top:
            names.add(name)
    for result in results:
        start=(root/result).resolve()
        if not start.is_relative_to(root) or not result.startswith('outputs/'):
            raise ValueError('Result directories must be inside this checkout under outputs/.')
        if not start.exists():continue
        for parent,dirs,children in os.walk(start,followlinks=False):
            relative=Path(parent).relative_to(root)
            dirs[:]=[d for d in dirs if not forbidden((relative/d/'placeholder').as_posix())
                     and d!='history' and not d.startswith('before-')]
            names.update((relative/f).as_posix() for f in children)
    accepted=[]; rejected=[]
    for name in sorted(names):
        path=root/name
        if not path.is_file():continue  # Do not stage user's unrelated deletions.
        reason=forbidden(name)
        if path.is_symlink():reason='symlink (not followed for synchronization)'
        if not reason and path.stat().st_size>50*1024**2:reason='over 50 MiB; transfer data separately'
        if reason:rejected.append((name,reason));continue
        if generated_duplicate(name):continue
        if name in files_cache_top or Path(name).suffix.lower() in TEXT_EXTENSIONS or Path(name).name in TENSOR_DATA or 'activation_monitoring/' in name:
            accepted.append(name)
    return accepted,rejected


files_cache_top={'.gitignore','.gitattributes','.githooks/pre-commit','.githooks/pre-push'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','stage','check'])
    parser.add_argument('--results',nargs='*',default=DEFAULT_RESULTS,
                        help='Result directories to include; bare --results includes no new ignored outputs')
    parser.add_argument('--staged',action='store_true',help='For check: inspect the index, not just HEAD')
    args=parser.parse_args()
    root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel']).decode().strip()).resolve()
    if args.action=='check':
        names=files(root,'ls-files','-z') if args.staged else files(root,'ls-tree','-r','--name-only','-z','HEAD')
        bad=[(n,forbidden(n)) for n in names if forbidden(n)]
        print(f'Checked {len(names)} Git paths; {len(bad)} forbidden artifacts.')
        for n,reason in bad:print(f'BLOCKED {n}: {reason}')
        return 1 if bad else 0
    accepted,rejected=candidates(root,args.results)
    tracked=files(root,'ls-files','-z')
    remove=[n for n in tracked if forbidden(n)]
    print(f'{len(accepted)} eligible code/result paths; {len(remove)} tracked artifacts to untrack (local files retained).')
    for n in remove:print('UNTRACK',n)
    for n,reason in rejected[:15]:print('SKIP',n,reason)
    if args.action=='plan':
        for n in accepted:print('STAGE',n)
        print('Read-only preview. Run stage to update the index, then inspect git diff --cached --stat.')
        return 0
    # Clean the index only; never remove the working copies or rewrite history.
    for start in range(0,len(remove),100):git(root,'rm','--cached','-f','--',*remove[start:start+100])
    for start in range(0,len(accepted),100):git(root,'add','-f','--',*accepted[start:start+100])
    bad=[n for n in files(root,'ls-files','-z') if forbidden(n)]
    if bad:raise ValueError('Unsafe indexed artifacts remain: '+repr(bad))
    print('Safe files staged. No commit or network transfer performed.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
