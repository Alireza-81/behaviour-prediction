"""Explicitly replace only the selected training run and its generated reviews."""
from pathlib import Path
import shutil


def prepare_run_outputs(output, run_name, monitor, *, overwrite=False):
    output=Path(output)
    if Path(run_name).name!=run_name or run_name in ('','.','..'):
        raise ValueError('Use a simple run name')
    if output.name!=run_name:
        raise ValueError('Output directory must belong to the selected run')
    paths=[]
    response=monitor.get('response_generation')
    if response:
        root=Path(response.get('output_root',output.parent/'reviews'))
        for suite in response['suites']:
            if Path(suite).name!=suite or suite in ('','.','..'):
                raise ValueError('Response suite names must be simple directory names')
            paths.extend(root/suite/f'{run_name}{suffix}' for suffix in ('.jsonl','.csv'))
    # Validate the entire removal set before modifying anything.
    for path in [output,*paths]:
        if path.is_symlink():raise ValueError(f'Refusing to overwrite a symlink: {path}')
    if output.exists() and not output.is_dir():raise ValueError(f'Output is not a directory: {output}')
    if any(p.exists() and not p.is_file() for p in paths):raise ValueError('Expected response sidecars to be files')
    nonempty=output.exists() and any(output.iterdir())
    reviews=[p for p in paths if p.exists()]
    if not overwrite:
        if nonempty:
            raise ValueError(f'Output directory is not empty: {output}. Use --overwrite --runs {run_name} to replace this run.')
        if reviews:
            raise ValueError(f'Responses already exist: {reviews[0]}. Use --overwrite --runs {run_name} to replace this run and its generated reviews.')
        return
    if output.exists():
        shutil.rmtree(output)
        print(f'Removed previous training outputs: {output}')
    for path in reviews:
        if path.exists():
            path.unlink()
            print(f'Removed previous generated reviews: {path}')
