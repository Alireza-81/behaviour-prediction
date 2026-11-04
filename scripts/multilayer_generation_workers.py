"""Isolated generation processes, private journals, and a single merging writer.

Only this coordinator writes the canonical journal/CSVs. Worker journals survive
interruptions and are merged on the next run before any missing work is assigned.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from behaviour_prediction.monitor.shared_mask import identity
from run_dynamic_gradient_activation_patch import atomic_json


@contextmanager
def generation_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'.generation.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another generation/selection process still owns this output folder. Stop it before resuming.') from None
        try:
            yield handle.fileno()
        finally:
            # Do not explicitly unlock: children inherit this same open-file
            # description, keeping the lock until ALL owners have closed it.
            pass


@contextmanager
def durable_row_write():
    # A Ctrl-C must not split one canonical JSON row and then append another
    # during shutdown. Deliver it after the row and registry update complete.
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


class JournalTail:
    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.pending = b''
        self.discarded_tail = False

    def rows(self, final=False):
        if not self.path.exists(): return
        with self.path.open('rb') as handle:
            handle.seek(self.offset)
            data = handle.read()
            self.offset = handle.tell()
        self.pending += data
        while b'\n' in self.pending:
            line, self.pending = self.pending.split(b'\n', 1)
            if line.strip(): yield json.loads(line)
        if final and self.pending:
            try:
                row = json.loads(self.pending)
            except json.JSONDecodeError:
                self.discarded_tail = True # incomplete final answer; leave source journal untouched
            else:
                yield row
            self.pending = b''


def recover_worker_journals(folder, accept):
    recovered = 0
    for path in sorted((folder/'workers').glob('*/worker-*/generated.jsonl')):
        tail = JournalTail(path)
        for row in tail.rows(final=True): recovered += bool(accept(row))
        if tail.discarded_tail: print(f'Ignored incomplete final worker row in {path}; completed rows preserved.', flush=True)
    return recovered


def partition_tasks(tasks, workers, condition_count):
    """Disjoint prompt assignments balanced by remaining answer count."""
    bins = [[] for _ in range(min(workers, len(tasks)))]; loads = [0]*len(bins)
    weighted = [(2*condition_count-len(task['saved']), index, task) for index,task in enumerate(tasks)]
    for weight, _, task in sorted(weighted, key=lambda item:(-item[0], item[1])):
        target = min(range(len(bins)), key=lambda i:loads[i])
        bins[target].append(task); loads[target] += weight
    return bins


def parallel_generate(folder, manifest, selected, tasks, workers, lock_fd, accept, refresh,
                      command_factory=None, poll_interval=.25):
    assignments = partition_tasks(tasks, workers, len(manifest['conditions']))
    if not assignments: return
    session = folder/'workers'/uuid.uuid4().hex
    session.mkdir(parents=True)
    processes = []; tails = []; logs = []
    environment = dict(os.environ, TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    environment.setdefault('MPLCONFIGDIR', '/tmp/behaviour-matplotlib')
    try:
        for index, assigned in enumerate(assignments):
            directory = session/f'worker-{index}';directory.mkdir()
            value = dict(manifest=manifest, selected=selected, tasks=assigned,
                         parent_pid=os.getpid(), worker=index,
                         mask_sha256=hashlib.sha256((folder/'mask.json').read_bytes()).hexdigest())
            value['task_id'] = identity(value, 'task_id')
            task_path = directory/'task.json';atomic_json(task_path,value)
            log = (directory/'worker.log').open('w');logs.append(log)
            command = command_factory(task_path) if command_factory else [sys.executable, str(Path(__file__).resolve()), str(task_path)]
            # Fresh Python interpreters: never fork an initialized CUDA context.
            # Inherited lock protects against a second coordinator even if this
            # coordinator is killed before its workers finish shutting down.
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                env=environment, pass_fds=(lock_fd,))
            processes.append(process);tails.append(JournalTail(directory/'generated.jsonl'))
            print(f'Worker {index+1}/{len(assignments)} started; {len(assigned)} prompts; log: {directory}/worker.log',flush=True)
        while True:
            for tail in tails:
                for row in tail.rows(): accept(row)
            refresh()
            failures = [(i,p.returncode) for i,p in enumerate(processes) if p.poll() is not None and p.returncode != 0]
            if failures:
                raise RuntimeError(f'Generation worker failure {failures}; completed answers preserved. Inspect {session}/worker-*/worker.log. Rerun with fewer workers if GPU memory was exhausted.')
            if all(p.poll() is not None for p in processes): break
            time.sleep(poll_interval)
    finally:
        # Covers Ctrl-C, validation failures, OOM, and normal completion. Only
        # independent worker processes are stopped; no model is shared.
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            for process in processes:
                if process.poll() is None: process.terminate()
            for process in processes:
                try: process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
            for log in logs: log.close()
            for tail in tails:
                for row in tail.rows(final=True): accept(row)
                if tail.discarded_tail: print(f'Worker stopped mid-answer: {tail.path}; that answer will resume from its start.',flush=True)
            refresh(force=True)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)



def worker(task_path):
    # Terminal Ctrl-C is handled by the coordinator, which terminates workers
    # and drains their complete journal rows before returning.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import torch
    from run_multilayer_activation_patch import models, generation_rows
    from run_gradient_activation_patch import digest
    value = json.loads(task_path.read_text())
    if identity(value,'task_id') != value['task_id']: raise ValueError('Worker task changed')
    m = value['manifest']
    if digest(m) != m['pilot_id']: raise ValueError('Worker manifest changed')
    folder = task_path.parents[3]
    if hashlib.sha256((folder/'mask.json').read_bytes()).hexdigest() != value['mask_sha256']:
        raise ValueError('Frozen mask changed after dispatch')
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    path = task_path.parent/'generated.jsonl'
    with path.open('x') as handle:
        for model,tokenizer in models(m):
            model.eval()
            for parameter in model.parameters(): parameter.requires_grad_(False)
            for row in generation_rows(m, value['selected'], value['tasks'], model, tokenizer,
                                       parent_alive=lambda:os.getppid()==value['parent_pid']):
                handle.write(json.dumps(row)+'\n');handle.flush();os.fsync(handle.fileno())
                print(f"Saved {row['probe_id']}/{row['direction']}/{row['condition']}",flush=True)


if __name__ == '__main__':
    worker(Path(sys.argv[1]))
