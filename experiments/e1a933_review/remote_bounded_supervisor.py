"""Run a frozen command, validate its declared receipt, and seal its outputs.

The supervisor waits on child exit; collectors use the existing inotify watcher.
Each job has a new directory and never overwrites an earlier run.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import traceback


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    spec_path = Path(sys.argv[1]).resolve()
    spec = json.loads(spec_path.read_text())
    job = spec_path.parent
    root = Path(spec['cwd']).resolve()
    output = Path(spec['output_dir']).resolve()
    archive = Path(spec['archive']).resolve()
    for path in (job, output, archive):
        if not path.is_relative_to(root):
            raise ValueError(f'Output outside declared root: {path}')
    if (job / 'status.json').exists() or output.exists() or archive.exists():
        raise FileExistsError('Immutable job/output already exists')
    state = dict(task_id=spec['task_id'], supervisor_pid=os.getpid(),
                 status='WAITING_GPU', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 scientific_decision='PENDING_INDEPENDENT_INTERPRETATION')

    def save():
        temp = job / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(job / 'status.json')

    save()
    try:
        gpu_lock = None
        if spec.get('gpu_lock'):
            gpu_lock = open(spec['gpu_lock'], 'a')
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
        state.update(status='RUNNING', compute_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        env = os.environ.copy()
        env.update(spec.get('env', {}))
        with (job / 'task.stdout.log').open('xb') as stdout, (job / 'task.stderr.log').open('xb') as stderr:
            child = subprocess.Popen(spec['argv'], cwd=root, env=env,
                                     stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
            state['training_pid'] = child.pid
            save()
            state['exit_code'] = child.wait()
        if gpu_lock is not None:
            gpu_lock.close()
        if state['exit_code']:
            raise RuntimeError(f"Command exited with {state['exit_code']}")
        receipt = json.loads((output / 'receipt.json').read_text())
        if receipt['status'] != spec['receipt_status']:
            raise RuntimeError('Unexpected receipt status')
        if 'expected_count' in spec:
            completed = receipt['completed']
            if len(completed) != spec['expected_count'] or len({json.dumps(x, sort_keys=True) for x in completed}) != len(completed):
                raise RuntimeError('Incomplete or duplicate completed arms')
        for name in spec.get('required_files', []):
            path = (output / name).resolve()
            if not path.is_relative_to(output) or not path.is_file():
                raise RuntimeError(f'Missing required output: {name}')
        records = []
        for path in sorted(output.rglob('*')):
            if path.is_symlink():
                raise ValueError(f'Symlink in result: {path}')
            if path.is_file():
                records.append(dict(path=str(path.relative_to(output)), bytes=path.stat().st_size, sha256=digest(path)))
        (output / 'ARTIFACT_MANIFEST.json').write_text(json.dumps(records, indent=2) + '\n')
        with tarfile.open(archive, 'x:gz', compresslevel=1) as handle:
            handle.add(output, arcname=output.name)
            handle.add(job / 'task.stdout.log', arcname='_job/task.stdout.log')
            handle.add(job / 'task.stderr.log', arcname='_job/task.stderr.log')
            handle.add(spec_path, arcname='_job/spec.json')
        outputs = [output / 'receipt.json', output / 'ARTIFACT_MANIFEST.json', archive]
        (job / 'results').mkdir()
        (job / 'results/sha256sums.txt').write_text(''.join(f'{digest(p)}  {p}\n' for p in outputs))
        state.update(status='SUCCEEDED', completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     archive_bytes=archive.stat().st_size, archive_sha256=digest(archive))
        save()
        (job / 'succeeded_at_utc.txt').write_text(state['completed_utc'] + '\n')
    except BaseException:
        state.update(status='FAILED', ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
