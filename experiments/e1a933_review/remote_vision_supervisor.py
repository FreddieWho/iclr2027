"""Persist a declared remote CUDA job and package verified outputs on success."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import traceback


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    task_path = Path(sys.argv[1]).resolve()
    task = json.loads(task_path.read_text())
    job_dir = task_path.parent
    base = job_dir.parent.parent
    matrix = base / 'cuda_matrix'
    state = {'task_id': task['task_id'], 'supervisor_pid': os.getpid(),
             'status': 'RUNNING', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'scientific_decision': 'PENDING_INDEPENDENT_INTERPRETATION'}

    def save():
        tmp = job_dir / 'status.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(job_dir / 'status.json')

    save()
    (job_dir / 'pid').write_text(str(os.getpid()) + '\n')
    (job_dir / 'started_at_utc.txt').write_text(state['started_utc'] + '\n')
    try:
        with (job_dir / 'task.stdout.log').open('xb') as stdout, (job_dir / 'task.stderr.log').open('xb') as stderr:
            child = subprocess.Popen(shlex.split(task['command']), cwd=base, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr)
            state['training_pid'] = child.pid
            save()
            code = child.wait()
        state['exit_code'] = code
        if code:
            raise RuntimeError(f'Training exited with {code}; see task.stderr.log')
        receipt = json.loads((matrix / 'receipt.json').read_text())
        if receipt['status'] != 'MATRIX_COMPLETE' or len(receipt['completed']) != 21:
            raise RuntimeError('Complete 21-arm receipt missing')
        for arm in receipt['arms']:
            for seed in receipt['seeds']:
                arm_dir = matrix / f'{arm}_s{seed}'
                result = json.loads((arm_dir / 'result.json').read_text())
                if sha(arm_dir / 'model.pt') != result['checkpoint_sha256']:
                    raise RuntimeError(f'Checkpoint mismatch: {arm_dir.name}')
                for name in ['predictions.npz', 'test_single_predictions.npz', 'history.json']:
                    if not (arm_dir / name).is_file():
                        raise RuntimeError(f'Missing {arm_dir.name}/{name}')
        records = [{'path': str(p.relative_to(matrix)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                   for p in sorted(matrix.rglob('*')) if p.is_file()]
        (matrix / 'ARTIFACT_MANIFEST.json').write_text(json.dumps(records, indent=2) + '\n')
        archive = base / 'cuda_matrix_results.tar.gz'
        if archive.exists():
            raise FileExistsError(archive)
        with tarfile.open(archive, 'x:gz', compresslevel=1) as tf:
            tf.add(matrix, arcname='cuda_matrix')
        for name in task['outputs']:
            path = Path(name)
            if not path.is_file() or not path.is_relative_to(base):
                raise RuntimeError(f'Invalid or missing declared output: {name}')
        (job_dir / 'results').mkdir()
        (job_dir / 'results' / 'sha256sums.txt').write_text(''.join(
            f'{sha(Path(name))}  {name}\n' for name in task['outputs']))
        state.update(status='SUCCEEDED', completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     archive_bytes=archive.stat().st_size, archive_sha256=sha(archive))
        save()
        (job_dir / 'succeeded_at_utc.txt').write_text(state['completed_utc'] + '\n')
    except BaseException:
        state.update(status='FAILED', traceback=traceback.format_exc(),
                     ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        raise


if __name__ == '__main__':
    main()
