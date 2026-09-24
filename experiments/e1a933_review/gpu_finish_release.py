"""Wait for local collection events and certify compute-artifact shutdown safety.

This is not scientific acceptance and never powers off the rented instance.
"""
import ctypes
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import tarfile
import traceback

BASE = Path('artifacts/e1a933_review/gpu_finish_20260924').resolve()
STATUS = BASE / 'GPU_RELEASE_STATUS.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(state):
    state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(STATUS)


def wait_for_collection():
    folder = BASE / 'sampling'
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    fd = libc.inotify_init1(os.O_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_init1')
    try:
        if libc.inotify_add_watch(fd, os.fsencode(folder), 0x8 | 0x80) < 0:
            raise OSError(ctypes.get_errno(), 'inotify_add_watch')
        while True:
            state = json.loads((folder / 'COLLECTION_STATUS.json').read_text())
            if state['status'] == 'COLLECTED_VERIFIED':
                return
            if state['status'] not in ['WAITING_FOR_REMOTE_COMPLETION_EVENT', 'COLLECTING']:
                raise RuntimeError(f'Sampling collection did not succeed: {state["status"]}')
            select.select([fd], [], [])
            os.read(fd, 65536)
    finally:
        os.close(fd)


def verify_task(name, output_name, expected):
    base = BASE / name
    collection = json.loads((base / 'COLLECTION_STATUS.json').read_text())
    if collection['status'] != 'COLLECTED_VERIFIED':
        raise RuntimeError(f'{name}: collection not verified')
    dest = base / 'extracted'
    output = dest / output_name
    if not output.exists():
        dest.mkdir(exist_ok=False)
        with tarfile.open(base / 'collected' / f'{output_name}.tar.gz') as archive:
            for entry in archive.getmembers():
                if entry.issym() or entry.islnk() or not (dest / entry.name).resolve().is_relative_to(dest):
                    raise ValueError(f'Unsafe archive entry {entry.name}')
            archive.extractall(dest)
    receipt = json.loads((output / 'receipt.json').read_text())
    if receipt['status'] != 'MATRIX_COMPLETE' or len(receipt['completed']) != expected:
        raise RuntimeError(f'{name}: incomplete matrix')
    records = json.loads((output / 'ARTIFACT_MANIFEST.json').read_text())
    for row in records:
        path = output / row['path']
        if not path.resolve().is_relative_to(output) or path.is_symlink():
            raise ValueError(f'Unsafe manifest path {path}')
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise RuntimeError(f'Hash mismatch: {path}')
    result_name = 'receipt.json' if name == 'football_v2' else 'result.json'
    result_files = list(output.glob(f'*/{result_name}'))
    if name != 'vlm' and len(result_files) != expected:
        raise RuntimeError(f'{name}: missing per-arm result files')
    if name == 'football_v2':
        if {p.parent.name for p in result_files} != set(receipt['completed']):
            raise RuntimeError('Football per-arm receipts do not match completed arms')
    for result_file in result_files:
        result = json.loads(result_file.read_text())
        if 'checkpoint_sha256' in result and sha(result_file.parent / 'model.pt') != result['checkpoint_sha256']:
            raise RuntimeError(f'Checkpoint receipt mismatch: {result_file}')
    verified = dict(status='PASS', expected=expected, completed=len(receipt['completed']), verified_files=len(records), path=str(output))
    (base / 'LOCAL_ARTIFACT_VERIFICATION.json').write_text(json.dumps(verified, indent=2) + '\n')
    return verified


def main():
    instance_lock = (BASE / 'release_finalizer.lock').open('a')
    fcntl.flock(instance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='WAITING_FOR_SAMPLING_COLLECTION', pid=os.getpid(), safe_to_shutdown=False,
                 scientific_acceptance='PENDING', wait_method='local inotify completion event; no remote polling')
    save(state)
    try:
        wait_for_collection()
        state['status'] = 'VERIFYING_LOCAL_ARTIFACTS'
        save(state)
        checks = {}
        for name, output, count in [('n03', 'n03_results', 6), ('football_v2', 'football_results_v2', 12),
                                    ('vlm', 'vlm_results', 272), ('sampling', 'sampling_results', 48)]:
            checks[name] = verify_task(name, output, count)
        original = json.loads((BASE.parent / 'remote_vision_20260924_ssh30891/COLLECTION_STATUS.json').read_text())
        if original['status'] != 'COLLECTED_VERIFIED':
            raise RuntimeError('Original 21-arm matrix not collected')
        sampling = Path(checks['sampling']['path'])
        rows = []
        for result in sorted(sampling.glob('*/result.json')):
            row = json.loads(result.read_text())
            rows.append(dict(run_directory=result.parent.name, **row))
        (BASE / 'SAMPLING_RESULTS_COLLECTED.json').write_text(json.dumps(dict(status='MATRIX_COMPLETE', rows=rows), indent=2) + '\n')
        state.update(status='SAFE_TO_SHUTDOWN', safe_to_shutdown=True, tasks=checks,
                     completed_training_arms_including_original=87, completed_vlm_requests=272,
                     scientific_acceptance='Sampling independent interpretation and final paper integration remain local CPU work',
                     instance_action='No shutdown issued; user can release instance after reading this verified receipt')
        save(state)
        lines = ['# GPU实例释放回执', '', '状态：**SAFE_TO_SHUTDOWN**。本轮已启动GPU任务均完成，必需结果已回收并通过逐文件SHA256校验。', '',
                 '- 原视觉21臂、N03六臂、O05十二臂、N02四十八臂，共87个训练臂。',
                 '- O06原生VLM完成272次独立请求。',
                 '- 无关机命令已执行；可由用户通过实例平台释放。',
                 '- 科学结论并未自动验收；采样解释、论文整合及剩余CPU事项可本地继续，不需要保留GPU。', '',
                 f'权威机器回执：`{STATUS}`。']
        Path('reports/e1a933_review/GPU_RELEASE_RECEIPT.md').write_text('\n'.join(lines) + '\n')
    except BaseException:
        state.update(status='BLOCKED_NOT_SAFE_TO_SHUTDOWN', safe_to_shutdown=False, traceback=traceback.format_exc())
        save(state)
        raise


if __name__ == '__main__':
    main()
