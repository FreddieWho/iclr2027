"""Wait on a remote process exit event, then collect declared, hashed outputs."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--supervisor-pid', type=int, required=True)
    args = parser.parse_args()
    skill = Path('/home/huyudi/.codex/skills/lan-ssh-workflow/scripts')
    sys.path.insert(0, str(skill))
    from lan_ssh_common import load_host, ssh_command
    task = json.loads(args.task.read_text())
    host = load_host(args.config, task['host_alias'])
    state = {'status': 'WAITING_FOR_REMOTE_COMPLETION_EVENT', 'task_id': task['task_id'],
             'wait_method': 'Linux inotify + blocking select; no timer or polling loop',
             'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}

    def save():
        tmp = args.state.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(args.state)

    save()
    try:
        code = '\n'.join([
            'import os,select,json,ctypes',
            'from pathlib import Path',
            f'job=Path({task["remote_dir"]!r})',
            'libc=ctypes.CDLL(None,use_errno=True)',
            'libc.inotify_init1.argtypes=[ctypes.c_int]',
            'libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32]',
            'fd=libc.inotify_init1(os.O_CLOEXEC)',
            'if fd<0: raise OSError(ctypes.get_errno(),"inotify_init1")',
            'wd=libc.inotify_add_watch(fd,os.fsencode(job),0x8|0x80)',
            'if wd<0: raise OSError(ctypes.get_errno(),"inotify_add_watch")',
            'while True:',
            '    status=json.loads((job/"status.json").read_text())',
            '    if status["status"]=="FAILED": break',
            '    if (job/"succeeded_at_utc.txt").exists() and (job/"results/sha256sums.txt").exists(): break',
            '    try:',
            f'        os.kill({args.supervisor_pid},0)',
            '    except ProcessLookupError:',
            '        status["status"]="INCOMPLETE_SUPERVISOR_EXITED"',
            '        break',
            '    select.select([fd],[],[])',
            '    os.read(fd,65536)',
            'os.close(fd)',
            'print(json.dumps(status))',
        ])
        command = ssh_command(host, '/root/miniconda3/bin/python -')
        waited = subprocess.run(command, input=code, capture_output=True, text=True)
        state['wait_returncode'] = waited.returncode
        state['wait_stderr'] = waited.stderr
        waited.check_returncode()
        remote = json.loads(waited.stdout)
        state['remote_status'] = remote
        if remote['status'] != 'SUCCEEDED':
            raise RuntimeError('Remote matrix did not complete successfully; outputs not collected as success')
        state['status'] = 'COLLECTING'
        save()
        command = [sys.executable, str(skill/'lan_ssh_collect.py'), '--config', str(args.config),
                   '--task', str(args.task), '--destination', str(args.destination), '--execute']
        collected = subprocess.run(command, text=True, capture_output=True)
        state['collect_returncode'] = collected.returncode
        state['collect_log'] = collected.stdout
        state['collect_stderr'] = collected.stderr
        if collected.returncode:
            raise RuntimeError('Artifact transfer or SHA256 verification failed')
        state.update(status='COLLECTED_VERIFIED',
                     scientific_decision='PENDING_INDEPENDENT_INTERPRETATION',
                     completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
    except BaseException:
        state.update(status='FAILED', traceback=traceback.format_exc(),
                     ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        raise


if __name__ == '__main__':
    main()
