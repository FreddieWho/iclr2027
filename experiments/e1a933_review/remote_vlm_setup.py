"""Prepare an isolated pinned VLM environment on the already-authorized host."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

root = Path(sys.argv[1]).resolve()
state_path = root / 'VLM_SETUP_STATUS.json'
state = {'status': 'RUNNING', 'pid': os.getpid(), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'model': 'Qwen/Qwen2.5-VL-3B-Instruct', 'revision': '66285546d2b821cf421d4f5eb2576359d3770cd3'}


def save():
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(state_path)


save()
try:
    venv = root / 'vlm_venv'
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(venv)], check=True)
    py = str(venv / 'bin/python')
    subprocess.run([py, '-m', 'pip', 'install', 'transformers==4.51.3', 'accelerate==1.6.0'], check=True)
    env = os.environ.copy()
    env['HF_HOME'] = str(root / 'hf_cache')
    code = ("from huggingface_hub import snapshot_download; "
            f"p=snapshot_download({state['model']!r},revision={state['revision']!r}); "
            "import pathlib,torch,transformers,json; "
            f"pathlib.Path({str(root / 'VLM_MODEL.json')!r}).write_text(json.dumps(dict(path=p,torch=torch.__version__,transformers=transformers.__version__),indent=2))")
    subprocess.run([py, '-c', code], env=env, check=True)
    state.update(status='SUCCEEDED', completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()
except BaseException:
    state.update(status='FAILED', traceback=traceback.format_exc())
    save()
    raise
