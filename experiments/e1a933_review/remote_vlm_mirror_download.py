"""Retrieve a fixed public snapshot; verify weight hashes from official HF pages."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

root = Path(sys.argv[1]).resolve()
os.environ.update(HF_ENDPOINT='https://hf-mirror.com', HF_HOME=str(root / 'hf_cache'), HF_HUB_DISABLE_XET='1')
state = dict(status='RUNNING', pid=os.getpid(), endpoint=os.environ['HF_ENDPOINT'],
             started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
state_path = root / 'VLM_MIRROR_STATUS.json'
revision = '66285546d2b821cf421d4f5eb2576359d3770cd3'
hashes = {'model-00001-of-00002.safetensors': '41a8895c164b4d32bae6b302f4603fcbc1797f32dafa45c7e9bcda23c6755df8',
          'model-00002-of-00002.safetensors': '365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4'}


def save():
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(state_path)


save()
try:
    from huggingface_hub import snapshot_download
    import torch
    import transformers
    path = Path(snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', revision=revision, max_workers=3))
    for name, expected in hashes.items():
        h = hashlib.sha256()
        with (path / name).open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError(f'Official weight SHA256 mismatch: {name}')
    result = dict(path=str(path), revision=revision, torch=torch.__version__, transformers=transformers.__version__,
                  weight_sha256=hashes, source='public mirror; both weight SHA256 checked against official fixed-revision HF pages',
                  official_sources=[f'https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/{revision}/{name}' for name in hashes])
    (root / 'VLM_MODEL.json').write_text(json.dumps(result, indent=2) + '\n')
    state.update(status='SUCCEEDED', completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()
except BaseException:
    state.update(status='FAILED', traceback=traceback.format_exc())
    save()
    raise
