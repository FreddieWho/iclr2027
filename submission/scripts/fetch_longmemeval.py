"""Fetch public author data with bounded retries; never bypass gated access."""
import argparse
import hashlib
import time
from pathlib import Path
import requests
from core import write_json
BASE='https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/'
NAMES=['longmemeval_s_cleaned.json','longmemeval_oracle.json']

def fetch(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);entries=[]
    for name in NAMES:
        dest=out/name
        if dest.exists():
            # Existing downloads are never silently overwritten; verify recorded hash.
            prior=out/'download_manifest.json'
            if not prior.exists():raise RuntimeError('Existing data without manifest: inspect manually, do not silently trust')
            old=__import__('json').loads(prior.read_text())['files']
            ent=next((x for x in old if x['name']==name),None)
            sha=hashlib.sha256(dest.read_bytes()).hexdigest()
            if not ent or sha!=ent['sha256']:raise RuntimeError('Existing data checksum mismatch')
            entries.append(ent);continue
        for attempt in range(3):
            try:
                with requests.get(BASE+name,stream=True,timeout=(15,120),allow_redirects=True) as r:
                    if r.status_code in (401,403):raise PermissionError('Dataset requires permission/login; no bypass attempted')
                    r.raise_for_status();h=hashlib.sha256();size=0
                    with (out/(name+'.part')).open('wb') as f:
                        for chunk in r.iter_content(1024*1024):
                            if chunk:size+=len(chunk);h.update(chunk);f.write(chunk)
                    (out/(name+'.part')).replace(dest)
                    ent={'name':name,'url':BASE+name,'resolved_url':r.url,'sha256':h.hexdigest(),'bytes':size,'etag':r.headers.get('ETag'),'download_unix':time.time(),'license_status':'Verify dataset card/terms before redistribution'}
                    entries.append(ent)
                    # Persist incremental success so a failed second file can resume safely.
                    write_json(out/'download_manifest.json',{'files':entries})
                    break
            except PermissionError:raise
            except (requests.RequestException,OSError):
                if attempt==2:raise
                time.sleep(2**(attempt+1))
    write_json(out/'download_manifest.json',{'files':entries})
    print(f'Downloaded/verified {len(entries)} files. Verify dataset license separately.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();fetch(a.out)
