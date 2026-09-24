"""Durable serial CPU runner: 12 O03 runs + 9 additional N01 runs."""
import argparse,json,subprocess,sys,traceback,hashlib,os
from pathlib import Path
import numpy as np,torch
from vision_protocol import generate,train_arm,analyze
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=20);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
receipt={'status':'RUNNING','pid':os.getpid(),'command':sys.argv,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'threads':8,'matrix':{'seeds':[803,805,806],'arms':['pretrained_static','pretrained_flip','random_static','random_flip','ordered','matched','shuffled_matched'],'epochs':a.epochs,'resolution':'native64 bilinear upsample224'},'completed':[]}
receipt['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('vision_protocol.py'),Path('experiments/f095_campaign/d04_vision_train.py'),Path('experiments/f095_campaign/u06_relation_distill.py')]}
def save():
    tmp=a.out/'receipt.tmp';tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(a.out/'receipt.json')
if (a.out/'receipt.json').exists():
    previous=json.loads((a.out/'receipt.json').read_text())
    if previous['status']=='PREPARED_NOT_TRAINED':
        if previous.get('data_sha256') != hashlib.sha256((a.out/'data.npz').read_bytes()).hexdigest():raise RuntimeError('Prepared data differs')
        receipt['prepared_source_sha256']=previous['source_sha256']
    else:
        for key in ['source_sha256','matrix']:
            if previous[key]!=receipt[key]:raise RuntimeError('Resume contract differs: '+key)
        if previous.get('data_sha256') != hashlib.sha256((a.out/'data.npz').read_bytes()).hexdigest():raise RuntimeError('Resume data differs')
save()
try:
    data=dict(np.load(a.out/'data.npz')) if (a.out/'data.npz').exists() else generate(a.out)
    receipt['data_sha256']=hashlib.sha256((a.out/'data.npz').read_bytes()).hexdigest();save()
    if a.prepare_only:
        receipt['status']='PREPARED_NOT_TRAINED';save();sys.exit(0)
    # Full three-seed O03 first, then the three N01 additions.
    for arms in [['pretrained_static','pretrained_flip','random_static','random_flip'],['ordered','matched','shuffled_matched']]:
        for seed in [803,805,806]:
            for arm in arms:
                out=a.out/f'{arm}_s{seed}'
                if not (out/'result.json').exists():
                    receipt['current']={'arm':arm,'seed':seed};save()
                    train_arm(data,out,arm,seed,a.epochs)
                receipt['completed'].append({'arm':arm,'seed':seed});save()
    analyze(a.out)
    receipt['status']='MATRIX_COMPLETE';receipt['scientific_decision']='PENDING_INDEPENDENT_INTERPRETATION';save()
except Exception:
    receipt['status']='FAILED';receipt['traceback']=traceback.format_exc();save();raise
