"""Six additional CUDA runs; existing O03/N01 matrix remains read-only."""
import argparse,json,hashlib,os,sys,time,traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
p=argparse.ArgumentParser();p.add_argument('--base-bundle',type=Path,required=True);p.add_argument('--baseline-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
os.environ['TORCH_HOME']=str(a.base_bundle.resolve()/'torch_cache')
import numpy as np,torch,torchvision
import vision_n03_protocol as protocol
import vision_cuda_protocol as shared
receipt={'status':'STARTING','scientific_decision':'PENDING_INDEPENDENT_INTERPRETATION','pid':os.getpid(),'command':sys.argv,'backend':'cuda','dtype':'FP32','epochs':20,'batch':32,'resolution':224,'seeds':[803,805,806],'arms':['geometry_bottleneck','generic_bottleneck'],'completed':[],'torch':torch.__version__,'torchvision':torchvision.__version__,'source_sha256':{Path(f).name:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [__file__,protocol.__file__,shared.__file__]},'comparison_contract':'All six new arms share ImageNet initialization and new bottleneck architecture. O03/N01 baselines are the complete same-GPU matrix; no CPU mixing.'}
a.out.mkdir(parents=True,exist_ok=False)
def save():
    tmp=a.out/'receipt.tmp';tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(a.out/'receipt.json')
save()
try:
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    manifest=json.loads((a.base_bundle/'manifest.json').read_text())
    data_sha=hashlib.sha256((a.base_bundle/'data.npz').read_bytes()).hexdigest()
    expected=next(i['sha256'] for i in manifest['files'] if i['path']=='data.npz')
    if data_sha!=expected:raise RuntimeError('Frozen base data differs')
    receipt['data_sha256']=data_sha;receipt['device_name']=torch.cuda.get_device_name(0)
    for seed in receipt['seeds']:
        for arm in ['pretrained_flip','matched']:
            old=json.loads((a.baseline_root/f'{arm}_s{seed}'/'result.json').read_text())
            if old.get('backend')!='cuda':raise RuntimeError('Baseline backend is not CUDA')
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    protocol.DEVICE=shared.DEVICE=torch.device('cuda')
    data=dict(np.load(a.base_bundle/'data.npz'));receipt['status']='RUNNING';save()
    for seed in receipt['seeds']:
        for arm in receipt['arms']:
            receipt['current']={'arm':arm,'seed':seed};save()
            protocol.train_arm(data,a.out/f'{arm}_s{seed}',arm,seed,epochs=20,res=224,batch=32)
            receipt['completed'].append(receipt['current']);save()
    analysis={}
    for baseline,base in [('generic_bottleneck',a.out),('pretrained_flip',a.baseline_root),('matched',a.baseline_root)]:
        rows=[]
        for seed in receipt['seeds']:
            aa=np.load(base/f'{baseline}_s{seed}'/'predictions.npz');bb=np.load(a.out/f'geometry_bottleneck_s{seed}'/'predictions.npz')
            assert np.array_equal(aa['labels'],bb['labels']) and np.array_equal(aa['parent'],bb['parent'])
            ao=(aa['logits']>0)==aa['labels'];bo=(bb['logits']>0)==bb['labels'];aj=ao[:,1:].all(1);bj=bo[:,1:].all(1);h=ao[:,1]&ao[:,2]&~ao[:,3]
            delta=bj.astype(float)-aj.astype(float);parent=aa['parent'];unique=np.unique(parent);rng=np.random.default_rng(953000+seed);boot=[]
            for _ in range(2000):
                ix=np.concatenate([np.flatnonzero(parent==v) for v in rng.choice(unique,len(unique),replace=True)]);boot.append(delta[ix].mean())
            rows.append({'seed':seed,'J_difference':float(delta.mean()),'parent_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'baseline110_n':int(h.sum()),'full_repair_n':int((bj&h).sum()),'migration_n':int((bo[:,3]&~bo[:,1:3].all(1)&h).sum()),'baseline111_n':int(aj.sum()),'111_regression_n':int((aj&~bj).sum())})
        analysis['geometry_bottleneck minus '+baseline]={'seeds':rows,'mean_J_difference':float(np.mean([r['J_difference'] for r in rows]))}
    (a.out/'paired_analysis.json').write_text(json.dumps(analysis,indent=2));receipt['status']='MATRIX_COMPLETE';save()
except Exception:
    receipt['status']='FAILED';receipt['traceback']=traceback.format_exc();save();raise
