"""Portable entry point. Default is preflight only; --execute starts all 21 runs."""
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
os.environ['TORCH_HOME']=str(a.bundle.resolve()/'torch_cache')
import numpy as np,torch,torchvision
import vision_cuda_protocol as protocol
receipt={'status':'PREFLIGHT','backend':'cuda','dtype':'FP32','scientific_decision':'NOT_RUN','command':sys.argv,'pid':os.getpid(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'epochs':20,'batch':32,'resolution':224,'seeds':[803,805,806],'arms':['pretrained_static','pretrained_flip','random_static','random_flip','ordered','matched','shuffled_matched'],'torch':torch.__version__,'torchvision':torchvision.__version__,'source_sha256':{Path(f).name:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [__file__,protocol.__file__]}}
a.out.mkdir(parents=True,exist_ok=False)
def save():
    tmp=a.out/'receipt.tmp';tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(a.out/'receipt.json')
save()
try:
    manifest=json.loads((a.bundle/'manifest.json').read_text())
    for item in manifest['files']:
        path=a.bundle/item['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('Bundle hash mismatch: '+item['path'])
    receipt['bundle_manifest_sha256']=hashlib.sha256((a.bundle/'manifest.json').read_bytes()).hexdigest()
    receipt['cuda_available']=torch.cuda.is_available()
    if not torch.cuda.is_available():
        receipt['status']='NOT_RUN_NO_CUDA';receipt['unrun']=['CUDA device allocation','224 batch32 FP32 forward/backward','GPU memory measurement','CPU versus CUDA numeric comparison','GPU throughput estimate','21-run CUDA training matrix'];save();sys.exit(0)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    protocol.DEVICE=torch.device('cuda')
    receipt['device_name']=torch.cuda.get_device_name(0);receipt['cuda_version']=torch.version.cuda
    receipt['cudnn_version']=torch.backends.cudnn.version();receipt['tf32']=False;receipt['amp']=False
    data=dict(np.load(a.bundle/'data.npz'));receipt['data_sha256']=hashlib.sha256((a.bundle/'data.npz').read_bytes()).hexdigest()
    # Preflight checks both class and matched-aux gradients, with no optimizer step.
    torch.manual_seed(803);net=protocol.DistillNet(10).to(protocol.DEVICE)
    x=protocol.prep(data['train_images'][:32],224);y=torch.tensor(data['train_labels'][:32],device=protocol.DEVICE)
    orbit,_=protocol.normalize_orbit(protocol.target_orbit(data['train_coords']))
    target=torch.tensor(orbit[:32],device=protocol.DEVICE,dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.monotonic()
    z,aux=net(x);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)+protocol.matched_mse(aux,target);loss.backward()
    torch.cuda.synchronize();receipt['preflight_seconds']=time.monotonic()-t;receipt['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    assert all(torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None)
    del net,x,y,target,orbit;torch.cuda.empty_cache()
    if not a.execute:
        receipt['status']='CUDA_PREFLIGHT_COMPLETE_MATRIX_NOT_RUN';save();sys.exit(0)
    receipt['status']='RUNNING';receipt['completed']=[];save()
    for arms in [['pretrained_static','pretrained_flip','random_static','random_flip'],['ordered','matched','shuffled_matched']]:
        for seed in [803,805,806]:
            for arm in arms:
                receipt['current']={'arm':arm,'seed':seed};save()
                protocol.train_arm(data,a.out/f'{arm}_s{seed}',arm,seed,epochs=20,res=224,batch=32)
                receipt['completed'].append(receipt['current']);save()
    protocol.analyze(a.out);receipt['status']='MATRIX_COMPLETE';receipt['scientific_decision']='PENDING_INDEPENDENT_INTERPRETATION';save()
except Exception:
    receipt['status']='FAILED';receipt['traceback']=traceback.format_exc();save();raise
