#!/usr/bin/env python3
"""N02 portable CUDA entry: explicit preflight or complete frozen 48-arm matrix."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import upgrade_protocol as protocol

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--execute',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--cache',type=Path);a=p.parse_args()
    if a.execute and a.preflight:p.error('Choose --execute or --preflight, not both')
    bundle=a.bundle.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    config=json.loads((bundle/'config.json').read_text())
    receipt=dict(status='PREFLIGHT',command=sys.argv,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),completed=[],expected_completed=30,scientific_decision='NOT_RUN',source_sha256={Path(f).name:protocol.filehash(f) for f in [__file__,protocol.__file__]},torch=torch.__version__,torchvision=torchvision.__version__,config=config)
    def save():
        tmp=out/'receipt.tmp';tmp.write_text(json.dumps(receipt,indent=2));tmp.replace(out/'receipt.json')
    save()
    try:
        manifest=json.loads((bundle/'manifest.json').read_text())
        for item in manifest['files']:
            assert protocol.filehash(bundle/item['path'])==item['sha256'],'Bundle hash mismatch: '+item['path']
        receipt['bundle_manifest_sha256']=protocol.filehash(bundle/'manifest.json')
        for name in ['config.json','data_manifest.json','manifest.json']:(out/name).write_bytes((bundle/name).read_bytes())
        data=dict(np.load(bundle/'data.npz'));receipt['data_sha256']=protocol.filehash(bundle/'data.npz')
        # Small geometry input is retained with the output so caches can be reconstructed.
        (out/'data.npz').write_bytes((bundle/'data.npz').read_bytes())
        if not torch.cuda.is_available():
            receipt['status']='NOT_RUN_NO_CUDA';receipt['unrun']=['4 distinct stem/input training-step preflights','native GPU image cache','30-arm training matrix'];save();return
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
        device='cuda';receipt.update(device_name=torch.cuda.get_device_name(0),device_total_bytes=torch.cuda.get_device_properties(0).total_memory,cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),tf32=False,amp=False)
        weight=bundle/'resnet18-f37072fd.pth';batch=config['batch'];probe_results=[];structure={}
        # One bounded probe per distinct stem/input shape; fresh disposable networks.
        for stem in ['standard','lowstride']:
            for res in [64,224]:
                net=protocol.net_for('pretrained',stem,803,weight).to(device);net.train();opt=torch.optim.Adam(net.parameters(),lr=config['lr'])
                raw=protocol.render_native(data['train_coords'][:batch],data['train_bg'][:batch],64,device)
                if res==224:raw=F.interpolate(raw,(224,224),mode='bilinear',align_corners=False)
                x=(raw-protocol.MEAN.to(device))/protocol.STD.to(device);y=torch.tensor(data['train_labels'][:batch],device=device)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic();steps=[]
                for repeat in range(3):
                    t=time.monotonic();opt.zero_grad(set_to_none=True);z=net(x).flatten();loss=F.binary_cross_entropy_with_logits(z,y);loss.backward();opt.step();torch.cuda.synchronize();steps.append(time.monotonic()-t)
                assert all(torch.isfinite(v).all() for v in net.parameters())
                probe_results.append(dict(stem=stem,res=res,batch=batch,step_seconds=steps,steady_step_seconds=float(np.mean(steps[1:])),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
                hooks=[];mac=[];shapes={}
                for name,m in net.named_modules():
                    if isinstance(m,torch.nn.Conv2d):hooks.append(m.register_forward_hook(lambda m,i,o:mac.append(int(o.numel()*m.kernel_size[0]*m.kernel_size[1]*m.in_channels/m.groups))))
                    if isinstance(m,torch.nn.Linear):hooks.append(m.register_forward_hook(lambda m,i,o:mac.append(int(o.numel()*m.in_features))))
                    if name in ['conv1','maxpool','layer1','layer2','layer3','layer4']:hooks.append(m.register_forward_hook(lambda m,i,o,name=name:shapes.update({name:list(o.shape)})))
                net.eval()
                with torch.inference_mode():net(x[:1])
                for h in hooks:h.remove()
                structure[f'{stem}_{res}']=dict(parameters=sum(p.numel() for p in net.parameters()),conv_linear_MACs=sum(mac),feature_maps=shapes)
                del net,opt,x,y,z,loss,raw;torch.cuda.empty_cache()
        receipt['preflight']=probe_results
        # Wide-net (178M params) memory probe at 64px; declarative only.
        wnet=protocol.net_for_wide('random','standard',803).to(device);wnet.train()
        wopt=torch.optim.Adam(wnet.parameters(),lr=config['lr'])
        raw=protocol.render_native(data['train_coords'][:batch],data['train_bg'][:batch],64,device)
        x=(raw-protocol.MEAN.to(device))/protocol.STD.to(device);y=torch.tensor(data['train_labels'][:batch],device=device)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        wopt.zero_grad(set_to_none=True);z=wnet(x).flatten();loss=F.binary_cross_entropy_with_logits(z,y);loss.backward();wopt.step();torch.cuda.synchronize()
        receipt['wide_preflight']=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),params=sum(p.numel() for p in wnet.parameters()))
        del wnet,wopt,x,y,z,loss,raw;torch.cuda.empty_cache()
        (out/'structure_matrix.json').write_text(json.dumps(structure,indent=2))
        # Exact observation identity is by construction; record numeric backend audit too.
        checks=[]
        for res in [64,224]:
            cpu=protocol.render_native(data['train_coords'][:4],data['train_bg'][:4],res,'cpu').numpy();gpu=protocol.render_native(data['train_coords'][:4],data['train_bg'][:4],res,'cuda').cpu().numpy()
            checks.append(dict(res=res,cpu_gpu_max_abs=float(np.abs(cpu-gpu).max()),cpu_gpu_mean_abs=float(np.abs(cpu-gpu).mean()),cpu_sha256=protocol.digest(cpu),gpu_sha256=protocol.digest(gpu)))
            assert np.abs(cpu-gpu).max()<=.8/(protocol.SS**2)+1e-5
        receipt['renderer_numeric_checks']=checks
        times={(r['stem'],r['res']):r['steady_step_seconds'] for r in probe_results};updates=int(np.ceil(len(data['train_labels'])/batch))*config['epochs']
        receipt['estimated_training_step_seconds_without_dev_eval_cache_io']=float(sum(times[(arm['stem'],protocol.resolution(arm['observation']))]*updates*len(config['seeds']) for arm in config['arms']))
        if not a.execute:
            receipt['status']='CUDA_PREFLIGHT_COMPLETE_MATRIX_NOT_RUN';save();return
        cache=a.cache.resolve() if a.cache else out.with_name(out.name+'_image_cache')
        receipt.update(status='RENDERING_CACHE',image_cache_path=str(cache));save();maps,cache_manifest=protocol.prepare_cache(data,cache,device)
        (out/'image_cache_manifest.json').write_text(json.dumps(cache_manifest,indent=2));receipt['cache_total_bytes']=sum(p.stat().st_size for p in cache.glob('*.npy'))
        # Report identical-image conflicting labels; never remove them from denominator.
        collisions={}
        for obs in ['A','C']:
            q=maps['quartet_'+obs].reshape(-1,4,3,protocol.resolution(obs),protocol.resolution(obs));labels=data['quartet_labels'];bad=[]
            for i in range(len(q)):
                if any(labels[i,j]!=labels[i,k] and np.array_equal(q[i,j],q[i,k]) for j in range(4) for k in range(j)):bad.append(i)
            collisions[obs]=dict(conflicting_identical_quartet_indices=bad,n=len(bad))
        (out/'observability_audit.json').write_text(json.dumps(dict(exact_native_raster_conflicts=collisions,n_quartets=len(data['quartet_labels']),qualification='No clipping by geometry construction; exact equality is a lower bound on unobservability, not a complete occlusion/quantization certificate.'),indent=2))
        receipt['status']='RUNNING';save()
        for seed in config['seeds']:
            for arm in config['arms']:
                name=protocol.arm_name(arm);receipt['current']=dict(arm=name,seed=seed);save()
                protocol.train_arm(data,maps,out/f'{name}_s{seed}',arm,seed,weight,epochs=config['epochs'],batch=batch,device=device)
                receipt['completed'].append(dict(arm=name,seed=seed));save()
        assert len(receipt['completed'])==30;protocol.analyze_upgrade(out,config['seeds'])
        receipt.update(status='MATRIX_COMPLETE',scientific_decision='PENDING_INDEPENDENT_INTERPRETATION',finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()
    except Exception:
        receipt['status']='FAILED';receipt['traceback']=traceback.format_exc();save();raise
if __name__=='__main__':main()
