"""Frozen O05 12-arm real-tracking partial-observation matrix (CUDA)."""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

class Raw(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Flatten(),nn.Linear(65,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,x):return self.net(x).squeeze(-1)
class Typed(nn.Module):
 def __init__(self):
  super().__init__();self.edge=nn.Sequential(nn.Linear(24,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU());self.head=nn.Sequential(nn.Linear(192,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,x):
  p=x[:,0:1].expand(-1,11,-1);r=x[:,1:2].expand(-1,11,-1);d=x[:,2:]
  rp=r[:,:,:2]-p[:,:,:2];dp=d[:,:,:2]-p[:,:,:2];dr=d[:,:,:2]-r[:,:,:2]
  length=rp.square().sum(-1,keepdim=True).sqrt();projection=(dp*rp).sum(-1,keepdim=True)/rp.square().sum(-1,keepdim=True).clamp_min(1e-8)
  perpendicular=(dp-rp*projection.clamp(0,1)).square().sum(-1,keepdim=True).sqrt()
  f=torch.cat([p,r,d,rp,dp,dr,length,projection,perpendicular],-1)
  h=self.edge(f);pool=torch.cat([h.mean(1),h.amax(1),h.amin(1)],1)
  return self.head(pool).squeeze(-1)

def digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
@torch.inference_mode()
def predict(model,x,batch=1024):
 model.eval();return torch.cat([model(x[i:i+batch]) for i in range(0,len(x),batch)]).cpu().numpy()
def temporal(y,pred,group,seconds):
 lags=[];missed=0;false=0;true_n=0;pred_n=0
 for g in np.unique(group):
  ix=np.flatnonzero(group==g);ix=ix[np.argsort(seconds[ix])];yy=y[ix];pp=pred[ix];tt=seconds[ix]
  # Missing history windows break natural trajectories.
  contiguous=np.diff(tt)<.21
  truth=[(float(tt[i+1]),int(yy[i+1])) for i in np.flatnonzero((yy[1:]!=yy[:-1])&contiguous)]
  alarm=[(float(tt[i+1]),int(pp[i+1])) for i in np.flatnonzero((pp[1:]!=pp[:-1])&contiguous)];used=set();true_n+=len(truth);pred_n+=len(alarm)
  for t,label in truth:
   candidates=[(abs(at-t),j,at-t) for j,(at,lab) in enumerate(alarm) if j not in used and lab==label and abs(at-t)<=.60001]
   if candidates:
    _,j,lag=min(candidates);used.add(j);lags.append(lag)
   else:missed+=1
  false+=len(alarm)-len(used)
 return {'true_transition_n':true_n,'predicted_transition_n':pred_n,'matched_n':len(lags),'missed':missed,'false_alarm_n':false,'lag_seconds_median':float(np.median(lags)) if lags else None,'lag_seconds_all':lags,'match_rule':'same destination, nearest within0.6s one-to-one; negative=early; .2s sampling; gaps broken'}
def metric(pred,y,group,seconds):
 return {'n':len(y),'accuracy':float((pred==y).mean()),'open_n':int(y.sum()),'false_open':int(((pred==1)&(y==0)).sum()),'false_blocked':int(((pred==0)&(y==1)).sum()),'natural_temporal':temporal(y,pred,group,seconds)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',default='cuda');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if (a.out/'summary.json').exists():raise FileExistsError('immutable matrix exists')
 d=np.load(a.data/'data.npz');manifest=json.loads((a.data/'manifest.json').read_text());assert hashlib.sha256((a.data/'data.npz').read_bytes()).hexdigest()==manifest['data_sha256']
 signature={'data_sha256':manifest['data_sha256'],'config':manifest['configs'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'torch':torch.__version__}
 signature_path=a.out/'run_manifest.json'
 if signature_path.exists():assert json.loads(signature_path.read_text())==signature,'partial run signature mismatch'
 else:signature_path.write_text(json.dumps(signature,indent=2))
 torch.set_num_threads(4);device=torch.device(a.device)
 if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required; no silent CPU substitution')
 tensors={s:torch.from_numpy(d[s+'_x']).float().to(device) for s in ['train','dev','test']};ys={s:torch.from_numpy(d[s+'_y']).float().to(device) for s in ['train','dev','test']}
 start=time.monotonic();result={'status':'RUNNING','completed':0,'expected':12,'data_manifest':manifest,'arms':{},'baselines':{}}
 for baseline in ['last','cv']:result['baselines'][baseline]=metric(d['test_'+baseline],d['test_y'],d['test_group'],d['test_seconds'])
 for encoder,cls in [('raw',Raw),('typed',Typed)]:
  for seed in [11,23,47]:
   for aug in ['static','change']:
    tag=f'{encoder}_{aug}_s{seed}';od=a.out/tag;od.mkdir(exist_ok=True)
    if (od/'receipt.json').exists():result['arms'][tag]=json.loads((od/'receipt.json').read_text());result['completed']+=1;continue
    torch.manual_seed(seed);model=cls().to(device);init=digest(model);opt=torch.optim.Adam(model.parameters(),lr=.001)
    add='repeat' if aug=='static' else 'change';x=torch.cat([tensors['train'],torch.from_numpy(d[add+'_x']).float().to(device)]);y=torch.cat([ys['train'],torch.from_numpy(d[add+'_y']).float().to(device)])
    rng=np.random.default_rng(seed);bce=nn.BCEWithLogitsLoss();best=float('inf');hist=[];orders=hashlib.sha256();arm_start=time.monotonic()
    for ep in range(300):
     model.train();order=rng.permutation(len(x));orders.update(order.tobytes());total=0
     for offset in range(0,len(order),512):
      ix=torch.as_tensor(order[offset:offset+512],device=device);opt.zero_grad(set_to_none=True);loss=bce(model(x[ix]),y[ix]);loss.backward();opt.step();total+=float(loss.detach())*len(ix)
     if (ep+1)%10==0:
      logits=predict(model,tensors['dev']);dev_loss=float(nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(logits),ys['dev'].cpu()))
      hist.append({'epoch':ep+1,'train_bce':total/len(x),'dev_bce':dev_loss})
      if dev_loss<best:best=dev_loss;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=ep+1
    model.load_state_dict(best_state);testlog=predict(model,tensors['test']);pred=testlog>0
    qx=torch.from_numpy(d['test_qx'].reshape(-1,13,5)).float().to(device);ql=predict(model,qx).reshape(-1,4);qy=d['test_qy'];ok=(ql>0)==qy;atomic=ok[:,1:3].all(1)
    metrics=metric(pred,d['test_y'],d['test_group'],d['test_seconds']);metrics['controlled_E']={'n':len(qy),'J_A_B_AB':float(ok[:,1:].all(1).mean()),'J_all4':float(ok.all(1).mean()),'atomic_both_correct_n':int(atomic.sum()),'conditional_AB_error_given_atomic_correct':float((~ok[:,3]&atomic).sum()/atomic.sum()) if atomic.any() else None}
    receipt={'encoder':encoder,'augmentation':aug,'seed':seed,'initial_sha256':init,'final_sha256':digest(model),'batch_order_sha256':orders.hexdigest(),'train_examples':len(x),'steps':300*int(np.ceil(len(x)/512)),'parameters':sum(p.numel() for p in model.parameters()),'dev_selected_epoch':best_epoch,'dev_loss':best,'metrics':metrics,'training_history':hist,'wall_seconds':time.monotonic()-arm_start}
    torch.save({'state':best_state,'encoder':encoder,'seed':seed,'selected_epoch':best_epoch},od/'model.pt');np.savez_compressed(od/'predictions.npz',test_logits=testlog,test_labels=d['test_y'],quartet_logits=ql,quartet_labels=qy,groups=d['test_group'],seconds=d['test_seconds'])
    (od/'receipt.json').write_text(json.dumps(receipt,indent=2));result['arms'][tag]=receipt;result['completed']+=1
    (a.out/'progress.json').write_text(json.dumps({'status':'RUNNING','completed':result['completed'],'expected':12,'last_arm':tag},indent=2));print(tag,'accuracy',metrics['accuracy'],'J',metrics['controlled_E']['J_A_B_AB'],flush=True)
 result.update({'status':'MATRIX_COMPLETE','wall_seconds':time.monotonic()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,'torch':torch.__version__,'device':str(device)})
 (a.out/'summary.json').write_text(json.dumps(result,indent=2))
 (a.out/'receipt.json').write_text(json.dumps({'status':'MATRIX_COMPLETE','completed':list(result['arms']),'expected':12,'summary':'summary.json'},indent=2))
 print('MATRIX_COMPLETE',result['completed'],flush=True)
if __name__=='__main__':main()
