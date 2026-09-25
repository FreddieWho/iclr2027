#!/usr/bin/env python3
"""Fresh-parent source/T1/T2 confirmation with one task-adapted runner.

Provenance: task oracles and candidate families are imported unchanged from
experiments/f095_campaign/u10_targets.py/u10_oracles.py and
experiments/last15h/shared/paths.py. The source task role/action contract comes
from a_source_compare.py. No sealed pool, dev512, visual oracle coordinate, or
old E bank is read. One command runs deterministic banks, training, selection,
and analysis; selected weights are not retained.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'experiments'/'f095_campaign'), str(ROOT/'experiments'/'discovery_campaign'),
                str(ROOT/'experiments'/'last15h'/'shared'), str(ROOT/'docs'/'iclr2027_discovery_campaign_20260917')]
from u10_oracles import tri_oracle, disk_oracle  # noqa: E402
from u10_targets import candidates_3pt  # noqa: E402
from paths import atomic_edits, oracle_at  # noqa: E402

ART = ROOT/'artifacts'/'e832_focus'/'route1'
THREADS = 4
LR = 1e-3
BOOT = 1000
ROUND2_SEEDS = (11, 23, 47)
TASKS = {
 'source': {'n':4, 'train_seed':832501, 'dev_seed':832502, 'test_seed':832503, 'edit_seed':832504},
 'T1': {'n':4, 'train_seed':832511, 'dev_seed':832512, 'test_seed':832513, 'edit_seed':832514},
 'T2': {'n':3, 'train_seed':832521, 'dev_seed':832522, 'test_seed':832523, 'edit_seed':832524},
}

def oracle(task, x, floor=0.0):
 if task=='source':
  y,m,a=oracle_at(np.asarray(x,float)); return y,m,a
 o=(tri_oracle if task=='T1' else disk_oracle)(np.asarray(x,float), max(floor,0.0))
 return int(o['label']),float(o['margin']),bool(o['ambiguous'])

def gen(task,n,seed):
 rng=np.random.default_rng(seed); X=[];Y=[];M=[]
 for i in range(n):
  for attempt in range(1000):
   x=rng.uniform(-.8,.8,size=(TASKS[task]['n'],2))
   if task=='T1' and abs(np.cross(x[1]-x[0],x[2]-x[0]))<.04: continue
   try: y,m,a=oracle(task,x,.005)
   except ValueError: continue
   if a: continue
   # Coordinate-space L-inf duplicate guard across this split.
   if X and np.min(np.abs(np.asarray(X)-x).max(2))<1e-4: continue
   X.append(x);Y.append(y);M.append(m);break
  else: raise RuntimeError(f'{task} generator failed at {i}')
 return np.asarray(X,np.float32),np.asarray(Y,np.int64),np.asarray(M,np.float32)

def candidates(task,x,rng):
 if task=='source': return atomic_edits(x,rng,radius=.10)
 if task=='T1':
  # Reuse the task's established single/pair/global/random families. Q moves
  # only as part of the generic candidate; Q-vertex role swaps are not grouped.
  from r02_search import candidates_for_scene
  return candidates_for_scene(x,rng,n_random=24)
 return candidates_3pt(x,rng,n_random=24)

def mine_flips(task,X,Y,seed,cap=4):
 rng=np.random.default_rng(seed); S=[];F=[];P=[]
 for i,(x,y) in enumerate(zip(X,Y)):
  n=0
  for e,_ in candidates(task,x,rng):
   try: yy,_,amb=oracle(task,x+e)
   except ValueError: continue
   if not amb and yy!=int(y): S.append(x+e);F.append(yy);P.append(i);n+=1
   if n>=cap: break
 return np.asarray(S,np.float32).reshape(-1,TASKS[task]['n'],2),np.asarray(F,np.int64),np.asarray(P,np.int64)

def mine_E(task,X,Y,seed,cap,candidate_draws):
 rng=np.random.default_rng(seed); rows=[]; nq=0
 for i,(x,y) in enumerate(zip(X,Y)):
  keeps=[]
  # More than one deterministic action draw avoids restricting E to four angles.
  for draw in range(candidate_draws):
   for e,fam in candidates(task,x,np.random.default_rng(seed+1009*i+draw)):
    try: yy,mm,amb=oracle(task,x+e,.005)
    except ValueError: continue
    if not amb and yy==int(y): keeps.append((np.asarray(e,float),fam,mm))
  # stable family/edit tuple deduplication
  seen=set(); uk=[]
  for e,f,m in keeps:
   key=(tuple(np.round(e.ravel(),8)),f)
   if key not in seen:
    seen.add(key);uk.append((e,f,m))
    if len(uk)>=12: break
  made=0
  for ai,(ea,fa,ma) in enumerate(uk):
   for eb,fb,mb in uk[ai+1:]:
    try: yab,mab,aab=oracle(task,x+ea+eb,.005)
    except ValueError: continue
    if aab or yab==int(y): continue
    rows.append({'parent':i,'ea':ea.astype(np.float32),'eb':eb.astype(np.float32),
                 'y0':int(y),'yab':int(yab),'m':np.array([ma,mb,mab],np.float32),
                 'fam_a':fa,'fam_b':fb});made+=1
    if made>=cap: break
   if made>=cap: break
  nq+=made
 if not rows: return None
 pa=np.asarray([r['parent'] for r in rows],np.int64); px=X[pa]
 states=np.stack([px,px+np.stack([r['ea'] for r in rows]),px+np.stack([r['eb'] for r in rows]),
                  px+np.stack([r['ea']+r['eb'] for r in rows])]).astype(np.float32)
 return {'states':states,'parent':pa,'y0':np.asarray([r['y0'] for r in rows]),
         'yab':np.asarray([r['yab'] for r in rows]),'margins':np.stack([r['m'] for r in rows]),
         'fam_a':np.asarray([r['fam_a'] for r in rows]),'fam_b':np.asarray([r['fam_b'] for r in rows])}

def perms(task):
 if task=='source':
  # Independent endpoint swaps: the established G8 action.
  return np.asarray([(0,1,2,3),(1,0,2,3),(0,1,3,2),(1,0,3,2),
                     (2,3,0,1),(3,2,0,1),(2,3,1,0),(3,2,1,0)],np.int64)
 if task=='T1':
  return np.asarray([p+(3,) for p in itertools.permutations(range(3))],np.int64)
 return np.asarray([(0,1,2),(1,0,2)],np.int64)

def apply_perm(task,x,p):
 p=np.asarray(p,np.int64)
 if task=='T1': return np.concatenate([x[...,p[:3],:],x[...,3:,:]],axis=-2)
 return x[...,p,:]

def six(task,x):
 n=x.shape[-2];i,j=np.triu_indices(n,1);return np.linalg.norm(x[...,i,:]-x[...,j,:],axis=-1)

def rich(task,x):
 if task=='source':
  p=x.reshape(-1,4,2); le=np.linalg.norm(p[:,0]-p[:,1],axis=1);ld=np.linalg.norm(p[:,2]-p[:,3],axis=1)
  cr=np.stack([np.linalg.norm(p[:,i]-p[:,j],axis=1) for i,j in ((0,2),(0,3),(1,2),(1,3))],1)
  mid=np.linalg.norm((p[:,0]+p[:,1])/2-(p[:,2]+p[:,3])/2,axis=1)
  u=p[:,1]-p[:,0];v=p[:,3]-p[:,2];s=np.abs(u[:,0]*v[:,1]-u[:,1]*v[:,0])/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-8)
  return np.c_[le,ld,cr,mid,s]
 if task=='T1':
  p=x.reshape(-1,4,2);v=p[:,:3];q=p[:,3]
  edges=np.stack([np.linalg.norm(v[:,i]-v[:,j],axis=1) for i,j in ((0,1),(0,2),(1,2))],1)
  qd=np.linalg.norm(q[:,None]-v,axis=2)
  sine=np.abs(np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]))/(np.prod(edges,axis=1)+1e-8)
  return np.c_[edges,qd,np.linalg.norm(q-v.mean(1),axis=1),sine]
 p=x.reshape(-1,3,2);a,b,c=p[:,0],p[:,1],p[:,2];d=np.linalg.norm(c[:,None]-p[:,:2],axis=1)
 return np.c_[np.linalg.norm(a-b,axis=1),d,np.linalg.norm((a+b)/2-c,axis=1),d.min(1),d.prod(1)]

def sort_features(task,f):
 f=np.asarray(f,float)
 if task=='source': return np.c_[np.sort(f[:,:2],1),np.sort(f[:,2:6],1),f[:,6:]]
 if task=='T1': return np.c_[np.sort(f[:,:3],1),np.sort(f[:,3:6],1),f[:,6:]]
 return np.c_[np.sort(f[:,:2],1),f[:,2:]]

def feat(task,arm,x):
 x=np.asarray(x,np.float32)
 if arm in ('raw','additive','repaired'):
  return x.reshape(len(x),-1)
 if arm=='six': return six(task,x)
 f=rich(task,x)
 return sort_features(task,f) if arm=='rich_sorted' else f

class MLP(nn.Module):
 def __init__(self,d,h=64):
  super().__init__();self.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,32),nn.ReLU(),nn.Linear(32,1))
 def forward(self,x):return self.net(x).reshape(-1)

class Typed(nn.Module):
 def __init__(self,task,repaired):
  super().__init__();self.task=task;self.phi=nn.Sequential(nn.Linear(2,48),nn.ReLU(),nn.Linear(48,48),nn.ReLU())
  d=48 if task=='source' else 96
  self.rho=nn.Linear(d,1) if not repaired else nn.Sequential(nn.Linear(d,16),nn.GELU(),nn.Linear(16,1))
 def forward(self,x):
  p=x.reshape(-1,self.task=='source' and 4 or (4 if self.task=='T1' else 3),2)
  if self.task=='source': h=self.phi(p.reshape(-1,2)).reshape(len(p),4,-1);z=h[:,:2].sum(1)+h[:,2:].sum(1)
  elif self.task=='T1': z=torch.cat([self.phi(p[:,:3].reshape(-1,2)).reshape(len(p),3,-1).sum(1),self.phi(p[:,3])],1)
  else: z=torch.cat([self.phi(p[:,:2].reshape(-1,2)).reshape(len(p),2,-1).sum(1),self.phi(p[:,2])],1)
  return self.rho(z).reshape(-1)

def build(task,arm):
 if arm in ('raw','six','rich_unsorted','rich_sorted'): return MLP(feat(task,arm,np.zeros((1,TASKS[task]['n'],2),np.float32)).shape[1])
 return Typed(task,arm=='repaired')

def stats(task,arm,X):
 f=feat(task,arm,X)
 if arm in ('additive','repaired'):
  p=X.reshape(-1,2)
  return np.tile(p.mean(0),X.shape[1]),np.tile(p.std(0)+1e-8,X.shape[1])
 return f.mean(0),f.std(1e-8) if False else f.std(0)+1e-8

def prep(task,arm,X,mu,sd):return ((feat(task,arm,X)-mu)/sd).astype(np.float32)
def nparam(m):return sum(p.numel() for p in m.parameters())
def macs(m,d):
 total=0
 def hook(mod,inp,out):return sum(inp0.shape[0]*mod.in_features*mod.out_features for inp0 in [inp[0].reshape(-1,inp[0].shape[-1])])
 hs=[]
 for q in m.modules():
  if isinstance(q,nn.Linear):hs.append(q.register_forward_hook(lambda mod,i,o:None))
 # deterministic analytical count by zero forward
 for q in m.modules():
  if isinstance(q,nn.Linear): total += q.in_features*q.out_features
 for h in hs:h.remove()
 return total*2 # batch of two convention normalized below is not valid; corrected by per-example hook
# use explicit per-example count by one forward below
def per_example_macs(m,d):
 c=[0]
 def h(mod,inp,out):c[0]+=int(np.prod(inp[0].shape[:-1]))*mod.in_features*mod.out_features
 hs=[q.register_forward_hook(h) for q in m.modules() if isinstance(q,nn.Linear)]
 with torch.no_grad():m(torch.zeros(1,d))
 for x in hs:x.remove()
 return c[0]

def train_one(task,arm,seed,sup,X,Y,F,dev,epochs):
 mu,sd=stats(task,arm,X);xc=torch.from_numpy(prep(task,arm,X,mu,sd));yc=torch.from_numpy(Y.astype(np.float32))
 if sup=='flip':xf=torch.from_numpy(prep(task,arm,F[0],mu,sd));yf=torch.from_numpy(F[1].astype(np.float32))
 dx=np.r_[prep(task,arm,dev[0],mu,sd),prep(task,arm,dev[2],mu,sd)];dy=np.r_[dev[1],dev[3]].astype(np.float32)
 torch.manual_seed(seed);m=build(task,arm);opt=torch.optim.Adam(m.parameters(),lr=LR);lossfn=nn.BCEWithLogitsLoss();t0=time.time()
 for _ in range(epochs):
  opt.zero_grad();loss=lossfn(m(xc),yc)
  if sup=='flip':loss=loss+lossfn(m(xf),yf)
  loss.backward();opt.step()
 m.eval()
 with torch.no_grad():devb=float(lossfn(m(torch.from_numpy(dx)),torch.from_numpy(dy)));trainb=float(lossfn(m(xc),yc))
 return m,mu,sd,{'dev_bce':devb,'train_bce':trainb,'seconds':time.time()-t0,'parameters':nparam(m),'inference_macs':per_example_macs(m,xc.shape[1]),'train_state_forwards':epochs*(len(X)+(len(F[1]) if sup=='flip' else 0)),'train_macs':per_example_macs(m,xc.shape[1])*epochs*(len(X)+(len(F[1]) if sup=='flip' else 0)),'threads':THREADS,'epoch':epochs,'lr':LR}

def predict(task,arm,m,mu,sd,x):return m(torch.from_numpy(prep(task,arm,x,mu,sd))).detach().numpy()
def group_logit(task,arm,m,mu,sd,x):
 pp=perms(task);z=[]
 for p in pp:z.append(predict(task,arm,m,mu,sd,apply_perm(task,x,p)))
 return np.mean(z,axis=0)
def metric(ok,pa):
 j=ok[1]&ok[2]&ok[3];atomic=ok[1]&ok[2]
 return {'n':len(j),'n_parents':len(np.unique(pa)),'J3':float(j.mean()),'J4':None,'A':float(ok[1].mean()),'B':float(ok[2].mean()),'AB':float(ok[3].mean()),'atomic_joint':float(atomic.mean()),'CCM_miss_given_atomic':float(1-j[atomic].mean()) if atomic.any() else None,'CCM_denominator':int(atomic.sum())}

def boot_delta(a,b,pa,seed):
 """Parent-cluster bootstrap of J3 differences (parents are the resampling units)."""
 d=a.astype(float)-b.astype(float);u=np.unique(pa);rng=np.random.default_rng(seed)
 group_means=np.asarray([d[pa==p].mean() for p in u]);vals=[]
 for _ in range(BOOT):vals.append(group_means[rng.integers(0,len(u),len(u))].mean())
 return {'estimate':float(group_means.mean()),'ci95':np.quantile(vals,[.025,.975]).tolist(),
         'n':len(d),'parents':len(u),'unit':'test parent (equal weight)'}

def run_task(task,settings,seeds):
 cfg=dict(TASKS[task]);cfg.update({'train_seed':seeds['train'],'dev_seed':seeds['dev'],
                                   'test_seed':seeds['test'],'edit_seed':seeds['edit']})
 X,Y,M=gen(task,settings['train_parents'],cfg['train_seed'])
 DX,DY,DM=gen(task,settings['dev_parents'],cfg['dev_seed'])
 DS=mine_flips(task,DX,DY,cfg['dev_seed']+1,3)
 T,TY,TM=gen(task,settings['test_parents'],cfg['test_seed'])
 bank=mine_E(task,T,TY,cfg['edit_seed'],settings['edit_cap'],settings['candidate_draws'])
 F=mine_flips(task,X,Y,cfg['train_seed']+1,4);dev=(DX,DY,DS[0],DS[1])
 base={'task':task,'train_seed':cfg['train_seed'],'dev_seed':cfg['dev_seed'],
       'test_seed':cfg['test_seed'],'edit_seed':cfg['edit_seed'],'settings':settings}
 if bank is None:
  return {**base,'status':'NOT_FEASIBLE','reason':'zero official E pairs',
          'E':0,'E_parents':0,'n_test_parents':len(T),
          'witness':{'candidates':'task legal action families','rule':'both preserve, sum flips'}}
 arms=['raw','additive','repaired','six','rich_unsorted','rich_sorted'];rows=[];oks={}
 for arm in arms:
  for seed in ROUND2_SEEDS:
   for sup in ('clean','flip'):
    m,mu,sd,rec=train_one(task,arm,seed,sup,X,Y,F,dev,settings['epochs'])
    for ga in (False,True):
     a=arm+('_groupavg' if ga else '');zz=np.stack([group_logit(task,arm,m,mu,sd,bank['states'][i]) if ga else predict(task,arm,m,mu,sd,bank['states'][i]) for i in range(4)])
     lab=np.stack([bank['y0'],bank['y0'],bank['y0'],bank['yab']]);ok=(zz>0)==(lab==1);oks[(a,seed,sup)]=ok
     rows.append({'arm':a,'seed':seed,'supervision':sup,**rec,'metric':metric(ok,bank['parent'])})
 contrasts=[]
 for seed in ROUND2_SEEDS:
  for sup in ('clean','flip'):
   def j(a):return (oks[(a,seed,sup)][1]&oks[(a,seed,sup)][2]&oks[(a,seed,sup)][3])
   for name,l,r in [('repaired-additive','repaired','additive'),('six-raw','six','raw'),('rich_unsorted-six','rich_unsorted','six'),('rich_sorted-rich_unsorted','rich_sorted','rich_unsorted'),('repaired_groupavg-repaired','repaired_groupavg','repaired'),('raw_groupavg-raw','raw_groupavg','raw')]:
    contrasts.append({'contrast':name,'seed':seed,'supervision':sup,'delta_J3':boot_delta(j(l),j(r),bank['parent'],9000+seed)})
 return {**base,'status':'COMPLETE','n_train':len(X),'n_train_flips':len(F[1]),
         'n_dev_clean':len(DX),'n_dev_flips':len(DS[1]),'n_test_parents':len(T),
         'E':len(bank['parent']),'E_parents':len(np.unique(bank['parent'])),
         'denominator_note':'unordered official E pairs; no directed-path doubling; ambiguity is not used to change denominator',
         'arms':rows,'contrasts':contrasts,'threads':THREADS}

def env_int(name,default):
 value=os.environ.get(name)
 if value is None:return default
 return int(value)

def parse_args(argv=None):
 p=argparse.ArgumentParser()
 p.add_argument('--tasks',nargs='+',choices=tuple(TASKS),default=['source','T1','T2'])
 p.add_argument('--out',type=Path,default=ART)
 p.add_argument('--task-prefix',default='')
 p.add_argument('--train-parents',type=int,default=env_int('ROUTE1_TRAIN_PARENTS',256))
 p.add_argument('--dev-parents',type=int,default=env_int('ROUTE1_DEV_PARENTS',64))
 p.add_argument('--test-parents',type=int,default=env_int('ROUTE1_TEST_PARENTS',256))
 p.add_argument('--edit-cap',type=int,default=env_int('ROUTE1_EDIT_CAP',6))
 p.add_argument('--candidate-draws',type=int,default=env_int('ROUTE1_CANDIDATE_DRAWS',3))
 p.add_argument('--epochs',type=int,default=env_int('ROUTE1_EPOCHS',240))
 for name in ('train','dev','test','edit'):
  p.add_argument(f'--{name}-seed',type=int,default=TASKS['source'][f'{name}_seed'])
 a=p.parse_args(argv)
 counts=(a.train_parents,a.dev_parents,a.test_parents,a.edit_cap,a.candidate_draws,a.epochs)
 if any(v<1 for v in counts):p.error('counts, edit cap, candidate draws, and epochs must be positive')
 if len(a.tasks)!=len(set(a.tasks)):p.error('tasks must be unique')
 return a

def merge_results(path,tasks):
 if path.exists():
  try: allr=json.loads(path.read_text())
  except (json.JSONDecodeError,OSError) as exc:raise RuntimeError(f'cannot merge existing {path}: {exc}')
  if not isinstance(allr,dict):raise RuntimeError(f'cannot merge non-object {path}')
 else:allr={}
 allr.update(tasks)
 path.write_text(json.dumps(allr,indent=2,sort_keys=True)+'\n')
 return allr

def main(argv=None):
 a=parse_args(argv)
 if Path(a.task_prefix).name!=a.task_prefix or not a.task_prefix.replace('_','').isalnum():
  raise ValueError('task-prefix must be empty or a filename-safe name')
 a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(THREADS);torch.manual_seed(20260925)
 settings={'train_parents':a.train_parents,'dev_parents':a.dev_parents,
            'test_parents':a.test_parents,'edit_cap':a.edit_cap,
            'candidate_draws':a.candidate_draws,'epochs':a.epochs}
 seeds={'train':a.train_seed,'dev':a.dev_seed,'test':a.test_seed,'edit':a.edit_seed}
 allr={}
 for task in a.tasks:
  r=run_task(task,settings,seeds);allr[task]=r
  (a.out/f'{a.task_prefix}{task}.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
  print(task,r['status'],r.get('E'),flush=True)
 merge_results(a.out/'results.json',allr)
if __name__=='__main__':main()
