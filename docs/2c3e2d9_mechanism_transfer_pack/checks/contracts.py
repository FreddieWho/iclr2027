"""Auditable minimal reproductions, NOT a full checkout or empirical rerun.

Legacy formulas are transcribed from GitHub commit
2c3e2d9640607c32aacf86c8758cd1e2f8ef317b, chiefly:
experiments/e832_focus/route1_cross_task/common_runner.py: rich/sort_features
and experiments/e832_focus/route2_visual/gpu_run.py: metrics.
The reference functions below are proposed test primitives, not new methods.
"""
from __future__ import annotations
import itertools
import numpy as np

T1_PERMS = [tuple(p) + (3,) for p in itertools.permutations(range(3))]
SOURCE_PERMS = [(0,1,2,3),(1,0,2,3),(0,1,3,2),(1,0,3,2),
                (2,3,0,1),(3,2,0,1),(2,3,1,0),(3,2,1,0)]
T2_PERMS = [(0,1,2),(1,0,2)]


def t1_legacy_rich(points: np.ndarray, sorted_features: bool = True) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64).reshape(-1,4,2)
    v, q = p[:,:3], p[:,3]
    edges = np.stack([np.linalg.norm(v[:,i]-v[:,j],axis=1)
                      for i,j in ((0,1),(0,2),(1,2))],1)
    qd = np.linalg.norm(q[:,None]-v,axis=2)
    a,b = v[:,1]-v[:,0], v[:,2]-v[:,0]
    # Exact legacy formula: despite its name it is not dimensionless sine.
    legacy_sine = np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])/(np.prod(edges,axis=1)+1e-8)
    if sorted_features:
        edges, qd = np.sort(edges,axis=1), np.sort(qd,axis=1)
    return np.c_[edges,qd,np.linalg.norm(q-v.mean(1),axis=1),legacy_sine]


def t2_legacy_rich(points: np.ndarray, sorted_features: bool = False) -> np.ndarray:
    p=np.asarray(points,dtype=np.float64).reshape(-1,3,2)
    a,b,c=p[:,0],p[:,1],p[:,2]
    # Legacy axis=1 collapses the endpoint axis, not x/y coordinates.
    d=np.linalg.norm(c[:,None]-p[:,:2],axis=1)
    f=np.c_[np.linalg.norm(a-b,axis=1),d,
            np.linalg.norm((a+b)/2-c,axis=1),d.min(1),d.prod(1)]
    return np.c_[np.sort(f[:,:2],1),f[:,2:]] if sorted_features else f


def t2_role_distances(points: np.ndarray) -> np.ndarray:
    p=np.asarray(points,dtype=np.float64).reshape(-1,3,2)
    d=np.linalg.norm(p[:,2,None,:]-p[:,:2,:],axis=-1)
    return np.c_[np.linalg.norm(p[:,0]-p[:,1],axis=-1),np.sort(d,axis=1)]


def point_in_triangle(points: np.ndarray) -> bool:
    p=np.asarray(points,dtype=np.float64).reshape(4,2)
    uv=np.linalg.solve(np.stack((p[1]-p[0],p[2]-p[0]),axis=1),p[3]-p[0])
    return bool(uv[0]>0 and uv[1]>0 and uv.sum()<1)


def crossing(points: np.ndarray) -> bool:
    a,b,c,d=np.asarray(points,dtype=np.float64).reshape(4,2)
    def orient(p,q,r):
        u,v=q-p,r-p
        return float(u[0]*v[1]-u[1]*v[0])
    return orient(a,b,c)*orient(a,b,d)<0 and orient(c,d,a)*orient(c,d,b)<0


def orbit_distance_representative(points: np.ndarray, perms: list[tuple[int,...]]) -> np.ndarray:
    """Whole-orbit lexicographic representative. Diagnostic, not smooth layer.

Do not sort unrelated distance columns independently. Equal representatives
imply congruence after a legal permutation, within ordinary Euclidean
(non-chiral) task assumptions. Ties/seams require a separate stability audit.
"""
    x=np.asarray(points,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=2 or not np.all(np.isfinite(x)):
        raise ValueError('Expected finite [n_points,2]')
    i,j=np.triu_indices(len(x),1)
    candidates=[]
    for perm in perms:
        if sorted(perm)!=list(range(len(x))):
            raise ValueError('Invalid permutation')
        xp=x[list(perm)]
        candidates.append(np.linalg.norm(xp[i]-xp[j],axis=1))
    return min(candidates,key=lambda row:tuple(row.tolist())).copy()


def t1_collision_pair() -> tuple[np.ndarray,np.ndarray]:
    # All coordinates and centroids are dyadic: the collision also survives
    # float32 for this implementation, rather than relying on a decimal tie.
    vertices=np.array([[-.75,0.],[.75,0.],[-.375,.75]])
    return np.vstack([vertices,[.1875,.09375]]), np.vstack([vertices,[-.1875,-.09375]])



def joint_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    logits,labels=np.asarray(logits),np.asarray(labels)
    if logits.shape!=labels.shape or logits.ndim!=2 or logits.shape[1]!=4:
        raise ValueError('Expected matching [n,4] P,A,B,AB arrays')
    ok=(logits>0)==(labels>0.5)
    return {'J3':float(ok[:,1:].all(1).mean()),'J4':float(ok.all(1).mean())}
