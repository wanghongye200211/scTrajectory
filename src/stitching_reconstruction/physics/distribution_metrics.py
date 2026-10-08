"""Common weighted empirical metrics, matching the frozen attempt169 protocol.

MMD is the biased V-statistic, averaged over three train-fitted RBF bandwidths.
W1 and W2 are exact OT of the finite empirical measures, not Sinkhorn losses.
No dynamics or OM modules are imported.
"""
import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


def metric_values(x,y,w,bandwidth,directions):
    import ot
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    w=np.full(len(x),1/len(x)) if w is None else np.asarray(w,dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('Invalid predictions, targets or masses')
    if bandwidth<=0 or not np.isfinite(bandwidth):raise ValueError('Invalid training bandwidth')
    w=w/w.sum();v=np.full(len(y),1/len(y))
    dxx=cdist(x,x,'sqeuclidean');dyy=cdist(y,y,'sqeuclidean');dxy=cdist(x,y,'sqeuclidean')
    def quad(a,M,b):return np.einsum('i,ij,j->',a,M,b,optimize=False)
    mmd2=np.mean([quad(w,np.exp(-dxx/(2*bandwidth*f)),w)+quad(v,np.exp(-dyy/(2*bandwidth*f)),v)
                 -2*quad(w,np.exp(-dxy/(2*bandwidth*f)),v) for f in (.5,1.,2.)])
    w1,log1=ot.emd2(w,v,np.sqrt(dxy),numItermax=1000000,log=True,numThreads=1)
    w22,log2=ot.emd2(w,v,dxy,numItermax=1000000,log=True,numThreads=1)
    if log1.get('warning') or log2.get('warning'):raise RuntimeError((log1,log2))
    xp=np.einsum('nd,kd->nk',x,directions,optimize=False);yp=np.einsum('nd,kd->nk',y,directions,optimize=False)
    swd=np.mean([wasserstein_distance(xp[:,i],yp[:,i],u_weights=w) for i in range(len(directions))])
    mx=np.einsum('n,nd->d',w,x,optimize=False);my=np.einsum('n,nd->d',v,y,optimize=False)
    cx=np.einsum('ni,n,nj->ij',x-mx,w,x-mx,optimize=False);cy=np.einsum('ni,n,nj->ij',y-my,v,y-my,optimize=False)
    energy=2*quad(w,np.sqrt(dxy),v)-quad(w,np.sqrt(dxx),w)-quad(v,np.sqrt(dyy),v)
    return dict(swd=float(swd),mmd2=max(0.,float(mmd2)),w1=float(w1),w2=math.sqrt(max(0.,w22)),
        energy=max(0.,float(energy)),mean_rmse=float(np.sqrt(np.mean((mx-my)**2))),
        covariance_rel_error=float(np.linalg.norm(cx-cy)/max(np.linalg.norm(cy),1e-12)),ess=float(1/np.sum(w*w)))
