"""Distribution-focused Stitching ablations; prior implementations stay sealed.

The transport field retains potential, entropy, and symmetric interaction.
New supervision uses visible distributions and visible OT paths only.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from geomloss.sinkhorn_divergence import scaling_parameters,log_weights,sinkhorn_loop,sinkhorn_cost
from .mass_control import MassStitching, configuration_for, integrate
from .experimental import loss as base_loss
from .model import ScalarMLP


ARMS=('baseline','sinkhorn','ot_velocity','ot_sinkhorn','ot_rollout',
      'time_ot','time_rollout','time_uniform')


def configuration(arm):
    assert arm in ARMS
    cfg=configuration_for('free')
    cfg.update(arm=arm,model_family='shape_losses',shape_weight=0.,nll_fraction=1.,
        ot_velocity_weight=0.,endpoint_weight=0.,endpoint_every=4,endpoint_steps=8,
        endpoint_particles=64,teacher_particles=64,teacher_groups=2,
        time_features=False,shape_blur_fraction=.10,checkpoint_every=500,
        shape_selection='visible independent rollout SWD only; no count term')
    if arm!='baseline':
        cfg.update(shape_weight=10.,nll_fraction=.1)
    if arm in ('ot_velocity','ot_sinkhorn','ot_rollout','time_ot','time_rollout','time_uniform'):
        cfg['ot_velocity_weight']=10.
    if arm=='ot_velocity':cfg.update(shape_weight=0.,nll_fraction=1.)
    if arm in ('ot_rollout','time_rollout','time_uniform'):cfg['endpoint_weight']=20.
    if arm in ('time_ot','time_rollout','time_uniform'):cfg['time_features']=True
    if arm=='time_uniform':cfg.update(growth_mode='uniform',hard_mass=True)
    return cfg


class ShapeStitching(MassStitching):
    def __init__(self,x,lm,grid,bw,obs,targets,cfg,curve_kind):
        super().__init__(x,lm,grid,bw,obs,targets,cfg,curve_kind)
        if cfg['time_features']:
            # Absolute normalized time plus Fourier coordinates and within-gap
            # phase. Spatial gradients still come from a scalar potential.
            dim=x.shape[-1]+8
            self.potential=ScalarMLP(dim,(64,64))
            self.growth=None if cfg['growth_mode']=='uniform' else ScalarMLP(dim,(64,64))

    def time_embedding(self,time):
        t=(time-self.observed_times[0])/(self.observed_times[-1]-self.observed_times[0])
        k=(torch.searchsorted(self.observed_times,time.contiguous(),right=True)-1).clamp(0,len(self.observed_times)-2)
        phase=(time-self.observed_times[k])/(self.observed_times[k+1]-self.observed_times[k])
        freq=time.new_tensor([1.,2.,4.])
        angle=2*math.pi*t[:,None]*freq
        # A raw phase jumps from 1 to 0 at knots and can create artificial field
        # discontinuities. This bump and its first derivative agree at knots.
        bump=torch.sin(math.pi*phase).square()
        return torch.cat((t[:,None],bump[:,None],angle.sin(),angle.cos()),-1)

    def coordinates(self,queries,time):
        if not self.cfg['time_features']:
            return torch.cat((queries,time[:,None,None].expand(*queries.shape[:2],1)),-1)
        return torch.cat((queries,self.time_embedding(time)[:,None,:].expand(*queries.shape[:2],8)),-1)

    def field(self,positions,lm,time,queries):
        if not self.cfg['time_features']:return super().field(positions,lm,time,queries)
        diff=queries[:,:,None,:]-positions[:,None,:,:]
        resp=F.softmax(self.log_components(positions,queries)+lm[:,None,:],-1)
        score=-(resp[:,:,:,None]*diff/self.bandwidth.square()).sum(2)
        v=-self.potential.gradient(self.coordinates(queries,time))[...,:positions.shape[-1]]-self.entropy*score
        if self.interaction is not None:
            dw=self.interaction.gradient(diff.square().sum(-1,keepdim=True)/positions.shape[-1])
            v=v-(lm.exp()[:,None,:,None]*2/positions.shape[-1]*diff*dw).sum(2)
        if self.growth is None:g=self.global_rate(time)[:,None].expand(queries.shape[:2])
        else:g=self.growth(self.coordinates(queries,time))
        return v,g


def sinkhorn_loss(x,lm,y,bw2,blur_fraction=.1):
    """Balanced debiased Sinkhorn on normalized empirical center measures.

    Normalize spatial scale using training-only kernel bandwidth. This is a
    center-cloud loss; the retained NLL supplies Gaussian-mixture smoothing.
    """
    scale=math.sqrt(float(bw2))
    x,y=x/scale,y/scale
    a=F.softmax(lm,-1);b=torch.full(y.shape[:2],1/y.shape[1],dtype=y.dtype,device=y.device)
    cost=lambda u,v:.5*(u[:,:,None,:]-v[:,None,:,:]).square().sum(-1)
    cxx,cyy=cost(x,x.detach()),cost(y,y.detach())
    cxy,cyx=cost(x,y.detach()),cost(y,x.detach())
    _,eps,schedule,rho=scaling_parameters(x,y,2,blur_fraction,None,5.,.7)
    # The installed legacy GeomLoss softmin flattens B*N and breaks B>1.
    # Preserve batch dimensions locally; library files are never modified.
    softmin=lambda e,c,f:-e*(f[:,None,:]-c/e).logsumexp(-1)
    previous=torch.is_grad_enabled()
    try:
        dual=sinkhorn_loop(softmin,log_weights(a),log_weights(b),cxx,cyy,cxy,cyx,schedule,rho,debias=True)
    finally:
        torch.set_grad_enabled(previous)
    return sinkhorn_cost(eps,rho,a,b,*dual,batch=True,debias=True).mean()


def teacher_loss(model,teacher,bw2):
    x0,x1,t0,t1,f=teacher
    xt=(1-f[:,None,None])*x0+f[:,None,None]*x1;t=t0+f*(t1-t0)
    lm=model.curve.value_rate(t)[0][:,None].expand(xt.shape[:2])-math.log(xt.shape[1])
    velocity,_=model.field(xt,lm,t,xt)
    # Displacement error weights long visible intervals as well as short ones.
    err=(t1-t0)[:,None,None]*velocity-(x1-x0)
    return err.square().sum(-1).mean()/bw2


def objective(model,oi,y,targets,ids,noise,cfg,support,radius,bw2,teacher=None,endpoint=None):
    total,parts=base_loss(model,oi,y,targets,ids,noise,cfg,support,radius,bw2)
    geometry=total.new_zeros(());fm=total.new_zeros(());end=total.new_zeros(())
    if cfg['shape_weight']:
        geometry=sinkhorn_loss(model.trajectories[oi],model.log_masses[oi],y,bw2,cfg['shape_blur_fraction'])
        total=total+(cfg['nll_fraction']-1)*parts['nll']+cfg['shape_weight']*geometry
    if teacher is not None:
        fm=teacher_loss(model,teacher,bw2)
        total=total+cfg['ot_velocity_weight']*fm
    if endpoint is not None:
        x,target,t0,t1=endpoint
        lm=model.curve.value_rate(t0)[0][:,None].expand(x.shape[:2])-math.log(x.shape[1])
        xr,lr=integrate(model,x,lm,t0,t1,cfg['endpoint_steps'])
        end=sinkhorn_loss(xr,lr,target,bw2,cfg['shape_blur_fraction'])
        total=total+cfg['endpoint_weight']*end
    return total,dict(parts,shape_sinkhorn=geometry,ot_displacement=fm,endpoint_sinkhorn=end)
