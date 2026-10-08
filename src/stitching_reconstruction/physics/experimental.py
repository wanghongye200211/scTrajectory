"""Controlled local-growth experiments; the sealed v1 implementation stays immutable.

A time-independent learned KDE bandwidth retains the exact induced continuity
identity. Centered growth uses empirical-center quadrature, not an exact Gaussian
expectation. Counts are captured-cell proxies. All losses use visible data only.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .model import ScalarMLP, Stitching

ARMS = {
 'fixed': dict(learn_bandwidth=False, growth_mode='free', interaction=True),
 'bandwidth': dict(learn_bandwidth=True, growth_mode='free', interaction=True),
 'local': dict(learn_bandwidth=True, growth_mode='centered', interaction=True),
 'uniform': dict(learn_bandwidth=True, growth_mode='uniform', interaction=True),
 'no_w': dict(learn_bandwidth=True, growth_mode='centered', interaction=False),
 'mmd': dict(learn_bandwidth=True, growth_mode='centered', interaction=True, data_loss='mmd'),
 'hybrid': dict(learn_bandwidth=True, growth_mode='centered', interaction=True, data_loss='hybrid'),
 'consistency': dict(learn_bandwidth=True, growth_mode='centered', interaction=True, consistency_weight=1.),
 'direct': dict(learn_bandwidth=True, growth_mode='centered', interaction=False, direct=True),
}

def configuration(arm):
    c=dict(arm=arm, learn_bandwidth=True, growth_mode='centered', interaction=True,
        direct=False, data_loss='nll', consistency_weight=0., mass_weight=20.,
        velocity_weight=1., growth_residual_weight=1., growth_penalty=.01,
        growth_variance_penalty=.01, residual_floor=.05, residual_alpha=.5,
        support_weight=1., ess_weight=2., ess_fraction=.2, bandwidth_prior=.001,
        mmd_weight=100., hidden=[64,64], interval_batch=8, data_batch_per_time=64)
    c.update(ARMS[arm]);return c

class VectorMLP(nn.Module):
    def __init__(self,d):
        super().__init__();self.layers=nn.ModuleList([nn.Linear(d+1,60),nn.Linear(60,60),nn.Linear(60,d)])
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
    def forward(self,x):
        for layer in self.layers[:-1]:
            y=F.silu(layer(x));x=x+y if x.shape[-1]==y.shape[-1] else y
        return self.layers[-1](x)

class FlexibleStitching(Stitching):
    def __init__(self,x,lm,grid,bw,obs,targets,cfg):
        nn.Module.__init__(self);self.cfg=cfg;d=x.shape[-1]
        self.trajectories=nn.Parameter(torch.as_tensor(x,dtype=torch.float32).clone())
        self.log_masses=nn.Parameter(torch.as_tensor(lm,dtype=torch.float32).clone())
        self.register_buffer('grid',torch.as_tensor(grid,dtype=torch.float32).clone())
        self.register_buffer('initial_bandwidth',torch.as_tensor(bw,dtype=torch.float32).clone())
        self.register_buffer('observed_times',torch.as_tensor(obs,dtype=torch.float32).clone())
        self.register_buffer('count_logmass',torch.as_tensor(np.log(targets),dtype=torch.float32))
        if cfg['learn_bandwidth']:
            self.raw_bandwidth=nn.Parameter(torch.log(torch.expm1(self.initial_bandwidth-.03)))
        self.raw_entropy=nn.Parameter(torch.tensor(math.log(math.expm1(.03)),dtype=torch.float32))
        self.potential=None if cfg['direct'] else ScalarMLP(d+1,(64,64))
        self.velocity=VectorMLP(d) if cfg['direct'] else None
        self.interaction=ScalarMLP(1,(16,16)) if cfg['interaction'] else None
        self.growth=ScalarMLP(d+1,(64,64)) if cfg['growth_mode']!='uniform' else None
    @property
    def bandwidth(self):
        return .03+F.softplus(self.raw_bandwidth) if self.cfg['learn_bandwidth'] else self.initial_bandwidth
    def global_rate(self,t):
        j=torch.searchsorted(self.observed_times,t.contiguous(),right=True)-1
        j=j.clamp(0,len(self.observed_times)-2)
        return (self.count_logmass[j+1]-self.count_logmass[j])/(self.observed_times[j+1]-self.observed_times[j])
    def field(self,positions,lm,time,queries):
        diff=queries[:,:,None,:]-positions[:,None,:,:]
        resp=F.softmax(self.log_components(positions,queries)+lm[:,None,:],dim=-1)
        score=-(resp[:,:,:,None]*diff/self.bandwidth.square()).sum(2)
        tx=torch.cat((queries,time[:,None,None].expand(*queries.shape[:2],1)),dim=-1)
        v=self.velocity(tx) if self.velocity is not None else -self.potential.gradient(tx)[...,:-1]
        v=v-self.entropy*score
        if self.interaction is not None:
            dw=self.interaction.gradient(diff.square().sum(-1,keepdim=True)/positions.shape[-1])
            v=v-(lm.exp()[:,None,:,None]*2/positions.shape[-1]*diff*dw).sum(2)
        if self.growth is None:g=self.global_rate(time)[:,None].expand(queries.shape[:2])
        else:
            g=self.growth(tx)
            if self.cfg['growth_mode']=='centered':
                tc=torch.cat((positions,time[:,None,None].expand(*positions.shape[:2],1)),dim=-1)
                mean=(F.softmax(lm,-1)*self.growth(tc)).sum(-1)
                g=g-mean[:,None]+self.global_rate(time)[:,None]
        return v,g


def gaussian_mmd(x,lm,y,h,bw2):
    """Exact RBF MMD2: diagonal Gaussian mixture versus empirical observations.

    Includes Gaussian kernel uncertainty analytically. No noisy KDE draws or
    held-time bandwidth. Biased empirical target term, same as evaluation.
    """
    w=F.softmax(lm,-1);xx=(x[:,:,None,:]-x[:,None,:,:]).square()
    xy=(x[:,:,None,:]-y[:,None,:,:]).square();yy=(y[:,:,None,:]-y[:,None,:,:]).square().sum(-1)
    result=x.new_zeros(())
    for factor in (.5,1.,2.):
        b=bw2*factor;vxx=b+2*h.square();vxy=b+h.square()
        kxx=torch.exp(-.5*(xx/vxx).sum(-1)+.5*torch.log(b/vxx).sum())
        kxy=torch.exp(-.5*(xy/vxy).sum(-1)+.5*torch.log(b/vxy).sum())
        kyy=torch.exp(-yy/(2*b))
        result=result+((w[:,:,None]*w[:,None,:]*kxx).sum((-1,-2))-2*(w*kxy.mean(-1)).sum(-1)+kyy.mean((-1,-2))).mean()/3
    return result


def loss(m,oi,y,targets,ids,noise,cfg,support,radius,bw2):
    xx,ll=m.trajectories[oi],m.log_masses[oi]
    nll=-m.log_prob(xx,ll,y).mean()
    mmd=gaussian_mmd(xx,ll,y,m.bandwidth,bw2) if cfg['data_loss']!='nll' else xx.new_zeros(())
    data=nll if cfg['data_loss']=='nll' else cfg['mmd_weight']*mmd+(nll*.25 if cfg['data_loss']=='hybrid' else 0.)
    mass=(torch.logsumexp(ll,-1)-targets.log()).square().mean()
    x0,x1=m.trajectories[ids],m.trajectories[ids+1];l0,l1=m.log_masses[ids],m.log_masses[ids+1]
    t0,t1=m.grid[ids],m.grid[ids+1];dt=t1-t0
    # Evaluate endpoints from inside the interval for piecewise constant count rate.
    ve,ge=m.field(torch.cat((x0,x1)),torch.cat((l0,l1)),torch.cat((t0+1e-5*dt,t1-1e-5*dt)),torch.cat((x0,x1)))
    v0,v1=ve.chunk(2);g0,g1=ge.chunk(2)
    xd=(x1-x0)/dt[:,None,None];rate=(l1-l0)/dt[:,None]
    a=torch.exp(.5*(l0+l1));w=(1-cfg['residual_floor'])*a+cfg['residual_floor']*a.mean(-1,keepdim=True)
    cv=(w*(xd-.5*(v0+v1)).square().sum(-1)).sum(-1)
    cg=(w*(rate-.5*(g0+g1)).square()).sum(-1)
    xm,lmm=.5*(x0+x1),.5*(l0+l1);q=xm+m.bandwidth*noise
    vt,gt=m.induced_fields(xm,lmm,xd,rate,q);vf,gf=m.field(xm,lmm,.5*(t0+t1),q)
    mv=(w*(vt-vf).square().sum(-1)).sum(-1);mg=(w*(gt-gf).square()).sum(-1)
    energy=(a*rate.square()).sum(-1);var=(rate-rate.mean(-1,keepdim=True)).square().mean(-1)
    alpha=cfg['residual_alpha'];terms=torch.stack(((1-alpha)*cv+alpha*mv,(1-alpha)*cg+alpha*mg,energy,var),1)
    terms=(dt[:,None]*terms).mean(0)*(len(m.grid)-1)/(m.grid[-1]-m.grid[0])
    sx,mask=support;dist=(xx[:,:,None,:]-sx[:,None,:,:]).square().sum(-1).masked_fill(~mask[:,None,:],1e9).min(-1).values.clamp_min(1e-12).sqrt()
    supp=F.relu(dist-radius[:,None]).square().mean()
    ess=1/F.softmax(m.log_masses,-1).square().sum(-1)
    ep=F.relu(math.log(cfg['ess_fraction']*m.log_masses.shape[-1])-ess.log()).square().mean()
    prior=(m.bandwidth.log()-m.initial_bandwidth.log()).square().mean()
    consistency=xx.new_zeros(())
    if cfg['consistency_weight']:
        # Differentiated one-step Heun rollout, endpoint identity pairing; tests
        # whether local residuals are sufficient. Uses sampled training intervals.
        xp=x0+dt[:,None,None]*v0;lp=l0+dt[:,None]*g0
        vp,gp=m.field(xp,lp,t1-1e-5*dt,xp)
        xr=x0+.5*dt[:,None,None]*(v0+vp);lr=l0+.5*dt[:,None]*(g0+gp)
        consistency=(w*((xr-x1).square().sum(-1)+(lr-l1).square())/dt[:,None].square()).sum(-1).mean()
    total=data+cfg['mass_weight']*mass+cfg['velocity_weight']*terms[0]+cfg['growth_residual_weight']*terms[1]+cfg['growth_penalty']*terms[2]+cfg['growth_variance_penalty']*terms[3]+cfg['support_weight']*supp+cfg['ess_weight']*ep+cfg['bandwidth_prior']*prior+cfg['consistency_weight']*consistency
    return total,dict(nll=nll,mmd=mmd,mass=mass,velocity_residual=terms[0],growth_residual=terms[1],support=supp,ess_penalty=ep,minimum_ess=ess.min(),bandwidth_mean=m.bandwidth.mean(),entropy=m.entropy,consistency=consistency)
