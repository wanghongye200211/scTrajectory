"""PyTorch/MPS implementation of unbalanced Stitching, independent of OM.

The scalar MLP and its exact analytic spatial derivative share parameters.
This avoids nested autograd dispatches on MPS while retaining gradients through
the force with respect to positions and network parameters. There is no detach
in the training force. Reference parity is checked against the original JAX code.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ScalarMLP(nn.Module):
    def __init__(self,dim,hidden):
        super().__init__();dims=[dim,*hidden,1]
        self.layers=nn.ModuleList(nn.Linear(a,b) for a,b in zip(dims[:-1],dims[1:]))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)

    def forward(self,x):
        for layer in self.layers[:-1]:
            v=F.silu(layer(x));x=x+v if x.shape[-1]==v.shape[-1] else v
        return self.layers[-1](x)[...,0]

    def gradient(self,x):
        cache=[]
        for layer in self.layers[:-1]:
            u=layer(x);same=x.shape[-1]==u.shape[-1]
            cache.append((u,layer.weight,same))
            y=F.silu(u);x=x+y if same else y
        g=self.layers[-1].weight[0].expand_as(x)
        for u,w,same in reversed(cache):
            s=torch.sigmoid(u);derivative=s*(1+u*(1-s))
            prev=(g*derivative)@w
            g=prev+g if same else prev
        return g


class Stitching(nn.Module):
    def __init__(self,positions,log_masses,grid,bandwidth,hidden=(64,64),interaction_hidden=(16,16)):
        super().__init__();d=positions.shape[-1]
        self.trajectories=nn.Parameter(torch.as_tensor(positions,dtype=torch.float32).clone())
        self.log_masses=nn.Parameter(torch.as_tensor(log_masses,dtype=torch.float32).clone())
        self.potential=ScalarMLP(d+1,hidden)
        self.interaction=ScalarMLP(1,interaction_hidden)
        self.growth=ScalarMLP(d+1,hidden)
        self.raw_entropy=nn.Parameter(torch.tensor(math.log(math.expm1(.03)),dtype=torch.float32))
        self.register_buffer('grid',torch.as_tensor(grid,dtype=torch.float32).clone())
        self.register_buffer('bandwidth',torch.as_tensor(bandwidth,dtype=torch.float32).clone())

    @property
    def entropy(self):return F.softplus(self.raw_entropy)

    def log_components(self,positions,queries):
        # All fields accept batch x points x coordinate arrays.
        diff=queries[:,:,None,:]-positions[:,None,:,:]
        return -.5*((diff/self.bandwidth)**2).sum(-1)-torch.log(self.bandwidth).sum()-.5*positions.shape[-1]*math.log(2*math.pi)

    def log_prob(self,positions,lm,queries):
        return torch.logsumexp(self.log_components(positions,queries)+F.log_softmax(lm,-1)[:,None,:],dim=-1)

    def field(self,positions,lm,time,queries):
        log_comp=self.log_components(positions,queries)
        responsibility=F.softmax(log_comp+lm[:,None,:],dim=-1)
        diff=queries[:,:,None,:]-positions[:,None,:,:]
        score=-(responsibility[:,:,:,None]*diff/self.bandwidth.square()).sum(2)
        tx=torch.cat((queries,time[:,None,None].expand(queries.shape[0],queries.shape[1],1)),dim=-1)
        grad_v=self.potential.gradient(tx)[...,:-1]
        radius2=diff.square().sum(-1,keepdim=True)/positions.shape[-1]
        dw=self.interaction.gradient(radius2)
        interaction=(torch.exp(lm)[:,None,:,None]*(2/positions.shape[-1])*diff*dw).sum(2)
        velocity=-grad_v-self.entropy*score-interaction
        growth=self.growth(tx)
        return velocity,growth

    def induced_fields(self,positions,lm,xdot,rate,queries):
        resp=F.softmax(self.log_components(positions,queries)+lm[:,None,:],dim=-1)
        return resp@xdot,(resp@rate[:,:,None])[...,0]


def objective(m,obs_idx,samples,targets,intervals,noise,cfg,support=None,radius=None):
    xx=m.trajectories[obs_idx];ll=m.log_masses[obs_idx]
    nll=-m.log_prob(xx,ll,samples).mean()
    mass=(torch.logsumexp(ll,dim=-1)-torch.log(targets)).square().mean()
    x0,x1=m.trajectories[intervals],m.trajectories[intervals+1]
    l0,l1=m.log_masses[intervals],m.log_masses[intervals+1]
    t0,t1=m.grid[intervals],m.grid[intervals+1];dt=t1-t0
    # One batched field call for the two endpoints reduces Metal dispatches.
    vv,gg=m.field(torch.cat((x0,x1)),torch.cat((l0,l1)),torch.cat((t0,t1)),torch.cat((x0,x1)))
    v0,v1=vv.chunk(2);g0,g1=gg.chunk(2)
    xd=(x1-x0)/dt[:,None,None];r=(l1-l0)/dt[:,None]
    a=torch.exp(.5*(l0+l1));floor=cfg.get('residual_floor',0.)
    guarded=(1-floor)*a+floor*a.sum(-1,keepdim=True)/a.shape[-1]
    cv=(guarded*(xd-.5*(v0+v1)).square().sum(-1)).sum(-1)
    cg=(guarded*(r-.5*(g0+g1)).square()).sum(-1)
    xm,lm=.5*(x0+x1),.5*(l0+l1);q=xm+m.bandwidth*noise
    vt,rt=m.induced_fields(xm,lm,xd,r,q)
    vf,gf=m.field(xm,lm,.5*(t0+t1),q)
    mv=(guarded*(vt-vf).square().sum(-1)).sum(-1)
    mg=(guarded*(rt-gf).square()).sum(-1)
    energy=(a*r.square()).sum(-1);variance=(r-r.mean(-1,keepdim=True)).square().mean(-1)
    alpha=cfg.get('residual_alpha',.5)
    terms=torch.stack(((1-alpha)*cv+alpha*mv,(1-alpha)*cg+alpha*mg,energy,variance,cv,cg,mv,mg),dim=1)
    terms=(dt[:,None]*terms).mean(0)*(len(m.grid)-1)/(m.grid[-1]-m.grid[0])
    supp=xx.new_zeros(())
    if cfg.get('support_weight',0.):
        support_x,support_mask=support
        d2=(xx[:,:,None,:]-support_x[:,None,:,:]).square().sum(-1)
        d2=d2.masked_fill(~support_mask[:,None,:],1e9)
        dist=torch.sqrt(torch.clamp(d2.min(-1).values,min=1e-12))
        # Applies only at observed training times; does not forbid new intervening states.
        supp=F.relu(dist-radius[:,None]).square().mean()
    weights=F.softmax(m.log_masses,dim=-1)
    ess=1/weights.square().sum(-1)
    target_ess=cfg.get('ess_fraction',.2)*weights.shape[-1]
    ess_penalty=F.relu(math.log(target_ess)-torch.log(ess)).square().mean()
    total=(nll+cfg['mass_weight']*mass+cfg['velocity_weight']*terms[0]
           +cfg['growth_residual_weight']*terms[1]+cfg['growth_penalty']*terms[2]
           +cfg['growth_variance_penalty']*terms[3]+cfg.get('support_weight',0.)*supp
           +cfg.get('ess_weight',0.)*ess_penalty)
    parts=dict(nll=nll,mass=mass,velocity_residual=terms[0],growth_residual=terms[1],
               growth_energy=terms[2],growth_variance=terms[3],center_velocity_residual=terms[4],
               center_growth_residual=terms[5],kde_velocity_residual=terms[6],kde_growth_residual=terms[7],
               support=supp,ess_penalty=ess_penalty,minimum_ess=ess.min(),entropy=m.entropy)
    return total,parts
