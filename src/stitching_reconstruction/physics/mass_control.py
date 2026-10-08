"""Mass-factorized Stitching experiments, separate from immutable earlier runs.

Captured counts define a positive global curve. Local growth is centered using
the CURRENT particle population, including during independent integration.
The global curve is chosen on visible counts only. No held-time data enter here.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.interpolate import PchipInterpolator
from .experimental import FlexibleStitching, configuration, loss as stitching_loss, gaussian_mmd
from .model import ScalarMLP


ARMS = ('free', 'hard_local', 'hard_uniform', 'hard_shrink', 'hard_rollout', 'otcfm')


def count_value(t, counts, q, kind):
    t, counts, q = np.asarray(t), np.asarray(counts), np.asarray(q)
    if kind == 'linear':
        return np.interp(q, t, counts)
    if kind == 'pchip':
        return np.exp(PchipInterpolator(t, np.log(counts))(q))
    return np.exp(np.interp(q, t, np.log(counts)))


def select_count_curve(t, counts):
    """Interior leave-one-visible-time-out log-count MAE; no held count access."""
    t, counts = np.asarray(t), np.asarray(counts)
    kinds = ('loglinear', 'linear', 'pchip')
    folds = {k: [] for k in kinds}
    for j in range(1, len(t)-1):
        take = np.arange(len(t)) != j
        for k in kinds:
            pred = float(count_value(t[take], counts[take], t[j], k))
            folds[k].append(dict(time=float(t[j]), observed=float(counts[j]),
                                 prediction=pred, absolute_log_error=float(abs(np.log(pred/counts[j])))))
    scores = {k: float(np.mean([v['absolute_log_error'] for v in folds[k]])) for k in kinds}
    # Three visible times give only one validation fold. Do not select on it.
    chosen = min(kinds, key=lambda k: scores[k]) if len(t) >= 5 else 'loglinear'
    return dict(selected=chosen, scores=scores, folds=folds,
                rule='minimum interior visible-time log-count MAE; loglinear fallback if fewer than 5 visible times')


class CountCurve(nn.Module):
    def __init__(self, times, masses, kind):
        super().__init__()
        tt, mm = np.asarray(times, dtype=float), np.asarray(masses, dtype=float)
        assert len(tt) >= 2 and np.all(np.diff(tt) > 0) and np.all(mm > 0)
        self.kind = kind
        self.register_buffer('times', torch.tensor(tt, dtype=torch.float32))
        self.register_buffer('values', torch.tensor(mm, dtype=torch.float32))
        coef = PchipInterpolator(tt, np.log(mm)).c.T
        self.register_buffer('coefficients', torch.tensor(coef, dtype=torch.float32))

    def value_rate(self, t):
        j = (torch.searchsorted(self.times, t.contiguous(), right=True)-1).clamp(0, len(self.times)-2)
        dt = self.times[j+1]-self.times[j]
        delta = t-self.times[j]
        a, b = self.values[j], self.values[j+1]
        if self.kind == 'linear':
            mass = a+(b-a)*delta/dt
            return mass.log(), (b-a)/dt/mass
        if self.kind == 'pchip':
            c = self.coefficients[j]
            logmass = ((c[..., 0]*delta+c[..., 1])*delta+c[..., 2])*delta+c[..., 3]
            return logmass, (3*c[..., 0]*delta+2*c[..., 1])*delta+c[..., 2]
        rate = (b.log()-a.log())/dt
        return a.log()+delta*rate, rate


def configuration_for(arm):
    assert arm in ARMS
    cfg = configuration('fixed')
    cfg.update(arm=arm, model_family='mass_control', hard_mass=arm != 'free',
               growth_mode='free' if arm == 'free' else 'uniform' if arm in ('hard_uniform', 'otcfm') else 'centered',
               max_local_rate=.5 if arm in ('hard_shrink', 'hard_rollout') else None,
               local_energy_weight=.1 if arm in ('hard_shrink', 'hard_rollout') else 0.,
               local_temporal_weight=.05 if arm in ('hard_shrink', 'hard_rollout') else 0.,
               rollout_weight=100. if arm == 'hard_rollout' else 0., rollout_every=4,
               rollout_particles=48, rollout_steps=8, checkpoint_every=500)
    return cfg


class MassStitching(FlexibleStitching):
    def __init__(self, x, lm, grid, bw, obs, targets, cfg, curve_kind):
        nn.Module.__init__(self)
        self.cfg = cfg
        d = x.shape[-1]
        self.trajectories = nn.Parameter(torch.as_tensor(x, dtype=torch.float32).clone())
        if cfg['hard_mass']:
            self.weight_logits = nn.Parameter(torch.as_tensor(lm, dtype=torch.float32).clone())
        else:
            self.raw_log_masses = nn.Parameter(torch.as_tensor(lm, dtype=torch.float32).clone())
        self.register_buffer('grid', torch.as_tensor(grid, dtype=torch.float32).clone())
        self.register_buffer('initial_bandwidth', torch.as_tensor(bw, dtype=torch.float32).clone())
        self.register_buffer('observed_times', torch.as_tensor(obs, dtype=torch.float32).clone())
        self.register_buffer('count_logmass', torch.as_tensor(np.log(targets), dtype=torch.float32))
        self.raw_entropy = nn.Parameter(torch.tensor(np.log(np.expm1(.03)), dtype=torch.float32))
        self.potential = ScalarMLP(d+1,(64,64))
        self.velocity = None
        self.interaction = ScalarMLP(1,(16,16))
        self.growth = None if cfg['growth_mode']=='uniform' else ScalarMLP(d+1,(64,64))
        self.curve = CountCurve(obs, targets, curve_kind)

    @property
    def log_masses(self):
        if self.cfg['hard_mass']:
            mass, _ = self.curve.value_rate(self.grid)
            return mass[:, None]+F.log_softmax(self.weight_logits, -1)
        return self.raw_log_masses

    def global_rate(self, t):
        return self.curve.value_rate(t)[1]

    def field(self, positions, lm, time, queries):
        # Reuse exactly the previous potential/entropy/interaction velocity.
        v, g = super().field(positions, lm, time, queries)
        bound = self.cfg.get('max_local_rate')
        if self.growth is not None and bound is not None:
            tx = torch.cat((queries, time[:, None, None].expand(*queries.shape[:2], 1)), -1)
            tc = torch.cat((positions, time[:, None, None].expand(*positions.shape[:2], 1)), -1)
            u = bound*torch.tanh(self.growth(tx)/bound)
            uc = bound*torch.tanh(self.growth(tc)/bound)
            g = self.global_rate(time)[:, None]+u-(F.softmax(lm, -1)*uc).sum(-1)[:, None]
        return v, g


def project_mass(model, lm, t):
    if not model.cfg['hard_mass']:
        return lm
    return F.log_softmax(lm, -1)+model.curve.value_rate(t)[0][:, None]


def integrate(model, x, lm, t0, t1, steps):
    """Differentiable Heun integration with exact total-mass projection.

    Intervals must not cross visible count-curve knots; callers split there.
    Projection changes only the common log-mass offset, not relative weights.
    """
    dt = (t1-t0)/steps
    lm = project_mass(model, lm, t0)
    for k in range(steps):
        t = t0+k*dt
        v0, g0 = model.field(x, lm, t+1e-6*dt, x)
        xp = x+dt[:, None, None]*v0
        lp = project_mass(model, lm+dt[:, None]*g0, t+dt)
        v1, g1 = model.field(xp, lp, t+dt*(1-1e-6), xp)
        x = x+.5*dt[:, None, None]*(v0+v1)
        lm = project_mass(model, lm+.5*dt[:, None]*(g0+g1), t+dt)
    return x, lm


def objective(model, oi, y, targets, ids, noise, cfg, support, radius, bw2, rollout=None):
    total, parts = stitching_loss(model, oi, y, targets, ids, noise, cfg, support, radius, bw2)
    reg = total.new_zeros(())
    smooth = total.new_zeros(())
    if cfg['local_energy_weight'] or cfg['local_temporal_weight']:
        xx, ll = model.trajectories, model.log_masses
        x0, x1, l0, l1 = xx[ids], xx[ids+1], ll[ids], ll[ids+1]
        t0, t1 = model.grid[ids], model.grid[ids+1]
        dt = t1-t0
        times = torch.cat((t0+1e-5*dt, t1-1e-5*dt))
        _, gg = model.field(torch.cat((x0,x1)), torch.cat((l0,l1)), times, torch.cat((x0,x1)))
        delta = gg-model.global_rate(times)[:, None]
        d0, d1 = delta.chunk(2)
        w = F.softmax(.5*(l0+l1), -1)
        reg = (w*.5*(d0.square()+d1.square())).sum(-1).mean()
        smooth = (w*((d1-d0)/dt[:,None]).square()).sum(-1).mean()
        total = total+cfg['local_energy_weight']*reg+cfg['local_temporal_weight']*smooth
    rl = total.new_zeros(())
    if rollout is not None and cfg['rollout_weight']:
        x, target_y, t0, t1 = rollout
        lm = model.curve.value_rate(t0)[0][:,None].expand(x.shape[:2])-np.log(x.shape[1])
        xr, lr = integrate(model, x, lm, t0, t1, cfg['rollout_steps'])
        rl = gaussian_mmd(xr, lr, target_y, xr.new_zeros(x.shape[-1]), bw2)
        total = total+cfg['rollout_weight']*rl
    return total, dict(parts, local_field_energy=reg, local_field_temporal=smooth, rollout_mmd=rl)


class CFM(nn.Module):
    """Local exact-OT CFM comparator, neural capacity near V+W+g (10k)."""
    def __init__(self, d, obs, masses, cfg, curve_kind):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([nn.Linear(d+1,90),nn.Linear(90,90),nn.Linear(90,d)])
        nn.init.zeros_(self.layers[-1].weight); nn.init.zeros_(self.layers[-1].bias)
        self.curve = CountCurve(obs, masses, curve_kind)

    def field(self, positions, lm, time, queries):
        x = torch.cat((queries,time[:,None,None].expand(*queries.shape[:2],1)), -1)
        for layer in self.layers[:-1]:
            z = F.silu(layer(x)); x = x+z if x.shape[-1] == z.shape[-1] else z
        return self.layers[-1](x), self.curve.value_rate(time)[1][:,None].expand(queries.shape[:2])
