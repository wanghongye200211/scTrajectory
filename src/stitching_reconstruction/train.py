"""Joint trajectory/field fitting, selected using visible validation only."""
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist
from scipy.special import softmax
from scipy.stats import wasserstein_distance
from .data import load_visible
from .io import (device, digest, environment, folders, fresh, read_json, source_hashes,
                 synchronize, tensor, write_json)
from .physics.mass_control import integrate, select_count_curve
from .physics.shape_losses import ShapeStitching, configuration, objective


def initialize(a, particles, nodes, seed):
    z, t, tr = a['x'], a['time'], a['train_rows']
    obs, targets = a['observed_times'], a['mass_targets']
    pools = [tr[t[tr] == q] for q in obs]
    rng = np.random.default_rng(seed)
    n = min(512, *map(len, pools))
    path = [z[rng.choice(pools[0], n, replace=False)]]
    for pool in pools[1:]:
        y = z[rng.choice(pool, n, replace=False)]
        rows, cols = linear_sum_assignment(cdist(path[-1], y, 'sqeuclidean'))
        np.testing.assert_array_equal(rows, np.arange(n))
        path.append(y[cols])
    take = rng.choice(n, particles, replace=n < particles)
    path = np.asarray(path)[:, take]
    # Merge again in the actual integration dtype: two float64 values can map
    # to one float32 knot, otherwise a zero-duration residual interval appears.
    grid = np.unique(np.r_[np.linspace(obs[0], obs[-1], nodes), obs].astype(np.float32))
    xx = []
    for q in grid:
        j = np.searchsorted(obs, q, side='right') - 1
        j = np.clip(j, 0, len(obs) - 2)
        f = (q - obs[j]) / (obs[j+1] - obs[j])
        xx.append((1 - f) * path[j] + f * path[j+1])
    lm = np.repeat(np.interp(grid, obs, np.log(targets))[:, None], particles, axis=1) - math.log(particles)
    pts = z[pools[0]]
    bw = np.maximum(1.06 * pts.std(0) * len(pts)**(-1 / (z.shape[1] + 4)), .12)
    curve = select_count_curve(obs, targets) if len(obs) > 2 else dict(selected='loglinear', scores={}, folds={}, rule='two-time fallback')
    return dict(x=np.array(xx, np.float32), lm=lm.astype(np.float32), grid=grid, bw=bw,
                obs=obs, targets=targets, dimension=z.shape[1], curve_kind=curve['selected']), curve


def fit(data, output_root, run, arm='baseline', steps=4000, particles=96, nodes=33,
        seed=20261008, devname='mps', config_path=None):
    if steps < 1 or particles < 2 or nodes < 2:
        raise ValueError('steps>=1, particles>=2 and nodes>=2 required')
    dev = device(devname)
    a, meta = load_visible(data)
    paths = folders(output_root, run)
    protocol_path = paths['results'] / 'protocol.json'
    fresh(protocol_path); fresh(paths['models'] / 'selected.pt')
    cfg = configuration(arm)
    if config_path:
        override = read_json(config_path)
        allowed = {'mass_weight', 'velocity_weight', 'growth_residual_weight', 'growth_penalty',
                   'growth_variance_penalty', 'residual_floor', 'residual_alpha', 'support_weight',
                   'ess_weight', 'ess_fraction', 'shape_weight', 'ot_velocity_weight', 'endpoint_weight',
                   'checkpoint_every', 'interval_batch', 'data_batch_per_time'}
        if not set(override) <= allowed:
            raise ValueError(f'Unsupported loss/config keys: {set(override) - allowed}')
        for key, value in override.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'Nonnegative finite config required: {key}')
        cfg.update(override)
    for key in ('checkpoint_every', 'interval_batch', 'data_batch_per_time'):
        if int(cfg[key]) != cfg[key] or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 <= cfg['residual_floor'] <= 1 or not 0 <= cfg['residual_alpha'] <= 1 or not 0 < cfg['ess_fraction'] <= 1:
        raise ValueError('Residual mixing/floor and ESS fraction out of range')
    cfg.update(seed=seed, steps=steps, device=str(dev), dtype='float32', fallback=False,
               learning_rate=.003, selection='visible-interval independent rollout SWD; no held data/counts')
    started = time.monotonic()
    init, curve = initialize(a, particles, nodes, seed)
    torch.manual_seed(seed)
    m = ShapeStitching(init['x'], init['lm'], init['grid'], init['bw'], init['obs'], init['targets'], cfg, init['curve_kind']).to(dev)
    z, tt = a['x'], a['time']
    tr, va, obs, targets = a['train_rows'], a['validation_rows'], init['obs'], init['targets']
    pools = [tr[tt[tr] == t] for t in obs]
    vpools = [va[tt[va] == t] for t in obs]
    sx = np.zeros((len(obs), max(map(len, pools)), z.shape[1]), np.float32)
    mask = np.zeros(sx.shape[:2], bool); radius = []
    for j, pool in enumerate(pools):
        sx[j, :len(pool)] = z[pool]; mask[j, :len(pool)] = True
        radius.append(max(np.quantile(cKDTree(z[pool]).query(z[pool], k=2)[0][:, 1], .95), np.linalg.norm(init['bw'])))
    support = (tensor(sx, dev), torch.as_tensor(mask, device=dev))
    radius = tensor(radius, dev)
    oi_np = np.array([np.argmin(abs(init['grid'] - t)) for t in obs])
    np.testing.assert_allclose(init['grid'][oi_np], obs, atol=1e-6)
    oi = torch.as_tensor(oi_np, device=dev); target = tensor(targets, dev)
    idx = np.random.default_rng(20260930).choice(tr, min(1000, len(tr)), False)
    bw2 = max(float(np.median(pdist(z[idx].astype(float), 'sqeuclidean'))), 1e-8)
    banks = []
    brng = np.random.default_rng(seed)
    if cfg['ot_velocity_weight']:
        for j in range(len(pools) - 1):
            n = min(512, len(pools[j]), len(pools[j+1]))
            xx, yy = z[brng.choice(pools[j], n, False)], z[brng.choice(pools[j+1], n, False)]
            rows, cols = linear_sum_assignment(cdist(xx, yy, 'sqeuclidean'))
            banks.append((xx[rows], yy[cols]))
    parameters = {key: p.numel() for key, p in m.named_parameters()}
    protocol = dict(config=cfg, initialization=curve, visible_sha256=meta['visible_sha256'],
                    source_hashes=source_hashes(), environment=environment(), mmd_bandwidth_squared=bw2,
                    parameter_counts=dict(total=sum(parameters.values()), by_parameter=parameters),
                    trajectory_particles=particles, trajectory_nodes=len(init['grid']),
                    time_scale=float(a['time_scale']), time_offset=float(a['time_offset']),
                    scope='Joint trajectory and field fitting. Auxiliary OT is a pseudo-pairing, not a measured lineage.')
    write_json(protocol_path, protocol)
    shutil.copytree(Path(__file__).parent, paths['models'] / 'source' / 'stitching_reconstruction',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    np.savez_compressed(paths['models'] / 'initial.npz', **init)
    timing = dict(initialization_seconds=time.monotonic() - started, validation_seconds=0.)
    vrng = np.random.default_rng(seed + 1999)
    vn = min(128, min(map(len, pools)))
    vs = tensor(np.stack([z[vrng.choice(p, vn, False)] for p in pools[:-1]]), dev)
    vy = [z[vrng.choice(p, min(256, len(p)), False)] for p in vpools[1:]]
    vl = tensor(np.repeat(np.log(targets[:-1, None] / vn), vn, axis=1), dev)
    t0, t1 = tensor(obs[:-1], dev), tensor(obs[1:], dev)
    dirs = vrng.normal(size=(64, z.shape[1])); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    @torch.no_grad()
    def validate():
        clock = time.monotonic()
        xx, ll = integrate(m, vs.clone(), vl.clone(), t0, t1, 32)
        xx, ll = xx.cpu().numpy(), ll.cpu().numpy()
        scores = [float(np.mean([wasserstein_distance(xx[j] @ d, y @ d, u_weights=softmax(ll[j])) for d in dirs])) for j, y in enumerate(vy)]
        if not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite visible validation rollout')
        timing['validation_seconds'] += time.monotonic() - clock
        return dict(visible_swd=float(np.mean(scores)), intervals=scores)

    def save(name, step):
        torch.save(dict(state_dict={k: v.detach().cpu() for k, v in m.state_dict().items()},
                        initial=init, config=cfg, step=step,
                        portable_metadata=dict(visible_sha256=meta['visible_sha256'],
                                               source_hashes=protocol['source_hashes'])), paths['models'] / name)

    def log(row):
        import json
        line = json.dumps(row, allow_nan=False)
        with (paths['logs'] / 'training.jsonl').open('a') as f:
            f.write(line + '\n')
        print(line, flush=True)

    rng = np.random.default_rng(seed + 100)
    optim = torch.optim.Adam(m.parameters(), lr=.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, steps, eta_min=.00015)
    synchronize(dev); start_fit = time.monotonic()
    score = validate()['visible_swd']; best_step = 0; save('selected.pt', 0)
    log(dict(step=0, visible_swd=score))
    for step in range(1, steps + 1):
        batch = np.stack([z[rng.choice(p, cfg['data_batch_per_time'], len(p) < cfg['data_batch_per_time'])] for p in pools])
        ii = rng.choice(len(init['grid']) - 1, min(cfg['interval_batch'], len(init['grid']) - 1), False)
        noise = rng.normal(size=(len(ii), *init['x'].shape[1:]))
        teacher = endpoint = None
        if banks:
            groups = []
            for _ in range(cfg['teacher_groups']):
                j = int(rng.integers(len(banks))); bx, by = banks[j]
                idx = rng.choice(len(bx), cfg['teacher_particles'], len(bx) < cfg['teacher_particles'])
                groups.append((bx[idx], by[idx], obs[j], obs[j+1], rng.random()))
            teacher = tuple(tensor(np.array([g[k] for g in groups]), dev) for k in range(5))
        if cfg['endpoint_weight'] and step % cfg['endpoint_every'] == 0:
            j = int(rng.integers(len(pools) - 1)); n = cfg['endpoint_particles']
            xx = tensor(z[rng.choice(pools[j], n, len(pools[j]) < n)][None], dev)
            yy = tensor(z[rng.choice(pools[j+1], n, len(pools[j+1]) < n)][None], dev)
            endpoint = (xx, yy, tensor([obs[j]], dev), tensor([obs[j+1]], dev))
        optim.zero_grad(set_to_none=True)
        value, parts = objective(m, oi, tensor(batch, dev), target, torch.as_tensor(ii, device=dev),
                                 tensor(noise, dev), cfg, support, radius, bw2, teacher, endpoint)
        if not torch.isfinite(value):
            raise FloatingPointError(f'Nonfinite training objective at step {step}')
        value.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 10., error_if_nonfinite=True)
        optim.step(); scheduler.step()
        if step == 1 or step % cfg['checkpoint_every'] == 0 or step == steps:
            val = validate()
            log(dict(step=step, loss=float(value.detach()), gradient_norm=float(gn),
                     **{k: float(v.detach()) for k, v in parts.items()}, **val))
            if val['visible_swd'] < score:
                score, best_step = val['visible_swd'], step; save('selected.pt', step)
    save('final.pt', steps); synchronize(dev)
    elapsed = time.monotonic() - start_fit
    report = dict(status='TRAINING_COMPLETE', selected_step=best_step, best_visible_swd=score,
                  selected_checkpoint=str(paths['models'] / 'selected.pt'),
                  checkpoint_sha256=digest(paths['models'] / 'selected.pt'),
                  training_wall_seconds=elapsed, timing=timing,
                  timing_scope='training, visible validation and checkpoint writes; initialization separate',
                  parameter_counts=protocol['parameter_counts'], device=str(dev), fallback=False)
    write_json(paths['results'] / 'training.json', report)
    (paths['models'] / 'README.md').write_text('# Reconstruction model\n\nSelected on visible validation only.\n'
        'Weights and optimized paths were fitted jointly. Checkpoint is for inference; optimizer state is not saved for exact resumption.\n'
        f'Protocol: {protocol_path}\n')
    return report
