"""Frozen-model exports, self-consistent rollouts, and held-time checks.

No optimizer or fitting is invoked here. The evolving rollout cohort supplies
the density and interaction force at every Heun stage; fitted future centers
are never used as a forcing population and no path resets are performed.
"""
import math
from pathlib import Path
import time
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.special import softmax, logsumexp
from .data import load_visible
from .io import (arrays, device, digest, environment, folders, fresh, load_model,
                 read_json, source_hashes, synchronize, tensor, write_json)
from .physics.mass_control import integrate
from .physics.distribution_metrics import metric_values


def bind_model(checkpoint, data, dev):
    a, meta = load_visible(data)
    m, cp = load_model(checkpoint, dev)
    if cp.get('portable_metadata', {}).get('visible_sha256') != meta['visible_sha256']:
        raise ValueError('Checkpoint is not bound to this visible input; use import-checkpoint with the original protocol for a legacy model')
    np.testing.assert_allclose(m.observed_times.cpu().numpy(), a['observed_times'], rtol=1e-6)
    np.testing.assert_allclose(cp['initial']['targets'], a['mass_targets'], rtol=1e-6)
    if m.trajectories.shape[-1] != a['x'].shape[-1]:
        raise ValueError('Checkpoint/data feature dimension mismatch')
    return m, cp, a, meta


def import_checkpoint(checkpoint, protocol, data, output_root, run):
    a, meta = load_visible(data)
    source = read_json(protocol)
    if source['input_sha256'] != meta.get('source_axis_sha256'):
        raise ValueError('Legacy training protocol does not match this sealed axis')
    m, cp = load_model(checkpoint, torch.device('cpu'))
    if cp['config'] != source['config']:
        raise ValueError('Legacy checkpoint/config mismatch')
    np.testing.assert_allclose(m.observed_times.numpy(), a['observed_times'])
    np.testing.assert_allclose(cp['initial']['targets'], a['mass_targets'])
    if cp['initial']['dimension'] != a['x'].shape[-1]:
        raise ValueError('Dimension mismatch')
    paths = folders(output_root, run)
    dest = paths['models'] / 'selected.pt'; fresh(dest)
    cp['portable_metadata'] = dict(visible_sha256=meta['visible_sha256'],
                                  legacy_checkpoint_sha256=digest(checkpoint), legacy_protocol_sha256=digest(protocol))
    torch.save(cp, dest)
    reloaded, _ = load_model(dest, torch.device('cpu'))
    for k, value in m.state_dict().items():
        torch.testing.assert_close(value, reloaded.state_dict()[k], rtol=0, atol=0)
    with torch.no_grad():
        x, lm, t = m.trajectories[:1], m.log_masses[:1], m.grid[:1]
        old = m.field(x, lm, t, x); new = reloaded.field(x, lm, t, x)
        parity = max(float((u-v).abs().max()) for u, v in zip(old, new))
    report = dict(status='IMPORTED_WITHOUT_TRAINING', source_checkpoint=str(Path(checkpoint).resolve()),
                  source_checkpoint_sha256=digest(checkpoint), imported_checkpoint_sha256=digest(dest),
                  state_exact_match=True, cpu_field_max_abs_difference=parity,
                  visible_sha256=meta['visible_sha256'], source_hashes=source_hashes())
    write_json(paths['results'] / 'import.json', report)
    (paths['models'] / 'README.md').write_text('# Imported frozen model\n\nAll tensor values retained exactly.\n'
        f'Source checkpoint: {checkpoint}\nSource protocol: {protocol}\n')
    return report


@torch.no_grad()
def trace(model, x, lm, times, dtmax):
    times = np.asarray(times, float)
    if dtmax <= 0 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('Strictly increasing times and positive dtmax required')
    if x.ndim != 3 or x.shape[0] != 1 or lm.shape != x.shape[:2]:
        raise ValueError('trace accepts one cohort with shapes [1,K,d] and [1,K]')
    knots = model.observed_times.detach().cpu().numpy()
    xs, ls = [x[0].cpu().numpy().copy()], [lm[0].cpu().numpy().copy()]
    for left, right in zip(times[:-1], times[1:]):
        splits = np.unique(np.r_[left, knots[(knots > left) & (knots < right)], right])
        for t0, t1 in zip(splits[:-1], splits[1:]):
            x, lm = integrate(model, x, lm, x.new_tensor([t0]), x.new_tensor([t1]),
                              max(1, math.ceil((t1 - t0) / dtmax)))
            if not torch.isfinite(x).all() or not torch.isfinite(lm).all() or x.abs().max() > 1e6 or lm.abs().max() > 100:
                raise FloatingPointError(f'Unstable rollout between {t0:g} and {t1:g}; no clipping or resets applied')
        xs.append(x[0].cpu().numpy().copy()); ls.append(lm[0].cpu().numpy().copy())
    return np.asarray(xs), np.asarray(ls)


def weighted_vector_rms(error, lm):
    """RMS vector norm (sum across d); keep distinct from coordinate RMSE."""
    return np.sqrt(np.sum(softmax(lm, axis=-1) * np.sum(error**2, axis=-1), axis=-1))


@torch.no_grad()
def export(checkpoint, data, output_root, run, devname='mps', seed=20261008):
    dev = device(devname)
    m, cp, a, meta = bind_model(checkpoint, data, dev)
    paths = folders(output_root, run)
    dest = paths['results'] / 'reconstruction.npz'; fresh(dest)
    x, lm, grid = m.trajectories.detach(), m.log_masses.detach(), m.grid.detach()
    dt = grid[1:] - grid[:-1]
    xd = (x[1:] - x[:-1]) / dt[:, None, None]
    rate = (lm[1:] - lm[:-1]) / dt[:, None]
    midpoint = .5 * (grid[1:] + grid[:-1])
    # Export in chunks to bound B*Q*K*d storage for high-dimensional inputs.
    values = {k: [] for k in ('center_model_v', 'center_model_g', 'center_induced_v',
                              'center_induced_g', 'query', 'query_induced_v', 'query_induced_g',
                              'query_model_v', 'query_model_g')}
    rng = np.random.default_rng(seed)
    for j in range(len(dt)):
        pos, mass = .5*(x[j:j+1]+x[j+1:j+2]), .5*(lm[j:j+1]+lm[j+1:j+2])
        tt = midpoint[j:j+1]
        cv, cg = m.field(pos, mass, tt, pos)
        iv, ig = m.induced_fields(pos, mass, xd[j:j+1], rate[j:j+1], pos)
        # One Gaussian draw per component; weighted by component mass for the expectation.
        q = pos + m.bandwidth * tensor(rng.normal(size=pos.shape), dev)
        qv, qg = m.induced_fields(pos, mass, xd[j:j+1], rate[j:j+1], q)
        fv, fg = m.field(pos, mass, tt, q)
        for key, value in zip(values, (cv, cg, iv, ig, q, qv, qg, fv, fg)):
            values[key].append(value[0].cpu().numpy())
    values = {k: np.asarray(v) for k, v in values.items()}
    scale, offset = float(a['time_scale']), float(a['time_offset'])
    # Keep normalized model units plus explicitly named physical-time versions.
    physical = {k + '_per_physical_time': v / scale for k, v in values.items() if k.endswith(('_v', '_g'))}
    np.savez_compressed(dest, x=x.cpu().numpy(), logmass=lm.cpu().numpy(), model_time=grid.cpu().numpy(),
                        physical_time=offset + scale * grid.cpu().numpy(), midpoint=midpoint.cpu().numpy(),
                        center_derivative=xd.cpu().numpy(), center_logmass_derivative=rate.cpu().numpy(),
                        center_derivative_per_physical_time=xd.cpu().numpy()/scale,
                        center_logmass_derivative_per_physical_time=rate.cpu().numpy()/scale,
                        bandwidth=m.bandwidth.cpu().numpy(), **values, **physical)
    err = values['query_induced_v'] - values['query_model_v']
    mid_lm = .5 * (lm[:-1] + lm[1:]).cpu().numpy()
    vrms = weighted_vector_rms(err, mid_lm)
    grms = np.sqrt(np.sum(softmax(mid_lm, axis=-1) * (values['query_induced_g'] - values['query_model_g'])**2, axis=-1))
    interval_dt = dt.cpu().numpy()
    report = dict(status='EXPORTED', checkpoint_sha256=digest(checkpoint), visible_sha256=meta['visible_sha256'],
                  reconstruction_sha256=digest(dest), source_hashes=source_hashes(), device=str(dev),
                  trajectory_velocity_residual_rms=float(np.sqrt(np.average(vrms**2, weights=interval_dt))),
                  trajectory_growth_residual_rms=float(np.sqrt(np.average(grms**2, weights=interval_dt))),
                  residual_scope='normalized-mass KDE midpoint diagnostic, one fixed draw per component; not the full training objective or biological truth',
                  physical_time='physical_time = time_offset + time_scale * model_time; v_physical=v_model/time_scale',
                  time_offset=offset, time_scale=scale,
                  interpretation='center derivative, mixture-induced Eulerian field and learned field are exported separately; derivatives have jumps at path knots')
    write_json(paths['results'] / 'export.json', report)
    return report


@torch.no_grad()
def validate(checkpoint, data, output_root, run, devname='mps', dtmax=.0125, max_cells=512, seed=20261005):
    dev = device(devname)
    if max_cells < 2 or dtmax <= 0:
        raise ValueError('max_cells>=2 and dtmax>0 required')
    paths = folders(output_root, run)
    dest = paths['results'] / 'validation.json'; fresh(dest)
    fresh(paths['results'] / 'rollouts.npz')
    m, cp, a, meta = bind_model(checkpoint, data, dev)
    start = time.monotonic()
    for p in m.parameters():
        p.requires_grad_(False)
    state_before = {k: v.detach().clone() for k, v in m.state_dict().items()}
    grid = m.grid.cpu().numpy().astype(float)
    # Same starting optimized cohort tests long-horizon self-consistency.
    x0, l0 = m.trajectories[:1].clone(), m.log_masses[:1].clone()
    xx, ll = trace(m, x0.clone(), l0.clone(), grid, dtmax)
    xc, lc = trace(m, x0.clone(), l0.clone(), grid, 2 * dtmax)
    native, native_lm = m.trajectories.cpu().numpy(), m.log_masses.cpu().numpy()
    error = np.sqrt(np.mean((xx - native)**2, axis=(1, 2)))
    wrms = weighted_vector_rms(xx - native, native_lm)
    tr, z, tt, obs = a['train_rows'], a['x'], a['time'], a['observed_times']
    support = z[tr]
    tree = cKDTree(support)
    radius = float(np.quantile(tree.query(support, k=2)[0][:, 1], .99))
    outside = tree.query(xx.reshape(-1, xx.shape[-1]))[0].reshape(xx.shape[:2]) > radius
    weights = softmax(ll, axis=1)
    outside_weight = np.sum(weights * outside, axis=1)
    rng = np.random.default_rng(seed)
    first_pool = tr[tt[tr] == obs[0]]
    first = rng.choice(first_pool, min(max_cells, len(first_pool)), False)
    rx, rl = trace(m, tensor(z[first][None], dev), tensor(np.full((1, len(first)), math.log(a['mass_targets'][0]/len(first))), dev), grid, dtmax)
    idx = np.random.default_rng(20260930).choice(tr, min(1000, len(tr)), False)
    bw2 = max(float(np.median(pdist(z[idx].astype(float), 'sqeuclidean'))), 1e-8)
    dirs = np.random.default_rng(20260941).normal(size=(64, z.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    visible_rows = []
    for q, target_mass in zip(obs, a['mass_targets']):
        j = int(np.argmin(abs(grid - q)))
        pool = a['validation_rows'][tt[a['validation_rows']] == q]
        take = rng.choice(pool, min(max_cells, len(pool)), False)
        vals = metric_values(rx[j], z[take], softmax(rl[j]), bw2, dirs)
        visible_rows.append(dict(time=float(q), metrics=vals,
            relative_mass_error=float(abs(np.exp(logsumexp(rl[j]))/target_mass - 1))))
    test_rows, cache = [], {}
    # Only now open the sealed test file. No training/selection depends on these values.
    test_path = Path(data) / 'test.npz'
    if meta.get('test_sha256'):
        if digest(test_path) != meta['test_sha256']:
            raise ValueError('Sealed test input hash mismatch')
        test = arrays(test_path)
        for k, q in enumerate(np.unique(test['time'])):
            if q <= obs[0] or q >= obs[-1] or np.any(np.isclose(obs, q, rtol=0, atol=1e-8)):
                raise ValueError('This release supports strictly interior whole-time holdouts only')
            prev = float(max(obs[obs < q])); nxt = float(min(obs[obs > q]))
            erng = np.random.default_rng(seed + k)
            pool = tr[tt[tr] == prev]
            si = erng.choice(pool, min(max_cells, len(pool)), False)
            target_pool = np.flatnonzero(test['time'] == q)
            ti = erng.choice(target_pool, min(max_cells, len(target_pool)), False)
            mass0 = a['mass_targets'][np.argmin(abs(obs-prev))]
            source = tensor(z[si][None], dev)
            initial_lm = tensor(np.full((1, len(si)), math.log(mass0/len(si))), dev)
            xp, lp = trace(m, source.clone(), initial_lm.clone(), [prev, float(q)], dtmax)
            xpc, lpc = trace(m, source.clone(), initial_lm.clone(), [prev, float(q)], 2*dtmax)
            w = softmax(lp[-1]); target_x = test['x'][ti]
            metrics = metric_values(xp[-1], target_x, w, bw2, dirs)
            coarse = metric_values(xpc[-1], target_x, softmax(lpc[-1]), bw2, dirs)
            near = z[tr[np.isin(tt[tr], [prev, nxt])]]
            ntree = cKDTree(near); nr = float(np.quantile(ntree.query(near, k=2)[0][:, 1], .99))
            out = ntree.query(xp[-1])[0] > nr
            row = dict(time=float(q), physical_time=float(a['time_offset'] + a['time_scale']*q),
                source_time=prev, metrics=metrics, n_source=len(si), n_target=len(ti),
                refinement_coordinate_rmse=float(np.sqrt(np.mean((xp[-1]-xpc[-1])**2))),
                refinement_w2_change=abs(metrics['w2']-coarse['w2']),
                outside_weight=float(w[out].sum()), outside_particle_fraction=float(out.mean()),
                max_particle_weight=float(w.max()), predicted_relative_mass=float(np.exp(logsumexp(lp[-1]))))
            if 'velocity_true_physical' in test:
                v, _ = m.field(tensor(xp[-1][None], dev), tensor(lp[-1][None], dev), tensor([q], dev), tensor(target_x[None], dev))
                truth = test['velocity_true_physical'][ti]
                row['known_velocity_coordinate_rmse'] = float(np.sqrt(np.mean((v[0].cpu().numpy()/float(a['time_scale'])-truth)**2)))
                row['known_velocity_scope'] = 'Independent supplied truth at held-time queries; density comes from the rolled population'
            test_rows.append(row)
            cache.update({f'held_{k}_prediction': xp[-1], f'held_{k}_logmass': lp[-1],
                          f'held_{k}_source': z[si], f'held_{k}_target': target_x,
                          f'held_{k}_source_rows': si, f'held_{k}_target_rows': ti})
    for key, value in m.state_dict().items():
        torch.testing.assert_close(value, state_before[key], rtol=0, atol=0)
    result_path = paths['results'] / 'rollouts.npz'
    np.savez_compressed(result_path, model_time=grid, physical_time=a['time_offset'] + a['time_scale']*grid,
        native_x=native, native_logmass=native_lm, integrated_x=xx, integrated_logmass=ll,
        coarse_x=xc, coarse_logmass=lc, real_initial_x=rx, real_initial_logmass=rl,
        coordinate_rmse=error, weighted_vector_rms=wrms, outside_weight=outside_weight,
        outside_particle_fraction=outside.mean(1), **cache)
    synchronize(dev)
    report = dict(status='VALIDATED_WITHOUT_TRAINING', device=str(dev), fallback=False,
        environment=environment(), elapsed_seconds=time.monotonic()-start,
        checkpoint_sha256=digest(checkpoint), visible_sha256=meta['visible_sha256'],
        test_sha256=meta.get('test_sha256'), source_hashes=source_hashes(), rollouts_sha256=digest(result_path),
        model_state_unchanged=True, dtmax=dtmax, coarse_dtmax=2*dtmax,
        same_initial_endpoint_coordinate_rmse=float(error[-1]),
        same_initial_endpoint_weighted_vector_rms=float(wrms[-1]),
        integration_refinement_endpoint_coordinate_rmse=float(np.sqrt(np.mean((xx[-1]-xc[-1])**2))),
        endpoint_relative_mass_native=float(np.exp(logsumexp(native_lm[-1]))),
        endpoint_relative_mass_integrated=float(np.exp(logsumexp(ll[-1]))),
        minimum_ess=float((1/np.sum(weights**2, axis=1)).min()),
        maximum_outside_weight=float(outside_weight.max()),
        maximum_outside_particle_fraction=float(outside.mean(1).max()),
        visible_real_initial_rollout=visible_rows, heldout=test_rows,
        normalized_metric_protocol='exact empirical W1/W2, three-bandwidth biased RBF MMD2; train-only bandwidth; CPU float64',
        support_scope='99th percentile training nearest-neighbor radius; descriptive only, not a biological boundary',
        mass_scope='initial and visible mass targets are capture proxies; heldout count is not scored',
        interpretation='Jointly learned fields checked after freezing, not independent evidence of biological velocity. Center-cohort rollout is a KDE particle approximation, not an exact Gaussian-mixture PDE solution.')
    write_json(dest, report)
    return report


def plot(data, output_root, run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths = folders(output_root, run)
    a, _ = load_visible(data)
    rpath = paths['results'] / 'rollouts.npz'
    report = read_json(paths['results'] / 'validation.json')
    if digest(rpath) != report['rollouts_sha256']:
        raise ValueError('Rollout cache hash mismatch')
    p = arrays(rpath)
    support = a['x'][a['train_rows']].astype(float)
    center = support.mean(0)
    _, _, vt = np.linalg.svd(support-center, full_matrices=False)
    projection = vt[:2].T
    if projection.shape[1] == 1:
        projection = np.c_[projection, np.zeros_like(projection)]
    def project(x):
        # Explicit contraction avoids platform BLAS floating-status warnings for
        # small batched matrices; coordinates and the PCA map are unchanged.
        out = np.einsum('...d,dk->...k', x-center, projection, optimize=False)
        if not np.isfinite(out).all():
            raise FloatingPointError('Nonfinite display projection')
        return out
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    xy = project(support)
    for ax, key, title, color in [(axes[0,0], 'native_x', 'Optimized trajectories (joint fit)', '#2664a8'),
                                  (axes[0,1], 'integrated_x', 'Frozen field rollout (same start)', '#d27622')]:
        ax.scatter(xy[:,0], xy[:,1], s=5, c='0.75', alpha=.4, rasterized=True)
        lines = project(p[key])
        for j in range(lines.shape[1]):
            ax.plot(lines[:,j,0], lines[:,j,1], color=color, alpha=.4, lw=.7)
        ax.set(title=title, xlabel='Train-fitted PC1', ylabel='Train-fitted PC2')
    all_xy = np.concatenate([project(p[k]).reshape(-1,2) for k in ('native_x','integrated_x')]+[xy])
    low, high = all_xy.min(0), all_xy.max(0); margin = np.maximum((high-low)*.05, .01)
    for ax in axes[0,:2]:
        ax.set_xlim(low[0]-margin[0], high[0]+margin[0]); ax.set_ylim(low[1]-margin[1], high[1]+margin[1])
    t = p['physical_time']
    axes[0,2].plot(t, p['coordinate_rmse'], color='#d27622', label='Native vs rollout')
    axes[0,2].plot(t, np.sqrt(np.mean((p['integrated_x']-p['coarse_x'])**2, axis=(1,2))), color='0.3', label='dt vs 2dt')
    axes[0,2].set(title='Full-dimensional coordinate RMSE', xlabel='Physical input time'); axes[0,2].legend()
    for key, label, color in [('native_logmass','Optimized','#2664a8'),('integrated_logmass','Same-start rollout','#d27622'),('real_initial_logmass','Real-cell start','#54854a')]:
        axes[1,0].plot(t, np.exp(logsumexp(p[key], axis=1)), label=label, color=color)
        axes[1,1].plot(t, 1/(softmax(p[key], axis=1)**2).sum(1)/p[key].shape[1], label=label, color=color)
    axes[1,0].scatter(a['time_offset']+a['time_scale']*a['observed_times'],a['mass_targets'], marker='x', c='black', label='Visible count proxy')
    axes[1,0].set(title='Total mass / initial count', xlabel='Physical input time'); axes[1,0].legend(fontsize=8)
    axes[1,1].set(title='Effective sample size / particle count', xlabel='Physical input time', ylim=(0,1.05))
    axes[1,2].plot(t, p['outside_weight'], label='Mass fraction')
    axes[1,2].plot(t, p['outside_particle_fraction'], label='Particle fraction')
    axes[1,2].set(title='Outside training support radius', xlabel='Physical input time', ylim=(0,1.05)); axes[1,2].legend()
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Stitching reconstruction audit — all optimized paths retained; projection for display only')
    base = paths['figures'] / 'reconstruction_audit'
    for suffix in ('.png', '.pdf', '.svg'):
        fig.savefig(base.with_suffix(suffix), dpi=180)
    plt.close(fig)
    write_json(base.with_suffix('.json'), dict(source_hashes=source_hashes(), rollouts_sha256=digest(rpath),
        plot_projection='PCA fitted only to visible training cells; no trajectory trimming',
        outputs={s:digest(base.with_suffix(s)) for s in ('.png','.pdf','.svg')}))
    base.with_suffix('.md').write_text('# Draft reconstruction audit\n\n'
        'Purpose: compare optimized paths and frozen self-consistent integration, mass, ESS and outliers.\n'
        f'Code: stitching_reconstruction/evaluate.py:plot\nInput: {rpath}\nCheckpoint SHA256: {report["checkpoint_sha256"]}\n'
        f'Reproduction: bash run.sh plot --data "{data}" --output-root "{output_root}" --run "{run}"\n'
        'All paths retained; shared PCA limits; statistics use the full feature space. This is not lineage or velocity ground truth.\n')
    return dict(figure=str(base.with_suffix('.png')))
