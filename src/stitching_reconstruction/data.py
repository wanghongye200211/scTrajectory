"""Visible-only training archive and physically separate evaluation archive.

Coordinates are supplied by the user; this package does not fit an encoder.
Use a representation fitted without held-time cells for an inductive claim.
"""
from pathlib import Path
import numpy as np
from .io import arrays, digest, fresh, read_json, write_json


def check_visible(a):
    x, t = a['x'], a['time']
    if x.ndim != 2 or x.shape[1] < 1 or t.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError('x must be finite N by d; time must be finite length N')
    tr, va = a['train_rows'], a['validation_rows']
    for rows in (tr, va):
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or len(set(rows)) != len(rows):
            raise ValueError('Split rows must be unique integer indices')
        if not len(rows) or rows.min() < 0 or rows.max() >= len(x):
            raise ValueError('Invalid split indices')
    if np.intersect1d(tr, va).size or len(tr) + len(va) != len(x):
        raise ValueError('Train and validation must partition the visible cells')
    obs = np.unique(t[tr])
    if len(obs) < 2 or not np.array_equal(obs, a['observed_times']) or not np.array_equal(obs, np.unique(t[va])):
        raise ValueError('Need at least two visible times, each with train and validation cells')
    if np.any(np.diff(obs.astype(np.float32)) <= 0):
        raise ValueError('Visible times collapse in float32; rescale time before training')
    if np.any(a['mass_targets'] <= 0) or len(a['mass_targets']) != len(obs):
        raise ValueError('Positive mass targets required at visible times')
    if not np.isfinite(a['mass_targets']).all() or float(a['time_scale']) <= 0 or not np.isfinite([a['time_scale'],a['time_offset'],a['reference_count']]).all() or float(a['reference_count']) <= 0:
        raise ValueError('Invalid mass targets or physical time scale')
    for q in obs:
        if (t[tr] == q).sum() < 2:
            raise ValueError('Need at least two training cells per visible time')
    return a


def load_visible(directory):
    directory = Path(directory)
    meta = read_json(directory / 'metadata.json')
    path = directory / 'visible.npz'
    if digest(path) != meta['visible_sha256']:
        raise ValueError('Visible data hash mismatch')
    # Deliberately never opens test.npz or reads its arrays/counts.
    return check_visible(arrays(path)), meta


def save_dataset(out, visible, test, metadata):
    out = Path(out)
    fresh(out / 'metadata.json')
    fresh(out / 'visible.npz')
    fresh(out / 'test.npz')
    check_visible(visible)
    if test is not None and np.intersect1d(visible['time'], test['time']).size:
        raise ValueError('Held-time cells overlap visible times')
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'visible.npz', **visible)
    if test is not None:
        np.savez_compressed(out / 'test.npz', **test)
    write_json(out / 'metadata.json', dict(metadata, visible_sha256=digest(out / 'visible.npz'),
               test_sha256=digest(out / 'test.npz') if test is not None else None))


def prepare(raw, out, held_times, seed=20261008, validation_fraction=.2):
    a = arrays(raw)
    x, physical = np.asarray(a['X'], np.float32), np.asarray(a['time'], float)
    if x.ndim != 2 or physical.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(physical).all():
        raise ValueError('Input X/time have invalid shapes or nonfinite values')
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    all_times = np.unique(physical)
    if not np.all(np.isin(held_times, all_times)):
        raise ValueError('Each requested held time must occur in input')
    held = np.isin(physical, held_times)
    obs = np.unique(physical[~held])
    if len(obs) < 2:
        raise ValueError('Need at least two visible times')
    offset, scale = float(obs[0]), float(obs[-1] - obs[0])
    t = (physical[~held] - offset) / scale
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for q in np.unique(t):
        rows = rng.permutation(np.flatnonzero(t == q))
        if len(rows) < 3:
            raise ValueError('Need at least three cells at each visible time')
        nval = max(1, min(len(rows) - 2, round(len(rows) * validation_fraction)))
        va.extend(rows[:nval]); tr.extend(rows[nval:])
    if ('count_times' in a) != ('counts' in a):
        raise ValueError('count_times and counts must be supplied together')
    if 'counts' in a:
        counts = []
        for q in obs:
            ids = np.flatnonzero(a['count_times'] == q)
            if len(ids) != 1:
                raise ValueError('Supply exactly one positive count for each visible physical time')
            counts.append(float(a['counts'][ids[0]]))
        counts = np.array(counts)
        count_scope = 'user-supplied count proxies; interpret according to sampling design'
    else:
        counts = np.array([(physical == q).sum() for q in obs], float)
        count_scope = 'captured cells before within-time splitting; not population census'
    visible = dict(x=x[~held], time=t, train_rows=np.array(tr), validation_rows=np.array(va),
                   observed_times=np.unique(t), mass_targets=counts / counts[0],
                   reference_count=np.array(counts[0]), time_offset=np.array(offset), time_scale=np.array(scale))
    test = dict(x=x[held], time=(physical[held] - offset) / scale) if held.any() else None
    if test is not None and 'velocity_true' in a:
        test['velocity_true_physical'] = a['velocity_true'][held]
    save_dataset(out, visible, test, dict(source_sha256=digest(raw), preparation='visible-only cell split; no fitted spatial transform',
                 seed=seed, count_scope=count_scope, input_coordinates='unchanged supplied feature coordinates'))


def import_axis(axis_dir, out):
    """Import existing sealed project inputs without changing coordinates or splits."""
    src = Path(axis_dir)
    manifest = read_json(src / 'shared_axis_manifest.json')
    if digest(src / 'shared_axis.npz') != manifest['sha256']:
        raise ValueError('Source axis hash mismatch')
    a = arrays(src / 'shared_axis.npz')
    original, t = a['original_time'].astype(float), a['model_time'].astype(float)
    i, j = int(t.argmin()), int(t.argmax())
    scale = (original[j] - original[i]) / (t[j] - t[i])
    offset = original[i] - scale * t[i]
    np.testing.assert_allclose(original, offset + scale * t, rtol=1e-7, atol=1e-7)
    obs = np.unique(t[a['full_train_rows']])
    counts = np.array([manifest['observed_counts'][str(float(q))] for q in obs])
    visible = dict(x=a['z'], time=t, train_rows=a['full_train_rows'], validation_rows=a['full_validation_rows'],
                   observed_times=obs, mass_targets=counts / counts[0], reference_count=np.array(counts[0]),
                   time_offset=np.array(offset), time_scale=np.array(scale))
    test = None
    hp = src / 'heldout_evaluation_only.npz'
    if hp.exists():
        ep = read_json(src / 'evaluation_protocol.json')
        if digest(hp) != ep['holdout_sha256']:
            raise ValueError('Source holdout hash mismatch')
        h = arrays(hp)
        test = dict(x=h['z'], time=h['model_time'])
    save_dataset(out, visible, test, dict(source_axis_sha256=manifest['sha256'], source_manifest=manifest,
                 preparation='identity copy of sealed feature coordinates, time mapping and visible splits',
                 input_coordinates='unchanged sealed coordinates; no inverse encoder available',
                 count_scope='captured-cell proxies, not measured population size'))


def demo(out):
    """Translation with known velocity, used only for software smoke testing."""
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True); fresh(out)
    rng = np.random.default_rng(37)
    times = np.repeat(np.array([0., .25, .5, .75, 1.]), 64)
    v = np.array([.8, -.3], np.float32)
    x = rng.normal(0, .3, (len(times), 2)) + times[:, None] * v
    np.savez_compressed(out, X=x.astype(np.float32), time=times,
                        velocity_true=np.broadcast_to(v, x.shape).copy())
