"""Portable paths, provenance and explicit device selection."""
from pathlib import Path
import hashlib
import json
import os
import platform
import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def device(name):
    if name == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise RuntimeError('Disable PYTORCH_ENABLE_MPS_FALLBACK for an auditable MPS run')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable; no silent CPU fallback. Use --device cpu explicitly for a smoke test.')
    torch.set_num_threads(2)
    return torch.device(name)


def synchronize(dev):
    if str(dev) == 'mps':
        torch.mps.synchronize()


def tensor(x, dev):
    return torch.as_tensor(x, dtype=torch.float32, device=dev)


def source_hashes():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*.py'))}


def environment():
    return dict(python=platform.python_version(), platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__, mps_fallback=False)


def folders(root, run):
    if not run or Path(run).name != run or run in ('.', '..'):
        raise ValueError('run must be a single directory name')
    root = Path(root).expanduser().resolve()
    paths = {key: root / key / run for key in ('models', 'results', 'logs')}
    paths['figures'] = root / 'figures' / '_drafts' / run
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def fresh(path):
    if Path(path).exists():
        raise FileExistsError(f'Immutable output already exists: {path}; choose a new run/output path.')


def load_model(path, dev):
    from .physics.shape_losses import ShapeStitching
    # These local project checkpoints include NumPy initialization arrays.
    cp = torch.load(path, map_location='cpu', weights_only=False)
    i, cfg = cp['initial'], cp['config']
    model = ShapeStitching(i['x'], i['lm'], i['grid'], i['bw'],
                          i['obs'], i['targets'], cfg, i['curve_kind'])
    model.load_state_dict(cp['state_dict'], strict=True)
    return model.to(dev).eval(), cp
