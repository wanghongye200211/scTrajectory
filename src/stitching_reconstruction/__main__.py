"""CLI; configure CPU libraries before importing NumPy/Torch."""
import os
from pathlib import Path
import tempfile
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
for key, sub in [('MPLCONFIGDIR', 'mpl'), ('KEOPS_CACHE_FOLDER', 'keops'), ('XDG_CACHE_HOME', 'cache')]:
    os.environ.setdefault(key, str(Path(tempfile.gettempdir()) / 'stitching_reconstruction' / sub))
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
import argparse
import json


def main():
    p = argparse.ArgumentParser(description='Stitching: joint reconstruction, then frozen-field validation')
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('demo', help='Generate small synthetic data with known velocity; does not train')
    d.add_argument('--output', required=True)
    d = sub.add_parser('prepare', help='Split generic fixed-coordinate NPZ into visible and held-time archives')
    d.add_argument('--input', required=True); d.add_argument('--output', required=True)
    d.add_argument('--holdout', nargs='*', type=float, default=[])
    d.add_argument('--seed', type=int, default=20261008)
    d.add_argument('--validation-fraction', type=float, default=.2)
    d = sub.add_parser('import-axis', help='Import an existing sealed project axis')
    d.add_argument('--axis-dir', required=True); d.add_argument('--output', required=True)
    d = sub.add_parser('import-checkpoint', help='Bind a legacy frozen checkpoint to imported inputs')
    d.add_argument('--checkpoint', required=True); d.add_argument('--protocol', required=True)
    for name in ('train', 'all'):
        d = sub.add_parser(name, help='Fit only' if name == 'train' else 'Fit, export, validate and plot')
        d.add_argument('--arm', choices=('baseline', 'time_rollout'), default='baseline')
        d.add_argument('--steps', type=int, default=4000)
        d.add_argument('--particles', type=int, default=96)
        d.add_argument('--nodes', type=int, default=33)
        d.add_argument('--seed', type=int, default=20261008)
        d.add_argument('--config', help='Optional JSON loss-weight overrides')
    for name in ('export', 'validate'):
        d = sub.add_parser(name)
        d.add_argument('--checkpoint', required=True)
    sub.add_parser('plot', help='Plot existing validated caches; no model fitting')
    for name in ('import-checkpoint', 'train', 'all', 'export', 'validate', 'plot'):
        d = sub.choices[name]
        d.add_argument('--data', required=True)
        d.add_argument('--output-root', required=True, help='Root containing models/results/logs/figures')
        d.add_argument('--run', required=True)
    for name in ('train', 'all', 'export', 'validate'):
        sub.choices[name].add_argument('--device', choices=('mps', 'cpu'), default='mps')
    for name in ('all', 'validate'):
        sub.choices[name].add_argument('--dtmax', type=float, default=.0125, help='Maximum model-time step; 2*dt used for refinement')
        sub.choices[name].add_argument('--max-cells', type=int, default=512)
    a = p.parse_args()
    from . import data
    result = None
    if a.command == 'demo':
        data.demo(a.output); result = dict(raw_data=a.output)
    elif a.command == 'prepare':
        data.prepare(a.input, a.output, a.holdout, a.seed, a.validation_fraction); result = dict(data=a.output)
    elif a.command == 'import-axis':
        data.import_axis(a.axis_dir, a.output); result = dict(data=a.output)
    elif a.command in ('train', 'all'):
        from .train import fit
        result = fit(a.data, a.output_root, a.run, a.arm, a.steps, a.particles, a.nodes, a.seed, a.device, a.config)
        if a.command == 'all':
            from .evaluate import export, validate, plot
            checkpoint = result['selected_checkpoint']
            export(checkpoint, a.data, a.output_root, a.run, a.device)
            result = validate(checkpoint, a.data, a.output_root, a.run, a.device, a.dtmax, a.max_cells)
            result.update(plot(a.data, a.output_root, a.run))
    else:
        from . import evaluate
        if a.command == 'import-checkpoint':
            result = evaluate.import_checkpoint(a.checkpoint, a.protocol, a.data, a.output_root, a.run)
        elif a.command == 'export':
            result = evaluate.export(a.checkpoint, a.data, a.output_root, a.run, a.device)
        elif a.command == 'validate':
            result = evaluate.validate(a.checkpoint, a.data, a.output_root, a.run, a.device, a.dtmax, a.max_cells)
        else:
            result = evaluate.plot(a.data, a.output_root, a.run)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
