"""Delivery QA: short synthetic smoke fits, then frozen GSE75748 replay.

This never resumes the paused research training queue. Numerical artifacts use
the project's existing data/models/results/logs/figures locations.
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from stitching_reconstruction import __main__  # Environment before numeric imports.
from stitching_reconstruction.data import demo, prepare, import_axis
from stitching_reconstruction.train import fit
from stitching_reconstruction.evaluate import import_checkpoint, export, validate, plot
from stitching_reconstruction.io import digest, read_json, write_json, fresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', required=True)
    ap.add_argument('--tag', default='20261008')
    args = ap.parse_args()
    project = Path(args.project_root).resolve()
    code = Path(__file__).resolve().parents[1]
    audit_path = project/'results'/f'reconstruction_release_{args.tag}'/'audit.json'
    fresh(audit_path)
    paused = project/'results/shape_losses_20261006/pause.json'
    pause_before = digest(paused)
    root_data = project/'data/processed'/f'reconstruction_release_{args.tag}'
    demo(root_data/'raw_demo.npz')
    prepare(root_data/'raw_demo.npz', root_data/'synthetic', [.5])
    smoke = {}
    for arm in ('baseline','time_rollout'):
        run = f'reconstruction_smoke_{arm}_{args.tag}'
        trained = fit(root_data/'synthetic', project, run, arm, steps=20, particles=16,
                      nodes=9, seed=20261008, devname='mps', config_path=code/'configs/smoke.json')
        cp = trained['selected_checkpoint']
        ex = export(cp, root_data/'synthetic', project, run)
        val = validate(cp, root_data/'synthetic', project, run, max_cells=64, dtmax=.025)
        fig = plot(root_data/'synthetic', project, run)
        smoke[arm] = dict(training=trained, export=ex, validation=val, **fig,
                          scope='20-update end-to-end software test; not a model quality benchmark')
    dataset = 'gse75748_ae10_heldout_h36'
    old_run = 'shape_time_rollout_4000_s20261006_shape_v1'
    old_results = project/'results'/dataset/old_run
    done = read_json(old_results/'complete.json')
    original = Path(done['selected_checkpoint'])
    if digest(original) != done['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint hash changed')
    original_before = digest(original)
    data = root_data/'gse75748'
    import_axis(project/'data/processed'/dataset, data)
    run = f'reconstruction_gse75748_frozen_{args.tag}'
    imp = import_checkpoint(original, old_results/'protocol.json', data, project, run)
    cp = project/'models'/run/'selected.pt'
    ex = export(cp, data, project, run)
    val = validate(cp, data, project, run)
    fig = plot(data, project, run)
    # Same historical heldout cohorts, kernel bandwidth and projection directions.
    historical = read_json(old_results/'shape_evaluation.json')
    max_metric_difference = max(abs(val['heldout'][0]['metrics'][k]-historical['metrics'][k]) for k in ('w2','w1','mmd2','swd'))
    if max_metric_difference > 2e-5:
        raise AssertionError(f'Frozen heldout replay mismatch: {max_metric_difference}')
    assert digest(original) == original_before
    assert digest(paused) == pause_before
    report = dict(status='NUMERICAL_QA_PASS', synthetic=smoke,
                  frozen_gse75748=dict(import_report=imp, export=ex, validation=val, **fig,
                                      historical_metric_max_abs_difference=max_metric_difference),
                  paused_research_campaign_unchanged=True, paused_file_sha256=pause_before,
                  original_checkpoint_unchanged=True, original_checkpoint_sha256=original_before,
                  scope='Synthetic-only smoke training plus existing-checkpoint inference. No new real-data fitting or quality-improvement claim.')
    write_json(audit_path, report)
    print(f'Delivery numerical QA saved: {audit_path}', flush=True)


if __name__ == '__main__':
    main()
