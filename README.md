# scTrajectory

**Single-cell trajectory reconstruction with population transport and local growth.**

scTrajectory jointly fits population trajectories, particle masses, a structured
velocity field and heterogeneous local growth from time-indexed cell snapshots.
It extends the trajectory/KDE residual approach of
[Stitching](https://github.com/BasisResearch/wasserstein-residuals).

**中文：单细胞轨迹重构与局部生长动力学。**
[中文方法与完整命令](docs/method_and_usage.zh-CN.md) ·
[Current results](docs/current_results.md) ·
[Verification record](VERIFIED_RUN.md)

## What it does

- Reconstructs weighted particle paths from fixed single-cell coordinates.
- Learns transport from a scalar potential, entropy and symmetric interactions.
- Learns a separate local growth field, with visible counts constraining mass.
- Exports optimized paths, path derivatives, KDE-induced fields and learned fields separately.
- Freezes the fitted model and integrates it again from initial conditions.
- Evaluates held-time distributions with exact empirical W1/W2 and RBF MMD²,
  together with mass, ESS, remote paths and step-size sensitivity.

This is a **research implementation**. Current pilots do not establish a general
accuracy advantage over classical OT or identify measured cell velocities.
Captured-cell counts are sampling proxies, and reconstructed paths are population
components rather than observed cell lineages. Local growth and auxiliary OT losses
are project extensions, not the original balanced Stitching experiment.

## Install

Tested with Python 3.9.25 and PyTorch 2.8.0 on macOS arm64. Apple MPS is the default
training/integration device; CPU evaluation statistics use float64.

```bash
git clone https://github.com/wanghongye200211/scTrajectory.git
cd scTrajectory
python -m pip install -e .
sctrajectory --help
```

Alternatively, install `requirements.txt` and use `bash run.sh`. Set `PYTHON` to
the desired interpreter if needed. Use `--device cpu` explicitly on systems without
MPS; there is no silent device fallback. CUDA has not been added to this release.

## Synthetic example

The example creates its own input and holds out the middle time point. Twenty
updates check the software workflow; they are not sufficient training for a
performance claim.

```bash
sctrajectory demo --output ./outputs/data/raw_demo.npz
sctrajectory prepare --input ./outputs/data/raw_demo.npz \
  --output ./outputs/data/demo --holdout 0.5
sctrajectory all --data ./outputs/data/demo --output-root ./outputs \
  --run demo_baseline --arm baseline --steps 20 --particles 16 --nodes 9 \
  --config configs/smoke.json --device mps
```

## Your data

Supply an NPZ containing `X[N,d]` and `time[N]`. Coordinates must already be
prepared; for strict held-time evaluation, fit the encoder or spatial transform
without held-time cells. Optional `count_times[T]` and `counts[T]` override captured
cell counts at visible times. Raw GEO download, transcriptome QC and encoder
training are outside this package.

```bash
sctrajectory prepare --input input.npz --output ./outputs/data/cells --holdout 36
sctrajectory train --data ./outputs/data/cells --output-root ./outputs \
  --run cells_baseline --arm baseline --steps 4000 --device mps
```

`baseline` uses the core free-local-growth model and soft trajectory safeguards.
`time_rollout` adds Sinkhorn, OT displacement supervision, a differentiated endpoint
rollout loss and richer time features. Both use consecutive visible-time OT
initialization; the extra OT teacher is not measured lineage information.

## Reconstruct, then validate the frozen field

```bash
CHECKPOINT=./outputs/models/cells_baseline/selected.pt
sctrajectory export --data ./outputs/data/cells --output-root ./outputs \
  --run cells_baseline --checkpoint "$CHECKPOINT" --device mps
sctrajectory validate --data ./outputs/data/cells --output-root ./outputs \
  --run cells_baseline --checkpoint "$CHECKPOINT" --device mps
sctrajectory plot --data ./outputs/data/cells --output-root ./outputs --run cells_baseline
```

The first stage jointly fits paths and fields. Validation uses the same frozen
field parameters but recomputes density from the evolving integration cohort,
without future fitted centers or path resets. It tests model consistency and
held-time interpolation, not independent biological velocity truth.

## Files and reproducibility

| Path | Purpose |
|---|---|
| `src/stitching_reconstruction/` | Portable implementation; import name retained for compatibility |
| `configs/` | Smoke-test settings and trajectory constraints |
| `tests/test_contracts.py` | Mathematical and pipeline contracts |
| `scripts/verify_delivery.py` | Optional replay against an existing research workspace |
| `SOURCE_MANIFEST.json` | Provenance of unchanged core physics modules |
| `docs/` | Equations, commands, results and project records |

Outputs are separated into `data`, `models`, `results`, `logs` and
`figures/_drafts` under the chosen output directory. Checkpoints, raw cell data,
machine-specific paths, caches and credentials are not bundled in the repository.

```bash
python -m unittest discover -s tests -v
```

Ten contracts cover analytic gradients, reaction continuity, held-time isolation,
rollout independence, analytic integration, checkpoint parity and time handling.
See [NOTICE.md](NOTICE.md) for attribution. An open-source license has not yet been
selected for the project; dependency licenses remain unchanged.
