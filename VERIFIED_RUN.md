# Verification record — 2026-10-08

The initial scTrajectory code derives from the independently packaged unbalanced
Stitching reconstruction workflow. Five core physics/metric files retain their
recorded SHA-256 hashes in `SOURCE_MANIFEST.json`.

- Ten mathematical/pipeline tests passed.
- `baseline` and `time_rollout` each completed a 20-update MPS synthetic smoke fit,
  followed by export, frozen integration, held-time checks and diagnostic plots.
- An existing GSE75748 selected `time_rollout` checkpoint was imported with
  exactly unchanged tensors and zero CPU field discrepancy.
- Matched 36 h holdout W2: **2.2399613556513716**;
  MMD²: **0.39587231551216706**.
- Replayed W1/W2/MMD²/SWD matched the historical report exactly.
- Same optimized initial cohort, final coordinate RMSE: **0.3269962966442108**.
- Maximum model-time step 0.0125 versus 0.025, final coordinate difference:
  **0.0005282546626403928**.
- Three diagnostic PNGs were visually inspected; PDF/SVG counterparts were exported.
- No real-data model was retrained during packaging or publication.

These are implementation and consistency checks, not a claim of improved prediction
or experimentally measured velocity. Only the synthetic input includes independent
velocity truth. See [current results](docs/current_results.md) for negative results
and the comparison boundary.

Original frozen GSE75748 checkpoint SHA-256:
`1564f220768da8146cdaa6813509f4eb963b4eebea0e2a4712ab831b393fff14`.

The original data, checkpoints and local audit records are maintained separately.
The repository provides synthetic reproduction and sealed-axis/checkpoint import
commands; it does not redistribute the cell-level data or pretrained weights.
