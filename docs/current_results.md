# Current experimental evidence

Status as of **2026-10-08**: preliminary, one pilot model seed per dataset.
The enhanced recipe was selected using visible-time validation before these
held-time results were evaluated. These datasets were previously used for
development; this is not a prospective benchmark. Additional seed confirmation
is incomplete. No new real-data training was performed for this repository upload.

All rows below use the existing sealed AE10 representation. Lower is better.

| Whole-time holdout | Baseline W2 | Enhanced W2 | Baseline MMD² | Enhanced MMD² |
|---|---:|---:|---:|---:|
| GSE114412 D4 | 1.243098 | 1.182430 | 0.001338 | 0.001054 |
| GSE132188 E14.5 | 2.848773 | 2.614035 | 0.096693 | 0.076443 |
| GSE75748 36 h | 2.209882 | 2.239961 | 0.389556 | 0.395872 |

The enhanced configuration is `time_rollout`. It improves these two shape metrics
on D4 and E14.5 relative to this campaign's baseline, but not on GSE75748.
Its W2 remains higher than the matched classical OT references on all three datasets.
This statement concerns W2; individual MMD² comparisons need not have the same ranking.

For GSE75748, the matched exact-OT reference has W2 **2.179457** and MMD² **0.377923**.
The enhanced prediction at 36 h remains behind the observed population mean;
adding local growth and auxiliary losses has not resolved this timing error.

## Trajectory consistency

The following coordinate RMS compares independent integration from a model's
optimized initial centers/masses with that same model's optimized final centers.
It is not ground-truth lineage or velocity error.

| Dataset | Baseline endpoint RMS | Enhanced endpoint RMS |
|---|---:|---:|
| GSE114412 | 0.6583 | 0.5123 |
| GSE132188 | 0.3719 | 0.3271 |
| GSE75748 | 0.3506 | 0.3270 |

The portable GSE75748 replay reproduces enhanced RMS **0.326996**, whereas changing
the maximum integration step from 0.0125 to 0.025 changes the endpoint by **0.000528**.
This separates the observed path mismatch from the tested step-size sensitivity;
the mismatch may involve both fitted residuals and the KDE/particle approximation.

Global counts constrain average mass, not uniquely identifiable biological birth
or death. Normalized W1/W2/MMD² do not score total mass. A large distance from
visible training snapshots between sampled times is a descriptive signal and may
also reflect a valid transition, not necessarily an erroneous trajectory.

## Source records

These tables summarize the preserved research records
`shape_losses_20261006/pilot_heldout_evaluation.json`,
`trajectory_review_20261006/summary.json` and
`reconstruction_gse75748_frozen_20261008/validation.json`.
The corresponding source hashes and selected checkpoint are retained by the project.
Cell-level input and checkpoint files are not bundled in this initial code repository.
