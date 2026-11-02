# Evaluation plan

Goal: results that survive a skeptical reviewer. "Better than existing work" is claimed only along
axes we can measure: harder evaluation, the still-person case reported separately, drift handled
explicitly, and open data + code + firmware. Raw accuracy is compared only on shared benchmarks.

## Tasks
- zone: empty / zoneA / zoneB / ...
- presence: empty vs occupied (people standing still count as occupied)
- motion: moving vs static (static = empty + still). What variance-style motion detectors measure.

## Splits (csi_eval.py), all grouped so a session never appears on both sides
| split | question it answers |
|---|---|
| session | does it work on a session it has not seen, same days? |
| day | does it work on a day it has not seen? (headline number) |
| temporal | accuracy vs days since training (forward chaining) |
| room | does it transfer to another room? (needs a second room) |
| leaky | random window split. Reported only to show how much a naive split inflates results. |

## Models and baselines
threshold (moving variance, P95 of empty x 1.1), logistic regression, random forest, small MLP.
Scalers and any data-dependent step are fit on training data only.

## Feature groups
motion (temporal dynamics), fp (per-subcarrier fingerprint), both, level (turbulence level only), all.
Baseline-referenced fingerprints (each day divided by that day's empty baseline) are compared with raw ones.
Hypothesis to test, not assume: raw fingerprints degrade across days (as reported for ESP32 fingerprint
positioning), baseline referencing recovers part of it, motion features are stable but blind to a still person.

## Metrics
Balanced accuracy (pooled, with a 95% session-bootstrap CI), mean and std across folds, macro F1,
per-class precision and recall, confusion matrices, false-positive rate on empty windows, accuracy
split by moving vs still, accuracy vs time gap. Report every split, including the bad ones.

## Experiments, in order
1. Pilot (docs/firmware_config.md): decide HT20 vs HT40 and gain handling from data, then freeze.
2. Leakage: leaky vs session vs day, same features. Expect leaky >> day.
3. Feature ablation: motion / fp / both / all, raw vs baseline-referenced.
4. Temporal decay: train on early days, test on later ones.
5. Moving vs still breakdown.
6. Cross-room, if a second room exists.
7. Public benchmark: the open ESP32 dataset from the temporal-generalization study (Zenodo, CC BY-NC).
   NOT IMPLEMENTED: the dataset's file format has not been inspected yet, so no loader exists.

## Reporting checklist
Firmware variant and commit hash, packet rate and loss per session, feature config, random seeds,
all splits including negative results, dataset and code released. Synthetic data (host/tools/synth_csi.py)
is for testing the pipeline only and must never appear in a results table.
