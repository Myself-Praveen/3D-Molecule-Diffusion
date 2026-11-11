# result.md — latest run (V8 BBB probe, rescue chunk 2)

Metrics below are read directly from the live run's log and checkpoints on
2026-11-11 ~13:40. This run is still training, so the epoch table grows and the
generation-probe rows are still empty by design (first probe fires at epoch 4000).

## 1. Run identity

| field | value |
|---|---|
| arm | V8 BBB probe — V7 no-min-SNR recipe, re-run under corrected generation selection |
| config (committed) | `configs/central_v8_bbb_probe.yaml` |
| config (live) | `configs/central_v8_bbb_probe_runtime.yaml` (epochs/patience extended 3900 → 4300) |
| checkpoints | `checkpoints/central_v8_bbb_probe/` |
| log | `outputs/train_v8_probe_cont.log` |
| launch | `setsid .venv/bin/python -u train.py --config ...runtime.yaml --resume --run_epochs 400` |
| PID / session | 3778711, own session (survives shell cleanup) |
| device / threads | CPU, 8 torch threads |
| resumed from | `last.pt` @ epoch 3917 |
| window | epochs 3918 → 4300 (383 epochs) |

## 2. Status

| field | value |
|---|---|
| state | running |
| elapsed this launch | 04:47 |
| CPU | ~791 % |
| epochs completed | 10 (3918 → 3927) |
| throughput | ~28 s/epoch |
| next snapshot | epoch 4000 (`save_every: 200`), ~34 min away |
| next gen probe | epoch 4000 (`eval_every: 100`), same point |
| ETA to 4300 | ~2.9 h |

## 3. Per-epoch metrics (this launch)

`train_pos` / `train_type` are denoising losses; `baseline` is the trivial
all-zero predictor; `val` is the held-out composite.

| epoch | train_pos | train_type | baseline | val_pos | val_type | val | lr |
|---|---|---|---|---|---|---|---|
| 3918 | 0.4884 | 0.5581 | 1.0000 | 0.6005 | 0.5498 | 0.8754 | 1.06e-08 |
| 3919 | 0.4783 | 0.5696 | 1.0000 | 0.5637 | 0.6486 | 0.8880 | 1.18e-08 |
| 3920 | 0.4814 | 0.5522 | 1.0000 | 0.6158 | 0.6204 | 0.9260 | 1.31e-08 |
| 3921 | 0.4844 | 0.5525 | 1.0000 | 0.5902 | 0.6159 | 0.8982 | 1.45e-08 |
| 3922 | 0.4838 | 0.5617 | 1.0000 | 0.5993 | 0.5854 | 0.8920 | 1.59e-08 |
| 3923 | 0.4703 | 0.5629 | 1.0000 | 0.5872 | 0.6995 | 0.9370 | 1.73e-08 |
| 3924 | 0.4816 | 0.5459 | 1.0000 | 0.6044 | 0.6439 | 0.9263 | 1.89e-08 |
| 3925 | 0.4977 | 0.5621 | 1.0000 | 0.6448 | 0.5845 | 0.9371 | 2.05e-08 |
| 3926 | 0.4787 | 0.5757 | 1.0000 | 0.6037 | 0.6062 | 0.9068 | 2.22e-08 |
| 3927 | 0.4692 | 0.5746 | 1.0000 | 0.5678 | 0.6908 | 0.9132 | 2.39e-08 |

Summary over these 10 epochs:

| statistic | value |
|---|---|
| mean train_pos | 0.4814 |
| mean train_type | 0.5615 |
| mean val | 0.9100 |
| best val in window | 0.8754 (epoch 3918) |
| worst val in window | 0.9371 (epoch 3925) |
| val spread | ±0.031 around the mean |
| lr range | 1.06e-08 → 2.39e-08 (cosine tail) |

Interpretation: the loss is flat and oscillating inside the ±3 % noise band of
the epoch-to-epoch validation estimate. The LR is at the tail of the original
3900-epoch cosine schedule (~1e-8), so the AdamW weight-update magnitude is
effectively nil. These 400 epochs are not a learning phase — they exist to give
the generation probe more checkpoints to draw from, not to improve val loss.

## 4. Checkpoint metrics

### `best.pt` — the current winner (epoch 3100, unchanged this launch)

| metric | value |
|---|---|
| stored epoch | 3100 |
| gen validity | 9.375 % |
| n_valid / n_samples | 6 / 64 |
| connected fraction | 33.33 % |
| nn_outside | 9.29 % |
| generation score | **−6.1693** |
| val_loss | 0.8473 |
| best_val_loss | 0.8473 |
| epochs_no_improve (at save) | 1656 |
| probe_settings | `num_samples 64, ddim_steps 50, size_source training_histogram` |
| probe score formula | `(validity/100)*conn_frac - nn_outside`, `-inf` when `n_valid < 4` |

### `last.pt` — live EMA state (epoch 3927, overwritten every epoch)

| metric | value |
|---|---|
| epoch | 3927 |
| val_loss | 0.9132 |
| best_val_loss | 0.8473 |
| best_gen_score | `None` (EMA snapshot never stores the probe) |
| epochs_no_improve | 2483 |
| gen_metrics | `None` |
| mtime | 2026-11-11 13:37 |

## 5. Molecule-level metrics (validity / ConnValidity / BBB% / QED)

The in-run probe reports only validity / ConnFrac / nn_outside and its first
firing in this chunk is epoch 4000, so molecule quality was measured with an
explicit out-of-schedule evaluation of the live weights.

**Evaluated checkpoint:** snapshot of `last.pt` taken while this chunk was at
**epoch 3940** (`val_loss 0.9071`).

### Protocol A — selection-probe protocol

200 molecules, 50 DDIM steps, `eta 0.0`, linear schedule, sizes from the
training histogram, relax on, seed 42.
Artifacts: `outputs/eval_v8_live/` (`metrics.json`, `smiles.txt`, `molecules.sdf`).

| metric | value | 95 % CI | n |
|---|---|---|---|
| **Validity** | **8.00 %** | [4.98, 12.60] | 16 / 200 |
| **ConnectedValidity** | **0.50 %** | [0.09, 2.78] | 1 / 200 |
| **BBB%** (of valid) | **12.50 %** | [3.50, 36.02] | 2 / 16 |
| **QED** (of valid) | **0.5856** | printed as n/a (n<50) | 16 |

Supporting fields from the same run:

| metric | value |
|---|---|
| ConnectedFrac | 6.25 % (1 of 16 valid is a single fragment) |
| BBB yield (of all 200) | 1.00 % |
| Uniqueness / Novelty | 100.0 % / 100.0 % |
| LogP | 0.853 |
| IntDiv_p | 0.874 |
| SNN | 0.164 |
| ScaffDiv / ScaffCov | 0.750 / 0.0016 |
| Lipinski% / Veber% | 93.75 / 87.5 |
| CNS_MPO | 3.22 |
| BondsPerMol | 21.25 |
| BondsPerAtom (rel. to real) | 0.835 (0.805 × reference 1.038) |
| MeanAtoms | 25.44 |
| FragmentsPerMol | 6.31 |
| NN_Outside% | 18.35 |
| Rg | 4.79 Å |
| `reportable` | `false` — n_valid 16 < `MIN_REPORTABLE_N` 50 |

**Caveat:** 16 valid molecules do not support a headline claim; the harness
itself sets `reportable: false`. Validity 8.0 % and ConnectedValidity 0.5 % sit
in the same regime as the epoch-3100 probe (9.375 % valid, 33.3 % of those
connected), which is what the flat val loss predicted — the chunk has not
changed molecule quality.

### Protocol B — project standard eval

200 molecules, 200 DDIM steps, `eta 1.0`, quadratic schedule, relax on, same
snapshot — the protocol used for the other arms' headline rows (v5/v6).
Artifacts: `outputs/eval_v8_live_std/`.

200 molecules, 200 DDIM steps, `eta 1.0`, quadratic schedule, same snapshot
(epoch 3940) — the protocol used for the other arms' headline rows (v5/v6).
Artifacts: `outputs/eval_v8_live_std/`.

| metric | value | 95 % CI | n |
|---|---|---|---|
| **Validity** | **14.50 %** | [10.29, 20.05] | 29 / 200 |
| **ConnectedValidity** | **6.00 %** | [3.47, 10.19] | 12 / 200 |
| **BBB%** (of valid) | **37.93 %** | [22.69, 56.00] | 11 / 29 |
| **QED** (of valid) | **0.5755** | printed as n/a (n<50) | 29 |

Supporting fields: ConnectedFrac 41.38 % (12 of 29 valid), BBB yield 5.50 %
of all 200, LogP 1.573, IntDiv_p 0.880, Uniqueness 100 %, Novelty 100 %,
BondsPerAtom 1.0028 (97 % of real), MeanAtoms 37.55, FragmentsPerMol 2.72,
NN_Outside% 1.42, Lipinski% 100, Veber% 75.86, `reportable: false`
(29 < 50).

### Protocol comparison

| | A (50 DDIM, eta 0) | B (200 DDIM, eta 1.0) |
|---|---|---|
| Validity | 8.00 % | 14.50 % |
| ConnectedValidity | 0.50 % | 6.00 % |
| ConnectedFrac | 6.25 % | 41.38 % |
| BBB% | 12.50 % | 37.93 % |
| QED | 0.586 | 0.575 |
| MeanAtoms | 25.4 | 37.6 |
| n_valid | 16 | 29 |

Protocol B is the number that should be compared with the other arms; the
8 % figure in protocol A is a consequence of 50-step, deterministic sampling,
not of the checkpoint. Even so, neither reaches the 50 % validity this project
reported for the V6 arm, and both are labelled non-reportable by the harness.

## 6. How to refresh this file

```
# live per-epoch tail
grep "^Epoch " outputs/train_v8_probe_cont.log | tail -n 20

# probe / save events (empty until epoch 4000)
grep -inE "probe|quick_valid|Saved best" outputs/train_v8_probe_cont.log

# checkpoint fields
.venv/bin/python -c "
import torch
for p in ['checkpoints/central_v8_bbb_probe/best.pt','checkpoints/central_v8_bbb_probe/last.pt']:
    c=torch.load(p,map_location='cpu',weights_only=False)
    print(p, {k:c.get(k) for k in ['epoch','val_loss','best_gen_score','gen_metrics','epochs_no_improve']})
"
```
