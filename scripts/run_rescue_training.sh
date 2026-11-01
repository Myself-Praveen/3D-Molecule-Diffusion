#!/usr/bin/env bash
# Rescue plan R7 — training execution.
#
# Runs the next training arm defined by rescueplan.md: the V5 recipe with
# `min_snr_gamma: 0.0` (lose the eps/min-SNR coupling so the low-noise steps
# that set bond lengths carry full weight) and `checkpoint.select_by:
# generation` (best.pt ranked on ConnFrac - NN-outside-window).
#
# Budget: one arm is ~33 h CPU at 8 threads — the measured optimum on this
# host (the default nproc//2 was ~18x slower for this workload). Run it as a
# background job and watch `checkpoints/central_v7_bbb_nosnr/` for best.pt.
#
#   bash scripts/run_rescue_training.sh                          # full arm
#   bash scripts/run_rescue_training.sh --resume --run_epochs 400  # 1-h chunks
set -euo pipefail

OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" \
.venv/bin/python train.py --config configs/central_v7_bbb_nosnr.yaml "$@"
