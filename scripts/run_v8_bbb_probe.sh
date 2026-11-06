#!/usr/bin/env bash
# V8 BBB — fix.md F11 training execution.
#
# The V7 no-min-SNR recipe (bit-identical) re-run under the corrected probe:
# `checkpoint.select_by: generation` now ranks on expected yield,
#
#     (validity/100) * conn_frac - nn_outside      [-inf if n_valid < 4]
#
# over 64 molecules at 50 DDIM steps drawn at real BBBP sizes. That is the
# whole difference from V7, whose best.pt was selected on 16 molecules at 20
# steps by a score that ignored validity (epoch 3400: probe validity 6.2 %,
# measured validity 33.0 %, connectivity 0.0 %).
#
# Budget: ~33 h CPU at 8 threads — the measured optimum on this host (the
# default nproc//2 was ~18x slower for this workload). Run it as a background
# job and watch `checkpoints/central_v8_bbb_probe/` for best.pt.
#
#   bash scripts/run_v8_bbb_probe.sh                            # full arm
#   bash scripts/run_v8_bbb_probe.sh --resume --run_epochs 400  # 1-h chunks
set -euo pipefail

OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}" \
.venv/bin/python train.py --config configs/central_v8_bbb_probe.yaml "$@"
