#!/usr/bin/env bash
# Run the V6 BBB geometry arm (radius normalization). One shot; not resumable.
#
# Ties out to the plan: V5 recipe (config inherited from V5 verbatim) plus
# Intervention 3, the coordinate system (radius_norm: atom_count, R(N) =
# 1.06 * N^(1/3)). V5 is already trained to 3900 epochs; this run launches
# V6 from scratch on the same budget so the coordinate-system delta is
# isolated.
#
# Expectation note (honest, from progress.md): radius normalization targets the
# fragmentation/over-bonded-soup failure (the one that survived budget, lambda2
# renorm, geofix, steric, guidance sweeps). It is the plan's "coordinate system"
# lever and is shipped-but-untrained in the repo today.
#
# Thread cap: the session's measured optimum on this host is 8 compute threads;
# the default (nproc//2 = 18) was measured ~18x slower. Keep OMP + MKL at 8.
#
set -euo pipefail

OMP_NUM_THREADS=8 \
MKL_NUM_THREADS=8 \
.venv/bin/python train.py \
  --config configs/central_v6_bbb_radius.yaml
