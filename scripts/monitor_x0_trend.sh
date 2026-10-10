#!/usr/bin/env bash
# Poll the probe-scale BBB checkpoint dir and append the low-t x0 recovery
# trend. Each 100th-epoch checkpoint is probed exactly once.
#
#   setsid nohup bash scripts/monitor_x0_trend.sh > logs/chain_x0_trend.log 2>&1 &
#
# The trend answers REMAINING TASK 1: if the low-t x0 MSE (t=20/50/100) drops
# below the trivial eps=0 baseline as the training budget grows, surface
# connectivity is bound by steps, not by the objective (→ train the full
# budget). If it plateaus above the baseline, soup != pure undertraining and
# the objective is the suspect (→ REMAINING TASK 3b).
set -u
cd "$(dirname "$0")/.."

CKPT_DIR=${CKPT_DIR:-checkpoints/bbb_longprobe}
TREND=${TREND:-logs/bbb_longprobe_probe_trend.log}
EVERY=${EVERY:-100}      # probe every Nth epoch checkpoint
SLEEP=${SLEEP:-300}      # poll interval (s)
MAX_POLLS=${MAX_POLLS:-600}

mkdir -p "$(dirname "$TREND")"
for _ in $(seq 1 "$MAX_POLLS"); do
  for f in "$CKPT_DIR"/epoch_*.pt; do
    [ -e "$f" ] || continue
    ep=$(basename "$f" .pt); ep=${ep#epoch_}
    [ $((ep % EVERY)) -eq 0 ] || continue
    grep -q "epoch $ep " "$TREND" 2>/dev/null && continue
    echo "=== epoch $ep  ($(date +%H:%M)) ===" >> "$TREND"
    .venv/bin/python scripts/x0_recovery_probe.py --only BBBP \
      --timesteps 20,50,100,200,900 \
      --bbbp-checkpoint "$f" >> "$TREND" 2>/dev/null
  done
  sleep "$SLEEP"
done
