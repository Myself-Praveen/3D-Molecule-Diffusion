#!/usr/bin/env bash
# Chain the Task 3b (lambda2 live-class renorm) acceptance steps so they run
# without a live session:
#
#   1. wait for train.py on configs/central_bbb_lambda2renorm.yaml to reach
#      epoch 1400 (or to exit early) -- decided from the checkpoint itself, not
#      from the log, per the "read progress from artifacts" rule (§15 item 0);
#   2. final 1000-molecule eval on best.pt, same protocol as every Table I row
#      (DDIM-200 / eta=1.0 / quadratic / relax) -> outputs/eval_bbb_renorm_final;
#   3. ConnV-LDAG diagnostic on best.pt -> logs/bbb_renorm_connv.log;
#   4. regenerate Figs 1-7 + Table I from real data (the new BBB renorm row
#      appears automatically once its metrics.json exists).
#
#   setsid nohup bash scripts/chain_bbb_renorm_final.sh \
#       > logs/chain_bbb_renorm_final.log 2>&1 &
#
# Every step's exit status is captured and reported; a failed step is recorded
# as failed rather than allowed to look like a pass.
set -u
cd "$(dirname "$0")/.."

CFG=${CFG:-configs/central_bbb_lambda2renorm.yaml}
CKPT_DIR=${CKPT_DIR:-checkpoints/bbb_renorm}
TARGET_EPOCH=${TARGET_EPOCH:-1400}
OUT=${OUT:-outputs/eval_bbb_renorm_final}
POLL=${POLL:-120}          # seconds between progress checks
MAX_POLLS=${MAX_POLLS:-180}  # 6 h ceiling

log() { echo "[$(date +%H:%M:%S)] $*"; }

epoch_now() {
  .venv/bin/python - "$CKPT_DIR/last.pt" <<'PY' 2>/dev/null || echo 0
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(ck.get("epoch") or 0))
PY
}

log "waiting for $CKPT_DIR to reach epoch $TARGET_EPOCH"
for i in $(seq 1 "$MAX_POLLS"); do
  ep=$(epoch_now)
  if [ "$ep" -ge "$TARGET_EPOCH" ]; then
    log "reached epoch $ep"; break
  fi
  if ! pgrep -f "config ${CFG}" > /dev/null; then
    log "train.py no longer running (last checkpoint epoch $ep)"
    break
  fi
  log "epoch $ep/$TARGET_EPOCH"
  sleep "$POLL"
done

ep=$(epoch_now)
if [ "$ep" -lt "$TARGET_EPOCH" ]; then
  log "WARNING: stopping short of $TARGET_EPOCH (epoch $ep) -- evaluating anyway"
fi

log "step 2: final eval on best.pt -> $OUT"
env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python generate_and_eval.py \
  --checkpoint "$CKPT_DIR/best.pt" --config "$CFG" \
  --num_samples 1000 --ddim_steps 200 --eta 1.0 --relax \
  --step_schedule quadratic --output_dir "$OUT" > logs/eval_bbb_renorm_final.log 2>&1
eval_rc=$?
log "eval exit=$eval_rc"
if [ ! -f "$OUT/metrics.json" ]; then
  log "FAIL: $OUT/metrics.json missing -- stopping before the ConnV step"
  exit 1
fi

log "step 3: ConnV-LDAG diagnostic -> logs/bbb_renorm_connv.log"
{
  echo "=== ConnV-LDAG on $CKPT_DIR/best.pt (epoch $ep) ==="
  .venv/bin/python eval_connv.py --ckpt "$CKPT_DIR/best.pt" --n-samples 256
} > logs/bbb_renorm_connv.log 2>&1
log "connv exit=$?"

log "step 4: regenerate Table I + Figs 1-7 from real data"
env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python \
  scripts/generate_paper_figures.py > logs/figures_bbb_renorm.log 2>&1
fig_rc=$?
log "figures exit=$fig_rc"
tail -2 logs/figures_bbb_renorm.log

log "summary"
.venv/bin/python - "$OUT/metrics.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); mm = m["metrics"]
print("  Validity %.1f | ConnV %.1f | BondsPerMol %.1f | BBB%% %.1f | valid %d/%d"
      % (mm["Validity"], mm["ConnectedValidity"], mm["BondsPerMol"],
         mm.get("BBB%", float("nan")), m["num_valid"], m["num_total"]))
PASS = (mm["ConnectedValidity"] > 0 and mm.get("BBB%", 0) > 50)
print("  acceptance (ConnV>0 and BBB%>>50):", "PASS" if PASS else "FAIL (soup != objective)")
PY
log "chain done (eval=$eval_rc figures=$fig_rc)"
