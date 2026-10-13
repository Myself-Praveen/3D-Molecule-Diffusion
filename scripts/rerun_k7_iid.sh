#!/usr/bin/env bash
# Task 4 (K=7 triage): re-run the K=7 IID arms under the *fixed* partitioner
# (commit 2137c68) and complete their sweep records.
#
# Why this exists: `outputs/sweep_K_mode/numclients7_modeiid/*/results.json`
# recorded `train_exit=1` for all three seeds in the original (buggy
# partitioner) run and again in the first re-run, which is why Table I carried
# a "train_exit=1, no checkpoint" placeholder row instead of metrics. The
# re-run below materializes each arm's config through the same sweep_lib
# helper the runner uses, trains it, evaluates it with the exact protocol of
# the other K rows (1000 samples, DDIM-200/eta 1.0/quadratic/relax), and
# rewrites results.json from the artifacts (train_exit, eval_exit, metrics)
# rather than from memory.
#
#   setsid nohup bash scripts/rerun_k7_iid.sh > logs/rerun_k7_iid.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"123 456"}
K=${K:-7}
SWEEP_DIR=outputs/sweep_K_mode/numclients${K}_modeiid

log() { echo "[$(date +%H:%M:%S)] $*"; }

for seed in $SEEDS; do
  RUN="$SWEEP_DIR/seed_${seed}"
  log "seed $seed: materializing config"
  .venv/bin/python - "$RUN" "$seed" "$K" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from sweep_lib import load_yaml, write_run_config
run_dir, seed, k = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
base = load_yaml("configs/fed_bbb_niid.yaml")
print("  config ->", write_run_config(
    base, {"fed.num_clients": k, "fed.mode": "iid"}, seed, run_dir, True))
PY

  log "seed $seed: training"
  env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python fed_train.py \
    --config "$RUN/config.yaml" > "$RUN/train.log" 2>&1
  train_rc=$?
  log "seed $seed: train exit=$train_rc"

  if [ "$train_rc" -eq 0 ]; then
    log "seed $seed: eval"
    env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python \
      generate_and_eval.py --checkpoint "$RUN/best_global.pt" \
      --config "$RUN/config.yaml" --num_samples 1000 --seed "$seed" \
      --ddim_steps 200 --eta 1.0 --step_schedule quadratic \
      --output_dir "$RUN/eval" --relax > "$RUN/eval.log" 2>&1
    eval_rc=$?
  else
    eval_rc=1
  fi
  log "seed $seed: eval exit=$eval_rc"

  .venv/bin/python - "$RUN" "$seed" "$K" "$train_rc" "$eval_rc" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from sweep_lib import REPO_ROOT, collect_results
run_dir, seed, k, trc, erc = (
    Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
    int(sys.argv[4]), int(sys.argv[5]))
run_dir = run_dir.resolve()          # argv may be relative; REPO_ROOT is not
cfg = run_dir / "config.yaml"
ckpt = run_dir / "best_global.pt"
res = {
    "sweep": "K_mode",
    "combo": {"fed.num_clients": k, "fed.mode": "iid"},
    "seed": seed,
    "config": str(cfg.relative_to(REPO_ROOT)),
    "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
    "train_exit": trc,
    "eval_exit": erc,
    "status": "ok" if trc == erc == 0 else "failed",
    **collect_results(run_dir, run_dir / "eval",
                      ["Validity", "ConnectedValidity", "BBB%", "ScaffDiv",
                       "server_loss"], True),
}
(run_dir / "results.json").write_text(json.dumps(res, indent=2))
m = res["metrics"]
print("  seed %d status=%s V=%s ConnV=%s Bonds=%s BBB%%=%s"
      % (seed, res["status"], m.get("Validity"), m.get("ConnectedValidity"),
         m.get("BondsPerMol"), m.get("BBB%")))
PY
done

log "all seeds done"
