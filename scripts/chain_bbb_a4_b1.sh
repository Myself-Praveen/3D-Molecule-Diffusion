#!/usr/bin/env bash
# Plan Phase 1/2 chain 1 (decision-tree order):
#   [A1]  seeded re-run of the calibrated-lens baseline (the first attempt
#         was generated with an unseeded torch RNG — generate_and_eval.py
#         only seeded the numpy atom-count rng until the fix — so it is
#         re-run here and the old numbers are superseded)
#   [A4]  re-eval with --perceive_bonds (orders only; Validity/ConnV/bonds
#         are invariant by construction — this measures the SMILES-derived
#         metrics on real bond orders instead of an all-single skeleton)
#   [B1]  fine guidance sweep w = 1.5, 2.5, 3.0 (the plan's tree values)
#   [B2]  last.pt (ep 7700, BBB% 24.0 at w=0) + mild guidance w=0.5
# All runs use the calibrated lens: cap 0.25 + min_fragment_atoms 2 +
# max_bonds_ratio 1.4 (ground-truth calibrated, commit 5c6a46b) and, since
# the seeding fix, seed 42 covers torch sampling too — same seed + same
# flags reproduce the identical metrics (verified byte-for-byte).
#   setsid nohup bash scripts/chain_bbb_a4_b1.sh > logs/chain_bbb_a4_b1.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

CKPT_BEST=checkpoints/bbb_full/best.pt
CKPT_LAST=checkpoints/bbb_full/last.pt
CFG=configs/central_bbb_full.yaml
LENS="--min_fragment_atoms 2 --bond_tolerance 0.25 --max_bonds_ratio 1.4"
PROTO="--num_samples 1000 --ddim_steps 200 --eta 1.0 --step_schedule quadratic --relax"
ORACLE="--bbb_oracle models/bbb_oracle.pt"

log() { echo "[$(date +%H:%M:%S)] $*"; }

run() {  # run <logname> <outdir> <extra args...>
  local name=$1 out=$2; shift 2
  log "START $name -> $out"
  env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python generate_and_eval.py \
    --checkpoint "$CKPT" --config "$CFG" $PROTO $LENS $ORACLE "$@" \
    --output_dir "$out" > "logs/eval_${name}.log" 2>&1
  local rc=$?
  log "DONE  $name exit=$rc"
  return $rc
}

CKPT=$CKPT_BEST
run a1_baseline outputs/eval_bbb_full_connfix_a1
run a4_perceive outputs/eval_bbb_connfix_a4 --perceive_bonds
run guidance_w1p5 outputs/eval_bbb_guidance_w1p5 --guidance_scale 1.5 --target_bbb 1
run guidance_w2p5 outputs/eval_bbb_guidance_w2p5 --guidance_scale 2.5 --target_bbb 1
run guidance_w3p0 outputs/eval_bbb_guidance_w3p0 --guidance_scale 3.0 --target_bbb 1
CKPT=$CKPT_LAST
run last_w0p5 outputs/eval_bbb_last_w05 --guidance_scale 0.5 --target_bbb 1

log "chain complete"
