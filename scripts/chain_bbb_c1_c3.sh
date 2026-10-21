#!/usr/bin/env bash
# Plan Phase 1/2 chain 2:
#   [C1]  quality floor: --min_qed 0.3 (raw table printed first by the tool,
#         so the filter's effect stays visible rather than laundered)
#   [C3]  type-temperature sweep T = 0.7 / 0.8 / 0.9 (CNS_MPO direction:
#         sharper, lower-MW type choices)
#   [D3]  T = 1.1 (IntDiv direction: flatter type posterior)
# All runs use the calibrated lens (cap 0.25 + frag 2 + ratio 1.4) and the
# same Table I protocol as chain 1 for comparability.
#   setsid nohup bash scripts/chain_bbb_c1_c3.sh > logs/chain_bbb_c1_c3.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

CKPT=checkpoints/bbb_full/best.pt
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

run minqed0p3 outputs/eval_bbb_minqed03 --min_qed 0.3
run temp0p7 outputs/eval_bbb_temp0p7 --type_temperature 0.7
run temp0p8 outputs/eval_bbb_temp0p8 --type_temperature 0.8
run temp0p9 outputs/eval_bbb_temp0p9 --type_temperature 0.9
run temp1p1 outputs/eval_bbb_temp1p1 --type_temperature 1.1

log "chain complete"
