#!/usr/bin/env bash
# Final measurement chain for the corrected plan (Steps 1.1 / 1.2 / 3.1).
#
# 1. Re-runs the steric arms under the CURRENT sampler code. The earlier arms
#    ran before the guidance weight became a ramp (steric_guidance_weight), so
#    mixing them into one table would compare two different interventions —
#    they are regenerated here under identical code.
# 2. Runs the plan's pre-RL evaluation (Step 3.1): 1000 samples, DDIM-200,
#    eta=1.0, quadratic, --relax (the Table I ruler), on best.pt.
#
# Resumable: an arm with metrics.json already present is skipped.
#
#   OMP_NUM_THREADS=8 bash scripts/chain_bbb_final_eval.sh

set -u

CKPT=${CKPT:-checkpoints/central_v5_bbb/best.pt}
CONFIG=${CONFIG:-configs/central_v5_bbb.yaml}
PYTHON=${PYTHON:-.venv/bin/python}
N_ARMS=${N_ARMS:-200}
N_PREREL=${N_PREREL:-1000}

python_bin=$PYTHON

run_eval () {
  local name=$1; shift
  local n=$1; shift
  local dir="outputs/$name"
  if [[ -f "$dir/metrics.json" ]]; then
    echo "== $name: already evaluated, skipping"
    return 0
  fi
  echo "== $name: n=$n (DDIM-200, eta 1.0, quadratic, relax)"
  "$python_bin" -u generate_and_eval.py \
    --checkpoint "$CKPT" --config "$CONFIG" \
    --num_samples "$n" --batch_size 100 \
    --ddim_steps 200 --eta 1.0 --step_schedule quadratic \
    --seed 42 --relax \
    --bbb_oracle models/bbb_oracle.pt \
    --output_dir "$dir" "$@" \
    > "$dir.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "== $name: FAILED (exit $rc), see $dir.log"
    return $rc
  fi
  "$python_bin" -u scripts/diagnose_bbb_clump.py \
    --eval_dir "$dir" --reference bbbp --reference_limit 200 \
    --out "$dir/clump.json" > "$dir.clump.log" 2>&1
  echo "== $name: done"
}

# --- 1. consistent steric arms (current ramp) -----------------------------
run_eval eval_v5_bbb_steric_w1 "$N_ARMS" --steric_weight 1
run_eval eval_v5_bbb_steric_w5 "$N_ARMS" --steric_weight 5

# --- 2. plan Step 3.1 pre-RL evaluation -----------------------------------
run_eval eval_pre_rl "$N_PREREL"

echo
echo "== summary"
"$python_bin" - <<'PY'
import json
from pathlib import Path

rows = []
for d in sorted(Path("outputs").glob("eval_v5_bbb_*")) + [Path("outputs/eval_pre_rl")]:
    if not (d / "metrics.json").exists():
        continue
    m = json.loads((d / "metrics.json").read_text())["metrics"]
    c = {}
    if (d / "clump.json").exists():
        c = json.loads((d / "clump.json").read_text()).get("generated", {})
    rows.append((d.name, m.get("Validity"), m.get("ConnectedValidity"),
                 m.get("BondsPerMol"), m.get("BBB%"), m.get("IntDiv_p"),
                 c.get("fragments_per_mol"), c.get("radius_of_gyration")))

print(f"{'arm':<28}{'V':>7}{'ConnV':>7}{'Bonds':>7}{'BBB%':>7}{'IntDiv':>8}{'frags':>7}{'Rgyr':>7}")
for r in rows:
    print(f"{r[0]:<28}{r[1]:>7.1f}{r[2]:>7.2f}{r[3]:>7.2f}{r[4]:>7.1f}"
          f"{r[5]:>8.3f}{(r[6] or float('nan')):>7.2f}{(r[7] or float('nan')):>7.2f}")
PY
