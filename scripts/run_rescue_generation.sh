#!/usr/bin/env bash
# Rescue plan R4/R5 — generation execution (no retraining).
#
# Runs the plan's suggested order against the V5 checkpoint and the unchanged
# Table-I protocol (1000 samples, DDIM-200, eta 1.0, quadratic, --relax):
#
#   1. R1  re-measure the geometry gate on evals already on disk. Free — no
#          generation. Prints bonds/atom (~1.0) and NN-outside-window (<=1%)
#          next to the reference, replacing the old "bonds/mol ~ 19" check.
#   2. R4  attractive connectivity guidance — the first sampler term that can
#          pull a fragment gap shut — at three weights.
#   3. R5  the knobs that were never non-zero in any recorded run:
#          valence_weight, ancestral DDPM (no --ddim_steps), and
#          keep_largest_fragment. Quick checks at n=200.
#
# Resumable: an arm whose metrics.json exists is skipped. Every step's exit
# status is captured, so a failed arm never looks like a pass.
#
#   bash scripts/run_rescue_generation.sh            # full sweep
#   N=200 bash scripts/run_rescue_generation.sh      # quick pass
set -uo pipefail

CKPT=${CKPT:-checkpoints/central_v5_bbb/best.pt}
CONFIG=${CONFIG:-configs/central_v5_bbb.yaml}
PYTHON=${PYTHON:-.venv/bin/python}
N=${N:-1000}
N_QUICK=${N_QUICK:-200}

mkdir -p outputs

run_diag () {
  local dir="outputs/$1"
  if [[ ! -f "$dir/molecules.sdf" ]]; then
    echo "== $1: no molecules.sdf, skipping re-measure"
    return 0
  fi
  echo "== $1: R1 geometry ruler"
  "$PYTHON" -u scripts/diagnose_bbb_clump.py \
    --eval_dir "$dir" --reference bbbp --reference_limit 200 \
    --out "$dir/clump.json" > "$dir.clump.log" 2>&1
  local rc=$?
  [[ $rc -ne 0 ]] && echo "== $1: diagnose FAILED (exit $rc), see $dir.clump.log"
  return 0
}

run_eval () {
  local name=$1; shift
  local n=$1; shift
  local dir="outputs/$name"
  if [[ -f "$dir/metrics.json" ]]; then
    echo "== $name: already evaluated, skipping"
    return 0
  fi
  echo "== $name: n=$n (DDIM-200, eta 1.0, quadratic, relax)"
  "$PYTHON" -u generate_and_eval.py \
    --checkpoint "$CKPT" --config "$CONFIG" \
    --num_samples "$n" --batch_size 100 \
    --ddim_steps 200 --eta 1.0 --step_schedule quadratic \
    --seed 42 --relax --bbb_oracle models/bbb_oracle.pt \
    --output_dir "$dir" "$@" > "$dir.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "== $name: FAILED (exit $rc), see $dir.log"
    return 0
  fi
  run_diag "$name"
  echo "== $name: done"
  return 0
}

run_eval_ancestral () {
  local name=$1
  local n=$2
  local dir="outputs/$name"
  if [[ -f "$dir/metrics.json" ]]; then
    echo "== $name: already evaluated, skipping"
    return 0
  fi
  echo "== $name: n=$n (ancestral DDPM, eta 1.0, quadratic, relax)"
  "$PYTHON" -u generate_and_eval.py \
    --checkpoint "$CKPT" --config "$CONFIG" \
    --num_samples "$n" --batch_size 100 \
    --eta 1.0 --step_schedule quadratic \
    --seed 42 --relax --bbb_oracle models/bbb_oracle.pt \
    --output_dir "$dir" > "$dir.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "== $name: FAILED (exit $rc), see $dir.log"
    return 0
  fi
  run_diag "$name"
  echo "== $name: done"
  return 0
}

# --- 1. R1: re-measure the ruler on existing arms (no generation) ---------
run_diag eval_pre_rl
run_diag eval_v6_bbb_full

# --- 2. R4: attractive connectivity guidance ------------------------------
run_eval rescue_r4_cw0p5 "$N" --connectivity_weight 0.5
run_eval rescue_r4_cw1p0 "$N" --connectivity_weight 1.0
run_eval rescue_r4_cw2p0 "$N" --connectivity_weight 2.0

# --- 3. R5: the never-non-zero knobs --------------------------------------
run_eval rescue_r5_valence_w0p5 "$N_QUICK" --valence_weight 0.5
run_eval rescue_r5_keep_largest "$N_QUICK" --keep_largest_fragment
run_eval_ancestral rescue_r5_ancestral "$N_QUICK"

echo
echo "== summary (fix.md schema v2: bonds/atom ~1.035, NN-outside <=1%)"
"$PYTHON" - <<'PY'
import json
from pathlib import Path

# fix.md F12/F8: one ruler per row, read from metrics.json. The geometry axis
# (bonds/atom, NN-outside, fragments) now ships inside evaluate(), so it is
# present even when clump.json was never run; clump.json stays optional and
# only fills gaps for pre-v2 rows. None renders as "—", never as nan/0.
arms = ["eval_pre_rl", "eval_v6_bbb_full", "rescue_r4_cw0p5",
        "rescue_r4_cw1p0", "rescue_r4_cw2p0", "rescue_r5_valence_w0p5",
        "rescue_r5_keep_largest", "rescue_r5_ancestral"]


def _f(value, spec=".1f"):
    return "—" if value is None else format(value, spec)


print(f"{'arm':<26}{'V%':>7}{'ConnV':>7}{'ConnF':>7}"
      f"{'b/atom':>8}{'nn_out%':>9}{'frags':>7}{'atoms%':>8}")
for name in arms:
    d = Path("outputs") / name
    if not (d / "metrics.json").exists():
        print(f"{name:<26}{'--':>7}")
        continue
    m = json.loads((d / "metrics.json").read_text())["metrics"]
    c = {}
    if (d / "clump.json").exists():
        c = json.loads((d / "clump.json").read_text()).get("generated", {})
    print(f"{name:<26}{_f(m.get('Validity')):>7}"
          f"{_f(m.get('ConnectedValidity'), '.2f'):>7}"
          f"{_f(m.get('ConnectedFrac')):>7}"
          f"{_f(m.get('BondsPerAtom') if m.get('BondsPerAtom') is not None
                else c.get('bonds_per_atom'), '.2f'):>8}"
          f"{_f(m.get('NN_Outside%') if m.get('NN_Outside%') is not None
                else c.get('nn_outside_window_pct'), '.2f'):>9}"
          f"{_f(m.get('FragmentsPerMol'), '.2f'):>7}"
          f"{_f(m.get('AtomsRetained%'), '.0f'):>8}")
PY
