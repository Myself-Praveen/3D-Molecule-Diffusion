#!/usr/bin/env bash
# BBBP geometry measurement arms (plan triage Step 2 + Interventions 1/2).
#
# Everything here runs on the finished V5 checkpoint
# (checkpoints/central_v5_bbb/best.pt, 3900 epochs) under ONE protocol, so the
# rows are directly comparable:
#
#   generate_and_eval.py --ddim_steps 200 --eta 1.0 --step_schedule quadratic
#                        --seed 42 --guidance_scale 2.0 (config default)
#
# Arms:
#   base          no intervention (the plan's triage baseline)
#   steric_w{1,5,10}  test-time steric guidance on the clean x0
#   ctemp0p5      coordinate temperature 0.5
#
# Each arm's eval dir also gets clump.json (scripts/diagnose_bbb_clump.py), so
# the fragment/structure signature is recorded next to the headline metrics.
#
# Resumable: an arm with metrics.json already present is skipped.
#
#   OMP_NUM_THREADS=8 bash scripts/chain_bbb_geometry_arms.sh
#   OMP_NUM_THREADS=8 N_SAMPLES=200 bash scripts/chain_bbb_geometry_arms.sh

set -u

CKPT=${CKPT:-checkpoints/central_v5_bbb/best.pt}
CONFIG=${CONFIG:-configs/central_v5_bbb.yaml}
OUT_ROOT=${OUT_ROOT:-outputs}
N_SAMPLES=${N_SAMPLES:-200}
N_BASE=${N_BASE:-400}
SEED=${SEED:-42}
STEPS=${STEPS:-200}

if [[ ! -f "$CKPT" ]]; then
  echo "missing checkpoint $CKPT" >&2
  exit 1
fi

python_bin=${PYTHON:-.venv/bin/python}

run_arm () {
  local name=$1; shift
  local dir="$OUT_ROOT/eval_v5_bbb_$name"
  if [[ -f "$dir/metrics.json" ]]; then
    echo "== $name: already evaluated ($dir), skipping"
    return 0
  fi
  echo "== $name: generating $N_SAMPLES molecules (DDIM-$STEPS, eta 1.0)"
  "$python_bin" generate_and_eval.py \
    --checkpoint "$CKPT" --config "$CONFIG" \
    --num_samples "$N_SAMPLES" --batch_size 100 \
    --ddim_steps "$STEPS" --eta 1.0 --step_schedule quadratic \
    --seed "$SEED" --bbb_oracle models/bbb_oracle.pt \
    --output_dir "$dir" "$@" \
    > "$dir.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "== $name: eval FAILED (exit $rc), see $dir.log"
    return $rc
  fi
  "$python_bin" scripts/diagnose_bbb_clump.py \
    --eval_dir "$dir" --reference bbbp --reference_limit 200 \
    --out "$dir/clump.json" > "$dir.clump.log" 2>&1
  echo "== $name: done -> $dir"
}

# Baseline first (more samples: it is the row every arm is compared against).
N_SAVED=$N_SAMPLES
N_SAMPLES=$N_BASE run_arm base
N_SAMPLES=$N_SAVED

# Intervention 1: steric guidance sweep (plan's w in {1, 5, 10}).
for w in 1 5 10; do
  run_arm "steric_w$w" --steric_weight "$w"
done

# Intervention 2: coordinate temperature (a mid-range value; eta is already 1.0).
run_arm ctemp0p5 --coord_temperature 0.5

echo
echo "== summary (n, Validity, ConnV, BondsPerMol, BBB%, fragments/mol)"
"$python_bin" - "$OUT_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for d in sorted(root.glob("eval_v5_bbb_*")):
    mpath = d / "metrics.json"
    if not mpath.exists():
        continue
    m = json.loads(mpath.read_text())
    met = m["metrics"]
    clump = {}
    cpath = d / "clump.json"
    if cpath.exists():
        clump = json.loads(cpath.read_text()).get("generated", {})
    rows.append((
        d.name.replace("eval_v5_bbb_", ""),
        m.get("num_total", 0),
        met.get("Validity", float("nan")),
        met.get("ConnectedValidity", float("nan")),
        met.get("BondsPerMol", float("nan")),
        met.get("BBB%", float("nan")),
        clump.get("fragments_per_mol", float("nan")),
        clump.get("radius_of_gyration", float("nan")),
    ))

hdr = ("arm", "n", "Valid", "ConnV", "Bonds", "BBB%", "frags", "Rgyr")
print(f"{hdr[0]:<14}{hdr[1]:>6}{hdr[2]:>8}{hdr[3]:>8}{hdr[4]:>8}{hdr[5]:>8}{hdr[6]:>8}{hdr[7]:>8}")
for r in rows:
    print(f"{r[0]:<14}{r[1]:>6d}{r[2]:>8.2f}{r[3]:>8.2f}{r[4]:>8.2f}"
          f"{r[5]:>8.2f}{r[6]:>8.2f}{r[7]:>8.2f}")
PY

echo
echo "Record the table above in progress.md (honest verdict, no cherry-picking)."
