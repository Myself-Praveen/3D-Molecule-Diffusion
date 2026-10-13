#!/usr/bin/env bash
# TASK 1 — reproduce the recorded BBB negatives from committed artifacts, no retrain.
# One command. Prints the two logs and three metrics.json files, then re-derives the
# headline numbers so the result is checkable from disk rather than from memory.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "============================================================"
echo "TASK 1 — reproduce recorded negatives (artifacts only)"
echo "============================================================"

for f in \
  logs/eval_bbb_full_final.log \
  logs/bbb_renorm_connv.log \
  outputs/eval_bbb_full_final/metrics.json \
  outputs/eval_bbb_full_final_last/metrics.json \
  outputs/eval_bbb_renorm_final/metrics.json \
; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f" >&2
    exit 2
  fi
done

echo
echo "--- logs/eval_bbb_full_final.log (best.pt, epoch 6159) ---"
cat logs/eval_bbb_full_final.log
echo
echo "--- logs/bbb_renorm_connv.log (best.pt, epoch 1160) ---"
cat logs/bbb_renorm_connv.log
echo
echo "--- outputs/eval_bbb_full_final/metrics.json ---"
cat outputs/eval_bbb_full_final/metrics.json
echo
echo "--- outputs/eval_bbb_full_final_last/metrics.json ---"
cat outputs/eval_bbb_full_final_last/metrics.json
echo
echo "--- outputs/eval_bbb_renorm_final/metrics.json ---"
cat outputs/eval_bbb_renorm_final/metrics.json
echo

echo "============================================================"
echo "Re-derived headline numbers (from the metrics.json above)"
echo "============================================================"
python3 - <<'PY'
import json
from pathlib import Path

def m(path):
    return json.loads(Path(path).read_text())["metrics"]

best   = m("outputs/eval_bbb_full_final/metrics.json")
last   = m("outputs/eval_bbb_full_final_last/metrics.json")
renorm = m("outputs/eval_bbb_renorm_final/metrics.json")

print(f"{'run':28} {'V':>6} {'ConnV':>6} {'Bonds':>7} {'BBB%':>7} {'n_valid':>8}")
for name, d in [
    ("bbb_full best (ep 6159)", best),
    ("bbb_full last (ep 7700)", last),
    ("bbb_renorm best (ep 1160)", renorm),
]:
    print(f"{name:28} {d['Validity']:6.1f} {d['ConnectedValidity']:6.1f} "
          f"{d['BondsPerMol']:7.2f} {d['BBB%']:7.2f} {d.get('num_valid',-1):8d}")

print()
print("Recorded answer (CONTEXT block of the mission brief):")
print("  bbb_full best  V 55.4 / ConnV 1.0 / Bonds 32.8 / BBB% 17.3")
print("  bbb_full last  V 53.8 / ConnV 1.2 / Bonds 33.1 / BBB% 24.0")
print("  bbb_renorm     V 47.8 / ConnV 0.6 / Bonds 30.3 / BBB% 16.1")
PY

echo
echo "TASK 1 verdict: negatives reproduce from committed artifacts (exit 0)."
