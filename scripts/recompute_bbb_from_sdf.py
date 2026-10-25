#!/usr/bin/env python
"""Recompute ``BBB%`` for an existing eval directory from its saved ``molecules.sdf``.

Why this exists
---------------
``BBBClassifier.predict_batch`` inferred each graph's node count from
``edge_index``. A molecule whose highest-index atom has no neighbour inside the
2.5 A distance gate therefore collated one atom short, ``global_mean_pool``
raised, and the blanket ``try/except`` in ``predict_batch`` turned the
*whole cohort* into a BBB% of 0.0 (progress.md section 17). The fixed model sets
``num_nodes`` explicitly and falls back to per-molecule scoring instead of
silently zeroing.

Evals produced before that fix carry a wrong ``BBB%`` while their molecule set
is correct, so this script re-scores the saved cohort with the fixed code path
and rewrites ``metrics.json``'s ``BBB%`` in place, recording the provenance in a
``bbb_recompute`` block. Molecules are read back from the SDF the eval wrote, so
the number describes exactly the cohort in that directory; nothing is
regenerated.

    .venv/bin/python scripts/recompute_bbb_from_sdf.py                 # every arm
    .venv/bin/python scripts/recompute_bbb_from_sdf.py --only eval_pre_rl
    .venv/bin/python scripts/recompute_bbb_from_sdf.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from rdkit import Chem, RDLogger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.bbb_classifier import BBBClassifier  # noqa: E402
from src.utils.evaluation import bbb_permeability_rate  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def load_oracle(path: Path) -> BBBClassifier:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    oracle = BBBClassifier(hidden_dim=int(ckpt.get("hidden_dim", 128)))
    oracle.load_state_dict(ckpt["model_state_dict"])
    return oracle.eval()


def rescore(eval_dir: Path, oracle: BBBClassifier) -> dict | None:
    sdf = eval_dir / "molecules.sdf"
    metrics_path = eval_dir / "metrics.json"
    if not sdf.exists() or not metrics_path.exists():
        return None
    mols = [m for m in Chem.SDMolSupplier(str(sdf), removeHs=False)
            if m is not None]
    if not mols:
        return None
    rate = bbb_permeability_rate(mols, oracle) * 100.0
    payload = json.loads(metrics_path.read_text())
    metrics = payload["metrics"]
    old = metrics.get("BBB%")
    metrics["BBB%"] = rate
    payload["bbb_recompute"] = {
        "reason": "predict_batch node-count inference zeroed whole cohorts "
                  "before the num_nodes fix (progress.md section 17)",
        "source": "molecules.sdf",
        "n_molecules": len(mols),
        "previous_value": old,
    }
    return {"metrics_path": metrics_path, "payload": payload,
            "old": old, "new": rate, "n": len(mols)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--oracle", default="models/bbb_oracle.pt")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Restrict to these directory names")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the table without rewriting metrics.json")
    args = ap.parse_args()

    oracle = load_oracle(Path(args.oracle))
    dirs = sorted(p for p in Path(args.outputs).iterdir() if p.is_dir())
    if args.only:
        wanted = set(args.only)
        dirs = [d for d in dirs if d.name in wanted]

    rows = []
    for d in dirs:
        result = rescore(d, oracle)
        if result is None:
            continue
        rows.append((d.name, result["n"], result["old"], result["new"]))
        if not args.dry_run:
            result["metrics_path"].write_text(
                json.dumps(result["payload"], indent=2) + "\n")

    print(f"{'arm':<34}{'n':>5}{'old BBB%':>10}{'new BBB%':>10}")
    for name, n, old, new in rows:
        flag = "" if old is not None and abs(old - new) < 1e-9 else "  <- changed"
        print(f"{name:<34}{n:>5}{(old if old is not None else float('nan')):>10.2f}"
              f"{new:>10.2f}{flag}")
    if args.dry_run:
        print("\n(dry run: metrics.json untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
