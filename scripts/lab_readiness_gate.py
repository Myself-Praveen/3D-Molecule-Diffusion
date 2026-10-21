#!/usr/bin/env python
"""Run the six-stage lab-readiness gate over an existing eval directory.

Reads an eval dir produced by ``generate_and_eval.py`` (``molecules.sdf``
preferred — 3D coordinates, which stages 1 and 2 require; ``smiles.txt`` as
fallback, which makes those stages report SKIP rather than guess), applies
the gate criteria from ``src/utils/lab_gate.py``, prints the verdict table
and writes ``lab_gate.json`` next to the eval outputs.

Usage:
    python scripts/lab_readiness_gate.py --eval-dir outputs/eval_bbb_full_final
    python scripts/lab_readiness_gate.py --eval-dir OUT --config configs/central_bbb.yaml
    # stage 4 (applicability domain) needs the training set: pass --config.

Exit codes (so chains/CI can gate on it):
    0 = PASS, 1 = FAIL, 2 = INCOMPLETE (a criterion was not measurable).

Note on the routes criterion: it is RELATIVE to the training set routed
under identical bounds (absolute 2-step rates are dataset-specific — ~1%
for BBBP-like space even for purchasable drugs), so pass --config for a
routes verdict rather than trusting an absolute number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rdkit import Chem  # noqa: E402

from src.utils.lab_gate import flatten_report, print_gate, run_gate  # noqa: E402


def load_eval_mols(eval_dir: Path) -> tuple[list, str]:
    """Molecules from an eval dir: molecules.sdf (3D) first, else smiles.txt."""
    sdf = eval_dir / "molecules.sdf"
    if sdf.exists():
        # Keep None entries: n_total must reflect what was written, and the
        # gate's denominators (valid only) do the right thing with them.
        return [m for m in Chem.SDMolSupplier(str(sdf))], "molecules.sdf"
    txt = eval_dir / "smiles.txt"
    if txt.exists():
        lines = [ln.strip() for ln in txt.read_text().splitlines() if ln.strip()]
        return [Chem.MolFromSmiles(ln) for ln in lines], "smiles.txt"
    raise FileNotFoundError(
        f"no molecules.sdf or smiles.txt in {eval_dir} — run generate_and_eval.py first"
    )


def load_train_mols(config_path: Path) -> list:
    """Training-set reference molecules for stage 4, from a config YAML.

    Uses the dataset's original ``data.smiles`` where available (BBBP/B3DB):
    re-parsing SMILES restores proper bond orders so the Morgan fingerprints
    compare real chemistry. QM9 has no SMILES column, so it falls back to the
    same distance-based conversion eval uses — slower, and noted on stdout.
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data") or {}
    name = str(data_cfg.get("dataset", "qm9")).lower()
    root = data_cfg.get("root", "data")
    if name == "bbbp":
        from src.dataset_bbb import load_bbbp

        full, _, _ = load_bbbp(root=root)
    elif name == "b3db":
        from src.dataset_bbb import load_b3db

        full, _, _ = load_b3db(root=root)
    else:
        from src.dataset import load_qm9

        full = load_qm9(root=root)

    refs: list[Chem.Mol] = []
    have_smiles = False
    for data in full:
        smi = getattr(data, "smiles", None)
        if smi:
            have_smiles = True
            m = Chem.MolFromSmiles(str(smi))
            if m is not None:
                refs.append(m)
    if not refs and not have_smiles:
        # QM9 path: distance-built graphs (bond orders not recoverable from
        # SMILES because there is none). Slow but correct-by-eval-convention.
        print(f"note: dataset {name!r} carries no SMILES — building the "
              "reference from 3D graphs (this can take a minute)")
        from generate_and_eval import _data_to_rdkit_mol

        refs = [m for m in (_data_to_rdkit_mol(d) for d in full)
                if m is not None]
    return refs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lab-readiness gate over an eval directory")
    parser.add_argument("--eval-dir", type=Path, required=True,
                        help="Directory with molecules.sdf / smiles.txt")
    parser.add_argument("--config", type=Path, default=None,
                        help="Training config YAML — enables stage 4 "
                             "(applicability domain) and the stage 6 "
                             "routability baseline; without it both SKIP")
    parser.add_argument("--out", type=Path, default=None,
                        help="Verdict JSON path (default: <eval-dir>/lab_gate.json)")
    args = parser.parse_args()

    if not args.eval_dir.is_dir():
        print(f"ERROR: {args.eval_dir} is not a directory", file=sys.stderr)
        return 2

    mols, source = load_eval_mols(args.eval_dir)
    print(f"loaded {len(mols)} molecules from {args.eval_dir / source}"
          + (" (valid-only; Validity itself is not recomputed)"
             if source == "smiles.txt" else ""))

    train_mols = []
    if args.config:
        train_mols = load_train_mols(args.config)
        print(f"training reference: {len(train_mols)} molecules "
              f"from {args.config}")
    else:
        print("no --config: stage 4 (applicability domain) and the stage 6 "
              "routability baseline will SKIP")

    gate = run_gate(mols, train_mols or None)
    print_gate(gate)

    print("stage metrics:")
    for k, v in sorted(flatten_report(gate["report"]).items()):
        print(f"  {k:<22} {v:>10.4f}" if isinstance(v, float)
              else f"  {k:<22} {v:>10}")

    out = args.out or (args.eval_dir / "lab_gate.json")
    payload = {
        **gate,
        "source": str(args.eval_dir / source),
        "config": str(args.config) if args.config else None,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[gate["overall"]]


if __name__ == "__main__":
    sys.exit(main())
