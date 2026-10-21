#!/usr/bin/env python
"""Calibrate the bond-perception chain on GROUND TRUTH before deploying it.

The plan's risk table requires: "Tightening _BOND_TOLERANCE reduces QM9
ground-truth validity below 99% -> test on QM9 ground truth first. Calibrate
tolerance so QM9 ground-truth stays >= 98%." This script is that test, and
it applies the same bar to BBBP ground truth (the actual target domain):

For each lens setting it rebuilds real molecules from their true 3D
coordinates through ``coords_and_types_to_mol`` and reports

    Validity%     — rebuild sanitized (the plan's >= 98% bar, exit 2 if missed)
    SingleFrag%   — one connected fragment (ground-truth ConnV proxy)
    BondsPerMol   — rebuilt bonds vs the dataset's true bond count
    PerceptionOK% — A4: fraction where DetermineBondOrders succeeded (A4 must
                    be neutral-or-positive on ground truth, per the plan)

Lens settings: legacy (Table I protocol) vs cap 0.35 vs cap 0.35 +
min_fragment_atoms 2 vs that + perception.

Usage:
    python scripts/calibrate_bond_tolerance.py [--n_qm9 1000] [--n_bbbp 500]

Exit codes: 0 = every DEPLOY-candidate lens holds QM9 >= 98% AND is
non-inferior to legacy (validity + single-fragment) on both datasets;
2 = a deploy candidate failed. Rejected candidates (the plan's 0.35 guess)
and the legacy baseline itself are recorded for comparison but do not
determine the exit status; legacy failing the absolute bar is a
pre-existing finding that gets recorded, not a blocker.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from rdkit import Chem  # noqa: E402

from src.utils.evaluation import coords_and_types_to_mol  # noqa: E402

BAR = 98.0  # plan's ground-truth validity floor (%)

# (label, kwargs for coords_and_types_to_mol)
# Calibration (first run, 21 Oct 2026, QM9 n=1000 / BBBP n=500): legacy
# 95.90% QM9 — every failure is "Explicit valence for C, 5", i.e. the
# legacy lens bonds a spurious 5th neighbour on REAL geometry. Caps fix
# those: 0.35 -> 97.90%, 0.30 -> 98.60%, 0.25 -> 99.90% QM9, with BBBP
# validity and single-fragment rate never below legacy at any cap (a cut
# TRUE bond would show up as a fragment split; it does not). The plan's
# assumed 0.35 misses its own >= 98% bar, so the calibrated deploy value
# is 0.25 — chosen by the plan's risk-table rule ("calibrate so QM9
# ground truth stays >= 98%"), not by the plan's illustrative number.
# Each entry: (label, kwargs, judge) — judge=False rows are candidates we
# expect to reject (recorded for the comparison) or the legacy baseline;
# exit status reflects ONLY judge=True rows (the deploy candidates).
LENSES = [
    ("legacy (Table I)", {}, False),
    ("cap 0.35 (plan's guess)", {"bond_tolerance": 0.35}, False),
    ("cap 0.30", {"bond_tolerance": 0.30}, False),
    ("cap 0.25 + frag 2 (DEPLOY)",
     {"bond_tolerance": 0.25, "min_fragment_atoms": 2}, True),
    ("cap 0.25 + frag 2 + perceive (DEPLOY)",
     {"bond_tolerance": 0.25, "min_fragment_atoms": 2,
      "perceive_orders": True}, True),
]


def _samples(data, n: int, seed: int = 0):
    """Deterministic sample of (pos, z, true_bonds) triples."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))[:n]
    out = []
    for i in idx:
        d = data[int(i)]
        pos = np.asarray(d.pos)
        z = np.asarray(d.z)
        n_true = None
        ei = getattr(d, "edge_index", None)
        if ei is not None:
            n_true = int(ei.shape[1]) // 2
        out.append((pos, z, n_true))
    return out


def measure(samples, **lens_kwargs) -> dict:
    valid = single = perceive_ok = 0
    bonds, true_bonds = [], []
    for pos, z, n_true in samples:
        mol = coords_and_types_to_mol(pos, z, **lens_kwargs)
        if mol is None:
            continue
        valid += 1
        if len(Chem.GetMolFrags(mol)) == 1:
            single += 1
        bonds.append(mol.GetNumBonds())
        if n_true is not None:
            true_bonds.append(n_true)
        if lens_kwargs.get("perceive_orders"):
            # success = a bond order beyond SINGLE was found (or the molecule
            # is trivially all-single-but-solved — detected as no exception
            # is impossible to see from outside, so count non-single as OK;
            # ground-truth molecules have double/aromatic bonds when correct)
            if any(b.GetBondType() != Chem.BondType.SINGLE
                   for b in mol.GetBonds()):
                perceive_ok += 1
    n = len(samples)
    return {
        "Validity%": 100.0 * valid / n if n else 0.0,
        "SingleFrag%": 100.0 * single / n if n else 0.0,
        "BondsPerMol": float(np.mean(bonds)) if bonds else 0.0,
        "TrueBondsPerMol": float(np.mean(true_bonds)) if true_bonds else float("nan"),
        "PerceptionOK%": 100.0 * perceive_ok / n if n else 0.0,
    }


def run_dataset(name: str, samples) -> list[tuple[str, dict, bool]]:
    print(f"\n=== {name}: {len(samples)} ground-truth molecules ===")
    rows = []
    for label, kwargs, judge in LENSES:
        m = measure(samples, **kwargs)
        rows.append((label, m, judge))
        line = (f"  {label:<32} Valid {m['Validity%']:6.2f}%  "
                f"1frag {m['SingleFrag%']:6.2f}%  "
                f"bonds {m['BondsPerMol']:6.2f}")
        if m["TrueBondsPerMol"] == m["TrueBondsPerMol"]:  # not NaN
            line += f" (true {m['TrueBondsPerMol']:.2f})"
        if kwargs.get("perceive_orders"):
            line += f"  perceiveOK {m['PerceptionOK%']:6.2f}%"
        print(line)
        if name == "QM9":  # the plan's absolute bar applies to QM9 only
            bar = "OK " if m["Validity%"] >= BAR else "FAIL"
            print(f"  {'':32} vs >= {BAR:.0f}% bar: {bar}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n_qm9", type=int, default=1000)
    ap.add_argument("--n_bbbp", type=int, default=500)
    args = ap.parse_args()

    from src.dataset import load_qm9
    from src.dataset_bbb import load_bbbp

    qm9 = load_qm9(root="data")
    qm9_samples = _samples(qm9, args.n_qm9, seed=0)

    train, val, test = load_bbbp(root="data/bbbp")
    bbbp_samples = _samples(list(train) + list(val) + list(test),
                            args.n_bbbp, seed=0)

    rows_q = run_dataset("QM9", qm9_samples)
    rows_b = run_dataset("BBBP", bbbp_samples)

    # Verdict per NON-legacy lens:
    #   1. absolute bar: QM9 ground-truth validity >= 98% (plan risk table);
    #   2. non-inferiority: never below legacy on either dataset, in
    #      validity AND single-fragment rate (a cap that cut true bonds
    #      would fragment real molecules — that is what this catches).
    # The legacy row failing bar 1 is a PRE-EXISTING finding (recorded, not
    # a deployment blocker for caps that strictly improve it).
    legacy = {"QM9": rows_q[0][1], "BBBP": rows_b[0][1]}
    failed = []
    for ds, rows in (("QM9", rows_q), ("BBBP", rows_b)):
        for label, m, judge in rows:
            if not judge:
                continue  # legacy baseline + rejected candidates are recorded only
            if ds == "QM9" and m["Validity%"] < BAR:
                failed.append(f"{ds} @ {label}: validity {m['Validity%']:.2f}% < {BAR:.0f}%")
            if m["Validity%"] < legacy[ds]["Validity%"]:
                failed.append(f"{ds} @ {label}: validity below legacy "
                              f"({m['Validity%']:.2f}% < {legacy[ds]['Validity%']:.2f}%)")
            if m["SingleFrag%"] < legacy[ds]["SingleFrag%"]:
                failed.append(f"{ds} @ {label}: single-frag below legacy "
                              f"({m['SingleFrag%']:.2f}% < {legacy[ds]['SingleFrag%']:.2f}%) "
                              "— a cap cut a TRUE bond")
    if failed:
        print("\nVERDICT: FAIL — do not deploy the flagged settings:")
        for f in failed:
            print(f"  {f}")
        return 2
    print(f"\nVERDICT: PASS — every DEPLOY lens holds QM9 ground truth >= "
          f"{BAR:.0f}% and is non-inferior to legacy on both datasets "
          "(validity and single-fragment).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
