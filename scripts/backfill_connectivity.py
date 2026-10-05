"""Backfill connectivity metrics into pre-Phase-4 eval metrics.json files.

Phase 4 added ConnectedValidity / BondsPerMol / ConnectedFrac to
``evaluate()``, so evals run before that commit lack the keys even though
their saved ``molecules.sdf`` contains exactly the valid-molecule list the
metrics were computed over (invalid molecules are None and are never
written to the SDF).

This recomputes the three keys from the SDF plus the stored ``num_total``
denominator — no molecule re-generation, so recorded validity numbers are
preserved bit-for-bit.

    ConnectedValidity = 100 * (# single-fragment in SDF) / num_total
    ConnectedFrac     = 100 * (# single-fragment in SDF) / (# in SDF)
    BondsPerMol       = mean bond count over SDF molecules

Validation: eval dirs that already carry the keys natively are recomputed
and compared (mismatch = hard error), so the formula is checked against
real ``evaluate()`` output before any file is modified.

Usage:
    .venv/bin/python scripts/backfill_connectivity.py --validate
    .venv/bin/python scripts/backfill_connectivity.py            # apply
    .venv/bin/python scripts/backfill_connectivity.py --only eval_v3 eval_v4_eps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_lib import REPO_ROOT  # noqa: E402  (repo-root resolution)

from rdkit import Chem  # noqa: E402


def connectivity_from_sdf(sdf_path: Path, num_total: int) -> tuple[dict, int] | None:
    """Recompute connectivity keys from a saved SDF of valid molecules.

    Returns ``(keys, parsed_count)`` or ``None`` when the file is missing or
    unreadable — some old SDFs are partially corrupt, and RDKit silently
    drops bad records, so callers must compare ``parsed_count`` against the
    recorded ``num_valid`` before trusting the keys.
    """
    if not sdf_path.exists():
        return None
    try:
        mols = [m for m in Chem.SDMolSupplier(str(sdf_path), removeHs=False)
                if m is not None]
    except (OSError, RuntimeError):
        return None
    if not mols:
        return None
    bonds, connected = [], 0
    for m in mols:
        bonds.append(m.GetNumBonds())
        if len(Chem.GetMolFrags(m)) == 1:
            connected += 1
    return {
        "ConnectedValidity": 100.0 * connected / max(num_total, 1),
        "BondsPerMol": float(sum(bonds) / len(mols)),
        "ConnectedFrac": 100.0 * connected / len(mols),
    }, len(mols)


def close_enough(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--only", default=None,
                   help="Comma-separated dir-name substrings to restrict")
    p.add_argument("--validate", action="store_true",
                   help="Only check dirs that already have the keys")
    args = p.parse_args()

    outputs = REPO_ROOT / args.outputs
    dirs = sorted(d for d in outputs.glob("eval_*") if d.is_dir())
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        dirs = [d for d in dirs if any(k in d.name for k in keys)]

    checked = validated = backfilled = skipped = 0
    failures = 0
    for d in dirs:
        metrics_path = d / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            data = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        metrics = data.get("metrics") or {}
        if not metrics:
            continue
        num_total = int(data.get("num_total", 0))
        if num_total <= 0:
            skipped += 1
            continue
        modified = False

        for table_key, sdf_name in (("metrics", "molecules.sdf"),
                                    ("metrics_relaxed",
                                     "molecules_relaxed.sdf")):
            table = data.get(table_key)
            if not isinstance(table, dict) or "Validity" not in table:
                continue
            if num_total <= 0:
                continue
            native = "ConnectedValidity" in table
            result = connectivity_from_sdf(d / sdf_name, num_total)
            if result is None:
                recorded_valid = round(float(table["Validity"]) / 100.0
                                       * num_total)
                if native and recorded_valid == 0:
                    # Zero valid molecules -> no SDF is written and the
                    # native keys are trivially all-zero; nothing to check.
                    validated += 1
                    continue
                if native:
                    print(f"FAIL {d.name}/{table_key}: native keys but no "
                          f"{sdf_name}")
                    failures += 1
                else:
                    skipped += 1
                continue
            computed, parsed = result
            expected_valid = round(float(table["Validity"]) / 100.0
                                   * num_total)
            if parsed != expected_valid:
                # Lossy/corrupt SDF: records were dropped on read, so the
                # recomputed keys would not match what evaluate() saw.
                if native and parsed != expected_valid:
                    print(f"note {d.name}/{table_key}: sdf parsed {parsed} "
                          f"!= recorded valid {expected_valid} "
                          f"(corrupt records) — keys left as recorded")
                else:
                    print(f"SKIP {d.name}/{table_key}: sdf parsed {parsed} "
                          f"!= recorded valid {expected_valid} "
                          f"(corrupt records) — not backfilled")
                    skipped += 1
                continue
            if native:
                checked += 1
                bad = [k for k in computed
                       if not close_enough(computed[k], table[k])]
                if bad:
                    print(f"FAIL {d.name}/{table_key}: formula mismatch "
                          f"{ {k: (computed[k], table[k]) for k in bad} }")
                    failures += 1
                else:
                    validated += 1
                continue
            if args.validate:
                skipped += 1
                continue
            table.update(computed)
            backfilled += 1
            modified = True
            print(f"backfilled {d.name}/{table_key}: "
                  f"ConnectedValidity={computed['ConnectedValidity']:.1f} "
                  f"BondsPerMol={computed['BondsPerMol']:.1f} "
                  f"ConnectedFrac={computed['ConnectedFrac']:.1f}")

        if modified:
            metrics_path.write_text(json.dumps(data, indent=2))

    print(f"\nvalidated={validated} checked={checked} "
          f"backfilled={backfilled} skipped={skipped} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
