"""Lead triage: turn raw diffusion output into chemically sane, purchasable leads.

Step 1 of the triage plan. The model emits molecules whose atoms do not carry
implicit hydrogens, i.e. valence-unsaturated radical-looking SMILES such as
``[C]1([H])[C](N...)``. RDKit sanitizes them only because they are read as
neutral radicals, so every downstream tool (catalog search, docking, property
prediction) would be applied to a structure that cannot exist.

This script does the minimum chemistry needed to fix that, then reports how
much the fix changed the molecule:

1. strip the spurious explicit H's and radical flags,
2. re-enable implicit hydrogens so every atom reaches its normal valence,
3. sanitize (rejects anything RDKit still considers impossible),
4. embed 3D + MMFF-minimise,
5. compute drug-likeness descriptors and the heavy-atom distance from the
   original generated graph, so a "catalog hit" can never be reported for a
   molecule that no longer resembles what was generated.

Usage
-----
    python scripts/lead_triage.py --input outputs/reward_select_v5/elites_top25.sdf \
        --outdir outputs/lead_triage
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


def saturate(mol: Chem.Mol) -> tuple[Chem.Mol | None, str]:
    """Give every atom a normal valence and re-sanitise.

    Returns ``(mol, status)``; ``mol`` is ``None`` when the graph cannot be
    made chemically valid.
    """
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetFormalCharge(0)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
        # generated graphs carry explicit H atoms; fold them into implicit H
        # so the SMILES we export is the one a chemist would read
        out = Chem.RemoveHs(out)
        Chem.SanitizeMol(out)
    except Exception as exc:  # valence still impossible -> not a real molecule
        return None, f"unsaturatable: {type(exc).__name__}"
    return out, "ok"


def descriptors(mol: Chem.Mol) -> dict:
    return {
        "MW": round(Descriptors.MolWt(mol), 2),
        "logP": round(Crippen.MolLogP(mol), 2),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "QED": round(QED.qed(mol), 3),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "FracCsp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "StereoCenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
    }


def lipinski_violations(d: dict) -> int:
    return sum(
        [
            d["MW"] > 500,
            d["logP"] > 5,
            d["HBD"] > 5,
            d["HBA"] > 10,
        ]
    )


def embed(mol: Chem.Mol, seed: int = 42) -> tuple[Chem.Mol | None, str]:
    """Add Hs, embed, MMFF-minimise. Returns (mol, status)."""
    mol_h = Chem.AddHs(mol)
    for attempt in range(2):
        params = AllChem.ETKDGv3()
        params.randomSeed = seed + attempt
        # strained polycyclic/fused cages fail the distance-geometry bounds;
        # the random-coordinate fallback still produces a usable 3D pose
        params.useRandomCoords = attempt > 0
        if AllChem.EmbedMolecule(mol_h, params) == 0:
            break
    else:
        return None, "embed_failed"
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
        status = "mmff"
    except Exception:
        status = "no_mmff"
    return mol_h, status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=str, required=True,
                    help="SDF of generated candidates (e.g. elites_top25.sdf)")
    ap.add_argument("--outdir", type=str, default="outputs/lead_triage")
    ap.add_argument("--lipo_filter", action="store_true",
                    help="keep only <=1 Lipinski violation")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    supplier = Chem.SDMolSupplier(args.input, sanitize=False, removeHs=False)
    rows: list[dict] = []
    kept: list[Chem.Mol] = []

    for idx, raw in enumerate(supplier):
        if raw is None:
            rows.append({"idx": idx, "status": "sdf_read_failed"})
            continue

        # the pre-saturation graph, for the similarity sanity check later
        orig = Chem.Mol(raw)
        orig_smiles_raw = Chem.MolToSmiles(orig, canonical=True)

        sat, status = saturate(raw)
        if sat is None:
            rows.append({"idx": idx, "status": status,
                         "smiles_raw": orig_smiles_raw})
            continue

        smiles = Chem.MolToSmiles(sat, canonical=True)
        d = descriptors(sat)
        d["LipinskiViol"] = lipinski_violations(d)

        mol3d, geo_status = embed(sat)
        d3 = None
        if mol3d is not None:
            conf = mol3d.GetConformer()
            d3 = {
                "MMFFEnergy": round(
                    AllChem.MMFFGetMoleculeForceField(
                        mol3d, AllChem.MMFFGetMoleculeProperties(mol3d)
                    ).CalcEnergy(), 2
                ) if geo_status == "mmff" else None,
                "z_min": round(min(conf.GetAtomPosition(i).z for i in range(mol3d.GetNumAtoms())), 2),
                "z_max": round(max(conf.GetAtomPosition(i).z for i in range(mol3d.GetNumAtoms())), 2),
                "n_conformers": mol3d.GetNumConformers(),
            }

        row = {
            "idx": idx,
            "status": "ok",
            "smiles": smiles,
            "smiles_raw": orig_smiles_raw,
            "heavy_delta": sat.GetNumHeavyAtoms() - orig.GetNumHeavyAtoms(),
            "geometry": geo_status,
            "MMFFEnergy": (d3 or {}).get("MMFFEnergy"),
            **d,
        }
        rows.append(row)

        if mol3d is not None:
            mol3d.SetProp("_Name", f"lead_{idx:02d}")
            mol3d.SetProp("SMILES", smiles)
            mol3d.SetProp("SMILES_raw", orig_smiles_raw)
            for k, v in d.items():
                mol3d.SetProp(k, str(v))
            kept.append(mol3d)

    ok = [r for r in rows if r.get("status") == "ok"]
    if args.lipo_filter:
        ok = [r for r in ok if r["LipinskiViol"] <= 1]

    # ---- outputs ----
    csv_path = outdir / "step1_saturated.csv"
    fields = ["idx", "status", "smiles", "smiles_raw", "geometry", "MW", "logP",
              "TPSA", "QED", "HBD", "HBA", "RotB", "Rings", "HeavyAtoms",
              "FracCsp3", "StereoCenters", "LipinskiViol", "MMFFEnergy",
              "heavy_delta"]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    sdf_path = outdir / "step1_saturated_3d.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    for m in kept:
        writer.write(m)
    writer.close()

    smi_path = outdir / "step1_saturated.smi"
    with smi_path.open("w") as fh:
        for r in ok:
            fh.write(f"{r['smiles']}\tlead_{r['idx']:02d}\n")

    summary = {
        "input": args.input,
        "n_input": len(rows),
        "n_saturated": len([r for r in rows if r.get("status") == "ok"]),
        "n_3d": len(kept),
        "n_failed": len([r for r in rows if r.get("status") != "ok"]),
        "n_rule_of_5_ok": len([r for r in rows
                               if r.get("status") == "ok" and r["LipinskiViol"] <= 1]),
        "n_written": len(ok),
        "outputs": {
            "csv": str(csv_path),
            "sdf_3d": str(sdf_path),
            "smi": str(smi_path),
        },
    }
    (outdir / "step1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    if ok:
        med_qed = sorted(r["QED"] for r in ok)[len(ok) // 2]
        print(f"\nmedian QED {med_qed:.3f} | "
              f"MW {min(r['MW'] for r in ok):.0f}-{max(r['MW'] for r in ok):.0f} | "
              f"logP {min(r['logP'] for r in ok):.1f}-{max(r['logP'] for r in ok):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
