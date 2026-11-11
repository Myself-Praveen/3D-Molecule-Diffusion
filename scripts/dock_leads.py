"""Step 3 of lead triage: dock the saturated leads into a CNS target.

Runs the standard AutoDock Vina protocol end to end, so the numbers are
reproducible rather than a single opaque run:

* receptor: PDB straight from RCSB, waters and crystallisation additives
  stripped, converted to PDBQT by ``mk_prepare_receptor.py`` (meeko),
* box: centred on the co-crystallised inhibitor (default 4EY7 / donepezil,
  the AChE gorge), configurable for the other targets in the triage plan,
* ligands: the step-1 3D SDF, converted per-molecule by
  ``mk_prepare_ligand.py`` with Gasteiger charges,
* scoring: best-pose Vina affinity, kept together with the pose file so the
  contacts can be inspected.

Usage
-----
    python scripts/dock_leads.py \
        --sdf outputs/lead_triage/step1_saturated_3d.sdf \
        --receptor_pdb outputs/lead_triage/docking/4EY7.pdb \
        --ligand_resname E20 --box 24 \
        --outdir outputs/lead_triage/docking
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

VINA_BIN = Path(".tools/vina_1.2.5")
AFFINITY_RE = re.compile(r"REMARK VINA RESULT:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


def run(cmd: list[str], log: Path) -> int:
    with log.open("a") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)


def strip_hetero(pdb_in: Path, pdb_out: Path, keep_resname: str) -> int:
    """Keep protein ATOM records only (drop water, ions, sugars, the ligand)."""
    kept = 0
    with pdb_in.open() as src, pdb_out.open("w") as dst:
        for line in src:
            tag = line[:6]
            if tag.startswith("ATOM") or tag.startswith("TER") or tag.startswith("END"):
                dst.write(line)
                kept += 1
    return kept


def ligand_centroid(pdb: Path, resname: str) -> tuple[float, float, float] | None:
    xs = ys = zs = 0.0
    n = 0
    for line in pdb.read_text().splitlines():
        if line.startswith("HETATM") and line[17:20].strip() == resname:
            xs += float(line[30:38]); ys += float(line[38:46]); zs += float(line[46:54])
            n += 1
    if not n:
        return None
    return xs / n, ys / n, zs / n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sdf", type=str, required=True)
    ap.add_argument("--receptor_pdb", type=str, required=True)
    ap.add_argument("--ligand_resname", type=str, default="E20",
                    help="co-crystallised ligand used to place the box")
    ap.add_argument("--box", type=float, default=24.0, help="box edge in Angstrom")
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--cpu", type=int, default=8)
    ap.add_argument("--outdir", type=str, default="outputs/lead_triage/docking")
    ap.add_argument("--limit", type=int, default=0, help="0 = all molecules")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (poses_dir := outdir / "poses").mkdir(exist_ok=True)
    (lig_dir := outdir / "ligands_pdbqt").mkdir(exist_ok=True)
    log = outdir / "docking.log"
    log.write_text("")

    if not VINA_BIN.exists():
        print(f"missing vina binary at {VINA_BIN}", file=sys.stderr)
        return 1

    # ---- receptor ----
    rec_pdb = Path(args.receptor_pdb)
    clean_pdb = outdir / f"{rec_pdb.stem}_protein.pdb"
    n_atoms = strip_hetero(rec_pdb, clean_pdb, args.ligand_resname)
    center = ligand_centroid(rec_pdb, args.ligand_resname)
    if center is None:
        print(f"ligand {args.ligand_resname} not found in {rec_pdb}", file=sys.stderr)
        return 1
    print(f"receptor: {n_atoms} atom lines, box centre "
          f"({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), edge {args.box} A")

    rec_pdbqt = outdir / f"{rec_pdb.stem}_receptor.pdbqt"
    box_file = outdir / f"{rec_pdb.stem}_vina_box.txt"
    if not rec_pdbqt.exists() or not box_file.exists():
        rc = run([".venv/bin/mk_prepare_receptor.py" if Path(".venv/bin/mk_prepare_receptor.py").exists()
                  else "mk_prepare_receptor.py",
                  "--read_pdb", str(clean_pdb),
                  "-o", str(outdir / f"{rec_pdb.stem}_receptor"),
                  "-p", "-v",
                  # crystal structures carry alternate conformers; picking A
                  # keeps one copy per residue instead of failing the whole prep
                  "--default_altloc", "A",
                  "--box_center", *[f"{c:.3f}" for c in center],
                  "--box_size", f"{args.box:.1f}", f"{args.box:.1f}", f"{args.box:.1f}"],
                 log)
        if rc != 0:
            print(f"receptor prep failed, see {log}", file=sys.stderr)
            return rc
    print(f"receptor pdbqt: {rec_pdbqt}")

    # ---- ligands ----
    supplier = Chem.SDMolSupplier(str(args.sdf), removeHs=False)
    rows = []
    for idx, mol in enumerate(supplier):
        if mol is None:
            continue
        if args.limit and len(rows) >= args.limit:
            break
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"lead_{idx:02d}"
        smiles = mol.GetProp("SMILES") if mol.HasProp("SMILES") else Chem.MolToSmiles(
            Chem.RemoveHs(mol))
        lig_pdbqt = lig_dir / f"{name}.pdbqt"
        if not lig_pdbqt.exists():
            single = outdir / "ligands_pdbqt" / f"{name}.sdf"
            w = Chem.SDWriter(str(single))
            w.write(mol)
            w.close()
            rc = run([".venv/bin/mk_prepare_ligand.py", "-i", str(single),
                      "-o", str(lig_pdbqt)], log)
            if rc != 0 or not lig_pdbqt.exists():
                rows.append({"name": name, "smiles": smiles, "status": "ligand_prep_failed"})
                print(f"{name}: ligand prep failed")
                continue

        pose = poses_dir / f"{name}_out.pdbqt"
        rc = run([str(VINA_BIN),
                  "--receptor", str(rec_pdbqt),
                  "--ligand", str(lig_pdbqt),
                  "--center_x", f"{center[0]:.3f}",
                  "--center_y", f"{center[1]:.3f}",
                  "--center_z", f"{center[2]:.3f}",
                  "--size_x", f"{args.box:.1f}",
                  "--size_y", f"{args.box:.1f}",
                  "--size_z", f"{args.box:.1f}",
                  "--exhaustiveness", str(args.exhaustiveness),
                  "--cpu", str(args.cpu),
                  "--num_modes", "9",
                  "--seed", "42",
                  "--out", str(pose)], log)
        if rc != 0 or not pose.exists():
            rows.append({"name": name, "smiles": smiles, "status": "vina_failed"})
            print(f"{name}: vina failed")
            continue

        m = AFFINITY_RE.search(pose.read_text()[:2000])
        if not m:
            rows.append({"name": name, "smiles": smiles, "status": "no_affinity"})
            continue
        aff, lb, ub = (float(x) for x in m.groups())
        n_modes = pose.read_text().count("MODEL ")
        rows.append({
            "name": name, "smiles": smiles, "status": "ok",
            "affinity": aff, "rmsd_lb": lb, "rmsd_ub": ub,
            "n_modes": n_modes, "pose": str(pose),
        })
        print(f"{name}: {aff:.2f} kcal/mol ({n_modes} modes)")

    ok = sorted([r for r in rows if r["status"] == "ok"],
                key=lambda r: r["affinity"])
    csv_path = outdir / "step3_docking.csv"
    fields = ["name", "smiles", "status", "affinity", "rmsd_lb", "rmsd_ub",
              "n_modes", "pose"]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "receptor_pdb": str(rec_pdb),
        "box_center": [round(c, 2) for c in center],
        "box_edge": args.box,
        "exhaustiveness": args.exhaustiveness,
        "n_docked": len(ok),
        "n_failed": len(rows) - len(ok),
        "best_affinity": ok[0]["affinity"] if ok else None,
        "median_affinity": (sorted(r["affinity"] for r in ok)[len(ok) // 2]
                            if ok else None),
        "outputs": {"csv": str(csv_path), "poses": str(poses_dir)},
    }
    (outdir / "step3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
