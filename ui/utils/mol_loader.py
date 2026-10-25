"""Data loader and caching utilities for the Molecular Visualization Suite."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdDepictor, rdmolops
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Standard covalent radii (Cordero et al., 2008)
COVALENT_RADII = {
    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39
}

DEFAULT_VALENCE_MAX = {
    0: 0, 1: 1, 6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1
}


def discover_evaluated_runs(outputs_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Scan outputs directory for evaluated runs and sort runs with molecules first."""
    if outputs_dir is None:
        outputs_dir = REPO_ROOT / "outputs"

    runs = {}

    # 1. Add Ground Truth BBBP Reference dataset
    bbbp_pt_path = REPO_ROOT / "data" / "bbbp" / "processed_3d.pt"
    if bbbp_pt_path.exists():
        runs["[Reference] BBBP Ground Truth"] = {
            "name": "[Reference] BBBP Ground Truth",
            "dir": str(bbbp_pt_path.parent),
            "metrics_path": None,
            "metrics": {
                "Validity": 100.0,
                "ConnV": 100.0,
                "ConnectedValidity": 100.0,
                "BBB%": 76.5,
                "BondsPerMol": 24.2,
                "QED": 0.54,
            },
            "sdf_path": None,
            "smiles_path": None,
            "clump_path": None,
            "has_molecules": True,
            "is_reference": True,
            "validity": 100.0,
            "conn_v": 100.0,
            "bbb_pct": 76.5,
            "bonds_per_mol": 24.2,
        }

    # 2. Discover outputs runs with metrics or SDFs
    raw_runs = {}
    for p in sorted(glob.glob(str(outputs_dir / "*" / "metrics.json"))):
        run_dir = Path(os.path.dirname(p))
        run_name = run_dir.name

        metrics = {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
                metrics = raw.get("metrics", {})
        except Exception:
            pass

        # Check SDF
        sdf_files = [f for f in run_dir.glob("molecules*.sdf") if f.stat().st_size > 0]
        sdf_path = str(sdf_files[0]) if sdf_files else None

        # Check SMILES
        smi_file = run_dir / "smiles.txt"
        smi_path = str(smi_file) if (smi_file.exists() and smi_file.stat().st_size > 0) else None

        # Check Clump
        clump_file = run_dir / "clump.json"
        clump_path = str(clump_file) if clump_file.exists() else None

        has_mols = (sdf_path is not None) or (smi_path is not None)

        raw_runs[run_name] = {
            "name": run_name,
            "dir": str(run_dir),
            "metrics_path": p,
            "metrics": metrics,
            "sdf_path": sdf_path,
            "smiles_path": smi_path,
            "clump_path": clump_path,
            "has_molecules": has_mols,
            "is_reference": False,
            "validity": metrics.get("Validity", 0.0),
            "conn_v": metrics.get("ConnV") or metrics.get("ConnectedValidity", 0.0),
            "bbb_pct": metrics.get("BBB%", 0.0),
            "bonds_per_mol": metrics.get("BondsPerMol", 0.0),
        }

    # Check for SDF runs without metrics.json
    for p in sorted(glob.glob(str(outputs_dir / "*" / "molecules*.sdf"))):
        run_dir = Path(os.path.dirname(p))
        run_name = run_dir.name
        if run_name not in raw_runs and os.path.getsize(p) > 0:
            raw_runs[run_name] = {
                "name": run_name,
                "dir": str(run_dir),
                "metrics_path": None,
                "metrics": {},
                "sdf_path": p,
                "smiles_path": str(run_dir / "smiles.txt") if (run_dir / "smiles.txt").exists() else None,
                "clump_path": str(run_dir / "clump.json") if (run_dir / "clump.json").exists() else None,
                "has_molecules": True,
                "is_reference": False,
                "validity": 0.0,
                "conn_v": 0.0,
                "bbb_pct": 0.0,
                "bonds_per_mol": 0.0,
            }

    # Sort runs: Runs with molecules first, then metrics-only runs
    mols_first = [k for k, v in raw_runs.items() if v["has_molecules"]]
    metrics_only = [k for k, v in raw_runs.items() if not v["has_molecules"]]

    for k in sorted(mols_first):
        runs[k] = raw_runs[k]
    for k in sorted(metrics_only):
        runs[k] = raw_runs[k]

    return runs


def load_bbbp_reference_molecules(max_mols: int = 150) -> List[Dict[str, Any]]:
    """Load real BBBP drug molecules with 3D conformers from data/bbbp/processed_3d.pt."""
    bbbp_pt_path = REPO_ROOT / "data" / "bbbp" / "processed_3d.pt"
    if not bbbp_pt_path.exists():
        return []

    try:
        d = torch.load(bbbp_pt_path, map_location="cpu", weights_only=False)
        items = d.get("items", [])
    except Exception:
        return []

    mols_data = []
    num_to_load = min(max_mols, len(items))
    for idx in range(num_to_load):
        item = items[idx]
        smi = getattr(item, "smiles", None)
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        mol_h = Chem.AddHs(mol)
        num_atoms = mol_h.GetNumAtoms()
        pos_tensor = item.pos.numpy()

        conf = Chem.Conformer(num_atoms)
        for i in range(min(num_atoms, pos_tensor.shape[0])):
            conf.SetAtomPosition(i, [float(c) for c in pos_tensor[i]])
        mol_h.AddConformer(conf, assignId=True)

        sdf_block = Chem.MolToMolBlock(mol_h)
        qed_val = float(getattr(item, "qed", 0.5))
        logp_val = float(getattr(item, "logp", 2.0))
        mw_val = float(getattr(item, "mw", 250.0))
        y_label = int(getattr(item, "y", 1))

        mols_data.append({
            "index": idx,
            "smiles": smi,
            "is_valid": True,
            "is_connected": True,
            "num_atoms": num_atoms,
            "num_bonds": mol_h.GetNumBonds(),
            "num_frags": 1,
            "radius_of_gyration": round(float(np.sqrt(np.mean(np.sum((pos_tensor - pos_tensor.mean(0)) ** 2, 1)))), 3),
            "valence_violations": 0,
            "qed": round(qed_val, 3),
            "logp": round(logp_val, 2),
            "mw": round(mw_val, 1),
            "bbb_prob": 1.0 if y_label == 1 else 0.0,
            "positions": pos_tensor[:num_atoms],
            "sdf_block": sdf_block,
            "mol": mol_h,
        })

    return mols_data


def load_molecules_from_sdf(sdf_path: str, max_mols: int = 500) -> List[Dict[str, Any]]:
    """Load 3D conformers from SDF file and compute per-molecule diagnostics."""
    if not os.path.exists(sdf_path) or os.path.getsize(sdf_path) == 0:
        return []

    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    mols_data = []

    for idx, mol in enumerate(suppl):
        if idx >= max_mols:
            break
        if mol is None:
            continue

        is_valid = True
        try:
            mol_copy = Chem.Mol(mol)
            Chem.SanitizeMol(mol_copy)
            can_smi = Chem.MolToSmiles(mol_copy)
        except Exception:
            is_valid = False
            try:
                can_smi = Chem.MolToSmiles(mol)
            except Exception:
                can_smi = f"mol_{idx}"

        frags = rdmolops.GetMolFrags(mol, asMols=False)
        num_frags = len(frags)
        is_connected = (num_frags == 1)

        conf = mol.GetConformer()
        positions = conf.GetPositions()
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        valence_violations = 0
        for atom in mol.GetAtoms():
            z = atom.GetAtomicNum()
            v_max = DEFAULT_VALENCE_MAX.get(z, 4)
            val = atom.GetExplicitValence() + atom.GetImplicitValence()
            if val > v_max:
                valence_violations += 1

        center = positions.mean(axis=0)
        rg = float(np.sqrt(np.mean(np.sum((positions - center) ** 2, axis=1))))

        qed_val = 0.0
        logp_val = 0.0
        mw = 0.0
        if is_valid:
            try:
                qed_val = float(QED.qed(mol_copy))
                logp_val = float(Descriptors.MolLogP(mol_copy))
                mw = float(Descriptors.MolWt(mol_copy))
            except Exception:
                pass

        sdf_block = Chem.MolToMolBlock(mol)

        mols_data.append({
            "index": idx,
            "smiles": can_smi,
            "is_valid": is_valid,
            "is_connected": is_connected,
            "num_atoms": num_atoms,
            "num_bonds": num_bonds,
            "num_frags": num_frags,
            "radius_of_gyration": round(rg, 3),
            "valence_violations": valence_violations,
            "qed": round(qed_val, 3),
            "logp": round(logp_val, 2),
            "mw": round(mw, 1),
            "positions": positions,
            "sdf_block": sdf_block,
            "mol": mol,
        })

    return mols_data


def load_molecules_for_run(run_info: Dict[str, Any], max_mols: int = 400) -> List[Dict[str, Any]]:
    """Universal molecule loader for any run, supporting SDF, SMILES, or Reference BBBP."""
    if run_info.get("is_reference"):
        return load_bbbp_reference_molecules(max_mols=max_mols)

    sdf_path = run_info.get("sdf_path")
    if sdf_path and os.path.exists(sdf_path) and os.path.getsize(sdf_path) > 0:
        return load_molecules_from_sdf(sdf_path, max_mols=max_mols)

    smi_path = run_info.get("smiles_path")
    if smi_path and os.path.exists(smi_path) and os.path.getsize(smi_path) > 0:
        smis = load_smiles_list(smi_path, max_lines=max_mols)
        mols = []
        for idx, smi in enumerate(smis):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol_h = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol_h, randomSeed=42)
            conf = mol_h.GetConformer() if mol_h.GetNumConformers() > 0 else None
            pos = conf.GetPositions() if conf else np.zeros((mol_h.GetNumAtoms(), 3))
            sdf_block = Chem.MolToMolBlock(mol_h) if conf else None
            frags = rdmolops.GetMolFrags(mol, asMols=False)

            mols.append({
                "index": idx,
                "smiles": smi,
                "is_valid": True,
                "is_connected": len(frags) == 1,
                "num_atoms": mol_h.GetNumAtoms(),
                "num_bonds": mol_h.GetNumBonds(),
                "num_frags": len(frags),
                "radius_of_gyration": round(float(np.sqrt(np.mean(np.sum((pos - pos.mean(0)) ** 2, 1)))), 3) if conf else 0.0,
                "valence_violations": 0,
                "qed": round(float(QED.qed(mol)), 3),
                "logp": round(float(Descriptors.MolLogP(mol)), 2),
                "mw": round(float(Descriptors.MolWt(mol)), 1),
                "positions": pos,
                "sdf_block": sdf_block,
                "mol": mol_h,
            })
        return mols

    return []


def load_smiles_list(smiles_path: str, max_lines: int = 1000) -> List[str]:
    """Load SMILES strings from file."""
    if not os.path.exists(smiles_path):
        return []
    with open(smiles_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()][:max_lines]


def load_clump_metrics(clump_path: str) -> Optional[Dict[str, Any]]:
    """Load clump diagnostic JSON."""
    if not clump_path or not os.path.exists(clump_path):
        return None
    try:
        with open(clump_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def generate_mock_trajectory_step(
    base_pos: np.ndarray,
    step_frac: float,
    seed: int = 42
) -> np.ndarray:
    """Simulate intermediate DDIM reverse-diffusion coordinates from noise to clean."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*base_pos.shape) * 3.0
    current_pos = (1.0 - step_frac) * noise + step_frac * base_pos
    return current_pos - current_pos.mean(axis=0, keepdims=True)
