"""MOSES-style evaluation metrics for generated molecules.

Implements all seven metrics from Section III-C of the paper:
  Validity, Uniqueness, Novelty, Internal Diversity (IntDiv_p), QED, LogP, SNN.

Also provides a bond-reconstruction helper that builds RDKit Mol objects from
predicted 3D coordinates, atom types, and bond predictions.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import rdMolDescriptors as MolDesc
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Geometry import Point3D

# QM9 atomic-number to RDKit-atom-symbol mapping.
# QM9 uses 1-indexed atomic numbers; padding symbol is 0 (→ '*').
_Z_TO_SYMBOL: dict[int, str] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N",
    8: "O", 9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si",
    15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    35: "Br", 53: "I",
}

# Bond-type index → RDKit BondType.
# Index 0 = no bond (we skip adding such bonds).
_BOND_TYPE_MAP: dict[int, Chem.BondType] = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}

# Distance thresholds for heuristic bond assignment when bond_logits are
# unavailable.  Values in Angstroms, tuned for QM9 molecules.
_DISTANCE_THRESHOLDS = [
    (1.8, Chem.BondType.SINGLE),
    (1.5, Chem.BondType.DOUBLE),
    (1.3, Chem.BondType.TRIPLE),
]


# ---------------------------------------------------------------------------
# Molecule construction
# ---------------------------------------------------------------------------

def coords_and_types_to_mol(
    pos: np.ndarray,
    atomic_numbers: np.ndarray,
    bond_logits: np.ndarray | None = None,
    distance_cutoff: float = 2.5,
    sanitize: bool = True,
) -> Chem.Mol | None:
    """Build an RDKit Mol from 3D coordinates and atom types.

    Args:
        pos: (N, 3) Cartesian coordinates in Ångströms.
        atomic_numbers: (N,) integer atomic numbers (QM9 convention).
        bond_logits: optional (P, 5) bond-type logits for candidate pairs.
            If ``None``, bonds are inferred purely from distances.
        distance_cutoff: maximum interatomic distance (Å) to consider a bond.
        sanitize: whether to call ``Chem.SanitizeMol`` after construction.

    Returns:
        An RDKit ``Mol`` or ``None`` if construction/sanitization fails.
    """
    mol = Chem.RWMol()

    # Add atoms (skip padding symbol '*')
    valid_atoms: list[int] = []  # indices into pos/atomic_numbers
    for i, z in enumerate(atomic_numbers):
        z_int = int(z)
        symbol = _Z_TO_SYMBOL.get(z_int, "*")
        if symbol == "*":
            continue  # skip padding
        atom = Chem.Atom(symbol)
        atom.SetNoImplicit(True)
        idx = mol.AddAtom(atom)
        valid_atoms.append(i)

    if mol.GetNumAtoms() == 0:
        return None

    # Determine connectivity
    if bond_logits is not None:
        _add_bonds_from_logits(mol, pos, atomic_numbers, valid_atoms,
                               bond_logits, distance_cutoff)
    else:
        _add_bonds_from_distance(mol, pos, valid_atoms, distance_cutoff)

    # Try to sanitise
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

    # Embed 3D coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for out_idx, orig_idx in enumerate(valid_atoms):
        x, y, z_pos = pos[orig_idx]
        conf.SetAtomPosition(out_idx, Point3D(float(x), float(y), float(z_pos)))
    conf.Set3D(True)
    mol.AddConformer(conf, assignId=True)
    return mol


def _add_bonds_from_logits(
    mol: Chem.RWMol,
    pos: np.ndarray,
    atomic_numbers: np.ndarray,
    valid_atoms: list[int],
    bond_logits: np.ndarray,
    distance_cutoff: float,
) -> None:
    """Add bonds using precomputed bond-type logits + distance gating."""
    N = len(valid_atoms)
    # bond_logits is expected to be (N, N, 5) or (P, 5) for candidate pairs.
    # For the general case we iterate over all valid-atom pairs.
    if bond_logits.ndim == 3:
        for i_loc in range(N):
            for j_loc in range(i_loc + 1, N):
                dist = np.linalg.norm(pos[valid_atoms[i_loc]] - pos[valid_atoms[j_loc]])
                if dist > distance_cutoff:
                    continue
                bt_idx = int(np.argmax(bond_logits[i_loc, j_loc]))
                if bt_idx == 0:
                    continue  # skip no-bond
                bt = _BOND_TYPE_MAP.get(bt_idx, Chem.BondType.SINGLE)
                mol.AddBond(i_loc, j_loc, bt)
    else:
        # bond_logits is (P, 5) for candidate pairs — fallback to distance
        _add_bonds_from_distance(mol, pos, valid_atoms, distance_cutoff)


def _add_bonds_from_distance(
    mol: Chem.RWMol,
    pos: np.ndarray,
    valid_atoms: list[int],
    distance_cutoff: float,
) -> None:
    """Heuristic bond assignment: strongest bond type within distance cutoff."""
    N = len(valid_atoms)
    for i_loc in range(N):
        for j_loc in range(i_loc + 1, N):
            dist = np.linalg.norm(pos[valid_atoms[i_loc]] - pos[valid_atoms[j_loc]])
            if dist > distance_cutoff:
                continue
            # Assign the strongest bond type whose threshold exceeds distance.
            bond_type = Chem.BondType.SINGLE  # default if within cutoff
            for thresh, bt in _DISTANCE_THRESHOLDS:
                if dist <= thresh:
                    bond_type = bt
                    break
            mol.AddBond(i_loc, j_loc, bond_type)


# ---------------------------------------------------------------------------
# MOSES metrics
# ---------------------------------------------------------------------------

def validity(mols: Sequence[Chem.Mol | None]) -> float:
    """Fraction of molecules that are not None (i.e. RDKit-parseable)."""
    valid = [m for m in mols if m is not None]
    return len(valid) / len(mols) if mols else 0.0


def uniqueness(mols: Sequence[Chem.Mol | None]) -> float:
    """Fraction of valid molecules that have unique SMILES strings."""
    valid = [m for m in mols if m is not None]
    if not valid:
        return 0.0
    smiles_set: set[str] = set()
    for m in valid:
        s = Chem.MolToSmiles(m)
        if s:
            smiles_set.add(s)
    return len(smiles_set) / len(valid)


def novelty(mols: Sequence[Chem.Mol | None], train_smiles: set[str]) -> float:
    """Fraction of valid unique molecules not in the training set."""
    valid = [m for m in mols if m is not None]
    if not valid:
        return 0.0
    novel = 0
    seen: set[str] = set()
    for m in valid:
        s = Chem.MolToSmiles(m)
        if s and s not in seen and s not in train_smiles:
            novel += 1
            seen.add(s)
    return novel / len(valid)


def internal_diversity(mols: Sequence[Chem.Mol | None], radius: int = 2) -> float:
    """Mean pairwise Tanimoto dissimilarity of Morgan fingerprints.

    IntDiv_p = 1 − (2 / N(N−1)) Σ_{i<j} sim(fp_i, fp_j).
    """
    valid = [m for m in mols if m is not None]
    if len(valid) < 2:
        return 0.0
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=2048)
        for m in valid
    ]
    total_sim = 0.0
    count = 0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            total_sim += DataStructs.TanimotoSimilarity(fps[i], fps[j])
            count += 1
    avg_sim = total_sim / count if count > 0 else 0.0
    return 1.0 - avg_sim


def mean_qed(mols: Sequence[Chem.Mol | None]) -> float:
    """Average Quantitative Estimation of Drug-likeness across valid molecules."""
    valid = [m for m in mols if m is not None]
    if not valid:
        return 0.0
    return float(np.mean([Descriptors.qed(m) for m in valid]))


def mean_logp(mols: Sequence[Chem.Mol | None]) -> float:
    """Average Crippen LogP across valid molecules."""
    valid = [m for m in mols if m is not None]
    if not valid:
        return 0.0
    return float(np.mean([Descriptors.MolLogP(m) for m in valid]))


def snn(
    mols: Sequence[Chem.Mol | None],
    train_mols: Sequence[Chem.Mol],
    radius: int = 2,
) -> float:
    """Similarity to Nearest Neighbor in training set (Tanimoto, Morgan FP).

    For each generated valid molecule, compute max Tanimoto similarity to any
    training molecule.  Return the average over generated valid molecules.
    """
    valid = [m for m in mols if m is not None]
    if not valid or not train_mols:
        return 0.0

    gen_fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=2048)
        for m in valid
    ]
    train_fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=2048)
        for m in train_mols
    ]
    similarities = []
    for gfp in gen_fps:
        best = max(DataStructs.TanimotoSimilarity(gfp, tfp) for tfp in train_fps)
        similarities.append(best)
    return float(np.mean(similarities))


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate(
    mols: Sequence[Chem.Mol | None],
    train_smiles: set[str],
    train_mols: Sequence[Chem.Mol],
) -> dict[str, float]:
    """Run the full MOSES metric suite and return a dict of results."""
    return {
        "Validity": validity(mols) * 100.0,
        "Uniqueness": uniqueness(mols) * 100.0,
        "Novelty": novelty(mols, train_smiles) * 100.0,
        "IntDiv_p": internal_diversity(mols),
        "QED": mean_qed(mols),
        "LogP": mean_logp(mols),
        "SNN": snn(mols, train_mols),
    }


def print_metrics(metrics: dict[str, float]) -> None:
    """Pretty-print the evaluation results."""
    print("=" * 50)
    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 50)
    for k, v in metrics.items():
        print(f"{k:<20} {v:>10.4f}")
    print("=" * 50)
