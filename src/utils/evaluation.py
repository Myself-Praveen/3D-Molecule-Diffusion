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

# Covalent radii (Cordero et al. 2008) in Å for the QM9 element set plus
# common extras. A pair is bonded iff ``dist <= r_i + r_j + tolerance``.
_COVALENT_RADII: dict[int, float] = {
    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39,
}
_BOND_TOLERANCE: float = 0.45

# Strategy 0.4 (docs/recommendation.md): per-element-pair bond tolerances.
# The scalar 0.45 was tuned to maximize ground-truth QM9 validity globally;
# element-pair-specific values recover borderline generated molecules:
# tighter where false positives dominate (H–H contacts bond spuriously),
# looser where false negatives do (C=O double bonds at ~1.23 Å sit close to
# the radius sum 1.42 + 0.45 cutoff). Keys are sorted ``(z_lo, z_hi)``;
# pairs not listed fall back to the scalar default. The values below encode
# the doc's qualitative guidance — recalibrate per pair on QM9 ground truth
# by maximizing per-pair bond recovery (see recommendation.md §0.4).
TOLERANCE_MATRIX: dict[tuple[int, int], float] = {
    (1, 1): 0.25,  # H–H: false positives (methyl H···H ≈ 1.78 Å contacts)
    (1, 6): 0.40,
    (1, 7): 0.40,
    (1, 8): 0.40,
    (1, 9): 0.35,
    (6, 6): 0.45,  # C–C: the calibrated scalar default
    (6, 7): 0.45,
    (6, 8): 0.50,  # C–O: false negatives (carbonyl C=O ≈ 1.21–1.23 Å)
    (6, 9): 0.40,
    (7, 7): 0.45,
    (7, 8): 0.45,
    (7, 9): 0.40,
    (8, 8): 0.40,
    (8, 9): 0.40,
    (9, 9): 0.40,
}


def get_bond_tolerance(z_i: int, z_j: int) -> float:
    """Tolerance for element pair ``(z_i, z_j)`` (Strategy 0.4).

    Symmetric in argument order; unlisted pairs (heavier elements) fall back
    to the scalar ``_BOND_TOLERANCE``.
    """
    key = (min(int(z_i), int(z_j)), max(int(z_i), int(z_j)))
    return TOLERANCE_MATRIX.get(key, _BOND_TOLERANCE)


# ---------------------------------------------------------------------------
# Molecule construction
# ---------------------------------------------------------------------------

def coords_and_types_to_mol(
    pos: np.ndarray,
    atomic_numbers: np.ndarray,
    bond_logits: np.ndarray | None = None,
    distance_cutoff: float = 2.5,
    sanitize: bool = True,
    min_fragment_atoms: int = 1,
) -> Chem.Mol | None:
    """Build an RDKit Mol from 3D coordinates and atom types.

    Args:
        pos: (N, 3) Cartesian coordinates in Ångströms.
        atomic_numbers: (N,) integer atomic numbers (QM9 convention).
        bond_logits: optional (P, 5) bond-type logits for candidate pairs.
            If ``None``, bonds are inferred purely from distances.
        distance_cutoff: maximum interatomic distance (Å) to consider a bond.
        sanitize: whether to call ``Chem.SanitizeMol`` after construction.
        min_fragment_atoms: connectivity post-processing (recommendation.md
            "Improving Connected Validity"): before bond assignment, drop
            provisional distance-graph fragments smaller than this (1 = off,
            legacy behavior; 2 removes isolated atoms; 3 also removes pairs).
            Unsupported together with dense (N, N, 5) ``bond_logits`` — that
            combination falls back to distance-based bonding on the kept
            atoms.

    Returns:
        An RDKit ``Mol`` or ``None`` if construction/sanitization fails.
    """
    valid_atoms: list[int] = [  # indices into pos/atomic_numbers
        i for i, z in enumerate(atomic_numbers)
        if _Z_TO_SYMBOL.get(int(z), "*") != "*"
    ]
    if not valid_atoms:
        return None

    # Connectivity post-processing: prune small provisional fragments BEFORE
    # bond assignment so disconnected specks never enter the molecule.
    if min_fragment_atoms > 1:
        kept = _keep_large_fragments(
            pos, atomic_numbers, valid_atoms, min_fragment_atoms, distance_cutoff,
        )
        if len(kept) != len(valid_atoms) and bond_logits is not None \
                and getattr(bond_logits, "ndim", 0) == 3:
            bond_logits = None  # dense logits index the UNpruned atom set
        valid_atoms = kept
        if not valid_atoms:
            return None

    mol = Chem.RWMol()
    for i in valid_atoms:
        atom = Chem.Atom(_Z_TO_SYMBOL[int(atomic_numbers[i])])
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)

    if mol.GetNumAtoms() == 0:
        return None

    # Determine connectivity
    if bond_logits is not None:
        _add_bonds_from_logits(mol, pos, atomic_numbers, valid_atoms,
                               bond_logits, distance_cutoff)
    else:
        _add_bonds_from_distance(mol, pos, atomic_numbers, valid_atoms,
                                 distance_cutoff)

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
    atomic_numbers: np.ndarray,
    valid_atoms: list[int],
    distance_cutoff: float,
) -> None:
    """Element-aware bond assignment via covalent radii + tolerance matrix.

    A pair is bonded iff ``dist <= r_i + r_j + get_bond_tolerance(i, j)``
    (and within the hard ``distance_cutoff`` cap) — Strategy 0.4's
    per-element-pair matrix replaces the old global +0.45 Å scalar. This
    excludes the contacts a plain distance cutoff bonds spuriously — methyl
    H···H ≈ 1.78 Å, benzene 1-3 C···C ≈ 2.42 Å — which made even REAL QM9
    molecules fail sanitization (ground-truth control: 0.7% valid at cutoff
    2.5 Å, 20.7% at 1.8 Å).

    Bond order is NOT inferable from distance alone (C–H 1.09 Å vs C≡C 1.20 Å
    overlap), so distance-derived bonds are SINGLE; bond types come from the
    bond head via ``_add_bonds_from_logits`` when available. Single-bond
    topology preserves valence legality (e.g. ethene as all-single has C
    valence 3 ≤ 4) while reproducing the correct molecular graph skeleton.
    """
    N = len(valid_atoms)
    for i_loc in range(N):
        for j_loc in range(i_loc + 1, N):
            i_orig, j_orig = valid_atoms[i_loc], valid_atoms[j_loc]
            dist = float(np.linalg.norm(pos[i_orig] - pos[j_orig]))
            if dist > distance_cutoff:
                continue
            ri = _COVALENT_RADII.get(int(atomic_numbers[i_orig]))
            rj = _COVALENT_RADII.get(int(atomic_numbers[j_orig]))
            if ri is None or rj is None:
                continue
            if dist <= ri + rj + get_bond_tolerance(
                    int(atomic_numbers[i_orig]), int(atomic_numbers[j_orig])):
                mol.AddBond(i_loc, j_loc, Chem.BondType.SINGLE)


def _keep_large_fragments(
    pos: np.ndarray,
    atomic_numbers: np.ndarray,
    valid_atoms: list[int],
    min_fragment_atoms: int,
    distance_cutoff: float,
) -> list[int]:
    """Indices of provisional fragments with ``>= min_fragment_atoms`` atoms.

    Connectivity post-processing (recommendation.md, "Improving Connected
    Validity" #2): build a provisional distance bond graph over the kept
    atoms (same covalent-radii + tolerance-matrix rule), take its connected
    components, and keep only fragments of at least ``min_fragment_atoms``
    atoms. Removes isolated atoms (and, at threshold 3, spurious pairs)
    before bond assignment.
    """
    n = len(valid_atoms)
    parent = list(range(n))  # union-find over positions in valid_atoms

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(n):
        for b in range(a + 1, n):
            i_orig, j_orig = valid_atoms[a], valid_atoms[b]
            dist = float(np.linalg.norm(pos[i_orig] - pos[j_orig]))
            if dist > distance_cutoff:
                continue
            ri = _COVALENT_RADII.get(int(atomic_numbers[i_orig]))
            rj = _COVALENT_RADII.get(int(atomic_numbers[j_orig]))
            if ri is None or rj is None:
                continue
            if dist <= ri + rj + get_bond_tolerance(
                    int(atomic_numbers[i_orig]), int(atomic_numbers[j_orig])):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

    sizes: dict[int, int] = {}
    for a in range(n):
        sizes[find(a)] = sizes.get(find(a), 0) + 1
    return [
        valid_atoms[a] for a in range(n) if sizes[find(a)] >= min_fragment_atoms
    ]


def apply_qed_filter(
    mols: Sequence[Chem.Mol | None], min_qed: float
) -> tuple[list[Chem.Mol | None], int]:
    """Rejection-sampling filter on QED (recommendation.md, "Improving QED" #3).

    Returns ``(filtered, n_dropped)`` where molecules with ``QED < min_qed``
    (and ``None`` entries) are replaced by ``None`` — positions are kept so
    per-sample rates stay honest. ``min_qed <= 0`` disables the filter.
    """
    if min_qed <= 0.0:
        return list(mols), 0
    filtered: list[Chem.Mol | None] = []
    dropped = 0
    for m in mols:
        keep = False
        if m is not None:
            try:
                keep = Descriptors.qed(m) >= min_qed
            except Exception:
                keep = False
        filtered.append(m if keep else None)
        dropped += 0 if keep else 1
    return filtered, dropped


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

def relax_molecule(mol: Chem.Mol | None, max_iters: int = 200) -> Chem.Mol | None:
    """Force-field relaxation of a generated molecule (Strategy 1).

    Runs MMFF94 minimization on a copy (falling back to UFF), snapping
    near-miss coordinates into chemically valid distances. Returns the
    relaxed copy, the original on force-field failure, or ``None`` for
    ``None`` input. Standard post-processing in 3D generation papers —
    always report raw AND relaxed validity side by side.
    """
    if mol is None:
        return None
    mol = Chem.Mol(mol)
    try:
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
    except Exception:
        ff = None
    if ff is None:
        try:
            ff = AllChem.UFFGetMoleculeForceField(mol)
        except Exception:
            return mol
    try:
        ff.Minimize(maxIts=max_iters)
    except Exception:
        pass
    return mol


def connectivity(mols: Sequence[Chem.Mol | None]) -> dict[str, float]:
    """Fragment-aware geometry metrics over valid molecules.

    Plain ``Validity`` (RDKit sanitization) passes disconnected single atoms,
    so a model emitting fragment soup can score ~99% while generating zero
    real molecules. These metrics close that loophole:

    - ``ConnectedValidity``: % of ALL generated molecules that are valid AND
      a single connected fragment (the honest headline number).
    - ``BondsPerMol``: mean bond count over valid molecules (QM9 truth ~19
      for 18-atom molecules; fragments score ~0-2, clumps ~26+).
    - ``ConnectedFrac``: % of valid molecules that are single-fragment.
    """
    valid = [m for m in mols if m is not None]
    total = len(mols)
    if not valid or total == 0:
        return {"ConnectedValidity": 0.0, "BondsPerMol": 0.0, "ConnectedFrac": 0.0}
    bonds, connected = [], 0
    for m in valid:
        try:
            bonds.append(m.GetNumBonds())
            if len(Chem.GetMolFrags(m)) == 1:
                connected += 1
        except Exception:
            bonds.append(0)
    return {
        "ConnectedValidity": 100.0 * connected / total,
        "BondsPerMol": float(sum(bonds) / len(bonds)),
        "ConnectedFrac": 100.0 * connected / len(valid),
    }


# ---------------------------------------------------------------------------
# Phase 4: BBB-specific metrics (implementation.md)
# ---------------------------------------------------------------------------

def _murcko_scaffolds(mols) -> list[str]:
    """Bemis-Murcko scaffold SMILES for valid molecules ('' if acyclic)."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    out = []
    for m in mols:
        if m is None:
            continue
        try:
            out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:
            out.append("__invalid__")
    return out


def bbb_permeability_rate(mols, classifier) -> float:
    """Fraction of valid generated molecules predicted BBB-permeable (p > 0.5)."""
    valid = [m for m in mols if m is not None]
    if not valid or classifier is None:
        return 0.0
    try:
        probs = classifier.predict_batch(valid)
    except Exception:
        return 0.0
    return float(sum(1 for p in probs if p > 0.5) / len(valid))


def scaffold_diversity(mols) -> float:
    """Unique Bemis-Murcko scaffolds / valid molecules."""
    valid = [m for m in mols if m is not None]
    if not valid:
        return 0.0
    return len(set(_murcko_scaffolds(valid))) / len(valid)


def scaffold_coverage(mols, train_mols) -> float:
    """Fraction of training-set scaffolds reproduced in generated molecules."""
    train_scaffs = set(_murcko_scaffolds(train_mols)) - {"__invalid__"}
    if not train_scaffs:
        return 0.0
    gen_scaffs = set(_murcko_scaffolds(m for m in mols if m is not None))
    return len(gen_scaffs & train_scaffs) / len(train_scaffs)


def lipinski_pass_rate(mols) -> float:
    """Fraction of valid molecules passing Lipinski's Rule of Five."""
    valid = [m for m in mols if m is not None]
    if not valid:
        return 0.0
    ok = 0
    for m in valid:
        try:
            if (Descriptors.MolWt(m) <= 500
                    and Descriptors.MolLogP(m) <= 5
                    and rdMolDescriptors.CalcNumLipinskiHBD(m) <= 5
                    and rdMolDescriptors.CalcNumLipinskiHBA(m) <= 10):
                ok += 1
        except Exception:
            continue
    return ok / len(valid)


def veber_pass_rate(mols) -> float:
    """Fraction passing Veber rules (tPSA ≤ 140, rotatable bonds ≤ 10)."""
    valid = [m for m in mols if m is not None]
    if not valid:
        return 0.0
    ok = 0
    for m in valid:
        try:
            if (rdMolDescriptors.CalcTPSA(m) <= 140
                    and rdMolDescriptors.CalcNumRotatableBonds(m) <= 10):
                ok += 1
        except Exception:
            continue
    return ok / len(valid)


def _ramp(x: float, low: float, high: float, invert: bool = False) -> float:
    """Piecewise-linear 0-1 desirability ramp between low and high."""
    if high <= low:
        return 1.0 if x <= low else 0.0
    s = (x - low) / (high - low)
    s = max(0.0, min(1.0, s))
    return 1.0 - s if invert else s


def cns_mpo_score(mols) -> float:
    """Average CNS MPO score (Wager et al., 2010) over valid molecules.

    Sums four Wager desirability functions computable in RDKit —
    MW (1.0 ≤360 → 0 at 500), cLogP (1.0 ≤3 → 0 at 5),
    HBD (1.0 at 0 → 0 at ≥2), tPSA (1.0 in 40–90, ramps to 0 at 20/140).
    Range 0–4 (the pKa and CLogD terms need proprietary predictors and are
    omitted; add ~2× for the rough 0–6 equivalent). Higher = BBB-favorable.
    """
    valid = [m for m in mols if m is not None]
    if not valid:
        return 0.0
    scores = []
    for m in valid:
        try:
            mw = Descriptors.MolWt(m)
            logp = Descriptors.MolLogP(m)
            hbd = rdMolDescriptors.CalcNumLipinskiHBD(m)
            tpsa = rdMolDescriptors.CalcTPSA(m)
            if 40.0 <= tpsa <= 90.0:
                tpsa_s = 1.0
            elif 20.0 <= tpsa < 40.0:
                tpsa_s = (tpsa - 20.0) / 20.0
            elif 90.0 < tpsa <= 140.0:
                tpsa_s = (140.0 - tpsa) / 50.0
            else:
                tpsa_s = 0.0
            scores.append(
                _ramp(mw, 360.0, 500.0, invert=True)
                + _ramp(logp, 3.0, 5.0, invert=True)
                + max(0.0, 1.0 - 0.5 * hbd)
                + tpsa_s
            )
        except Exception:
            continue
    return float(sum(scores) / len(scores)) if scores else 0.0


def evaluate(
    mols: Sequence[Chem.Mol | None],
    train_smiles: set[str],
    train_mols: Sequence[Chem.Mol],
    bbb_classifier=None,
) -> dict[str, float]:
    """Run the full MOSES + BBB metric suite and return a dict of results."""
    return {
        "Validity": validity(mols) * 100.0,
        "Uniqueness": uniqueness(mols) * 100.0,
        "Novelty": novelty(mols, train_smiles) * 100.0,
        "IntDiv_p": internal_diversity(mols),
        "QED": mean_qed(mols),
        "LogP": mean_logp(mols),
        "SNN": snn(mols, train_mols),
        # Phase 4 BBB-specific metrics (-1 sentinel when no oracle given).
        "BBB%": (bbb_permeability_rate(mols, bbb_classifier) * 100.0
                 if bbb_classifier is not None else -1.0),
        "ScaffDiv": scaffold_diversity(mols),
        "ScaffCov": scaffold_coverage(mols, train_mols),
        "Lipinski%": lipinski_pass_rate(mols) * 100.0,
        "Veber%": veber_pass_rate(mols) * 100.0,
        "CNS_MPO": cns_mpo_score(mols),
    }


def print_metrics(metrics: dict[str, float]) -> None:
    """Pretty-print the evaluation results."""
    print("=" * 50)
    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 50)
    for k, v in metrics.items():
        print(f"{k:<20} {v:>10.4f}")
    print("=" * 50)
