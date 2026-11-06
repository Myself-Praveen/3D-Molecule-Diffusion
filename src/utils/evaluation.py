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

from src.guidance import COVALENT_RADII as _COVALENT_RADII

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
# The table itself lives in ``src/guidance.py`` (imported above and re-exported
# here) so the test-time steric guidance and the ruler that scores its output
# can never disagree about an element's radius.
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
    # BBBP-era elements (S, Cl, Br): pairs the QM9 calibration never saw,
    # so they previously fell back to the 0.45 scalar. Values are tighter
    # than the scalar (heavier atoms have longer radii sums, so a generous
    # scalar over-bonds their contacts) while leaving every QM9-era pair —
    # and therefore every QM9 eval — bit-for-bit unchanged.
    (6, 16): 0.40,  # C–S
    (6, 17): 0.38,  # C–Cl
    (6, 35): 0.40,  # C–Br
    (7, 16): 0.38,  # N–S
    (16, 16): 0.35,  # S–S
}


def get_bond_tolerance(z_i: int, z_j: int,
                       cap: float | None = None) -> float:
    """Tolerance for element pair ``(z_i, z_j)`` (Strategy 0.4).

    Symmetric in argument order; unlisted pairs (heavier elements) fall back
    to the scalar ``_BOND_TOLERANCE``.

    ``cap`` (plan workstream A2) tighten-caps the result: ``min(base, cap)``.
    Tighten-only by construction — a cap can never *loosen* a calibrated
    pair (H–H stays 0.25 under cap 0.35), and ``cap=None`` reproduces the
    legacy behaviour exactly, so Table I rows remain regenerable.
    """
    key = (min(int(z_i), int(z_j)), max(int(z_i), int(z_j)))
    base = TOLERANCE_MATRIX.get(key, _BOND_TOLERANCE)
    if cap is not None and cap < base:
        return float(cap)
    return base


# ---------------------------------------------------------------------------
# Reporting contract (fix.md F5-F8): counts, CIs, size-normalized geometry
# ---------------------------------------------------------------------------

#: Bonds per atom of real BBBP molecules, measured by this repo's own ruler
#: (`load_bbbp` -> `coords_and_types_to_mol` -> mean bonds / mean atoms).
#: 47.3 bonds / 45.9 atoms. Every bond verdict is relative to THIS, never to
#: QM9's absolute ~19 bonds/mol (fix.md F2: that comparison inverted the
#: "over-bonded soup" diagnosis — these models are UNDER-bonded).
BONDS_PER_ATOM_REFERENCE: float = 1.035

#: Means over fewer than this many valid molecules are not reported as
#: measurements (`print_metrics` renders them ``n/a (n=…)``). Rates whose
#: denominator is all generated samples stay valid at any n.
MIN_REPORTABLE_N: int = 50

#: Keys that are counts, never suppressed for small n.
COUNTS: tuple[str, ...] = ("n_valid", "n_total", "n_bbb")

#: Rates whose denominator is *all generated samples* (or a structural
#: ratio), so they remain meaningful at any n.
TOTAL_RATE_KEYS: tuple[str, ...] = (
    "Validity", "Validity_ci95", "ConnectedValidity",
    "ConnectedValidity_ci95", "ConnectedFrac", "BBB%", "BBB%_ci95",
    "BBB_yield%", "BBB_yield%_ci95", "AtomsRetained%",
    "MeanAtoms_requested", "MeanAtoms_drawn",
)

#: Canonical column order for every table / LaTeX row. A stored row can go
#: stale when a metric is patched after the fact (fix.md F8), so rows are
#: *rendered* from this order instead of frozen at eval time.
METRIC_ORDER: tuple[str, ...] = (
    "Validity", "ConnectedValidity", "BBB%", "BBB_yield%",
    "BondsPerAtom", "MeanAtoms", "AtomsRetained%", "FragmentsPerMol",
    "NN_Outside%", "QED", "LogP", "IntDiv_p", "ScaffDiv",
    "Uniqueness", "Novelty", "SNN", "ScaffCov", "Lipinski%",
    "Veber%", "CNS_MPO", "n_valid", "n_total",
)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval in percent for ``k`` successes out of ``n``.

    fix.md F6: no percentage is reported without one. Returns ``(nan, nan)``
    when ``n == 0`` (nothing was measured, which is not 0 %).
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    k = max(0, min(int(k), int(n)))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100.0 * max(0.0, centre - half),
            100.0 * min(1.0, centre + half))


def ks_two_sample(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Two-sample Kolmogorov-Smirnov statistic (no scipy dependency).

    fix.md F16: validity is a size-conditional rate, so the *surviving* size
    distribution has to be compared against the one that was requested.
    Returns ``None`` when either sample is empty.
    """
    a = sorted(float(x) for x in a)
    b = sorted(float(x) for x in b)
    if not a or not b:
        return None
    points = sorted(set(a) | set(b))
    na, nb = len(a), len(b)
    best = 0.0
    ia = ib = 0
    for x in points:
        while ia < na and a[ia] <= x:
            ia += 1
        while ib < nb and b[ib] <= x:
            ib += 1
        best = max(best, abs(ia / na - ib / nb))
    return float(best)


def wilson_ci_or_none(k: int, n: int) -> list[float] | None:
    """``wilson_ci`` as a JSON-safe list, or ``None`` when undefined."""
    lo, hi = wilson_ci(k, n)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return [lo, hi]


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
    bond_tolerance: float | None = None,
    max_bonds_ratio: float = 0.0,
    perceive_orders: bool = False,
    keep_largest_fragment: bool = False,
    stats: dict | None = None,
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
        bond_tolerance: plan workstream A2 — tighten-cap (Å) applied to the
            effective per-pair tolerance via ``min(base, cap)``. ``None``
            (default) is the legacy Table I protocol; 0.35 is the BBBP-era
            setting the plan asks for. Tighten-only: calibrated pairs can
            only stay or tighten, never loosen.
        max_bonds_ratio: plan workstream A3 — reject the molecule when
            ``n_bonds > ratio * 0.95 * n_atoms`` (0 = off, legacy). 1.4 is
            the plan's setting: real chemistry sits near ratio 1.0
            (~19 bonds / 20 heavy atoms), so a molecule far above that is
            over-bonded soup and must count as INVALID rather than inflate
            BondsPerMol. Rejection is honest rejection: it lowers validity
            for the survivors' sake and never re-labels a rejected molecule.
        perceive_orders: plan workstream A4 — after the single-bond
            topology sanitizes, attempt ``rdDetermineBonds.
            DetermineBondOrders`` (bond ORDERS only; connectivity is the
            covalent-radii graph above and is never re-perceived). On any
            failure the single-bond molecule is returned unchanged, so
            Validity/ConnectedValidity are invariant by construction while
            SMILES-derived metrics (QED, LogP, SNN, scaffolds) get real
            bond orders instead of an all-single skeleton. Only applied
            when ``sanitize=True``.
        keep_largest_fragment: plan Intervention 2's spatial-clustering
            fallback — keep only the largest provisional distance-graph
            component (satellite specks are pruned), regardless of
            ``min_fragment_atoms``. Off by default; a post-hoc correction,
            not a generation fix.
        stats: optional dict filled in place with the atom accounting of this
            call (fix.md F3): ``requested_atoms``, ``kept_atoms``,
            ``pruned_atoms`` and ``pruned_fraction``. Post-processing that
            DELETES atoms (`keep_largest_fragment`, `min_fragment_atoms`) is
            otherwise invisible in every downstream metric — R5-best's 59.5 %
            "validity" came from pruning 40 % of each molecule's atoms.

    Returns:
        An RDKit ``Mol`` or ``None`` if construction/sanitization fails
        (or the ``max_bonds_ratio`` gate rejects the molecule).
    """
    requested_atoms = len(atomic_numbers)

    def _record(kept: int) -> None:
        if stats is None:
            return
        stats["requested_atoms"] = requested_atoms
        stats["kept_atoms"] = int(kept)
        stats["pruned_atoms"] = requested_atoms - int(kept)
        stats["pruned_fraction"] = (requested_atoms - int(kept)) / max(
            requested_atoms, 1)

    valid_atoms: list[int] = [  # indices into pos/atomic_numbers
        i for i, z in enumerate(atomic_numbers)
        if _Z_TO_SYMBOL.get(int(z), "*") != "*"
    ]
    if not valid_atoms:
        _record(0)
        return None

    # Connectivity post-processing: prune small provisional fragments BEFORE
    # bond assignment so disconnected specks never enter the molecule.
    if min_fragment_atoms > 1 or keep_largest_fragment:
        kept = _keep_large_fragments(
            pos, atomic_numbers, valid_atoms, min_fragment_atoms, distance_cutoff,
            bond_tolerance=bond_tolerance,
            keep_largest=keep_largest_fragment,
        )
        if len(kept) != len(valid_atoms) and bond_logits is not None \
                and getattr(bond_logits, "ndim", 0) == 3:
            bond_logits = None  # dense logits index the UNpruned atom set
        valid_atoms = kept
        if not valid_atoms:
            _record(0)
            return None

    mol = Chem.RWMol()
    for i in valid_atoms:
        atom = Chem.Atom(_Z_TO_SYMBOL[int(atomic_numbers[i])])
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)

    if mol.GetNumAtoms() == 0:
        _record(0)
        return None

    # Determine connectivity
    if bond_logits is not None:
        _add_bonds_from_logits(mol, pos, atomic_numbers, valid_atoms,
                               bond_logits, distance_cutoff)
    else:
        _add_bonds_from_distance(mol, pos, atomic_numbers, valid_atoms,
                                 distance_cutoff, bond_tolerance=bond_tolerance)

    # Try to sanitise
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            _record(0)
            return None

    # Embed 3D coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for out_idx, orig_idx in enumerate(valid_atoms):
        x, y, z_pos = pos[orig_idx]
        conf.SetAtomPosition(out_idx, Point3D(float(x), float(y), float(z_pos)))
    conf.Set3D(True)
    mol.AddConformer(conf, assignId=True)

    # A4: bond-order upgrade on a COPY so a mid-solve failure can never
    # corrupt the already-valid single-bond molecule. Runs after the
    # conformer is embedded — DetermineBondOrders uses the 3D geometry
    # (the stage-1 perception code documents the same requirement).
    if sanitize and perceive_orders:
        from rdkit.Chem import rdDetermineBonds

        attempt = Chem.RWMol(mol)
        try:
            rdDetermineBonds.DetermineBondOrders(attempt, charge=0)
            Chem.SanitizeMol(attempt)
            mol = attempt
        except Exception:
            pass  # keep the honest single-bond topology

    # A3: over-bonded rejection gate (see docstring; 0 = off).
    if max_bonds_ratio > 0.0:
        n_atoms = mol.GetNumAtoms()
        if mol.GetNumBonds() > max_bonds_ratio * 0.95 * n_atoms:
            _record(0)
            return None
    _record(mol.GetNumAtoms())
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
    bond_tolerance: float | None = None,
) -> None:
    """Element-aware bond assignment via covalent radii + tolerance matrix.

    A pair is bonded iff ``dist <= r_i + r_j + get_bond_tolerance(i, j, cap)``
    (and within the hard ``distance_cutoff`` cap) — Strategy 0.4's
    per-element-pair matrix replaces the old global +0.45 Å scalar, with the
    optional ``bond_tolerance`` tighten-cap (plan A2) applied on top. This
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
                    int(atomic_numbers[i_orig]), int(atomic_numbers[j_orig]),
                    cap=bond_tolerance):
                mol.AddBond(i_loc, j_loc, Chem.BondType.SINGLE)


def _keep_large_fragments(
    pos: np.ndarray,
    atomic_numbers: np.ndarray,
    valid_atoms: list[int],
    min_fragment_atoms: int,
    distance_cutoff: float,
    bond_tolerance: float | None = None,
    keep_largest: bool = False,
) -> list[int]:
    """Indices of provisional fragments with ``>= min_fragment_atoms`` atoms.

    Connectivity post-processing (recommendation.md, "Improving Connected
    Validity" #2): build a provisional distance bond graph over the kept
    atoms (same covalent-radii + tolerance-matrix rule), take its connected
    components, and keep only fragments of at least ``min_fragment_atoms``
    atoms. Removes isolated atoms (and, at threshold 3, spurious pairs)
    before bond assignment.

    ``keep_largest`` (plan Intervention 2, the "spatial clustering
    fallback") additionally prunes every component except the largest one, so
    a molecule with a main body plus satellite specks is reduced to its main
    body. Off by default: it changes what the molecule IS rather than how it
    is perceived, so it is a diagnostic/fallback knob, never a headline
    metric setting.
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
                    int(atomic_numbers[i_orig]), int(atomic_numbers[j_orig]),
                    cap=bond_tolerance):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

    sizes: dict[int, int] = {}
    for a in range(n):
        sizes[find(a)] = sizes.get(find(a), 0) + 1
    threshold = max(int(min_fragment_atoms), 1)
    if keep_largest and sizes:
        # Exactly one component survives: the largest, with ties broken by the
        # lowest root index so the output is deterministic. ``threshold`` is
        # still honoured, so a molecule whose every component is below
        # ``min_fragment_atoms`` prunes to nothing rather than keeping junk.
        best = max(sizes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        if sizes[best] < threshold:
            return []
        return [valid_atoms[a] for a in range(n) if find(a) == best]
    return [
        valid_atoms[a] for a in range(n) if sizes[find(a)] >= threshold
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


def apply_veber_filter(
    mols: Sequence[Chem.Mol | None],
) -> tuple[list[Chem.Mol | None], int]:
    """Rejection filter on the Veber rules (plan workstream C2).

    Returns ``(filtered, n_dropped)`` where Veber-failing molecules (and
    ``None`` entries) are replaced by ``None`` — positions are kept so
    per-sample rates stay honest. Because the filtered table's ``Veber%`` is
    100 by construction, the caller must record ``veber_pass_rate`` *before*
    calling this and report it as ``Veber_filtered%`` so the true pass rate
    stays visible next to it.
    """
    filtered: list[Chem.Mol | None] = []
    dropped = 0
    for m in mols:
        keep = m is not None and veber_pass(m)
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
    """Fraction of valid *unique* molecules not in the training set.

    MOSES definition (fix.md F14): numerator and denominator both count
    unique SMILES. The previous version counted novel molecules but divided
    by ALL valid molecules, so the number was wrong the moment duplicates
    appeared (masked today only because every arm reports 100 % uniqueness).
    """
    valid = [m for m in mols if m is not None]
    if not valid:
        return 0.0
    unique: set[str] = set()
    for m in valid:
        s = Chem.MolToSmiles(m)
        if s:
            unique.add(s)
    if not unique:
        return 0.0
    novel = sum(1 for s in unique if s not in train_smiles)
    return novel / len(unique)


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


def connectivity(mols: Sequence[Chem.Mol | None],
                 reference_bonds_per_atom: float | None = None,
                 ) -> dict[str, float | int | list[float] | None]:
    """Fragment-aware, size-normalized geometry metrics over valid molecules.

    Plain ``Validity`` (RDKit sanitization) passes disconnected single atoms,
    so a model emitting fragment soup can score ~99% while generating zero
    real molecules. These metrics close that loophole:

    - ``ConnectedValidity``: % of ALL generated molecules that are valid AND
      a single connected fragment (the honest headline number).
    - ``BondsPerAtom``: mean bonds / mean atoms over valid molecules. THIS is
      the comparable bond number (real BBBP = 1.035). ``BondsPerMol`` is an
      absolute count that only means something next to ``MeanAtoms`` — it was
      previously compared to QM9's ~19 for 18-atom molecules while these arms
      generate 46-atom BBBP molecules (fix.md F2).
    - ``BondsPerAtom_rel``: ``BondsPerAtom / reference`` (1.0 = real).
    - ``ConnectedFrac``: % of valid molecules that are single-fragment.
    - ``FragmentsPerMol``: mean connected components per valid molecule.

    ``None`` means *not measured* (no valid molecule); ``0.0`` means a
    measured zero. ``ConnectedValidity`` stays defined whenever any sample
    exists — it is a rate over ``n_total`` — but ``ConnectedFrac`` and every
    mean are undefined without a valid molecule (fix.md F10).
    """
    valid = [m for m in mols if m is not None]
    total = len(mols)
    if total == 0:
        return {
            "ConnectedValidity": None, "ConnectedValidity_ci95": None,
            "BondsPerMol": None, "BondsPerAtom": None,
            "BondsPerAtom_rel": None, "MeanAtoms": None,
            "ConnectedFrac": None, "FragmentsPerMol": None,
            "n_valid": 0, "n_total": 0,
        }
    bonds: list[int] = []
    atoms: list[int] = []
    frags: list[int] = []
    connected = 0
    for m in valid:
        try:
            atoms.append(int(m.GetNumAtoms()))
            bonds.append(int(m.GetNumBonds()))
            n_frag = len(Chem.GetMolFrags(m))
            frags.append(int(n_frag))
            connected += int(n_frag == 1)
        except Exception:
            continue

    mean_atoms = float(np.mean(atoms)) if atoms else None
    mean_bonds = float(np.mean(bonds)) if bonds else None
    bpa = (mean_bonds / mean_atoms
           if mean_bonds is not None and mean_atoms else None)
    ref = (reference_bonds_per_atom
           if reference_bonds_per_atom is not None
           else BONDS_PER_ATOM_REFERENCE)
    return {
        "ConnectedValidity": 100.0 * connected / total,
        "ConnectedValidity_ci95": wilson_ci_or_none(connected, total),
        "BondsPerMol": mean_bonds,
        "BondsPerAtom": bpa,
        "BondsPerAtom_rel": (bpa / ref) if bpa is not None and ref else None,
        "MeanAtoms": mean_atoms,
        "ConnectedFrac": (100.0 * connected / len(valid)) if valid else None,
        "FragmentsPerMol": float(np.mean(frags)) if frags else None,
        "n_valid": len(valid),
        "n_total": total,
    }


def _nearest_neighbour_stats(
    pos: np.ndarray,
    atomic_numbers: np.ndarray,
    bond_tolerance: float | None = None,
) -> tuple[float | None, float | None]:
    """``(NN-outside %, NN median distance Å)`` for one molecule.

    Returns ``(None, None)`` when the molecule has fewer than two atoms: a
    single atom has no nearest neighbour, so it cannot be scored 0 %
    (fix.md F13). Atoms whose element is absent from the covalent-radii table
    CANNOT bond by construction, so they count as outside instead of being
    skipped.
    """
    pos = np.asarray(pos, dtype=float)
    z = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    n = len(z)
    if n < 2:
        return (None, None)
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    nearest = dist.min(axis=1)
    nearest_index = dist.argmin(axis=1)

    outside = 0
    for i in range(n):
        j = int(nearest_index[i])
        ri = _COVALENT_RADII.get(int(z[i]))
        rj = _COVALENT_RADII.get(int(z[j]))
        if ri is None or rj is None:
            outside += 1  # cannot bond through the ruler at all
            continue
        window = ri + rj + get_bond_tolerance(
            int(z[i]), int(z[j]), cap=bond_tolerance)
        if nearest[i] > window:
            outside += 1
    return (100.0 * outside / n, float(np.median(nearest)))


def nn_outside_window_pct(
    pos: np.ndarray,
    atomic_numbers: np.ndarray,
    bond_tolerance: float | None = None,
) -> float:
    """% of atoms whose *nearest* neighbour lies outside the bond window.

    Rescue plan R1's geometry axis. The window is the eval's own
    ``r_i + r_j + get_bond_tolerance(i, j)``, so this ruler and the bond
    builder cannot disagree. An atom with no in-window neighbour can never
    form a bond, so it is always a fragment tip — real BBBP measures 0.1 %,
    V5 7.0 %, V6 57.4 % (rescueplan.md §1).

    Args:
        pos: ``(N, 3)`` coordinates in Å.
        atomic_numbers: ``(N,)`` integer atomic numbers.
        bond_tolerance: optional tighten-cap, exactly as the bond builder.

    Returns:
        Percentage in ``[0, 100]``; ``nan`` for fewer than two atoms or an
        empty input (undefined, not a perfect score).
    """
    pct, _ = _nearest_neighbour_stats(pos, atomic_numbers, bond_tolerance)
    return float("nan") if pct is None else pct


def geometry_metrics(
    mols: Sequence[Chem.Mol | None],
    bond_tolerance: float | None = None,
) -> dict[str, float | None]:
    """The geometry axis, inside ``metrics.json`` (fix.md F12).

    ``bonds_per_atom``, ``nn_outside_window_pct``, ``nn_p50`` and
    ``radius_of_gyration`` used to live only in ``clump.json``, produced by a
    second script that frequently was not run — so half the rows of the
    rescue summary were ``nan``. They are computed here from the embedded
    conformer (already present on every built molecule) so one eval run
    carries both rulers.

    ``NN_Outside%`` is the per-molecule mean (each molecule weighted once).
    """
    outs: list[float] = []
    p50s: list[float] = []
    rgs: list[float] = []
    for m in mols:
        if m is None:
            continue
        try:
            conf = m.GetConformer()
            n = m.GetNumAtoms()
            pos = np.array([[conf.GetAtomPosition(i).x,
                             conf.GetAtomPosition(i).y,
                             conf.GetAtomPosition(i).z] for i in range(n)])
            z = np.array([a.GetAtomicNum() for a in m.GetAtoms()], dtype=int)
        except Exception:
            continue
        pct, p50 = _nearest_neighbour_stats(pos, z, bond_tolerance)
        if pct is not None:
            outs.append(pct)
            p50s.append(p50)
            centre = pos.mean(axis=0)
            rgs.append(float(np.sqrt(((pos - centre) ** 2).sum(axis=1).mean())))
    return {
        "NN_Outside%": float(np.mean(outs)) if outs else None,
        "NN_p50": float(np.mean(p50s)) if p50s else None,
        "Rg": float(np.mean(rgs)) if rgs else None,
    }


def geometry_only_metrics(
    mols: Sequence[Chem.Mol | None],
    bond_tolerance: float | None = None,
) -> dict[str, float | None]:
    """The relaxed table's payload: geometry only (fix.md F7).

    MMFF/UFF relaxation moves coordinates and nothing else: the bond graph,
    and therefore ``Validity``, ``ConnectedValidity``, ``BondsPerMol``,
    ``QED``, ``BBB%`` and every scaffold metric, is fixed at construction.
    Reporting those keys a second time produced a byte-identical duplicate
    table whose only real effect was to let one stale copy drift
    (``eval_pre_rl``: BBB 34.17 % raw vs 0.0 % relaxed).
    """
    return geometry_metrics(mols, bond_tolerance=bond_tolerance)


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


def bbb_rates(mols, classifier) -> dict[str, float | int | list[float] | None]:
    """BBB+ rate over valid molecules AND over all generated samples.

    fix.md F4/F9: ``BBB%`` divided by the *valid* count, so two arms with 40 %
    and 5 % validity were compared on denominators that differ by 8x —
    ``34.2 %`` of 398 vs ``1.4 %`` of 503 hid a 19x difference in molecules a
    lab actually receives. ``BBB_yield%`` is the comparable number.

    ``None`` means "not measured" (no oracle); there is no ``-1`` sentinel
    that can leak into arithmetic or be swallowed by a falsy check.
    """
    valid = [m for m in mols if m is not None]
    n_total = len(mols)
    if classifier is None:
        return {"BBB%": None, "BBB%_ci95": None, "BBB_yield%": None,
                "BBB_yield%_ci95": None, "n_bbb": None,
                "n_valid": len(valid), "n_total": n_total}
    if not valid:
        return {"BBB%": None, "BBB%_ci95": None, "BBB_yield%": 0.0,
                "BBB_yield%_ci95": wilson_ci_or_none(0, n_total),
                "n_bbb": 0, "n_valid": 0, "n_total": n_total}
    try:
        probs = classifier.predict_batch(valid)
    except Exception:
        return {"BBB%": None, "BBB%_ci95": None, "BBB_yield%": None,
                "BBB_yield%_ci95": None, "n_bbb": None,
                "n_valid": len(valid), "n_total": n_total}
    k = int(sum(1 for p in probs if p > 0.5))
    return {
        "BBB%": 100.0 * k / len(valid),
        "BBB%_ci95": wilson_ci_or_none(k, len(valid)),
        "BBB_yield%": 100.0 * k / n_total if n_total else None,
        "BBB_yield%_ci95": wilson_ci_or_none(k, n_total),
        "n_bbb": k,
        "n_valid": len(valid),
        "n_total": n_total,
    }


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


def veber_pass(m: Chem.Mol) -> bool:
    """Veber rules for one molecule: tPSA ≤ 140 Å², rotatable bonds ≤ 10."""
    try:
        return (rdMolDescriptors.CalcTPSA(m) <= 140
                and rdMolDescriptors.CalcNumRotatableBonds(m) <= 10)
    except Exception:
        return False


def veber_pass_rate(mols) -> float:
    """Fraction of valid molecules passing Veber rules (tPSA ≤ 140, rot ≤ 10)."""
    valid = [m for m in mols if m is not None]
    if not valid:
        return 0.0
    ok = sum(1 for m in valid if veber_pass(m))
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
    *,
    stats: Sequence[dict] | None = None,
    reference: dict | None = None,
) -> dict:
    """Run the full MOSES + BBB + geometry metric suite (schema v2).

    Every rate carries its denominator (and a Wilson CI), every mean carries
    the count it was taken over, and unmeasured quantities are ``None``
    instead of a sentinel or a fabricated 0.

    Args:
        mols: generated molecules, ``None`` for construction failures.
        train_smiles / train_mols: the reference set (fix.md F1: 1628 BBBP
            molecules parsed from SMILES, not the 20 that survived the old
            distance-only converter).
        bbb_classifier: optional BBB oracle.
        stats: per-sample ``{requested_atoms, kept_atoms}`` dicts from
            ``coords_and_types_to_mol(stats=...)``; enables
            ``AtomsRetained%`` and ``SizeMatch_KS`` (fix.md F3/F16).
        reference: reference-set geometry (``BondsPerAtom`` etc.) used for the
            explicit ``BondsPerAtom_rel`` ratio.
    """
    valid = [m for m in mols if m is not None]
    n_total = len(mols)
    n_valid = len(valid)

    out: dict = {
        "Validity": (100.0 * n_valid / n_total) if n_total else None,
        "Validity_ci95": wilson_ci_or_none(n_valid, n_total),
        "Uniqueness": uniqueness(mols) * 100.0,
        "Novelty": (novelty(mols, train_smiles) * 100.0) if train_smiles else None,
        "IntDiv_p": internal_diversity(mols),
        "QED": mean_qed(mols) if valid else None,
        "LogP": mean_logp(mols) if valid else None,
        "SNN": (snn(mols, train_mols) if train_mols else None),
        "ScaffDiv": scaffold_diversity(mols) if valid else None,
        "ScaffCov": (scaffold_coverage(mols, train_mols)
                     if train_mols else None),
        "Lipinski%": lipinski_pass_rate(mols) * 100.0 if valid else None,
        "Veber%": veber_pass_rate(mols) * 100.0 if valid else None,
        "CNS_MPO": cns_mpo_score(mols) if valid else None,
    }
    out.update(connectivity(
        mols,
        reference_bonds_per_atom=((reference or {}).get("BondsPerAtom")),
    ))
    out.update(geometry_metrics(mols))
    out.update(bbb_rates(mols, bbb_classifier))

    # F3: post-hoc atom deletion is part of the result, not a hidden step.
    # The retention denominator is the atoms of molecules that were actually
    # BUILT — a construction failure is Validity's business, and counting its
    # atoms as "pruned" would silently understate retention.
    drawn = [int(s["requested_atoms"]) for s in stats] if stats else []
    pairs = ([(int(s["requested_atoms"]), int(s["kept_atoms"]))
              for s, m in zip(stats, mols) if m is not None] if stats else [])
    req_valid = [r for r, _ in pairs]
    kept_valid = [k for _, k in pairs]
    out["MeanAtoms_drawn"] = float(np.mean(drawn)) if drawn else None
    out["MeanAtoms_requested"] = (float(np.mean(req_valid))
                                  if req_valid else None)
    out["AtomsRetained%"] = (100.0 * sum(kept_valid) / sum(req_valid)
                             if req_valid and sum(req_valid) else None)
    # F16: validity is size-conditional, so compare surviving sizes against
    # the sizes that were requested.
    out["SizeMatch_KS"] = (
        ks_two_sample(req_valid, [m.GetNumAtoms() for m in valid])
        if req_valid and valid else None)
    return out


def _fmt_value(v) -> str:
    """One metric cell: floats at 4 dp, ``None`` as an em dash."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_value(x) for x in v) + "]"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def latex_row(metrics: dict, order: Sequence[str] = METRIC_ORDER) -> str:
    """Render a LaTeX table row from a metric dict (never stored).

    fix.md F8: a row frozen at eval time goes stale as soon as a metric is
    patched afterwards — ``eval_pre_rl`` still carries a row whose BBB column
    reads 0.0000 while ``metrics`` says 34.17. Rendering from a canonical
    order also means the column count no longer depends on which flags an
    arm happened to use.
    """
    cells = ["---" if metrics.get(k) is None else _fmt_value(metrics.get(k))
             for k in order]
    return "  &  ".join(cells) + r" \\"


def print_metrics(metrics: dict) -> None:
    """Pretty-print the evaluation results, honest about denominators.

    fix.md F5: means over 2 molecules were printed at 4 decimals next to
    means over 1000. Anything averaged over fewer than
    ``MIN_REPORTABLE_N`` valid molecules is rendered ``n/a (n=…)``.
    """
    n_valid = int(metrics.get("n_valid") or 0)
    n_total = int(metrics.get("n_total") or 0)
    small = n_valid < MIN_REPORTABLE_N
    print("=" * 58)
    print(f"{'Metric':<22} {'Value':>10}   n={n_valid}/{n_total}")
    print("-" * 58)
    for k, v in metrics.items():
        if (small and k not in TOTAL_RATE_KEYS and k not in COUNTS
                and isinstance(v, float)):
            shown = f"n/a (n={n_valid})"
        else:
            shown = _fmt_value(v)
        print(f"{k:<22} {shown:>10}")
    if small:
        print(f"-" * 58)
        print(f"  means suppressed: n_valid={n_valid} < {MIN_REPORTABLE_N}")
    print("=" * 58)
