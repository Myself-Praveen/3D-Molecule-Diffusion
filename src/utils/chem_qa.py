"""Chemistry quality-assurance for generated molecules (lab-readiness stack).

Why this module exists
----------------------
The honest-ruler metrics in ``src/utils/evaluation.py`` answer "is this a
connected, sanitizable molecule?". They do not answer the question a
synthesis lab actually asks: *can a chemist order this and make it?*

The gap is specific and was measured, not assumed (progress.md §15 items 3,
4, 13): the best BBB arm reaches ConnectedValidity 1.5% with 29.9 bonds/mol
against ~19 for real BBBP. Three objective hypotheses (step budget, the
lambda_2 demand side, explicit shape supervision) were each excluded by
direct A/B, so the remaining work is a feasibility stack around the model
rather than another loss term.

This module implements that stack in five stages, each independently usable:

1. **Chemistry perception** — bond orders and stereochemistry, which the
   distance-only ruler cannot supply.
2. **Geometry QA** — bond-length deviation from covalent norms, idealization,
   and force-field strain energy.
3. **Synthetic accessibility** — a 1-10 complexity score plus a PAINS
   structural-alert check.
4. **Applicability domain** — nearest-neighbour Tanimoto to the training set.
5. **ADMET proxies** — ESOL solubility and hERG/CYP/P-gp structural alerts.

Stage 6 (retrosynthesis feasibility) lives in ``src/utils/retrosynthesis.py``
because it carries its own reaction-template library.

Design rules, chosen to match the honesty conventions of this repo:

* Every entry point accepts ``None`` and returns a safe sentinel rather than
  raising — a generated soup must never crash the eval that is measuring it.
* Scores are labelled for what they *are*. ``synthetic_complexity_score`` is
  explicitly NOT the published Ertl SA score (that needs a PubChem-derived
  fragment table we do not ship); it is a self-contained complexity proxy
  whose values are comparable within this repo. The ADMET terms are
  structural alerts, not predictors. Saying so in the docstring is the point.
* Nothing here changes the measured Validity/ConnectedValidity ruler. These
  are additive gates that run *after* a molecule has been accepted by the
  honest ruler.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdDetermineBonds, rdMolDescriptors

# Covalent radii (Cordero et al. 2008, Å) — same table the honest ruler uses,
# duplicated as the ideal-bond-length reference for geometry QA.
_COVALENT_RADII: dict[int, float] = {
    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39,
}

# Bond-order dependent shrinkage applied to the radius sum when comparing a
# measured bond length against its ideal. Aromatic and double bonds are
# shorter than the single-bond radius sum; triple bonds shorter still. These
# are coarse literature-typical corrections (±5%), not fitted parameters —
# geometry QA measures *deviation*, and the correction only stops every C=O
# from being flagged as a 10% stretch.
_BOND_LENGTH_FACTOR: dict[Chem.BondType, float] = {
    Chem.BondType.SINGLE: 1.00,
    Chem.BondType.DOUBLE: 0.93,
    Chem.BondType.TRIPLE: 0.87,
    Chem.BondType.AROMATIC: 0.94,
}


def ideal_bond_length(bond: Chem.Bond, mol: Chem.Mol) -> float | None:
    """Ideal length (Å) for ``bond`` from covalent radii + bond-order factor.

    Returns ``None`` when either atom's element is absent from the radius
    table (metals, exotic types) so callers can skip rather than guess.
    """
    a, b = bond.GetBeginAtom(), bond.GetEndAtom()
    ra = _COVALENT_RADII.get(a.GetAtomicNum())
    rb = _COVALENT_RADII.get(b.GetAtomicNum())
    if ra is None or rb is None:
        return None
    factor = _BOND_LENGTH_FACTOR.get(bond.GetBondType(), 1.00)
    return (ra + rb) * factor


# ---------------------------------------------------------------------------
# Stage 1a: bond-order perception
# ---------------------------------------------------------------------------

def perceive_bond_orders(mol: Chem.Mol | None, charge: int = 0) -> tuple[Chem.Mol | None, str]:
    """Assign bond orders to a molecule that has connectivity but single bonds.

    The honest ruler builds bonds from covalent radii, which yields correct
    *connectivity* but only single-bond *order* — bond order is not inferable
    from distance (C-H at 1.09 Å and C#C at 1.20 Å overlap). Every molecule
    the ruler accepts is therefore missing C=C, C#N, aromatic rings and
    carbonyls, which is exactly the information a chemist needs to draw the
    compound and a retrosynthesis engine needs to propose a route.

    This restores bond orders from the 3D geometry using RDKit's
    ``DetermineBondOrders``: it keeps the perceived connectivity fixed and
    solves for the bond orders that satisfy octet/valence rules given the
    measured geometry. This is a perception step, not a repair — it never
    adds or removes bonds, so ConnectedValidity is unaffected.

    Returns ``(mol, status)`` where status is one of ``"ok"``,
    ``"already_ordered"``, ``"no_conformer"``, ``"failed"``. On failure the
    input molecule is returned unchanged (single bonds) so the caller can
    still report it honestly rather than dropping it.
    """
    if mol is None:
        return None, "failed"

    # Already carries bond orders? Leave it alone. Checked before the
    # conformer test because a SMILES-built molecule has bond orders but no
    # conformer, and "already ordered" is the more useful diagnosis.
    if any(b.GetBondType() != Chem.BondType.SINGLE for b in mol.GetBonds()):
        return mol, "already_ordered"

    if mol.GetNumConformers() == 0:
        return mol, "no_conformer"

    work = Chem.RWMol(mol)
    try:
        rdDetermineBonds.DetermineBondOrders(work, charge=charge)
    except Exception:
        # Perception failed (over-coordinated soup, hypervalent atoms, ...).
        # Return the single-bonded input unchanged rather than raising.
        return mol, "failed"

    try:
        Chem.SanitizeMol(work)
    except Exception:
        return mol, "failed"
    return work, "ok"


def bond_order_summary(mol: Chem.Mol | None) -> dict[str, float]:
    """Fraction of bonds by order — a direct read on whether perception worked.

    A molecule that stayed all-single after ``perceive_bond_orders`` reports
    ``NonSingleFrac`` 0.0, which at cohort level is the honest measure of how
    much chemical information the model actually produced.
    """
    empty = {"NonSingleFrac": 0.0, "AromaticFrac": 0.0, "MeanBondOrder": 1.0}
    if mol is None or mol.GetNumBonds() == 0:
        return empty
    orders = [b.GetBondTypeAsDouble() for b in mol.GetBonds()]
    non_single = sum(1 for o in orders if o > 1.0 + 1e-6)
    aromatic = sum(1 for b in mol.GetBonds() if b.GetIsAromatic())
    n = len(orders)
    return {
        "NonSingleFrac": non_single / n,
        "AromaticFrac": aromatic / n,
        "MeanBondOrder": float(sum(orders) / n),
    }


# ---------------------------------------------------------------------------
# Stage 1b: stereochemistry
# ---------------------------------------------------------------------------

def assign_stereochemistry(mol: Chem.Mol | None) -> tuple[Chem.Mol | None, str]:
    """Assign stereochemistry from the generated 3D coordinates.

    A 3D generator produces a specific conformer, so it *does* determine
    stereochemistry — but the ruler never reads it back, which means every
    generated stereocentre is discarded and the molecule handed to a lab has
    undefined configuration. For drug candidates that is the difference
    between one compound and a pair of enantiomers with different activity.

    Returns ``(mol, status)`` with status ``"ok"``/``"no_conformer"``/
    ``"failed"``.
    """
    if mol is None:
        return None, "failed"
    if mol.GetNumConformers() == 0:
        return mol, "no_conformer"
    work = Chem.Mol(mol)
    try:
        Chem.AssignStereochemistryFrom3D(work)
    except Exception:
        return mol, "failed"
    return work, "ok"


def stereo_summary(mol: Chem.Mol | None) -> dict[str, float]:
    """Defined/undefined stereocentre counts.

    ``StereoDefinedFrac`` is 1.0 when every potential stereocentre has been
    assigned from the 3D structure. ``StereocentresPerMol`` is the mean
    number of potential centres per molecule — a cohort-level signal for how
    much stereochemical content the generator produces.
    """
    if mol is None:
        return {"StereocentresPerMol": 0.0, "StereoDefinedFrac": 0.0}
    try:
        centres = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        undefined = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
    except Exception:
        return {"StereocentresPerMol": 0.0, "StereoDefinedFrac": 0.0}
    defined = max(0, centres - undefined)
    frac = (defined / centres) if centres > 0 else 1.0
    return {"StereocentresPerMol": float(centres), "StereoDefinedFrac": frac}


def full_perception(mol: Chem.Mol | None) -> Chem.Mol | None:
    """Convenience: bond orders + stereochemistry, best effort.

    Returns ``None`` only when the input is ``None``. A molecule whose
    perception fails partway is returned in its best available state, so a
    caller that just wants "the most chemical version of this" never loses
    the sample.
    """
    if mol is None:
        return None
    mol, _ = perceive_bond_orders(mol)
    mol, _ = assign_stereochemistry(mol)
    return mol


# ---------------------------------------------------------------------------
# Stage 2: geometry QA
# ---------------------------------------------------------------------------

def bond_length_deviation(mol: Chem.Mol | None) -> dict[str, float]:
    """How far the 3D geometry sits from covalent-norm bond lengths.

    This is the direct quantitative read on the failure the project has been
    chasing. A denoiser that beats the trivial x0 predictor at every
    timestep (progress.md §13.17) still produced "over-bonded soup" at 32.8
    bonds/mol against ~19 real. Validity and ConnectedValidity say *whether*
    the molecule is connected; this says whether its geometry is actually
    right. A molecule can be sanitizable and single-fragment while having
    every bond stretched 30% — which no chemist would recognise as the
    compound drawn on paper.

    Returns ``MeanBondStrainPct`` (mean |measured − ideal| / ideal, in %,
    over bonds whose elements are known), ``MaxBondStrainPct``, and
    ``n_bonds_measured``. ``None`` and bond-free input give 0.0 strain.
    """
    if mol is None or mol.GetNumBonds() == 0 or mol.GetNumConformers() == 0:
        return {"MeanBondStrainPct": 0.0, "MaxBondStrainPct": 0.0,
                "n_bonds_measured": 0}
    conf = mol.GetConformer()
    strains: list[float] = []
    for bond in mol.GetBonds():
        ideal = ideal_bond_length(bond, mol)
        if ideal is None or ideal <= 0.0:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pi, pj = conf.GetAtomPosition(i), conf.GetAtomPosition(j)
        measured = ((pi.x - pj.x) ** 2 + (pi.y - pj.y) ** 2
                    + (pi.z - pj.z) ** 2) ** 0.5
        strains.append(100.0 * abs(measured - ideal) / ideal)
    if not strains:
        return {"MeanBondStrainPct": 0.0, "MaxBondStrainPct": 0.0,
                "n_bonds_measured": 0}
    return {
        "MeanBondStrainPct": float(sum(strains) / len(strains)),
        "MaxBondStrainPct": float(max(strains)),
        "n_bonds_measured": len(strains),
    }


def strain_energy(mol: Chem.Mol | None) -> float | None:
    """MMFF94 steric energy in kcal/mol (UFF fallback), or ``None``.

    An independent geometry check that does not assume the covalent-radius
    model the bond builder already uses — MMFF94 scores the full force field
    including angles, torsions and non-bonded contacts, so a strained ring
    or a clash between non-bonded atoms shows up here even when every bond
    length is correct.

    Reported as an absolute energy for a single conformer. It is meaningful
    for *ranking* geometries and for flagging pathological output (values in
    the hundreds of kcal/mol indicate a real clash), not as a free energy.
    Returns ``None`` when no force field can be constructed — which is itself
    the signal for a molecule MMFF cannot type.
    """
    if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0:
        return None
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = (AllChem.MMFFGetMoleculeForceField(mol, props)
              if props is not None else None)
    except Exception:
        ff = None
    if ff is None:
        try:
            ff = AllChem.UFFGetMoleculeForceField(mol)
        except Exception:
            return None
    try:
        return float(ff.CalcEnergy())
    except Exception:
        return None


def idealize_bond_lengths(mol: Chem.Mol | None, iterations: int = 200,
                          step: float = 0.5) -> Chem.Mol | None:
    """Snap bond lengths toward covalent norms, preserving connectivity.

    A geometry-repair step that is deliberately narrow. It moves atoms along
    existing bond vectors so each bond approaches its ideal length; it never
    adds, removes or reorders a bond, so Validity and ConnectedValidity are
    invariant by construction — the same structural guarantee
    ``relax_molecule`` lacks (progress.md §13.13 showed MMFF relaxation cannot
    rescue an invalid molecule, because the bond graph is fixed before
    relaxation runs and never re-derived).

    What this *can* do: take a molecule whose connectivity is already correct
    but whose coordinates are off (the "coordinate precision, not clumping"
    diagnosis in §9) and pull it toward chemically realistic bond lengths.
    What it *cannot* do: fix a wrong bond graph. A fragment soup stays a
    fragment soup, and that is the correct behaviour — the honest ruler must
    keep scoring it as disconnected.

    Returns a copy with a single conformer; ``None`` for ``None`` or for
    input with no coordinates.
    """
    if mol is None or mol.GetNumConformers() == 0 or mol.GetNumBonds() == 0:
        return mol
    work = Chem.Mol(mol)
    conf = work.GetConformer()
    n = work.GetNumAtoms()

    for _ in range(max(1, int(iterations))):
        disp = np.zeros((n, 3), dtype=float)
        moved = 0.0
        for bond in work.GetBonds():
            ideal = ideal_bond_length(bond, work)
            if ideal is None or ideal <= 0.0:
                continue
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            pi = np.array([conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z])
            pj = np.array([conf.GetAtomPosition(j).x,
                           conf.GetAtomPosition(j).y,
                           conf.GetAtomPosition(j).z])
            vec = pj - pi
            length = float(np.linalg.norm(vec))
            if length < 1e-8:
                continue
            # Fraction of the error to close this pass (damped to stay stable
            # when an atom participates in several bonds at once).
            delta = step * (length - ideal) / length
            disp[i] += delta * vec
            disp[j] -= delta * vec
            moved = max(moved, abs(delta) * length)
        if moved < 1e-6:
            break
        for k in range(n):
            p = conf.GetAtomPosition(k)
            conf.SetAtomPosition(k, Chem.rdGeometry.Point3D(
                p.x + float(disp[k, 0]),
                p.y + float(disp[k, 1]),
                p.z + float(disp[k, 2])))
    return work
