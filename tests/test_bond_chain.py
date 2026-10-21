"""Plan workstreams A2/A3/A4 — the bond-perception chain.

A2: ``bond_tolerance`` tighten-cap on the tolerance matrix (Table I
    protocol must stay regenerable with the default).
A3: ``max_bonds_ratio`` over-bonded rejection gate (off by default).
A4: ``perceive_orders`` bond-order upgrade — must leave Validity
    invariant (fallback keeps the single-bond topology on failure) and
    must be neutral-or-positive on real ground-truth geometry.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from src.utils.evaluation import (
    TOLERANCE_MATRIX,
    coords_and_types_to_mol,
    get_bond_tolerance,
)


def _embed(smiles: str, seed: int = 42):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, smiles
    m = Chem.AddHs(m)
    assert AllChem.EmbedMolecule(m, randomSeed=seed) == 0, smiles
    return m


def _as_coords(mol: Chem.Mol):
    conf = mol.GetConformer()
    pos = np.array([[conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z]
                    for i in range(mol.GetNumAtoms())])
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    return pos, z


# ---------------------------------------------------------------------------
# A2: tighten-cap semantics
# ---------------------------------------------------------------------------

class TestBondToleranceCap:
    def test_no_cap_is_legacy_exactly(self):
        for (a, b), v in TOLERANCE_MATRIX.items():
            assert get_bond_tolerance(a, b) == v
            assert get_bond_tolerance(a, b, cap=None) == v

    def test_cap_tightens_scalar_and_matrix_entries(self):
        assert get_bond_tolerance(6, 6, cap=0.35) == 0.35   # matrix 0.45
        assert get_bond_tolerance(6, 8, cap=0.35) == 0.35   # matrix 0.50
        assert get_bond_tolerance(35, 53, cap=0.35) == 0.35  # unlisted 0.45

    def test_cap_is_tighten_only_never_loosens(self):
        # H–H is calibrated at 0.25; a looser cap must not raise it.
        assert get_bond_tolerance(1, 1, cap=0.35) == 0.25
        assert get_bond_tolerance(1, 9, cap=0.45) == 0.35  # stays at base

    def test_bbbp_pairs_no_longer_fall_back_to_scalar(self):
        # Pre-A2 these were unlisted and fell back to 0.45.
        assert get_bond_tolerance(6, 16) == 0.40
        assert get_bond_tolerance(6, 17) == 0.38
        assert get_bond_tolerance(6, 35) == 0.40
        assert get_bond_tolerance(7, 16) == 0.38
        assert get_bond_tolerance(16, 16) == 0.35

    def test_qm9_pairs_unchanged_by_bbbp_additions(self):
        """The QM9-era pairs must be bit-for-bit what Table I was built with."""
        qm9 = {(1, 1): 0.25, (1, 6): 0.40, (1, 7): 0.40, (1, 8): 0.40,
               (1, 9): 0.35, (6, 6): 0.45, (6, 7): 0.45, (6, 8): 0.50,
               (6, 9): 0.40, (7, 7): 0.45, (7, 8): 0.45, (7, 9): 0.40,
               (8, 8): 0.40, (8, 9): 0.40, (9, 9): 0.40}
        for key, val in qm9.items():
            assert TOLERANCE_MATRIX[key] == val, key

    def test_cap_cuts_borderline_contact_in_builder(self):
        """C···C at 1.92 Å bonds under the legacy tolerance (cutoff
        1.52+0.45 = 1.97) but not under cap 0.35 (cutoff 1.87) — the
        clumped-soup contact window the plan targets. Real C–C bonds
        (≤ ~1.54 Å) sit far below both, so the cap cannot cut them."""
        pos = np.array([[0.0, 0.0, 0.0], [1.92, 0.0, 0.0]])
        z = np.array([6, 6])
        loose = coords_and_types_to_mol(pos, z)               # legacy
        tight = coords_and_types_to_mol(pos, z, bond_tolerance=0.35)
        assert loose is not None and loose.GetNumBonds() == 1
        # No bond left: two isolated atoms remain a (disconnected) mol —
        # the builder never re-labels that; ConnectedValidity scores it.
        assert tight is not None and tight.GetNumBonds() == 0

    def test_cap_keeps_real_bonds(self):
        """Ground-truth methane must survive the cap (regression guard)."""
        pos = np.array([
            [0.0000, 0.0000, 0.0000],
            [0.6296, 0.6296, 0.6296],
            [-0.6296, -0.6296, 0.6296],
            [-0.6296, 0.6296, -0.6296],
            [0.6296, -0.6296, -0.6296],
        ])
        z = np.array([6, 1, 1, 1, 1])
        mol = coords_and_types_to_mol(pos, z, bond_tolerance=0.35)
        assert mol is not None
        assert mol.GetNumBonds() == 4
        assert len(Chem.GetMolFrags(mol)) == 1


# ---------------------------------------------------------------------------
# A3: over-bonded rejection gate
# ---------------------------------------------------------------------------

class TestMaxBondsRatio:
    # Regular tetrahedron, edge 1.5 Å: ALL six C–C pairs within bonding
    # range (an axis-aligned box only bonds 3 pairs — its diagonals sit at
    # 2.12 Å, past even the loose cutoff).
    _TETRA = np.array([[0.5303, 0.5303, 0.5303],
                       [0.5303, -0.5303, -0.5303],
                       [-0.5303, 0.5303, -0.5303],
                       [-0.5303, -0.5303, 0.5303]])
    _TETRA_Z = np.array([6, 6, 6, 6])

    def test_default_off_is_legacy(self):
        # Fully-connected C4: 6 bonds on 4 atoms would be rejected at 1.4,
        # but with the default (0) it must survive for Table I parity.
        mol = coords_and_types_to_mol(self._TETRA, self._TETRA_Z)
        assert mol is not None and mol.GetNumBonds() == 6

    def test_gate_rejects_over_bonded(self):
        # 6 bonds > 1.4 * 0.95 * 4 = 5.32 -> reject
        assert coords_and_types_to_mol(
            self._TETRA, self._TETRA_Z, max_bonds_ratio=1.4) is None

    def test_gate_keeps_normal_chemistry(self):
        mol = _embed("c1ccccc1O")
        pos, z = _as_coords(mol)
        rebuilt = coords_and_types_to_mol(pos, z, max_bonds_ratio=1.4)
        assert rebuilt is not None  # 13 atoms, 13 bonds (incl. Hs): under gate

    def test_zero_disables(self):
        assert coords_and_types_to_mol(
            self._TETRA, self._TETRA_Z, max_bonds_ratio=0.0) is not None


# ---------------------------------------------------------------------------
# A4: bond-order perception
# ---------------------------------------------------------------------------

class TestPerceiveOrders:
    def test_default_stays_single_bond(self):
        mol = _embed("c1ccccc1")
        pos, z = _as_coords(mol)
        rebuilt = coords_and_types_to_mol(pos, z)
        assert rebuilt is not None
        assert all(b.GetBondType() == Chem.BondType.SINGLE
                   for b in rebuilt.GetBonds())

    def test_perceive_recovers_aromatic_orders(self):
        """The whole point of A4: benzene must stop reading as cyclohexane."""
        mol = _embed("c1ccccc1")
        pos, z = _as_coords(mol)
        upgraded = coords_and_types_to_mol(pos, z, perceive_orders=True)
        assert upgraded is not None
        assert any(b.GetIsAromatic() for b in upgraded.GetBonds()), \
            "DetermineBondOrders did not aromatize a real benzene geometry"

    def test_failure_falls_back_not_crashes(self):
        """Odd-electron fragment: sanitizes (radical-legal) but no valid
        closed-shell bond ordering exists -> single-bond topology returned,
        never None-by-perception and never an exception."""
        # C with only 3 H at real C-H distances: 9 valence electrons total.
        pos = np.array([
            [0.0000, 0.0000, 0.0000],
            [1.0900, 0.0000, 0.0000],
            [-0.5450, 0.9440, 0.0000],
            [-0.5450, -0.9440, 0.0000],
        ])
        z = np.array([6, 1, 1, 1])
        plain = coords_and_types_to_mol(pos, z)
        perceived = coords_and_types_to_mol(pos, z, perceive_orders=True)
        assert plain is not None and perceived is not None
        assert perceived.GetNumBonds() == plain.GetNumBonds() == 3

    def test_validity_invariant_on_ground_truth_suite(self):
        """Perception must never turn a valid rebuild into None (the plan's
        'test A4 first' requirement, enforced as a suite invariant)."""
        for smi in ("CCO", "c1ccccc1", "CC(=O)O", "c1ccncc1", "CCN",
                    "c1ccc2ccccc2c1", "CC(=O)OC"):
            mol = _embed(smi)
            pos, z = _as_coords(mol)
            a = coords_and_types_to_mol(pos, z)
            b = coords_and_types_to_mol(pos, z, perceive_orders=True)
            assert a is not None, smi
            assert b is not None, f"perception invalidated {smi}"
            # Same connectivity either way (orders may only be upgraded).
            assert b.GetNumAtoms() == a.GetNumAtoms(), smi

    def test_perceive_requires_sanitize(self):
        # sanitize=False path must skip perception entirely (documented).
        mol = _embed("c1ccccc1")
        pos, z = _as_coords(mol)
        raw = coords_and_types_to_mol(pos, z, sanitize=False,
                                      perceive_orders=True)
        assert raw is not None
        assert all(b.GetBondType() == Chem.BondType.SINGLE
                   for b in raw.GetBonds())


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
