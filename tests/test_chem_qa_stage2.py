"""Stage 2 of the lab-readiness stack: geometry QA.

Validity and ConnectedValidity say *whether* a molecule is connected; they
say nothing about whether its geometry is right. The project's central
negative result is a model that beats the trivial x0 predictor at every
timestep (progress.md §13.17) yet still yields over-bonded soup at 32.8
bonds/mol against ~19 real. These tests pin down the measurement that
quantifies that gap, and the repair step that closes it *without* touching
the bond graph.

The critical invariant, and the reason this differs from ``relax_molecule``:
``idealize_bond_lengths`` never adds, removes or reorders a bond, so
Validity and ConnectedValidity are unchanged by construction (§13.13 showed
MMFF relaxation cannot rescue connectivity because the bond graph is fixed
before relaxation runs and never re-derived).
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from src.utils.chem_qa import (
    bond_length_deviation,
    idealize_bond_lengths,
    strain_energy,
)


def _embed(smiles: str, seed: int = 42) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert mol is not None
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _scale(mol: Chem.Mol, factor: float) -> Chem.Mol:
    """Uniformly scale all coordinates about the centroid.

    Scaling the whole molecule multiplies every interatomic distance by
    ``factor`` exactly, so it produces a controlled, known bond strain
    without changing connectivity — the right way to build a test fixture
    for a strain metric.
    """
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    n = out.GetNumAtoms()
    pts = np.array([[conf.GetAtomPosition(k).x,
                     conf.GetAtomPosition(k).y,
                     conf.GetAtomPosition(k).z] for k in range(n)])
    centre = pts.mean(axis=0)
    pts = centre + (pts - centre) * factor
    for k in range(n):
        conf.SetAtomPosition(k, Chem.rdGeometry.Point3D(*pts[k]))
    return out


class TestBondLengthDeviation:
    def test_optimized_molecule_is_low_strain(self):
        out = bond_length_deviation(_embed("c1ccccc1"))
        assert out["n_bonds_measured"] == 12  # 6 C-C + 6 C-H
        assert out["MeanBondStrainPct"] < 5.0
        assert out["MaxBondStrainPct"] < 10.0

    def test_strain_scales_with_stretch(self):
        good = _embed("c1ccccc1")
        mild = bond_length_deviation(_scale(good, 1.3))["MeanBondStrainPct"]
        severe = bond_length_deviation(_scale(good, 1.5))["MeanBondStrainPct"]
        assert 25.0 < mild < 35.0    # 30% stretch reads ~30%
        assert 45.0 < severe < 55.0  # 50% stretch reads ~50%
        assert severe > mild

    def test_compression_is_detected_like_stretch(self):
        # Strain is |measured - ideal|, so a squashed molecule scores too.
        out = bond_length_deviation(_scale(_embed("CCO"), 0.7))
        assert out["MeanBondStrainPct"] > 20.0

    def test_max_is_at_least_mean(self):
        out = bond_length_deviation(_embed("C[C@H](N)C(=O)O"))
        assert out["MaxBondStrainPct"] >= out["MeanBondStrainPct"]

    def test_none_and_empty_are_safe(self):
        assert bond_length_deviation(None) == {
            "MeanBondStrainPct": 0.0, "MaxBondStrainPct": 0.0,
            "n_bonds_measured": 0,
        }
        lonely = Chem.RWMol()
        lonely.AddAtom(Chem.Atom("C"))
        assert bond_length_deviation(lonely)["n_bonds_measured"] == 0

    def test_no_conformer_is_safe(self):
        assert bond_length_deviation(Chem.MolFromSmiles("CCO"))[
            "n_bonds_measured"] == 0


class TestStrainEnergy:
    def test_optimized_molecule_is_low_energy(self):
        energy = strain_energy(_embed("c1ccccc1"))
        assert energy is not None
        assert energy < 50.0  # kcal/mol; MMFF benzene is near its minimum

    def test_stretch_raises_energy_monotonically(self):
        good = _embed("c1ccccc1")
        e1 = strain_energy(_scale(good, 1.3))
        e2 = strain_energy(_scale(good, 1.5))
        e0 = strain_energy(good)
        assert e0 < e1 < e2

    def test_none_and_no_conformer_return_none(self):
        assert strain_energy(None) is None
        assert strain_energy(Chem.MolFromSmiles("CCO")) is None

    def test_value_is_finite(self):
        energy = strain_energy(_embed("C[C@H](N)C(=O)O"))
        assert energy is not None and np.isfinite(energy)


class TestIdealizeBondLengths:
    def test_reduces_strain_to_near_zero(self):
        stretched = _scale(_embed("c1ccccc1"), 1.3)
        before = bond_length_deviation(stretched)["MeanBondStrainPct"]
        after = bond_length_deviation(idealize_bond_lengths(stretched))
        assert before > 25.0
        assert after["MeanBondStrainPct"] < 1.0

    def test_connectivity_is_invariant(self):
        """The invariant that makes this safe: no bond is ever added or lost."""
        mol = _embed("C[C@H](N)C(=O)O")
        fixed = idealize_bond_lengths(_scale(mol, 1.4))
        assert fixed.GetNumAtoms() == mol.GetNumAtoms()
        assert fixed.GetNumBonds() == mol.GetNumBonds()
        assert len(Chem.GetMolFrags(fixed)) == 1

    def test_fragment_soup_stays_disconnected(self):
        """A soup must NOT be repaired into a molecule — the honest ruler has
        to keep scoring it as disconnected. Idealization only fixes lengths
        on bonds that already exist."""
        soup = Chem.CombineMols(Chem.MolFromSmiles("C"), Chem.MolFromSmiles("C"))
        conf = Chem.Conformer(2)
        conf.SetAtomPosition(0, Chem.rdGeometry.Point3D(0.0, 0.0, 0.0))
        conf.SetAtomPosition(1, Chem.rdGeometry.Point3D(5.0, 0.0, 0.0))
        soup.RemoveAllConformers()
        soup.AddConformer(conf)
        fixed = idealize_bond_lengths(soup)
        assert fixed is not None
        assert len(Chem.GetMolFrags(fixed)) == 2  # still two fragments

    def test_is_idempotent_on_good_geometry(self):
        mol = _embed("c1ccccc1")
        once = idealize_bond_lengths(mol)
        twice = idealize_bond_lengths(once)
        d1 = bond_length_deviation(once)["MeanBondStrainPct"]
        d2 = bond_length_deviation(twice)["MeanBondStrainPct"]
        assert abs(d1 - d2) < 1e-6

    def test_leaves_a_conformer(self):
        fixed = idealize_bond_lengths(_embed("CCO"))
        assert fixed.GetNumConformers() == 1

    def test_none_and_no_conformer_are_safe(self):
        assert idealize_bond_lengths(None) is None
        noconf = Chem.MolFromSmiles("CCO")
        out = idealize_bond_lengths(noconf)
        assert out.GetNumConformers() == 0  # returned unchanged

    def test_does_not_change_stereochemistry(self):
        """Repairing geometry must not invert a stereocentre."""
        mol = _embed("C[C@H](N)C(=O)O")
        fixed = idealize_bond_lengths(_scale(mol, 1.2))
        Chem.AssignStereochemistryFrom3D(fixed)
        assert Chem.MolToSmiles(Chem.RemoveHs(fixed)) == \
            Chem.MolToSmiles(Chem.RemoveHs(mol))
