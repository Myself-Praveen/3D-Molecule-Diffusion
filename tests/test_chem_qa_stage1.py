"""Stage 1 of the lab-readiness stack: bond-order perception + stereochemistry.

The honest ruler builds bonds from covalent radii, which yields correct
connectivity but only single-bond *order* (bond order is not inferable from
distance — C-H at 1.09 A and C#C at 1.20 A overlap). So every molecule the
ruler accepts is missing C=C, C#N, aromatic rings and carbonyls. These tests
pin down the perception step that restores them, and the stereochemistry
read-back the ruler discards.

The key fixture is ``_single_bonded``: a molecule with correct 3D geometry
but every bond forced to SINGLE — exactly what ``coords_and_types_to_mol``
emits. Perception must recover the true bond orders from that.
"""

from rdkit import Chem
from rdkit.Chem import AllChem

from src.utils.chem_qa import (
    assign_stereochemistry,
    bond_order_summary,
    full_perception,
    ideal_bond_length,
    perceive_bond_orders,
    stereo_summary,
)


def _single_bonded(smiles: str, seed: int = 42) -> Chem.RWMol:
    """3D-embedded molecule with every bond forced SINGLE (the ruler's output)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert mol is not None
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    return rw


def _canonical(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


class TestBondOrderPerception:
    def test_benzene_recovers_aromatic_rings(self):
        out, status = perceive_bond_orders(_single_bonded("c1ccccc1"))
        assert status == "ok"
        assert _canonical(out) == "c1ccccc1"
        summary = bond_order_summary(out)
        # Benzene: 6 aromatic C-C + 6 C-H. Half the bonds are non-single.
        assert summary["AromaticFrac"] > 0.4
        assert summary["NonSingleFrac"] > 0.4

    def test_carbonyl_recovers_double_bond(self):
        out, status = perceive_bond_orders(_single_bonded("CC(C)=O"))
        assert status == "ok"
        assert _canonical(out) == "CC(C)=O"
        # Exactly one C=O among 10 bonds (8 heavy + 2 C-H... 3+1+1 C-H = 9 total).
        assert bond_order_summary(out)["NonSingleFrac"] > 0.0

    def test_nitrile_recovers_triple_bond(self):
        out, status = perceive_bond_orders(_single_bonded("CC#N"))
        assert status == "ok"
        assert _canonical(out) == "CC#N"
        orders = {b.GetBondType() for b in out.GetBonds()}
        assert Chem.BondType.TRIPLE in orders

    def test_already_ordered_is_a_noop(self):
        mol = Chem.MolFromSmiles("CC(C)=O")  # SMILES-built: already has C=O
        out, status = perceive_bond_orders(mol)
        assert status == "already_ordered"
        assert _canonical(out) == "CC(C)=O"

    def test_no_conformer_reported_not_raised(self):
        mol = Chem.RWMol(Chem.MolFromSmiles("CCO"))  # no 3D conformer
        for bond in mol.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
        out, status = perceive_bond_orders(mol)
        assert status == "no_conformer"
        assert out is not None  # returned unchanged, not dropped

    def test_none_input_is_safe(self):
        out, status = perceive_bond_orders(None)
        assert out is None
        assert status == "failed"

    def test_overcoordinated_soup_fails_without_raising(self):
        # A hypervalent atom (N with 5 bonds) cannot satisfy valence rules;
        # perception must report failure and hand back the input untouched.
        soup = Chem.RWMol()
        idx = [soup.AddAtom(Chem.Atom("N")) for _ in range(6)]
        for i in idx[1:]:
            soup.AddBond(idx[0], i, Chem.BondType.SINGLE)
        conf = Chem.Conformer(6)
        for k, i in enumerate(idx):
            conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(k * 1.5, 0.0, 0.0))
        soup.AddConformer(conf)
        out, status = perceive_bond_orders(soup)
        assert status == "failed"
        assert out is not None  # soup is still reportable, just not ordered

    def test_perception_never_changes_connectivity(self):
        """Perception restores ORDER only — the molecule graph is untouched."""
        original = _single_bonded("c1ccccc1")
        out, status = perceive_bond_orders(original)
        assert status == "ok"
        assert out.GetNumAtoms() == original.GetNumAtoms()
        assert out.GetNumBonds() == original.GetNumBonds()


class TestStereochemistry:
    def test_chiral_centre_read_from_3d(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))  # L-alanine
        assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
        out, status = assign_stereochemistry(mol)
        assert status == "ok"
        summary = stereo_summary(out)
        assert summary["StereocentresPerMol"] == 1.0
        assert summary["StereoDefinedFrac"] == 1.0
        # The assigned configuration must round-trip as a specific enantiomer.
        assert "@" in _canonical(out)

    def test_achiral_molecule_has_no_centres(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
        out, _ = assign_stereochemistry(mol)
        summary = stereo_summary(out)
        assert summary["StereocentresPerMol"] == 0.0
        assert summary["StereoDefinedFrac"] == 1.0  # vacuously fully defined

    def test_none_input_is_safe(self):
        out, status = assign_stereochemistry(None)
        assert out is None
        assert status == "failed"
        assert stereo_summary(None) == {
            "StereocentresPerMol": 0.0, "StereoDefinedFrac": 0.0,
        }

    def test_two_centres_both_defined(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O.C[C@H](N)C(=O)O"))
        assert AllChem.EmbedMolecule(mol, randomSeed=11) == 0
        out, _ = assign_stereochemistry(mol)
        summary = stereo_summary(out)
        assert summary["StereocentresPerMol"] == 2.0
        assert summary["StereoDefinedFrac"] == 1.0


class TestBondOrderSummary:
    def test_empty_and_none(self):
        assert bond_order_summary(None)["NonSingleFrac"] == 0.0
        lonely = Chem.RWMol()
        lonely.AddAtom(Chem.Atom("C"))
        assert bond_order_summary(lonely)["MeanBondOrder"] == 1.0

    def test_ethane_is_all_single(self):
        summary = bond_order_summary(Chem.MolFromSmiles("CC"))
        assert summary["NonSingleFrac"] == 0.0
        assert summary["AromaticFrac"] == 0.0
        assert summary["MeanBondOrder"] == 1.0


class TestIdealBondLength:
    def test_single_bond_is_radius_sum(self):
        mol = Chem.MolFromSmiles("CC")
        bond = mol.GetBondWithIdx(0)
        # C-C: 0.76 + 0.76 = 1.52
        assert abs(ideal_bond_length(bond, mol) - 1.52) < 1e-6

    def test_double_bond_is_shorter_than_single(self):
        ethene = Chem.MolFromSmiles("C=C")
        single = Chem.MolFromSmiles("CC")
        assert (ideal_bond_length(ethene.GetBondWithIdx(0), ethene)
                < ideal_bond_length(single.GetBondWithIdx(0), single))

    def test_unknown_element_returns_none(self):
        # Iron is absent from the radius table. Build a real Fe-C bond so the
        # lookup path is exercised (a bare [Fe] has no bonds to query).
        mol = Chem.MolFromSmiles("[Fe]C")
        bond = mol.GetBondWithIdx(0)
        assert {bond.GetBeginAtom().GetAtomicNum(),
                bond.GetEndAtom().GetAtomicNum()} == {26, 6}
        assert ideal_bond_length(bond, mol) is None


class TestFullPerception:
    def test_composes_orders_and_stereo(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))
        assert AllChem.EmbedMolecule(mol, randomSeed=3) == 0
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            atom.SetIsAromatic(False)
        for bond in rw.GetBonds():
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)
        out = full_perception(rw)
        assert out is not None
        assert "@" in _canonical(out)          # stereo survived
        assert "=O" in _canonical(out) or "C(=O)" in _canonical(out)  # C=O restored

    def test_none_stays_none(self):
        assert full_perception(None) is None
