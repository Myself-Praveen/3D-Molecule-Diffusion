"""Connectivity metrics: validity must not reward fragment soup."""

from rdkit import Chem

from src.utils.evaluation import connectivity


def _mol(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None
    return m


class TestConnectivity:
    def test_connected_molecule(self):
        out = connectivity([_mol("CCO"), _mol("c1ccccc1")])
        assert out["ConnectedValidity"] == 100.0
        assert out["ConnectedFrac"] == 100.0
        assert out["BondsPerMol"] > 2.0

    def test_fragment_soup_penalized(self):
        # One Mol object with disconnected atoms (the generated-soup case):
        # sanitizes fine (Validity=100) but scores 0% connected, 0 bonds/mol.
        soup = Chem.CombineMols(_mol("C"), _mol("C"))
        Chem.SanitizeMol(soup)
        assert len(Chem.GetMolFrags(soup)) == 2
        out = connectivity([soup])
        assert out["ConnectedValidity"] == 0.0
        assert out["BondsPerMol"] == 0.0

    def test_empty(self):
        assert connectivity([]) == {
            "ConnectedValidity": 0.0, "BondsPerMol": 0.0, "ConnectedFrac": 0.0,
        }
        assert connectivity([None])["ConnectedValidity"] == 0.0
