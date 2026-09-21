"""Connectivity metrics: validity must not reward fragment soup."""

import torch
from rdkit import Chem

from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.egnn import EquivariantGenerator
from src.sampling import sample_molecules
from src.utils.evaluation import connectivity, relax_molecule


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


class TestRelax:
    def test_none_passthrough(self):
        assert relax_molecule(None) is None

    def test_valid_molecule_stays_valid(self):
        mol = Chem.MolFromSmiles("CCO")
        out = relax_molecule(mol)
        assert out is not None
        Chem.SanitizeMol(out)


class TestQuadraticSchedule:
    def _tiny(self):
        model = EquivariantGenerator(num_types=10, node_dim=8, edge_dim=8,
                                     num_layers=1, time_dim=8)
        device = "cpu"
        return (model, CenteredDDPM(num_steps=100, device=device),
                TypeDDPM(num_steps=100, device=device))

    def test_quadratic_runs(self):
        model, c_ddpm, t_ddpm = self._tiny()
        torch.manual_seed(0)
        pos, z = sample_molecules(model, c_ddpm, t_ddpm, torch.tensor([6, 7]),
                                  device="cpu", ddim_steps=8,
                                  step_schedule="quadratic")
        assert pos.shape == (13, 3) and z.shape == (13,)
        assert torch.isfinite(pos).all()

    def test_bad_schedule_raises(self):
        import pytest as _pytest

        model, c_ddpm, t_ddpm = self._tiny()
        with _pytest.raises(ValueError):
            sample_molecules(model, c_ddpm, t_ddpm, torch.tensor([6]),
                             device="cpu", ddim_steps=5,
                             step_schedule="bogus")
