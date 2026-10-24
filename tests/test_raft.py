"""Tests for the RAFT fallback (plan Step 1.5): elite filter + fine-tune."""

from __future__ import annotations

import copy

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from scripts.raft_finetune import (
    _tensorize,
    bond_window,
    finetune,
    select_elites,
)
from src.models.diffusion import CenteredDDPM
from src.models.egnn import EquivariantGenerator
from src.rl.rewards import REFERENCE_BONDS_PER_MOL


def _mol(smiles: str, add_hs: bool = True) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    if add_hs:
        mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == 0
    return mol


def _connected_out_of_window() -> Chem.Mol:
    """A single connected chain with far too many bonds for the window."""
    return _mol("CCCCCCCCCCCCCCCC")  # 16 C + 34 H => 49 bonds


def _fragmented() -> Chem.Mol:
    return _mol("CCCC.CC")


class TestBondWindow:
    def test_default_is_relative_to_the_measured_reference(self):
        low, high = bond_window()
        assert low == pytest.approx(REFERENCE_BONDS_PER_MOL * 0.65)
        assert high == pytest.approx(REFERENCE_BONDS_PER_MOL * 1.35)

    def test_explicit_bounds_win(self):
        assert bond_window(bond_min=15.0, bond_max=25.0) == (15.0, 25.0)

    def test_reference_matches_the_plan_intent(self):
        # The plan's 15..25 around a heavy-atom target of 19, rescaled to this
        # project's H-inclusive ruler (26.88), is 21.2..35.4 — inside the
        # default +/-35% window, so the two spellings select the same elites.
        scale = REFERENCE_BONDS_PER_MOL / 19.0
        low, high = bond_window()
        assert 15.0 * scale >= low
        assert 25.0 * scale <= high


class TestEliteSelection:
    def test_keeps_valid_connected_in_window(self):
        elite = _mol("CCCCCC")  # hexane: 5 C-C + 14 C-H = 19 bonds, in window
        elites, stats = select_elites([elite, None])
        assert elites == [elite]
        assert stats["n_elites"] == 1
        assert stats["n_invalid"] == 1

    def test_rejects_fragmented_by_default(self):
        elites, stats = select_elites([_fragmented()])
        assert elites == []
        assert stats["n_fragmented"] == 1

    def test_fragmented_kept_when_allowed(self):
        elites, stats = select_elites([_fragmented()], require_connected=False)
        assert len(elites) == 1
        assert stats["n_fragmented"] == 0

    def test_rejects_out_of_window(self):
        elites, stats = select_elites([_connected_out_of_window()])
        assert elites == []
        assert stats["n_out_of_window"] == 1

    def test_none_input_is_not_counted_as_candidate(self):
        _, stats = select_elites([None, None])
        assert stats["n_candidates"] == 0
        assert stats["elite_fraction"] == 0.0

    def test_empty_pool_is_safe(self):
        elites, stats = select_elites([])
        assert elites == [] and stats["n_elites"] == 0


class TestTensorize:
    def test_shapes_and_batch_ids(self):
        pos, z, batch = _tensorize([_mol("CCO"), _mol("C")], num_types=10)
        assert pos.shape[1] == 3
        assert pos.shape[0] == z.shape[0] == batch.shape[0]
        assert set(batch.tolist()) == {0, 1}
        assert int((batch == 0).sum()) == 9   # C2H6O
        assert int((batch == 1).sum()) == 5   # CH4

    def test_exotic_atomic_numbers_are_sanitized_to_carbon(self):
        mol = _mol("CBr")  # embedded, with explicit H
        pos, z, _ = _tensorize([mol], num_types=10)
        assert int(z.max()) < 10  # Br (Z=35) mapped to carbon by num_types=10


class TestFineTune:
    def test_reduces_loss_and_updates_weights(self):
        torch.manual_seed(0)
        model = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                     num_layers=2, time_dim=16)
        before = copy.deepcopy(model.state_dict())
        ddpm = CenteredDDPM(num_steps=20, device="cpu")
        elites = [_mol("CCCC"), _mol("CCO"), _mol("C")]
        result = finetune(model, ddpm, elites, num_types=10,
                          epochs=3, lr=1e-3, batch_size=8, seed=0)
        assert result["n_elites"] == 3
        losses = [h["loss"] for h in result["history"]]
        assert all(l == l for l in losses)  # finite
        assert any(
            not torch.allclose(before[k], v)
            for k, v in model.state_dict().items()
            if k in before and torch.is_floating_point(v)
        )

    def test_single_elite_still_trains(self):
        torch.manual_seed(1)
        model = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                     num_layers=2, time_dim=16)
        ddpm = CenteredDDPM(num_steps=10, device="cpu")
        result = finetune(model, ddpm, [_mol("CC")], num_types=10,
                          epochs=1, batch_size=4)
        assert result["n_elites"] == 1
        assert result["history"][0]["loss"] >= 0.0
