"""Tests for the sampler/eval interventions of the BBB geometry plan.

Intervention 2 — coordinate temperature and the largest-fragment fallback.
Intervention 3 gets its own file (tests/test_radius_norm.py) because it also
touches the DDPM forward process and the trainers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.egnn import EquivariantGenerator
from src.sampling import sample_molecules
from src.utils.evaluation import coords_and_types_to_mol


def _model(**kwargs) -> EquivariantGenerator:
    torch.manual_seed(0)
    return EquivariantGenerator(
        num_types=10, node_dim=16, edge_dim=16, num_layers=2, time_dim=16, **kwargs
    )


def _counts(n_mols: int, n_atoms: int) -> torch.Tensor:
    return torch.full((n_mols,), n_atoms, dtype=torch.long)


class TestCoordinateTemperature:
    def test_zero_is_bit_identical(self):
        m = _model()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        torch.manual_seed(3)
        pos_a, z_a = sample_molecules(m, c, tp, _counts(2, 5), device="cpu",
                                      ddim_steps=5)
        torch.manual_seed(3)
        pos_b, z_b = sample_molecules(m, c, tp, _counts(2, 5), device="cpu",
                                      ddim_steps=5, coord_temperature=0.0)
        assert torch.equal(pos_a, pos_b) and torch.equal(z_a, z_b)

    def test_positive_temperature_changes_the_sample(self):
        m = _model()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        torch.manual_seed(5)
        pos_a, _ = sample_molecules(m, c, tp, _counts(1, 6), device="cpu",
                                    ddim_steps=6)
        torch.manual_seed(5)
        pos_b, _ = sample_molecules(m, c, tp, _counts(1, 6), device="cpu",
                                    ddim_steps=6, coord_temperature=1.0)
        assert not torch.allclose(pos_a, pos_b)

    def test_sample_stays_centered_and_finite(self):
        """The perturbation must not break the per-molecule CoM discipline."""
        m = _model()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        pos, _ = sample_molecules(m, c, tp, torch.tensor([5, 7]), device="cpu",
                                  ddim_steps=6, coord_temperature=0.5)
        assert torch.isfinite(pos).all()
        com = pos[:5].mean(0)
        assert com.abs().max().item() < 1e-4

    def test_ancestral_path_supported(self):
        m = _model()
        c = CenteredDDPM(num_steps=4, device="cpu")
        tp = TypeDDPM(num_steps=4, device="cpu")
        pos, _ = sample_molecules(m, c, tp, _counts(1, 4), device="cpu",
                                  coord_temperature=0.3)
        assert torch.isfinite(pos).all()


class TestLargestFragmentFallback:
    @staticmethod
    def _chain_plus_speck():
        # 4 carbons on a 1.54 Å chain (one fragment) + 1 carbon 20 Å away.
        pos = np.array([
            [0.0, 0.0, 0.0], [1.54, 0.0, 0.0],
            [3.08, 0.0, 0.0], [4.62, 0.0, 0.0],
            [20.0, 20.0, 0.0],
        ])
        z = np.array([6, 6, 6, 6, 6])
        return pos, z

    def test_off_by_default_keeps_every_fragment(self):
        pos, z = self._chain_plus_speck()
        mol = coords_and_types_to_mol(pos, z, sanitize=False)
        assert mol is not None
        assert mol.GetNumAtoms() == 5

    def test_keeps_only_the_largest_component(self):
        pos, z = self._chain_plus_speck()
        mol = coords_and_types_to_mol(
            pos, z, sanitize=False, keep_largest_fragment=True,
        )
        assert mol is not None
        assert mol.GetNumAtoms() == 4

    def test_all_disconnected_reduces_to_one_atom(self):
        pos = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        z = np.array([6, 6])
        mol = coords_and_types_to_mol(
            pos, z, sanitize=False, keep_largest_fragment=True,
        )
        assert mol is not None
        assert mol.GetNumAtoms() == 1

    def test_single_fragment_is_unchanged(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])
        z = np.array([6, 6])
        base = coords_and_types_to_mol(pos, z, sanitize=False)
        kept = coords_and_types_to_mol(
            pos, z, sanitize=False, keep_largest_fragment=True,
        )
        assert base.GetNumAtoms() == kept.GetNumAtoms() == 2
