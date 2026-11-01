"""Tests for attractive connectivity guidance (rescue plan R4).

The plan's measured failure is that V5's inter-fragment gaps sit 0.1-0.5 Å
*outside* the eval bond window while every existing guidance term is
repulsive. The properties pinned here:

- a molecule at real covalent geometry feels (numerically) zero attraction,
- a near-miss pair inside ``window + pad`` is pulled *toward* the covalent
  distance — the only sampler term that can close a fragment gap,
- pairs beyond ``window + pad`` are left alone (no long-range collapse),
- the term is attractive only: it never pushes a pair apart,
- the same contract as ``steric_guidance`` (w <= 0 no-op, capped step,
  CoM-preserving, clamped).
"""

from __future__ import annotations

import pytest
import torch

from src.guidance import (
    connectivity_energy,
    connectivity_guidance,
    connectivity_guidance_weight,
)
from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.egnn import EquivariantGenerator
from src.sampling import sample_molecules


def _pair(distance: float, z_left: int = 6, z_right: int = 6):
    pos = torch.tensor([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    batch = torch.zeros(2, dtype=torch.long)
    z = torch.tensor([z_left, z_right], dtype=torch.long)
    return pos, batch, z


class TestEnergy:
    def test_pairs_at_the_covalent_distance_are_free(self):
        # C-C radii sum = 1.52 Å: zero gap -> zero energy and zero gradient.
        pos, batch, z = _pair(1.52)
        assert connectivity_energy(pos, batch, atom_types=z).item() == pytest.approx(
            0.0, abs=1e-9
        )
        assert torch.allclose(
            connectivity_guidance(pos, batch, atom_types=z, weight=1.0),
            pos,
            atol=1e-6,
        )

    def test_real_geometry_is_numerically_free(self):
        """A 3-carbon chain at real bond lengths costs ~nothing (the plan's
        Finding A: real bonds are 1.52-1.54 Å, the target is the radii sum)."""
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.54, 0.0, 0.0], [3.08, 0.0, 0.0]]
        )
        batch = torch.zeros(3, dtype=torch.long)
        z = torch.full((3,), 6, dtype=torch.long)
        assert connectivity_energy(pos, batch, atom_types=z).item() < 1e-3

    def test_near_miss_gap_inside_the_pad_is_attracted(self):
        # C-C window = 1.97 A; 2.2 A is 0.23 outside it, inside the 0.5 pad.
        pos, batch, z = _pair(2.2)
        assert connectivity_energy(pos, batch, atom_types=z).item() > 0.0

        guided = connectivity_guidance(pos, batch, atom_types=z, weight=1.0)
        d_before = (pos[0] - pos[1]).norm().item()
        d_after = (guided[0] - guided[1]).norm().item()
        assert d_after < d_before, "a near-miss pair must be pulled closer"

    def test_pairs_beyond_the_pad_are_left_alone(self):
        """No long-range attraction: 3.0 A > window + pad (2.47 A for C-C)."""
        pos, batch, z = _pair(3.0)
        assert connectivity_energy(pos, batch, atom_types=z).item() == 0.0
        assert torch.equal(
            connectivity_guidance(pos, batch, atom_types=z, weight=1.0), pos
        )

    def test_term_is_attractive_only(self):
        """A pair *inside* the covalent distance must not be pushed further
        apart (relu gate); steric guidance owns overlap repulsion."""
        pos, batch, z = _pair(1.0)
        assert connectivity_energy(pos, batch, atom_types=z).item() == pytest.approx(
            0.0, abs=1e-9
        )
        assert torch.equal(
            connectivity_guidance(pos, batch, atom_types=z, weight=1.0), pos
        )

    def test_molecules_do_not_interact_across_the_batch(self):
        pos = torch.tensor([
            [0.0, 0.0, 0.0], [2.2, 0.0, 0.0],        # molecule 0: near-miss gap
            [0.0, 60.0, 0.0], [2.2, 60.0, 0.0],      # molecule 1: identical gap
        ])
        batch = torch.tensor([0, 0, 1, 1])
        z = torch.full((4,), 6, dtype=torch.long)
        one = connectivity_energy(pos[:2], batch[:2], atom_types=z[:2]).item()
        two = connectivity_energy(pos, batch, atom_types=z).item()
        assert two == pytest.approx(2.0 * one, rel=1e-6)

    def test_empty_input_is_finite(self):
        e = connectivity_energy(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long))
        assert e.item() == 0.0


class TestGuidanceStep:
    def test_weight_zero_is_a_noop(self):
        pos, batch, z = _pair(2.2)
        assert connectivity_guidance(pos, batch, atom_types=z, weight=0.0) is pos

    def test_step_is_capped(self):
        pos, batch, z = _pair(2.2)
        out = connectivity_guidance(
            pos, batch, atom_types=z, weight=1e3, max_step=0.1
        )
        assert (out - pos).abs().max().item() <= 0.1 + 1e-6

    def test_clamp_respected(self):
        pos, batch, z = _pair(2.2)
        out = connectivity_guidance(
            pos, batch, atom_types=z, weight=10.0, max_step=10.0, clamp=0.5
        )
        assert out.abs().max().item() <= 0.5 + 1e-6

    def test_centre_of_mass_is_preserved(self):
        pos, batch, z = _pair(2.2)
        out = connectivity_guidance(pos, batch, atom_types=z, weight=1.0)
        # Symmetric pair: the two displacements must cancel.
        assert (out - pos).sum(dim=0).abs().max().item() < 1e-6

    def test_single_atom_batch_is_noop(self):
        pos = torch.zeros(1, 3)
        out = connectivity_guidance(pos, torch.zeros(1, dtype=torch.long), weight=1.0)
        assert out is pos


class TestRamp:
    def test_zero_before_the_ramp_point(self):
        for step_idx in (0, 3, 6):
            assert connectivity_guidance_weight(step_idx, 10, 0.7, 1.0) == 0.0
        assert connectivity_guidance_weight(7, 10, 0.7, 1.0) > 0.0

    def test_zero_max_weight_is_off(self):
        assert connectivity_guidance_weight(9, 10, 0.7, 0.0) == 0.0

    def test_linear_ramp_over_the_tail(self):
        w8 = connectivity_guidance_weight(8, 10, 0.7, 1.0)
        w9 = connectivity_guidance_weight(9, 10, 0.7, 1.0)
        assert 0.0 < w8 < w9 == 1.0


class TestSamplerIntegration:
    def _tiny_model(self):
        torch.manual_seed(0)
        return EquivariantGenerator(
            num_types=10, node_dim=16, edge_dim=16, num_layers=2, time_dim=16,
        )

    def test_guidance_runs_and_stays_finite(self):
        m = self._tiny_model()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        pos, _ = sample_molecules(
            m, c, tp, torch.tensor([6, 7]), device="cpu", ddim_steps=6,
            connectivity_weight=2.0, connectivity_start_frac=0.5,
        )
        assert pos.shape == (13, 3)
        assert torch.isfinite(pos).all()

    def test_weight_zero_is_bit_identical(self):
        m = self._tiny_model()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        torch.manual_seed(11)
        pos_a, z_a = sample_molecules(
            m, c, tp, torch.tensor([5, 6]), device="cpu", ddim_steps=5)
        torch.manual_seed(11)
        pos_b, z_b = sample_molecules(
            m, c, tp, torch.tensor([5, 6]), device="cpu", ddim_steps=5,
            connectivity_weight=0.0,
        )
        assert torch.equal(pos_a, pos_b) and torch.equal(z_a, z_b)

    def test_guidance_changes_the_trajectory(self):
        m = self._tiny_model()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        torch.manual_seed(7)
        pos_a, _ = sample_molecules(
            m, c, tp, torch.tensor([6]), device="cpu", ddim_steps=6)
        torch.manual_seed(7)
        pos_b, _ = sample_molecules(
            m, c, tp, torch.tensor([6]), device="cpu", ddim_steps=6,
            connectivity_weight=20.0, connectivity_start_frac=0.0,
        )
        assert not torch.allclose(pos_a, pos_b)
