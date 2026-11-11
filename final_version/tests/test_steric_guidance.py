"""Tests for test-time steric guidance (plan Intervention 1, corrected).

The properties pinned here are the ones the original plan got wrong:

- real covalent bonds must feel **zero** force (the plan's clash window
  actually repelled every bond at 1.2–1.8 Å),
- a genuine atomic overlap must be pushed apart (the plan's window applied
  zero force to those),
- the energy must be zero for a real molecule and positive for a clump,
- guidance must act on the clean x0 (guided output differs, trajectory runs,
  and w=0 is a bit-exact no-op).
"""

from __future__ import annotations

import pytest
import torch

from src.guidance import (
    COVALENT_RADII,
    atom_radii,
    steric_energy,
    steric_guidance,
    steric_guidance_weight,
)
from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.egnn import EquivariantGenerator
from src.sampling import sample_molecules


def _ethane_like() -> tuple[torch.Tensor, torch.Tensor]:
    """Two carbons + six hydrogens at real bond lengths (all single bonds)."""
    pos = torch.tensor([
        [-0.77, 0.0, 0.0],    # C
        [0.77, 0.0, 0.0],     # C   C–C = 1.54 Å
        [-1.20, 0.89, 0.0],   # H   C–H  = 1.09 Å
        [-1.20, -0.45, 0.78],
        [-1.20, -0.45, -0.78],
        [1.20, 0.89, 0.0],
        [1.20, -0.45, 0.78],
        [1.20, -0.45, -0.78],
    ])
    z = torch.tensor([6, 6, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    return pos, z


class TestRadiiLookup:
    def test_radii_follow_atomic_number(self):
        z = torch.tensor([6, 1, 8], dtype=torch.long)
        r = atom_radii(z, 3, "cpu")
        assert torch.allclose(
            r, torch.tensor([COVALENT_RADII[6], COVALENT_RADII[1], COVALENT_RADII[8]])
        )

    def test_exotic_and_missing_types_fall_back_to_carbon(self):
        # Z=53 (I) is in the table; Z=30 (Zn) is not; class 0 is padding.
        z = torch.tensor([53, 30, 0], dtype=torch.long)
        r = atom_radii(z, 3, "cpu")
        assert r[0].item() == pytest.approx(COVALENT_RADII[53])
        assert r[1].item() == pytest.approx(COVALENT_RADII[6])
        assert r[2].item() == pytest.approx(COVALENT_RADII[6])

    def test_none_types_use_fallback(self):
        r = atom_radii(None, 4, "cpu")
        assert torch.allclose(r, torch.full((4,), COVALENT_RADII[6]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            atom_radii(torch.tensor([6, 6]), 3, "cpu")


class TestEnergy:
    def test_real_bonds_are_free(self):
        """A molecule at real geometry must have exactly zero overlap energy —
        i.e. guidance cannot fight correct chemistry."""
        pos, z = _ethane_like()
        e = steric_energy(pos, torch.zeros(8, dtype=torch.long), atom_types=z)
        assert e.item() == pytest.approx(0.0, abs=1e-8)

    def test_single_overlap_is_positive_and_pushed_apart(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])  # 0.4 Å < 1.14 Å
        batch = torch.zeros(2, dtype=torch.long)
        z = torch.tensor([6, 6], dtype=torch.long)
        e = steric_energy(pos, batch, atom_types=z)
        assert e.item() > 0.0

        guided = steric_guidance(pos, batch, atom_types=z, weight=1.0)
        d_before = (pos[0] - pos[1]).norm()
        d_after = (guided[0] - guided[1]).norm()
        assert d_after > d_before, "guidance must increase an overlapping pair's distance"

    def test_clump_of_atoms_costs_more_than_a_chain(self):
        """The soup signature (many atoms inside one contact shell) must cost
        strictly more than the same atoms spread out."""
        z = torch.full((8,), 6, dtype=torch.long)
        batch = torch.zeros(8, dtype=torch.long)
        clump = torch.randn(8, 3, generator=torch.Generator().manual_seed(0)) * 0.15
        chain = torch.stack([
            torch.tensor([1.54 * i, 0.0, 0.0]) for i in range(8)
        ])
        assert steric_energy(clump, batch, atom_types=z).item() > \
            steric_energy(chain, batch, atom_types=z).item()

    def test_molecules_do_not_interact_across_the_batch(self):
        """Per-molecule blocks: two separate molecules never produce cross pairs."""
        pos = torch.tensor([
            [0.0, 0.0, 0.0], [1.54, 0.0, 0.0],      # molecule 0 (bonded, free)
            [0.0, 50.0, 0.0], [1.54, 50.0, 0.0],    # molecule 1 (bonded, free)
        ])
        batch = torch.tensor([0, 0, 1, 1])
        z = torch.full((4,), 6, dtype=torch.long)
        assert steric_energy(pos, batch, atom_types=z).item() == 0.0

    def test_gradient_is_finite_at_coincident_atoms(self):
        pos = torch.zeros(2, 3, requires_grad=True)
        batch = torch.zeros(2, dtype=torch.long)
        e = steric_energy(pos, batch)
        (g,) = torch.autograd.grad(e, pos)
        assert torch.isfinite(g).all()

    def test_sigma_scale_is_the_strength_knob(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]])  # compressed C–C
        batch = torch.zeros(2, dtype=torch.long)
        z = torch.tensor([6, 6], dtype=torch.long)
        assert steric_energy(pos, batch, atom_types=z, sigma_scale=0.6).item() == 0.0
        assert steric_energy(pos, batch, atom_types=z, sigma_scale=0.95).item() > 0.0

    def test_empty_input_is_finite(self):
        e = steric_energy(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long))
        assert e.item() == 0.0

    def test_r_clash_adds_an_absolute_severe_overlap_term(self):
        """Plan T-mode: r_clash=1.1 charges a 1.0 A contact that the C-H
        contact distance (0.81 A) alone would ignore."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        batch = torch.zeros(2, dtype=torch.long)
        z = torch.tensor([1, 1], dtype=torch.long)  # H-H contact = 0.62 A
        base = steric_energy(pos, batch, atom_types=z).item()
        with_clash = steric_energy(
            pos, batch, atom_types=z, r_clash=1.1
        ).item()
        assert base == 0.0 and with_clash > 0.0


class TestGuidanceRamp:
    def test_zero_before_the_ramp_point(self):
        # progress = step_idx / 9; 0, 0.33, 0.67 are all <= 0.7, 7/9 = 0.78 is not
        for step_idx in (0, 3, 6):
            assert steric_guidance_weight(step_idx, 10, 0.7, 1.0) == 0.0
        assert steric_guidance_weight(7, 10, 0.7, 1.0) > 0.0

    def test_zero_max_weight_is_off(self):
        assert steric_guidance_weight(9, 10, 0.7, 0.0) == 0.0

    def test_linear_ramp_over_the_tail(self):
        # step_idx 8,9 of 10 -> progress 8/9, 9/9 = .889, 1.0
        w8 = steric_guidance_weight(8, 10, 0.7, 1.0)
        w9 = steric_guidance_weight(9, 10, 0.7, 1.0)
        assert 0.0 < w8 < w9 == 1.0
        assert w8 == pytest.approx((8 / 9 - 0.7) / 0.3, rel=1e-6)

    def test_ramps_on_step_idx_not_t_cur(self):
        """step_idx counts up while t_cur counts down: a late step_idx must
        get weight even though the corresponding t_cur is small."""
        num_steps = 10
        late_step = 9
        assert steric_guidance_weight(late_step, num_steps, 0.7, 1.0) > 0.0
        # The old construction (gate on a small t_cur AND a high frac) is
        # unsatisfiable — this documents why the index must be step_idx.
        assert steric_guidance_weight(1, num_steps, 0.7, 1.0) == 0.0

    def test_single_step_sampling_has_a_finite_weight(self):
        assert steric_guidance_weight(0, 1, 0.7, 1.0) <= 1.0


class TestGuidanceStep:
    def test_weight_zero_is_a_noop(self):
        pos, z = _ethane_like()
        batch = torch.zeros(8, dtype=torch.long)
        out = steric_guidance(pos, batch, atom_types=z, weight=0.0)
        assert out is pos  # identity, not a copy

    def test_r_clash_forwarded(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        batch = torch.zeros(2, dtype=torch.long)
        z = torch.tensor([1, 1], dtype=torch.long)
        plain = steric_guidance(pos, batch, atom_types=z, weight=1.0)
        clashing = steric_guidance(
            pos, batch, atom_types=z, weight=1.0, r_clash=1.1
        )
        assert torch.equal(plain, pos)  # 1.0 A is free for H-H alone
        assert (clashing - pos).abs().max().item() > 0.0

    def test_step_is_capped(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
        batch = torch.zeros(2, dtype=torch.long)
        out = steric_guidance(pos, batch, weight=1e3, max_step=0.25)
        assert (out - pos).abs().max().item() <= 0.25 + 1e-6

    def test_clamp_respected(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        batch = torch.zeros(2, dtype=torch.long)
        out = steric_guidance(pos, batch, weight=10.0, max_step=10.0, clamp=0.2)
        assert out.abs().max().item() <= 0.2 + 1e-6

    def test_single_atom_batch_is_noop(self):
        pos = torch.zeros(1, 3)
        out = steric_guidance(pos, torch.zeros(1, dtype=torch.long), weight=1.0)
        assert out is pos


class TestSamplerIntegration:
    def _tiny_model(self, **kwargs):
        torch.manual_seed(0)
        return EquivariantGenerator(
            num_types=10, node_dim=16, edge_dim=16, num_layers=2, time_dim=16,
            **kwargs,
        )

    def test_guidance_runs_and_stays_finite(self):
        m = self._tiny_model()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        pos, z = sample_molecules(
            m, c, tp, torch.tensor([6, 7]), device="cpu", ddim_steps=6,
            steric_weight=5.0, steric_start_frac=0.5,
        )
        assert pos.shape == (13, 3)
        assert torch.isfinite(pos).all()

    def test_guidance_changes_the_trajectory(self):
        """Same seed + same model: w>0 must move atoms relative to w=0 (the
        guidance is live, not decorative)."""
        m = self._tiny_model()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        torch.manual_seed(7)
        pos_a, _ = sample_molecules(
            m, c, tp, torch.tensor([6]), device="cpu", ddim_steps=6)
        torch.manual_seed(7)
        pos_b, _ = sample_molecules(
            m, c, tp, torch.tensor([6]), device="cpu", ddim_steps=6,
            steric_weight=20.0, steric_start_frac=0.0,
        )
        assert not torch.allclose(pos_a, pos_b)

    def test_ramp_late_in_sampling_changes_the_trajectory(self):
        """With the default 0.7 ramp start the last third of steps must still
        bite. ``steric_r_clash`` guarantees a nonzero energy regardless of
        what the random model happened to predict."""
        m = self._tiny_model()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        torch.manual_seed(21)
        pos_a, _ = sample_molecules(
            m, c, tp, torch.tensor([6]), device="cpu", ddim_steps=6)
        torch.manual_seed(21)
        pos_b, _ = sample_molecules(
            m, c, tp, torch.tensor([6]), device="cpu", ddim_steps=6,
            steric_weight=20.0, steric_start_frac=0.7, steric_r_clash=6.0,
        )
        assert not torch.allclose(pos_a, pos_b)

    def test_ramp_schedule_is_what_gates_the_tail(self):
        """ramp_start_frac beyond 1.0 disables guidance entirely (weight
        never exceeds max_weight * 0), so the sample matches the baseline."""
        m = self._tiny_model()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        torch.manual_seed(33)
        pos_a, _ = sample_molecules(
            m, c, tp, torch.tensor([5]), device="cpu", ddim_steps=5)
        torch.manual_seed(33)
        pos_b, _ = sample_molecules(
            m, c, tp, torch.tensor([5]), device="cpu", ddim_steps=5,
            steric_weight=20.0, steric_start_frac=1.5, steric_r_clash=6.0,
        )
        assert torch.equal(pos_a, pos_b)

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
            steric_weight=0.0,
        )
        assert torch.equal(pos_a, pos_b) and torch.equal(z_a, z_b)

    def test_ancestral_ddpm_path_is_supported(self):
        m = self._tiny_model()
        c = CenteredDDPM(num_steps=4, device="cpu")
        tp = TypeDDPM(num_steps=4, device="cpu")
        pos, z = sample_molecules(
            m, c, tp, torch.tensor([4]), device="cpu",
            steric_weight=2.0, steric_start_frac=0.0,
        )
        assert torch.isfinite(pos).all()
