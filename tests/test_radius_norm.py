"""Tests for radius-aware coordinate normalization (plan Intervention 3).

The point of the intervention is that the *same* noise scale means the same
relative corruption for a 9-atom and a 40-atom molecule. The original plan's
formulation (scale the noise by the ground-truth molecular radius) cannot run
at sampling time, where no ground-truth conformer exists; here the scale is
``R(N) = coeff · N^(1/3)``, a function of the atom count only, and the whole
forward/reverse process runs in that normalized space.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.objectives import x0_geometry_consistency, x0_valence_penalty
from src.sampling import sample_molecules


class _ZeroModel(torch.nn.Module):
    """Denoiser stub: predicts zero noise and constant type logits.

    With ``eps = 0`` the DDIM update is fully deterministic given the initial
    draw, so two runs that differ only by the coordinate normalization must be
    related by exactly the per-molecule scale ``R(N)``.
    """

    num_types = 10
    self_condition = False
    knn_schedule = None
    objective = "eps"

    def forward(self, z, pos, edge_index, t, batch, cond=None,
                x0_estimate=None, edge_index_per_layer=None):
        return (
            torch.zeros_like(pos),
            torch.zeros(pos.size(0), self.num_types, device=pos.device, dtype=pos.dtype),
            None,
        )


class TestRadiusFunction:
    def test_off_is_identity(self):
        ddpm = CenteredDDPM(num_steps=10, device="cpu")
        r = ddpm.molecule_radius(torch.tensor([1, 10, 100]))
        assert torch.allclose(r, torch.ones(3))

    def test_atom_count_cube_root(self):
        ddpm = CenteredDDPM(num_steps=10, device="cpu", radius_norm="atom_count",
                            radius_coeff=1.06)
        r = ddpm.molecule_radius(torch.tensor([1, 8, 27, 41]))
        expected = 1.06 * torch.tensor([1.0, 2.0, 3.0, 41 ** (1 / 3)])
        assert torch.allclose(r, expected, rtol=1e-6)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="radius_norm"):
            CenteredDDPM(num_steps=10, device="cpu", radius_norm="molecular")

    def test_nonpositive_coeff_raises(self):
        with pytest.raises(ValueError, match="radius_coeff"):
            CenteredDDPM(num_steps=10, device="cpu", radius_norm="atom_count",
                         radius_coeff=0.0)

    def test_from_config_round_trip(self):
        cfg = {
            "num_steps": 50, "beta_start": 1e-4, "beta_end": 0.02,
            "schedule": "cosine", "radius_norm": "atom_count",
            "radius_coeff": 1.2,
        }
        ddpm = CenteredDDPM.from_config(cfg, device="cpu")
        assert ddpm.radius_norm == "atom_count"
        assert ddpm.radius_coeff == pytest.approx(1.2)
        # Missing keys default to the legacy (unnormalized) behaviour.
        plain = CenteredDDPM.from_config(
            {"num_steps": 50, "beta_start": 1e-4, "beta_end": 0.02}, device="cpu")
        assert plain.radius_norm == "none"


class TestTransform:
    def test_normalize_denormalize_round_trip(self):
        ddpm = CenteredDDPM(num_steps=10, device="cpu", radius_norm="atom_count")
        batch = torch.tensor([0, 0, 0, 1, 1])
        pos = torch.randn(5, 3, generator=torch.Generator().manual_seed(0))
        back = ddpm.denormalize(ddpm.normalize(pos, batch), batch)
        assert torch.allclose(back, pos, atol=1e-6)

    def test_off_is_a_passthrough(self):
        ddpm = CenteredDDPM(num_steps=10, device="cpu")
        pos = torch.randn(4, 3)
        batch = torch.zeros(4, dtype=torch.long)
        assert ddpm.normalize(pos, batch) is pos
        assert ddpm.denormalize(pos, batch) is pos

    def test_add_noise_equals_unscaled_noise_on_normalized_input(self):
        """The normalized forward process is exactly the old process applied to
        x0/R — same schedule, same noise draw."""
        torch.manual_seed(0)
        plain = CenteredDDPM(num_steps=50, device="cpu", schedule="cosine")
        normed = CenteredDDPM(num_steps=50, device="cpu", schedule="cosine",
                              radius_norm="atom_count")
        batch = torch.tensor([0, 0, 0, 0, 1, 1])
        pos = torch.randn(6, 3, generator=torch.Generator().manual_seed(1))
        t = torch.tensor([10, 30])

        torch.manual_seed(7)
        x_norm, eps_norm = normed.add_noise(pos, t, batch)
        scale = normed.per_atom_radius(batch)
        torch.manual_seed(7)
        x_plain, eps_plain = plain.add_noise(pos / scale, t, batch)

        assert torch.allclose(x_norm, x_plain, atol=1e-6)
        assert torch.allclose(eps_norm, eps_plain, atol=1e-6)

    def test_noise_stays_centered_per_molecule(self):
        normed = CenteredDDPM(num_steps=50, device="cpu", radius_norm="atom_count")
        batch = torch.tensor([0, 0, 0, 1, 1, 1])
        pos = torch.randn(6, 3, generator=torch.Generator().manual_seed(2))
        _, eps = normed.add_noise(pos, torch.tensor([20, 20]), batch)
        assert eps[:3].mean(0).abs().max().item() < 1e-6
        assert eps[3:].mean(0).abs().max().item() < 1e-6


class TestSamplerSpace:
    def test_reverse_process_returns_angstrom(self):
        """Same seed, zero-noise denoiser: the normalized run must equal the
        legacy run scaled by exactly R(N) per molecule."""
        torch.manual_seed(0)
        model = _ZeroModel()
        tp = TypeDDPM(num_steps=20, device="cpu")
        counts = torch.tensor([8, 27])

        torch.manual_seed(4)
        pos_plain, _ = sample_molecules(
            model, CenteredDDPM(num_steps=20, device="cpu"), tp, counts,
            device="cpu", ddim_steps=5,
        )
        torch.manual_seed(4)
        normed = CenteredDDPM(num_steps=20, device="cpu",
                              radius_norm="atom_count")
        pos_norm, _ = sample_molecules(
            model, normed, tp, counts, device="cpu", ddim_steps=5,
        )

        batch = torch.tensor([0] * 8 + [1] * 27)
        scale = normed.per_atom_radius(batch)
        assert torch.allclose(pos_norm, pos_plain * scale, atol=1e-4)

        # Per-molecule CoM discipline survives the scaling.
        assert pos_norm[:8].mean(0).abs().max().item() < 1e-4
        assert pos_norm[8:].mean(0).abs().max().item() < 1e-4


class TestObjectivesStayInAngstrom:
    def test_geometry_term_needs_the_scale(self):
        target = torch.tensor([
            [0.0, 0.0, 0.0], [1.54, 0.0, 0.0], [3.08, 0.0, 0.0],
        ])
        batch = torch.zeros(3, dtype=torch.long)
        alpha_bars = torch.ones(1)
        t = torch.zeros(1, dtype=torch.long)
        # x0 = noisy_pos (alpha_bar = 1, zero noise prediction) = target/2,
        # i.e. what a radius-normalized model emits.
        noisy_pos = target / 2.0
        zero = torch.zeros_like(target)
        kw = dict(
            noise_pred=zero, noisy_pos=noisy_pos, t=t, batch=batch,
            target_pos=target, alpha_bars=alpha_bars, tau=200,
        )
        assert x0_geometry_consistency(**kw).item() > 0.0
        scale = torch.full((3, 1), 2.0)
        assert x0_geometry_consistency(**kw, x0_scale=scale).item() == \
            pytest.approx(0.0, abs=1e-6)

    def test_valence_penalty_accepts_the_scale(self):
        target = torch.tensor([
            [0.0, 0.0, 0.0], [1.54, 0.0, 0.0], [3.08, 0.0, 0.0],
        ])
        batch = torch.zeros(3, dtype=torch.long)
        t = torch.zeros(1, dtype=torch.long)
        z = torch.tensor([6, 6, 6])
        logits = torch.zeros(3, 10)
        out = x0_valence_penalty(
            torch.zeros_like(target), target / 2.0, logits, t, batch, z,
            num_types=10, type_to_z={i: i for i in range(10)},
            alpha_bars=torch.ones(1), lambda_weight=1.0, tau=200,
            x0_scale=torch.full((3, 1), 2.0),
        )
        assert torch.isfinite(out)


class TestResumeGuard:
    def test_guard_covers_radius_norm(self):
        from train import train_diffusion

        src = inspect.getsource(train_diffusion)
        assert "radius_norm" in src
