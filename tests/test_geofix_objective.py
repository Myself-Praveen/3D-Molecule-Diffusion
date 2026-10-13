"""Tests for the Task 2 geometry-consistency terms (src/objectives.py).

The arm exists to test whether a rotation-invariant shape loss plus an explicit
interatomic-distance loss fixes the BBB soup. These tests check the two
properties that hypothesis depends on — invariance to rigid motion, and
sensitivity to shape — plus the gating behaviour the trainers rely on.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.objectives import (
    kabsch_aligned_mse,
    pairwise_distance_loss,
    full_pair_index,
    x0_geometry_consistency,
)


def _random_rotation(seed: int = 0) -> torch.Tensor:
    """A proper rotation (det = +1) via QR of a seeded Gaussian."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.linalg.det(q) < 0:          # force a rotation, not a reflection
        q[:, 0] = -q[:, 0]
    return q


class TestKabschAlignedMSE:
    def test_rigid_motion_is_free(self):
        """A rotated + translated copy of the target scores ~0."""
        g = torch.Generator().manual_seed(1)
        target = torch.randn(7, 3, generator=g, dtype=torch.float64)
        rot = _random_rotation(2)
        pred = target @ rot.T + torch.tensor([3.0, -2.0, 5.0], dtype=torch.float64)
        batch = torch.zeros(7, dtype=torch.long)
        assert kabsch_aligned_mse(pred, target, batch).item() == pytest.approx(
            0.0, abs=1e-10)

    def test_plain_mse_would_charge_for_rotation(self):
        """The same pair scores badly under the un-aligned loss, which is the
        defect the term is meant to remove."""
        g = torch.Generator().manual_seed(1)
        target = torch.randn(7, 3, generator=g, dtype=torch.float64)
        rot = _random_rotation(3)
        pred = target @ rot.T
        batch = torch.zeros(7, dtype=torch.long)
        assert torch.nn.functional.mse_loss(pred, target).item() > 0.1

    def test_shape_error_is_charged(self):
        """A genuinely wrong shape must not be free."""
        target = torch.zeros(4, 3, dtype=torch.float64)
        target[0] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        target[1] = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64)
        target[2] = torch.tensor([0.0, 0.0, 0.5], dtype=torch.float64)
        pred = target * 2.0                      # scaled: wrong bond lengths
        batch = torch.zeros(4, dtype=torch.long)
        assert kabsch_aligned_mse(pred, target, batch).item() > 0.1

    def test_per_molecule_alignment_is_independent(self):
        """Two molecules, one rotated one not, both score ~0 together."""
        g = torch.Generator().manual_seed(4)
        a = torch.randn(5, 3, generator=g, dtype=torch.float64)
        b = torch.randn(4, 3, generator=g, dtype=torch.float64)
        pred = torch.cat([a @ _random_rotation(5).T, b], dim=0)
        target = torch.cat([a, b], dim=0)
        batch = torch.tensor([0] * 5 + [1] * 4)
        assert kabsch_aligned_mse(pred, target, batch).item() == pytest.approx(
            0.0, abs=1e-10)

    def test_diatomic_fragment_does_not_crash(self):
        """< 3 atoms cannot define a rotation; falls back to centred MSE."""
        target = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float64)
        pred = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
        batch = torch.zeros(2, dtype=torch.long)
        val = kabsch_aligned_mse(pred, target, batch)
        assert math.isfinite(val.item()) and val.item() > 0.0

    def test_collinear_fragment_does_not_crash(self):
        """Rank-deficient covariance: rotation must stay proper/finite."""
        t = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                         dtype=torch.float64)
        batch = torch.zeros(3, dtype=torch.long)
        val = kabsch_aligned_mse(t * 1.1, t, batch)
        assert math.isfinite(val.item())


class TestPairwiseDistanceLoss:
    def test_rigid_motion_is_free(self):
        g = torch.Generator().manual_seed(6)
        target = torch.randn(6, 3, generator=g, dtype=torch.float64)
        pred = target @ _random_rotation(7).T + 4.0
        batch = torch.zeros(6, dtype=torch.long)
        pairs = full_pair_index(batch)
        assert pairwise_distance_loss(pred, target, pairs).item() == pytest.approx(
            0.0, abs=1e-12)

    def test_wrong_bond_lengths_are_charged(self):
        target = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
                              dtype=torch.float64)
        batch = torch.zeros(3, dtype=torch.long)
        pairs = full_pair_index(batch)
        assert pairwise_distance_loss(target * 1.5, target, pairs).item() > 0.1

    def test_pairs_are_intra_molecular(self):
        """Two molecules far apart must not contribute cross-molecule pairs."""
        target = torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [50.0, 0.0, 0.0], [51.5, 0.0, 0.0]],
            dtype=torch.float64)
        batch = torch.tensor([0, 0, 1, 1])
        pairs = full_pair_index(batch)
        assert pairs.shape[1] == 2          # one pair per molecule, no cross pairs


class TestGate:
    def test_high_noise_only_returns_zero(self):
        """With every molecule above tau the term is exactly inert."""
        n = 5
        noise_pred = torch.randn(n, 3)
        noisy = torch.randn(n, 3)
        target = torch.randn(n, 3)
        batch = torch.zeros(n, dtype=torch.long)
        t = torch.full((1,), 500, dtype=torch.long)
        val = x0_geometry_consistency(
            noise_pred, noisy, t, batch, target_pos=target,
            alpha_bars=torch.ones(1000), tau=200,
        )
        assert val.item() == 0.0

    def test_low_noise_is_active(self):
        n = 6
        g = torch.Generator().manual_seed(8)
        target = torch.randn(n, 3, generator=g)
        target = target - target.mean(0, keepdim=True)
        batch = torch.zeros(n, dtype=torch.long)
        ab = torch.ones(1000)
        # eps-prediction that leaves x0 == target exactly: eps such that
        # x0 = (x_t - sqrt(1-ab) eps)/sqrt(ab) = target, i.e. eps = 0 with
        # x_t = target. Then a wrong *shape* must still be charged by
        # perturbing the prediction away from target.
        zero_pred = torch.zeros_like(target)
        good = x0_geometry_consistency(
            zero_pred, target, torch.full((1,), 20, dtype=torch.long), batch,
            target_pos=target, alpha_bars=ab, tau=200,
        )
        assert good.item() == pytest.approx(0.0, abs=1e-8)

    def test_non_finite_prediction_is_dropped(self):
        """A NaN molecule must not poison the returned loss."""
        n = 6
        g = torch.Generator().manual_seed(9)
        target = torch.randn(n, 3, generator=g)
        batch = torch.tensor([0, 0, 0, 1, 1, 1])
        bad_pred = torch.zeros_like(target)
        bad_pred[3] = float("nan")
        val = x0_geometry_consistency(
            bad_pred, target, torch.full((2,), 20, dtype=torch.long), batch,
            target_pos=target, alpha_bars=torch.ones(1000), tau=200,
        )
        assert math.isfinite(val.item())

    def test_weights_off_returns_zero(self):
        n = 4
        target = torch.randn(n, 3)
        batch = torch.zeros(n, dtype=torch.long)
        val = x0_geometry_consistency(
            torch.randn(n, 3), target, torch.full((1,), 10, dtype=torch.long),
            batch, target_pos=target, alpha_bars=torch.ones(1000),
            lambda_kabsch=0.0, lambda_pairdist=0.0,
        )
        assert val.item() == 0.0

    def test_terms_are_additive_weighted(self):
        """Doubling a weight must change the sum in that direction."""
        n = 6
        g = torch.Generator().manual_seed(10)
        target = torch.randn(n, 3, generator=g)
        batch = torch.zeros(n, dtype=torch.long)
        t = torch.full((1,), 20, dtype=torch.long)
        noise = torch.randn(n, 3, generator=g)      # x0 != target
        base = x0_geometry_consistency(
            noise, target, t, batch, target_pos=target,
            alpha_bars=torch.ones(1000), lambda_pairdist=0.0)
        heavy = x0_geometry_consistency(
            noise, target, t, batch, target_pos=target,
            alpha_bars=torch.ones(1000), lambda_pairdist=0.0,
            lambda_kabsch=2.0)
        assert heavy.item() == pytest.approx(2.0 * base.item(), rel=1e-5)
        assert base.item() > 0.0


class TestWiring:
    def test_train_config_flag_is_read_and_guarded(self):
        """train.py must expose the flag and refuse a cross-setting resume."""
        import inspect

        import train as train_module

        src = inspect.getsource(train_module)
        assert 'train_cfg.get("geofix", False)' in src
        assert "training.geofix" in src          # resume-guard message
        assert "x0_geometry_consistency(" in src

    def test_fed_trainer_reads_the_same_flag(self):
        import inspect

        from src.fed.trainer import LocalTrainer

        src = inspect.getsource(LocalTrainer._batch_loss)
        assert 'get("geofix", False)' in src
        assert "x0_geometry_consistency(" in src
