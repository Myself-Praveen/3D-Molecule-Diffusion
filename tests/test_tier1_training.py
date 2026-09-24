"""Tests for Tier 1 training enhancements (docs/recommendation.md 1.1-1.6).

- 1.1 EMA of weights (src/training_utils.EMA)
- 1.2 min-SNR-gamma timestep weighting (src/training_utils.min_snr_weight)
- 1.4 warmup+cosine LR schedule (src/training_utils.build_warmup_cosine_scheduler)
- 1.5 random SO(3) rotation augmentation (random_rotation_matrix / rotate_batch)
- 1.6 sigmoid noise schedule (CenteredDDPM/TypeDDPM schedule="sigmoid")
"""

from __future__ import annotations

import pytest
import torch.nn.functional as F
import torch

from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.training_utils import (
    EMA,
    build_warmup_cosine_scheduler,
    min_snr_weight,
    random_rotation_matrix,
    rotate_batch,
)


# ---------------------------------------------------------------------------
# 1.1 EMA
# ---------------------------------------------------------------------------

class TestEMA:
    def test_update_matches_formula(self):
        m = torch.nn.Linear(4, 4)
        ema = EMA(m, decay=0.9, warmup=0)  # warmup off: exact formula
        w0 = m.weight.detach().clone()
        with torch.no_grad():
            m.weight.add_(1.0)
        ema.update(m)
        expected = 0.9 * w0 + 0.1 * m.weight.detach()
        assert torch.allclose(ema.shadow["weight"], expected)

    def test_warmup_ramps_decay(self):
        m = torch.nn.Linear(2, 2)
        ema = EMA(m, decay=0.9999, warmup=10)
        # First update: d = (1+1)/(10+1) ≈ 0.18 — far below the target decay.
        w0 = m.weight.detach().clone()
        with torch.no_grad():
            m.weight.add_(1.0)
        ema.update(m)
        frac = (ema.shadow["weight"] - w0).abs().max().item()
        assert frac > 0.1  # large step absorbed early

    def test_state_dict_roundtrip(self):
        m = torch.nn.Linear(2, 2)
        ema = EMA(m, decay=0.9, warmup=0)
        with torch.no_grad():
            m.weight.add_(1.0)
        ema.update(m)
        ema2 = EMA(m, decay=0.9, warmup=0)
        ema2.load_state_dict(ema.state_dict())
        assert torch.equal(ema2.shadow["weight"], ema.shadow["weight"])

    def test_copy_to_loads_shadow(self):
        m = torch.nn.Linear(2, 2)
        ema = EMA(m, decay=0.9, warmup=0)
        with torch.no_grad():
            m.weight.add_(5.0)
        ema.update(m)
        with torch.no_grad():
            m.weight.add_(5.0)  # raw weights now differ from shadow
        ema.copy_to(m)
        assert torch.equal(m.weight.detach(), ema.shadow["weight"])

    def test_off_by_default_in_train_py_config(self):
        """ema_decay absent from config ⇒ EMA never constructed (train.py)."""
        import yaml
        with open("configs/central.yaml") as f:
            cfg = yaml.safe_load(f)
        assert float(cfg["training"].get("ema_decay", 0.0)) == 0.0


# ---------------------------------------------------------------------------
# 1.2 min-SNR-gamma weighting
# ---------------------------------------------------------------------------

class TestMinSNRWeight:
    @pytest.fixture
    def ddpm(self):
        return CenteredDDPM(num_steps=1000, schedule="cosine", device="cpu")

    def test_low_noise_gets_capped_weight(self, ddpm):
        """min-SNR caps the effective weight of easy low-noise timesteps."""
        t = torch.tensor([0, 250, 500, 999])
        w = min_snr_weight(t, ddpm.alpha_bars, gamma=5.0)
        # t=0 has huge SNR ⇒ w = γ/SNR ≪ 1; t=999 has tiny SNR ⇒ w ≈ 1.
        assert w[0] < 0.01
        assert w[-1] > 0.9
        assert w[-1] > w[0]

    def test_mean_one_normalization(self, ddpm):
        torch.manual_seed(0)
        t = torch.randint(0, 1000, (64,))
        w = min_snr_weight(t, ddpm.alpha_bars, gamma=5.0)
        assert w.mean().item() == pytest.approx(1.0, abs=1e-5)

    def test_weights_bounded_after_normalization(self, ddpm):
        """Raw w ≤ 1; the mean-1 rescale can lift values but stays tight."""
        torch.manual_seed(0)
        t = torch.randint(0, 1000, (128,))
        w = min_snr_weight(t, ddpm.alpha_bars, gamma=5.0)
        # Every sampled weight is within the bounded set [min_w, max_w]:
        # the rescale factor is 1/mean(w_raw) ≤ 1/mean over uniform t. For
        # uniform t with γ=5, mean(w_raw) ≈ 0.5-0.9 ⇒ rescale ≤ ~2.5.
        assert (w > 0).all()
        assert (w <= 2.5 + 1e-6).all()
        assert w.mean().item() == pytest.approx(1.0, abs=1e-5)

    def test_per_molecule_broadcast_through_batch(self, ddpm):
        t = torch.tensor([0, 999])
        batch = torch.repeat_interleave(torch.arange(2), torch.tensor([3, 4]))
        w = min_snr_weight(t, ddpm.alpha_bars, batch=batch, gamma=5.0)
        assert w.shape == (7,)
        assert torch.allclose(w[:3], w[0].expand(3))
        assert w[-1] > w[0]

    def test_gamma_1_is_p2_weighting(self, ddpm):
        """γ=1 collapses to 1/(1+SNR) up to the mean-1 rescale — the P2
        weighting of Choi et al. (verified via the ratio between timesteps,
        which the shared rescale factor cancels)."""
        t = torch.tensor([100, 500, 900])
        w = min_snr_weight(t, ddpm.alpha_bars, gamma=1.0)
        raw = []
        for ti in t.tolist():
            ab = ddpm.alpha_bars[ti].item()
            snr = ab / (1 - ab)
            raw.append(min(1.0, 1.0 / snr))
        raw_t = torch.tensor(raw)
        expected = raw_t / raw_t.mean()
        assert torch.allclose(w, expected, rtol=1e-3)

    def test_off_by_default_in_config(self):
        import yaml
        with open("configs/central.yaml") as f:
            cfg = yaml.safe_load(f)
        assert float(cfg["training"].get("min_snr_gamma", 0.0)) == 0.0


# ---------------------------------------------------------------------------
# 1.4 warmup + cosine
# ---------------------------------------------------------------------------

class TestWarmupCosine:
    def _opt(self):
        return torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)

    def test_reaches_peak_after_warmup(self):
        opt = self._opt()
        sch = build_warmup_cosine_scheduler(opt, total_epochs=100, warmup_epochs=10)
        for _ in range(10):
            opt.step()
            sch.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3, rel=1e-6)

    def test_zero_warmup_returns_none(self):
        assert build_warmup_cosine_scheduler(self._opt(), 100, 0) is None

    def test_decays_after_warmup(self):
        opt = self._opt()
        sch = build_warmup_cosine_scheduler(opt, total_epochs=100, warmup_epochs=10)
        for _ in range(55):  # 10 warmup + 45 into cosine
            opt.step()
            sch.step()
        assert opt.param_groups[0]["lr"] < 1e-3 * 0.75


# ---------------------------------------------------------------------------
# 1.5 rotation augmentation
# ---------------------------------------------------------------------------

class TestRotationAugment:
    def test_matrix_is_special_orthogonal(self):
        torch.manual_seed(0)
        R = random_rotation_matrix(8)
        assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand(8, 3, 3), atol=1e-5)
        assert torch.allclose(torch.det(R), torch.ones(8), atol=1e-5)

    def test_rotate_preserves_geometry(self):
        batch = torch.repeat_interleave(torch.arange(3), torch.tensor([3, 4, 5]))
        torch.manual_seed(1)
        pos = torch.randn(12, 3)
        r = rotate_batch(pos, batch)
        # norms preserved per atom
        assert torch.allclose(
            torch.linalg.vector_norm(pos, dim=-1),
            torch.linalg.vector_norm(r, dim=-1), atol=1e-5,
        )
        # pairwise distances preserved within each molecule
        for g in range(3):
            idx = torch.where(batch == g)[0]
            d0 = torch.cdist(pos[idx], pos[idx])
            d1 = torch.cdist(r[idx], r[idx])
            assert torch.allclose(d0, d1, atol=1e-4)

    def test_actually_rotates(self):
        batch = torch.zeros(4, dtype=torch.long)
        torch.manual_seed(2)
        pos = torch.randn(4, 3)
        r = rotate_batch(pos, batch)
        assert not torch.allclose(pos, r, atol=1e-4)


# ---------------------------------------------------------------------------
# 1.6 sigmoid schedule
# ---------------------------------------------------------------------------

class TestSigmoidSchedule:
    def test_monotone_decay(self):
        ab = CenteredDDPM(num_steps=500, schedule="sigmoid", device="cpu").alpha_bars
        assert (ab[1:] <= ab[:-1]).all()

    def test_matches_reference_construction(self):
        """Recompute betas from the doc formula and compare ᾱ."""
        T = 200
        t = torch.linspace(-3, 3, T, dtype=torch.float64)
        raw = torch.sigmoid(t)
        betas = ((raw - raw.min()) / (raw.max() - raw.min())) * 0.02 + 1e-4
        expected = torch.cumprod(1.0 - betas, dim=0)
        ddpm = CenteredDDPM(num_steps=T, schedule="sigmoid", device="cpu")
        assert torch.allclose(ddpm.alpha_bars, expected.float(), atol=1e-6)

    def test_more_signal_than_cosine_at_low_noise(self):
        """Sigmoid's flat low-noise end keeps ᾱ higher than cosine early on."""
        cos = CenteredDDPM(num_steps=500, schedule="cosine", device="cpu")
        sig = CenteredDDPM(num_steps=500, schedule="sigmoid", device="cpu")
        assert sig.alpha_bars[50] > cos.alpha_bars[50]

    def test_type_ddpm_shares_sigmoid_schedule(self):
        c = CenteredDDPM(num_steps=100, schedule="sigmoid", device="cpu")
        t = TypeDDPM(num_steps=100, schedule="sigmoid", device="cpu")
        assert torch.allclose(c.alpha_bars, t.alpha_bars)

    def test_unknown_schedule_lists_all_options(self):
        with pytest.raises(ValueError, match="sigmoid"):
            CenteredDDPM(num_steps=10, schedule="cubic", device="cpu")
