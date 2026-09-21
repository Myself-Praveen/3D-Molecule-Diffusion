"""Tests for validity_80_plan.md strategies.

Covers the retraining-path additions:
- Strategy 2: cosine noise schedule (CenteredDDPM / TypeDDPM ``schedule=``).
- Strategy 4: per-edge sigmoid attention gates in EGNNLayer / EquivariantGenerator.
- Strategy 6: quadratic DDIM step grid in ``sample_molecules`` (wiring-level
  smoke test; the sampling math itself is unchanged by the grid).
- V3 config sanity: new keys present, checkpoint dir isolated from geo_esc.
"""

from __future__ import annotations

import math

import pytest
import torch
import yaml

from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.egnn import EquivariantGenerator
from src.sampling import ddim_timestep_grid, sample_molecules


# ---------------------------------------------------------------------------
# Strategy 2: cosine schedule
# ---------------------------------------------------------------------------

class TestCosineSchedule:
    def test_alpha_bars_match_reference_formula(self):
        """ᾱ_t must equal f(t)/f(0) with f(t) = cos²((t/T + s)/(1+s) · π/2)."""
        T, s = 1000, 0.008
        ddpm = CenteredDDPM(num_steps=T, schedule="cosine", device="cpu")
        for t in (0, 1, 250, 500, 999):
            f_t = math.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
            f_0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
            expected = f_t / f_0
            assert ddpm.alpha_bars[t].item() == pytest.approx(expected, abs=1e-4)

    def test_cosine_ends_near_zero(self):
        ddpm = CenteredDDPM(num_steps=1000, schedule="cosine", device="cpu")
        assert ddpm.alpha_bars[-1].item() < 0.05
        assert ddpm.alpha_bars[0].item() > 0.99

    def test_cosine_preserves_more_signal_midway(self):
        """Cosine keeps a larger ᾱ than linear at mid-schedule (the point of S2)."""
        lin = CenteredDDPM(num_steps=1000, schedule="linear", device="cpu")
        cos = CenteredDDPM(num_steps=1000, schedule="cosine", device="cpu")
        assert cos.alpha_bars[500].item() > lin.alpha_bars[500].item()

    def test_linear_matches_legacy_construction(self):
        """schedule='linear' (default) must reproduce the original schedule."""
        legacy = torch.linspace(1e-4, 0.02, 500)
        ddpm = CenteredDDPM(num_steps=500, beta_start=1e-4, beta_end=0.02,
                            schedule="linear", device="cpu")
        assert torch.allclose(ddpm.betas, legacy)
        assert torch.allclose(ddpm.alpha_bars, torch.cumprod(1 - legacy, dim=0))

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="schedule"):
            CenteredDDPM(num_steps=10, schedule="quadratic", device="cpu")

    def test_type_ddpm_shares_schedule(self):
        coord = CenteredDDPM(num_steps=200, schedule="cosine", device="cpu")
        typ = TypeDDPM(num_steps=200, schedule="cosine", device="cpu")
        assert torch.allclose(coord.alpha_bars, typ.alpha_bars)


# ---------------------------------------------------------------------------
# Strategy 4: attention gates
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_inputs():
    pos = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0],
    ])
    z = torch.tensor([6, 6, 8, 1], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    t = torch.tensor([50])
    n = pos.size(0)
    row, col = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    edge_index = torch.stack([row.reshape(-1), col.reshape(-1)])
    return pos, z, batch, t, edge_index


class TestAttentionGates:
    def test_attention_changes_forward(self, tiny_inputs):
        pos, z, batch, t, edge_index = tiny_inputs
        kwargs = dict(num_types=10, node_dim=16, edge_dim=16,
                      num_layers=2, time_dim=16)
        torch.manual_seed(0)
        plain = EquivariantGenerator(**kwargs).eval()
        torch.manual_seed(0)
        attn = EquivariantGenerator(**kwargs, use_attention=True).eval()
        with torch.no_grad():
            _, logits_plain, _ = plain(z, pos, edge_index, t, batch)
            _, logits_attn, _ = attn(z, pos, edge_index, t, batch)
        # Shared seed ⇒ shared edge_mlp weights, and the gate multiplies the
        # message stream, so the node embeddings — hence the type logits —
        # must differ. (noise_pred can't be used: the coordinate head is
        # zero-initialized, so it outputs exactly 0 for every variant.)
        assert not torch.allclose(logits_plain, logits_attn)

    def test_attention_layer_is_rotation_equivariant(self, tiny_inputs):
        """Attention must not break E(3) equivariance (gate is per-edge scalar)."""
        pos, z, batch, t, edge_index = tiny_inputs
        model = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                     num_layers=2, time_dim=16,
                                     use_attention=True).eval()
        A = torch.randn(3, 3)
        Q, _ = torch.linalg.qr(A)
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        with torch.no_grad():
            noise_orig, _, _ = model(z, pos, edge_index, t, batch)
            noise_rot, _, _ = model(z, pos @ Q.T, edge_index, t, batch)
        assert torch.allclose(noise_rot, noise_orig @ Q.T, atol=1e-4)

    def test_attention_off_by_default(self):
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16)
        assert not any("attn_mlp" in k for k in m.state_dict())

    def test_attention_state_dict_superset_back_compatible(self):
        """Turning attention on adds keys only; legacy checkpoints still load."""
        kwargs = dict(num_types=10, node_dim=16, edge_dim=16,
                      num_layers=2, time_dim=16)
        plain_keys = set(EquivariantGenerator(**kwargs).state_dict())
        attn_keys = set(
            EquivariantGenerator(**kwargs, use_attention=True).state_dict()
        )
        assert plain_keys < attn_keys
        assert all("attn_mlp" in k for k in attn_keys - plain_keys)


# ---------------------------------------------------------------------------
# Strategy 6: quadratic DDIM grid (smoke through the sampler)
# ---------------------------------------------------------------------------

class TestQuadraticDDIMSampling:
    @pytest.fixture
    def small_setup(self):
        torch.manual_seed(0)
        model = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                     num_layers=2, time_dim=16)
        coord = CenteredDDPM(num_steps=20, device="cpu")
        typ = TypeDDPM(num_steps=20, device="cpu")
        return model, coord, typ

    @pytest.mark.parametrize("schedule", ["linear", "quadratic"])
    def test_sampling_shapes(self, small_setup, schedule):
        model, coord, typ = small_setup
        pos, z = sample_molecules(
            model, coord, typ,
            torch.tensor([4, 3], dtype=torch.long),
            device="cpu", ddim_steps=6, eta=0.0, step_schedule=schedule,
        )
        assert pos.shape == (7, 3)
        assert z.shape == (7,)
        assert torch.isfinite(pos).all()

    def test_quadratic_grid_is_dense_at_low_noise(self):
        """Strategy 6 unit test: quadratic spacing concentrates low-t steps.

        With num_steps=20, ddim_steps=6: linear grid = [0,3,6,9,12,15,19];
        quadratic squars the noise level, packing half its points below t=4.
        """
        lin = ddim_timestep_grid(20, 6, "linear")
        quad = ddim_timestep_grid(20, 6, "quadratic")
        assert lin == [19, 15, 12, 9, 6, 3, 0]
        assert quad[-1] == 0 and quad[0] == 19
        low_noise_lin = sum(1 for t in lin if 0 < t <= 4)
        low_noise_quad = sum(1 for t in quad if 0 < t <= 4)
        assert low_noise_quad > low_noise_lin

    def test_quadratic_differs_from_linear(self, small_setup):
        """The grid must change DDIM trajectories (not a no-op flag).

        Stochastic DDIM (eta=0.5): per-step noise is scaled by
        sigma(t)·eps with t from the grid, so different grids give different
        realizations even from an identical seed. (With eta=0 the update
        telescopes through x0 predictions and a tiny untrained model makes
        the comparison fragile.)
        """
        model, coord, typ = small_setup
        counts = torch.tensor([4], dtype=torch.long)
        torch.manual_seed(123)
        pos_lin, _ = sample_molecules(model, coord, typ, counts, device="cpu",
                                      ddim_steps=6, eta=0.5, step_schedule="linear")
        torch.manual_seed(123)
        pos_quad, _ = sample_molecules(model, coord, typ, counts, device="cpu",
                                       ddim_steps=6, eta=0.5, step_schedule="quadratic")
        assert not torch.allclose(pos_lin, pos_quad)

    def test_unknown_step_schedule_raises(self, small_setup):
        model, coord, typ = small_setup
        with pytest.raises(ValueError, match="step_schedule"):
            sample_molecules(model, coord, typ, torch.tensor([4]),
                             device="cpu", ddim_steps=4,
                             step_schedule="cubic")


# ---------------------------------------------------------------------------
# V3 config sanity
# ---------------------------------------------------------------------------

class TestV3Config:
    @pytest.fixture
    def v3_cfg(self):
        with open("configs/central_v3_full.yaml") as f:
            return yaml.safe_load(f)

    def test_strategies_present(self, v3_cfg):
        assert v3_cfg["diffusion"]["schedule"] == "cosine"          # S2
        assert v3_cfg["model"]["node_dim"] == 256                   # S3
        assert v3_cfg["model"]["num_layers"] == 8                   # S3
        assert v3_cfg["model"]["use_attention"] is True             # S4
        assert v3_cfg["training"]["kNN"] == 8                       # S5
        assert v3_cfg["training"]["valence_loss_weight"] == 0.0     # λ₂ removed

    def test_checkpoint_dir_isolated_from_live_run(self, v3_cfg):
        """V3 must not clobber the geo_esc run's checkpoints/ directory."""
        assert v3_cfg["checkpoint"]["dir"] != "checkpoints"
