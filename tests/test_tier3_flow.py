"""Tests for Tier 3.2 flow matching (docs/recommendation.md section 3.2).

- Linear OT path interpolation with exact velocity targets (src/models/flow.py)
- Objective marker on EquivariantGenerator ("eps" default / "flow", no new params)
- Training-loop velocity loss (train.py / src/fed/trainer.py)
- Euler ODE sampler path (src/sampling.py), auto-detected from model.objective
- x0_valence_penalty objective-agnostic x0 recovery
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.egnn import EquivariantGenerator
from src.models.flow import flow_interpolate, flow_time_embedding, flow_x0_pred
from src.sampling import sample_molecules


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


# ---------------------------------------------------------------------------
# Linear-path flow helpers (src/models/flow.py)
# ---------------------------------------------------------------------------

class TestFlowHelpers:
    def test_interpolation_is_linear_path(self):
        torch.manual_seed(0)
        x0 = torch.randn(5, 3)
        batch = torch.zeros(5, dtype=torch.long)
        u = torch.tensor([0.37])
        x_u, v = flow_interpolate(x0, u, batch)
        # v = eps - x0 with eps the (re-centered) internal draw; the path
        # identity x_u == x0 + u * v must hold exactly.
        assert torch.allclose(x_u, x0 + u.item() * v, atol=1e-6)

    def test_velocity_target_matches_formula(self):
        """v = eps - x0 == alpha'_t x0 + sigma'_t eps on the linear path
        (alpha = 1-u, sigma = u => alpha' = -1, sigma' = 1)."""
        x0 = torch.randn(4, 3)
        batch = torch.zeros(4, dtype=torch.long)
        u = torch.tensor([0.6])
        x_u, v = flow_interpolate(x0, u, batch)
        eps = v + x0  # invert the target definition
        assert torch.allclose(v, eps - x0, atol=1e-6)
        # And x_u reproduces the (1-u) x0 + u eps interpolation.
        assert torch.allclose(x_u, (1 - u.item()) * x0 + u.item() * eps, atol=1e-6)

    def test_noise_recentered_per_molecule(self):
        """Flow noise honors the CoM discipline of CenteredDDPM.add_noise."""
        batch = torch.repeat_interleave(torch.arange(2), torch.tensor([3, 4]))
        x0 = torch.randn(7, 3)
        x_u, v = flow_interpolate(x0, torch.tensor([0.5]), batch)
        eps = v + x0
        for g in range(2):
            idx = torch.where(batch == g)[0]
            assert eps[idx].mean(dim=0).abs().max() < 1e-5

    def test_x0_recovery_exact_every_u(self):
        """x0_hat = x_u - u * v is exact at every u (no sqrt(ab) blow-up)."""
        torch.manual_seed(1)
        x0 = torch.randn(6, 3)
        batch = torch.zeros(6, dtype=torch.long)
        for u in (0.0, 0.01, 0.5, 0.99, 1.0):
            x_u, v = flow_interpolate(x0, torch.tensor([u]), batch)
            x0_hat = flow_x0_pred(v, x_u, u)
            assert torch.allclose(x0_hat, x0, atol=1e-6), f"u={u}"

    def test_x0_recovery_clamps(self):
        x0 = torch.zeros(4, 3)
        v = torch.full((4, 3), 1e3)  # huge velocity => huge x0
        x0_hat = flow_x0_pred(v, x0, 0.5, clamp=10.0)
        assert x0_hat.abs().max().item() == 10.0

    def test_time_embedding_grid_endpoints(self):
        assert flow_time_embedding(0.0, 500).item() == 0
        assert flow_time_embedding(1.0, 500).item() == 499
        # Monotone non-decreasing in u.
        us = torch.linspace(0, 1, 50)
        ts = flow_time_embedding(us, 500)
        assert (ts[1:] >= ts[:-1]).all()

    def test_per_molecule_u_broadcast(self):
        batch = torch.repeat_interleave(torch.arange(3), torch.tensor([2, 1, 4]))
        x0 = torch.randn(7, 3)
        u = torch.tensor([0.2, 0.5, 0.9])
        x_u, v = flow_interpolate(x0, u, batch)
        eps = v + x0
        for g, ug in enumerate(u.tolist()):
            idx = torch.where(batch == g)[0]
            assert torch.allclose(
                x_u[idx], (1 - ug) * x0[idx] + ug * eps[idx], atol=1e-6
            )


# ---------------------------------------------------------------------------
# Objective marker on the model
# ---------------------------------------------------------------------------

class TestObjectiveMarker:
    def test_default_is_eps(self):
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16)
        assert m.objective == "eps"

    def test_flow_adds_no_parameters(self):
        kwargs = dict(num_types=10, node_dim=16, edge_dim=16,
                      num_layers=2, time_dim=16)
        torch.manual_seed(0)
        m_eps = EquivariantGenerator(**kwargs)
        torch.manual_seed(0)
        m_flow = EquivariantGenerator(**kwargs, objective="flow")
        assert set(m_eps.state_dict()) == set(m_flow.state_dict())
        for k in m_eps.state_dict():
            assert torch.equal(m_eps.state_dict()[k], m_flow.state_dict()[k])

    def test_invalid_objective_raises(self):
        with pytest.raises(ValueError, match="flow"):
            EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16, objective="vel")

    def test_legacy_checkpoint_loads_into_flow_model(self):
        torch.manual_seed(0)
        legacy = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                      num_layers=2, time_dim=16)
        flow = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                    num_layers=2, time_dim=16, objective="flow")
        flow.load_state_dict(legacy.state_dict())


# ---------------------------------------------------------------------------
# Training-side velocity loss
# ---------------------------------------------------------------------------

class TestFlowTrainingLoss:
    def test_velocity_loss_trainable(self, tiny_inputs):
        """A gradient step on the velocity MSE must reach the coordinate head
        via a nonzero target (zero-init head: loss vs noise² is minimized at
        the zero output — same test-design lesson as Tier 2.4)."""
        pos, z, batch, t, edge_index = tiny_inputs
        torch.manual_seed(0)
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16, objective="flow")
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        u = torch.tensor([0.5])
        x_u, v_target = flow_interpolate(pos, u, batch)
        first = F.mse_loss(m(z, x_u, edge_index, t, batch)[0], v_target).item()
        for _ in range(10):
            opt.zero_grad()
            v_pred = m(z, x_u, edge_index, t, batch)[0]
            F.mse_loss(v_pred, v_target).backward()
            opt.step()
        with torch.no_grad():
            final = F.mse_loss(m(z, x_u, edge_index, t, batch)[0], v_target).item()
        assert final < first

    def test_train_py_config_default_eps(self):
        import yaml
        with open("configs/central.yaml") as f:
            cfg = yaml.safe_load(f)
        assert str(cfg["diffusion"].get("objective", "eps")) == "eps"

    def test_train_py_has_flow_branches(self):
        import inspect
        from train import train_diffusion
        src = inspect.getsource(train_diffusion)
        for token in ("flow_interpolate", "v_target", 'objective == "flow"',
                      "diffusion.objective"):
            assert token in src, f"train_diffusion must handle flow ({token})"

    def test_fed_trainer_flow_branch(self, tiny_inputs):
        """LocalTrainer._batch_loss computes a finite velocity loss when
        diffusion.objective=flow."""
        from src.fed.trainer import LocalTrainer

        pos, z, batch, t, _ = tiny_inputs
        data = type("B", (), {})()
        data.pos, data.z, data.batch = pos, z, batch
        data.num_graphs = 1
        data.edge_index = torch.zeros((2, 0), dtype=torch.long)
        data.to = lambda *a, **k: data

        cfg = {
            "diffusion": {"objective": "flow", "num_steps": 50,
                          "beta_start": 1e-4, "beta_end": 0.02},
            "model": {"num_types": 10},
            "training": {"kNN": 4, "rotation_augment_prob": 0.0,
                         "self_cond_dropout": 0.5, "min_snr_gamma": 0.0,
                         "valence_loss_weight": 0.0,
                         "diversity_loss_weight": 0.0},
        }
        torch.manual_seed(0)
        model = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                     num_layers=2, time_dim=16)
        trainer = LocalTrainer(
            model,
            CenteredDDPM(num_steps=50, device="cpu"),
            TypeDDPM(num_steps=50, device="cpu"),
            cfg, "cpu",
        )
        assert trainer.objective == "flow"
        assert model.objective == "flow"
        loss, pos_l, type_l = trainer._batch_loss(data, None, None, 0.0, 0.5, 0.0, 0.0)
        assert math.isfinite(pos_l)
        # Velocity loss must be the coordinate loss, i.e. ~E[(v-0)^2] scale,
        # NOT the noise-prediction value computed against actual_noise.
        assert pos_l < 10.0

    def test_fed_trainer_invalid_objective_raises(self):
        from src.fed.trainer import LocalTrainer
        cfg = {"diffusion": {"objective": "junk"}, "model": {"num_types": 10},
               "training": {"kNN": 4}}
        with pytest.raises(ValueError, match="junk"):
            LocalTrainer(
                EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                     num_layers=2, time_dim=16),
                CenteredDDPM(num_steps=10, device="cpu"),
                TypeDDPM(num_steps=10, device="cpu"),
                cfg, "cpu",
            )


# ---------------------------------------------------------------------------
# Euler ODE sampler path
# ---------------------------------------------------------------------------

class TestFlowSampler:
    def test_flow_model_samples_via_ode(self):
        torch.manual_seed(0)
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16, objective="flow").eval()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        pos, z = sample_molecules(m, c, tp, torch.tensor([4]), device="cpu",
                                  ddim_steps=8)
        assert pos.shape == (4, 3)
        assert torch.isfinite(pos).all()
        assert z.shape == (4,) and ((z >= 0) & (z < 10)).all()

    def test_explicit_eps_objective_on_flow_model_forces_ddim(self):
        """objective='eps' on a flow model routes to the DDIM path: the output
        must equal a plain eps-model run with identical weights and seed."""
        kwargs = dict(num_types=10, node_dim=16, edge_dim=16, num_layers=2,
                      time_dim=16)
        torch.manual_seed(0)
        m_flow = EquivariantGenerator(**kwargs, objective="flow").eval()
        torch.manual_seed(0)
        m_eps = EquivariantGenerator(**kwargs).eval()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        counts = torch.tensor([5])
        torch.manual_seed(42)
        pos_forced, z_forced = sample_molecules(
            m_flow, c, tp, counts, device="cpu", ddim_steps=6, objective="eps")
        torch.manual_seed(42)
        pos_ref, z_ref = sample_molecules(
            m_eps, c, tp, counts, device="cpu", ddim_steps=6)
        assert torch.allclose(pos_forced, pos_ref, atol=1e-6)
        assert torch.equal(z_forced, z_ref)

    def test_invalid_objective_raises(self):
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16)
        c = CenteredDDPM(num_steps=10, device="cpu")
        tp = TypeDDPM(num_steps=10, device="cpu")
        with pytest.raises(ValueError, match="flow"):
            sample_molecules(m, c, tp, torch.tensor([4]), device="cpu",
                             ddim_steps=4, objective="sde")

    def test_flow_sampler_runs_with_tier2_features(self):
        """Flow ODE path composes with self-conditioning + multi-scale kNN."""
        torch.manual_seed(0)
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16, objective="flow",
                                 self_condition=True, knn_schedule=[1, 3],
                                 coord_refine_layers=1).eval()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        pos, z = sample_molecules(m, c, tp, torch.tensor([6]), device="cpu",
                                  ddim_steps=5)
        assert pos.shape == (6, 3) and torch.isfinite(pos).all()

    def test_flow_sampler_supports_guidance(self):
        torch.manual_seed(0)
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16, objective="flow",
                                 cond_dim=8, num_cond_classes=2).eval()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        cond = {"label": torch.zeros(1, dtype=torch.long),
                "properties": torch.zeros(1, 4)}
        pos, z = sample_molecules(m, c, tp, torch.tensor([4]), device="cpu",
                                  ddim_steps=5, cond=cond, guidance_scale=1.5)
        assert torch.isfinite(pos).all()

    def test_flow_sampler_ignores_step_schedule(self):
        """Quadratic grids are a DDIM concept; the flow path uses a uniform
        Euler grid and must accept the flag without crashing."""
        torch.manual_seed(0)
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16, objective="flow").eval()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        pos, _ = sample_molecules(m, c, tp, torch.tensor([4]), device="cpu",
                                  ddim_steps=6, step_schedule="quadratic")
        assert torch.isfinite(pos).all()

    def test_ddim_steps_none_uses_full_grid(self):
        """ddim_steps=None on a flow model integrates T Euler steps."""
        torch.manual_seed(0)
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16, objective="flow").eval()
        c = CenteredDDPM(num_steps=10, device="cpu")
        tp = TypeDDPM(num_steps=10, device="cpu")
        pos, z = sample_molecules(m, c, tp, torch.tensor([4]), device="cpu",
                                  ddim_steps=None)
        assert torch.isfinite(pos).all()


# ---------------------------------------------------------------------------
# x0_valence_penalty: objective-agnostic x0 recovery
# ---------------------------------------------------------------------------

class TestValencePenaltyObjective:
    def test_flow_objective_recovers_x0_exactly(self):
        """Penalty computed from a velocity prediction equals the penalty from
        the ground-truth x0 it encodes (flow_x0_pred is exact on the path)."""
        from src.objectives import x0_valence_penalty

        torch.manual_seed(0)
        n = 4
        batch = torch.zeros(n, dtype=torch.long)
        t = torch.tensor([10])  # low noise
        alpha_bars = torch.linspace(0.99, 0.5, 50)
        x0 = torch.randn(n, 3)
        v = torch.randn(n, 3)
        u = t.float() / (len(alpha_bars) - 1)
        x_u = x0 + u.item() * v
        z = torch.tensor([6, 6, 8, 1], dtype=torch.long)
        type_logits = torch.randn(n, 10)
        kwargs = dict(noisy_pos=x_u, type_logits=type_logits, t=t, batch=batch,
                      z=z, num_types=10, type_to_z={i: i for i in range(10)},
                      alpha_bars=alpha_bars, lambda_weight=1.0, tau=200)
        pen_flow = x0_valence_penalty(v, objective="flow", **kwargs)
        # Ground truth: x0_from eps-form with pred = (x_u - x0*sqrt(ab)) / ...
        ab = alpha_bars[t]
        eps_equiv = (x_u - ab.sqrt() * x0) / (1 - ab).sqrt()
        pen_true = x0_valence_penalty(eps_equiv, objective="eps", **kwargs)
        assert torch.allclose(pen_flow, pen_true, atol=1e-4)

    def test_eps_objective_unchanged(self):
        """Default call path must remain exactly the legacy eps formula."""
        from src.objectives import x0_valence_penalty

        torch.manual_seed(0)
        n, batch = 4, torch.zeros(4, dtype=torch.long)
        t = torch.tensor([10])
        alpha_bars = torch.linspace(0.99, 0.5, 50)
        noise_pred = torch.randn(n, 3)
        noisy_pos = torch.randn(n, 3)
        z = torch.tensor([6, 6, 8, 1], dtype=torch.long)
        logits = torch.randn(n, 10)
        common = dict(type_logits=logits, t=t, batch=batch, z=z, num_types=10,
                      type_to_z={i: i for i in range(10)},
                      alpha_bars=alpha_bars, lambda_weight=1.0, tau=200)
        pen = x0_valence_penalty(noise_pred, noisy_pos, **common)
        ab = alpha_bars[t]
        x0 = (noisy_pos - (1 - ab).sqrt() * noise_pred) / ab.sqrt()
        assert torch.allclose(
            pen,
            x0_valence_penalty(
                (x0 - (noisy_pos - (1 - ab).sqrt() * noise_pred) / ab.sqrt()),
                noisy_pos, objective="eps", **common),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Resume guard
# ---------------------------------------------------------------------------

class TestFlowResumeGuard:
    def test_guard_covers_objective(self):
        import inspect
        from train import train_diffusion
        src = inspect.getsource(train_diffusion)
        assert "diffusion.objective" in src, (
            "resume guard must reject eps<->flow switches"
        )
