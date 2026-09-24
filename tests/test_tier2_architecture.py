"""Tests for Tier 2 architectural improvements (docs/recommendation.md 2.1-2.4).

- 2.1 Self-conditioning (x0 estimate fed back; 50% dropout at train time)
- 2.2 Bond-head supervision vs QM9 ground-truth connectivity
- 2.3 Multi-scale per-layer kNN graphs
- 2.4 Coordinate refinement head (zero-init => identity at init)
"""

from __future__ import annotations

import pytest
import torch

from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.egnn import EquivariantGenerator
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
# 2.1 Self-conditioning
# ---------------------------------------------------------------------------

class TestSelfConditioning:
    def test_off_by_default(self):
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16)
        assert m.self_condition is False
        assert not any("x0_proj" in k for k in m.state_dict())

    def test_on_adds_keys_only(self):
        """Legacy checkpoints still load into a self-condition model minus
        the new keys; the flag only ADDS parameters."""
        kwargs = dict(num_types=10, node_dim=16, edge_dim=16,
                      num_layers=2, time_dim=16)
        base_keys = set(EquivariantGenerator(**kwargs).state_dict())
        sc_keys = set(
            EquivariantGenerator(**kwargs, self_condition=True).state_dict()
        )
        assert base_keys < sc_keys
        assert all("x0_proj" in k for k in sc_keys - base_keys)

    def test_x0_input_changes_type_logits(self, tiny_inputs):
        """With the shared-seed trick, x0=None vs x0=given must differ on the
        node/type stream. (noise_pred can't distinguish: coord head is
        zero-init at init.)"""
        pos, z, batch, t, edge_index = tiny_inputs
        kwargs = dict(num_types=10, node_dim=16, edge_dim=16,
                      num_layers=2, time_dim=16, self_condition=True)
        torch.manual_seed(0)
        m = EquivariantGenerator(**kwargs).eval()
        with torch.no_grad():
            _, l_none, _ = m(z, pos, edge_index, t, batch, x0_estimate=None)
            _, l_x0, _ = m(z, pos, edge_index, t, batch,
                           x0_estimate=torch.randn(4, 3))
        assert not torch.allclose(l_none, l_x0)

    def test_sampler_auto_detects_self_conditioning(self):
        """use_self_conditioning=None auto-enables for self_condition models;
        passing False disables it. Both paths must run cleanly."""
        torch.manual_seed(0)
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16, self_condition=True)
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        counts = torch.tensor([4])
        pos_a, z_a = sample_molecules(m, c, tp, counts, device="cpu",
                                      ddim_steps=4, use_self_conditioning=None)
        pos_b, z_b = sample_molecules(m, c, tp, counts, device="cpu",
                                      ddim_steps=4, use_self_conditioning=False)
        assert pos_a.shape == pos_b.shape == (4, 3)
        assert torch.isfinite(pos_a).all() and torch.isfinite(pos_b).all()

    def test_forced_sc_on_plain_model_is_noop(self):
        """use_self_conditioning=True on a non-SC model falls back safely."""
        torch.manual_seed(0)
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16)
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        pos, z = sample_molecules(m, c, tp, torch.tensor([4]), device="cpu",
                                  ddim_steps=4, use_self_conditioning=True)
        assert torch.isfinite(pos).all()


# ---------------------------------------------------------------------------
# 2.2 Bond head supervision
# ---------------------------------------------------------------------------

class TestBondHeadSupervision:
    def test_label_extraction(self):
        from train import _bond_pair_labels

        # 4 atoms; true bonds 0-1 and 2-3 (undirected).
        true_edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        pair_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3, 0, 2],
            [1, 2, 0, 3, 0, 3, 1, 2, 3, 1],
        ])
        labels = _bond_pair_labels(pair_index, true_edges, torch.zeros(4))
        expected = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1, 0, 0])
        assert torch.equal(labels, expected)

    def test_bond_head_logits_shape(self, tiny_inputs):
        pos, z, batch, t, edge_index = tiny_inputs
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16)
        _, _, h = m(z, pos, edge_index, t, batch)
        pair_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        logits = m.predict_bond_logits(h, pos, pair_index)
        assert logits.shape == (3, 5)  # BOND_CLASSES = 5

    def test_bond_loss_trainable(self, tiny_inputs):
        """A single gradient step on the bond head must reduce its loss
        (head is detached from the trunk, so only head params move)."""
        pos, z, batch, t, edge_index = tiny_inputs
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16)
        pair_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
        targets = torch.tensor([1, 1, 0])
        opt = torch.optim.Adam(m.bond_head.parameters(), lr=1e-2)
        _, _, h = m(z, pos, edge_index, t, batch)
        logits = m.predict_bond_logits(h.detach(), pos, pair_index)
        first = torch.nn.functional.cross_entropy(logits, targets).item()
        for _ in range(20):
            opt.zero_grad()
            logits = m.predict_bond_logits(h.detach(), pos, pair_index)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            opt.step()
        _, _, h2 = m(z, pos, edge_index, t, batch)
        with torch.no_grad():
            final = torch.nn.functional.cross_entropy(
                m.predict_bond_logits(h2.detach(), pos, pair_index), targets
            ).item()
        assert final < first


# ---------------------------------------------------------------------------
# 2.3 Multi-scale kNN
# ---------------------------------------------------------------------------

class TestMultiScaleKNN:
    def test_schedule_changes_type_logits(self, tiny_inputs):
        from src.utils.graph import build_knn_graph

        pos, z, batch, t, _ = tiny_inputs
        torch.manual_seed(0)
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16, knn_schedule=[1, 3])
        e_small = build_knn_graph(pos, batch, k=1)
        e_big = build_knn_graph(pos, batch, k=3)
        with torch.no_grad():
            _, la, _ = m(z, pos, e_big, t, batch,
                         edge_index_per_layer=[e_small, e_big])
            _, lb, _ = m(z, pos, e_big, t, batch,
                         edge_index_per_layer=[e_big, e_big])
        assert not torch.allclose(la, lb)

    def test_no_schedule_keeps_legacy_graph(self, tiny_inputs):
        pos, z, batch, t, edge_index = tiny_inputs
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16)
        assert m.knn_schedule is None
        with torch.no_grad():
            out_a, _, _ = m(z, pos, edge_index, t, batch)
            out_b, _, _ = m(z, pos, edge_index, t, batch,
                            edge_index_per_layer=[edge_index, edge_index])
        assert torch.allclose(out_a, out_b)
        assert torch.allclose(out_a[1], out_b[1])

    def test_sampler_builds_per_layer_graphs(self):
        torch.manual_seed(0)
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16, knn_schedule=[1, 3])
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        pos, z = sample_molecules(m, c, tp, torch.tensor([4]), device="cpu",
                                  ddim_steps=4)
        assert torch.isfinite(pos).all()

    def test_schedule_length_mismatch_raises_clearly(self, tiny_inputs):
        """Fewer graphs than layers must raise an IndexError, not silently
        reuse the wrong graph — surfaced here as documented behavior."""
        pos, z, batch, t, edge_index = tiny_inputs
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=3, time_dim=16, knn_schedule=[1, 3])
        with pytest.raises(IndexError):
            m(z, pos, edge_index, t, batch,
              edge_index_per_layer=[edge_index, edge_index])


# ---------------------------------------------------------------------------
# 2.4 Coordinate refinement head
# ---------------------------------------------------------------------------

class TestCoordRefinement:
    def test_identity_at_init(self, tiny_inputs):
        """Zero-init refine layers => refinement is the identity at init;
        noise_pred must exactly equal the main stack's output."""
        pos, z, batch, t, edge_index = tiny_inputs
        torch.manual_seed(0)
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16,
                                 coord_refine_layers=2).eval()
        with torch.no_grad():
            noise_full, _, _ = m(z, pos, edge_index, t, batch)
            h, updated_pos = m.encode(z, pos, edge_index, t, batch)
        assert torch.allclose(noise_full, updated_pos - pos, atol=1e-6)

    def test_zero_layers_is_exact_legacy(self, tiny_inputs):
        pos, z, batch, t, edge_index = tiny_inputs
        kwargs = dict(num_types=10, node_dim=16, edge_dim=16,
                      num_layers=2, time_dim=16)
        torch.manual_seed(0)
        m0 = EquivariantGenerator(**kwargs, coord_refine_layers=0).eval()
        torch.manual_seed(0)
        m_default = EquivariantGenerator(**kwargs).eval()
        with torch.no_grad():
            n0, _, _ = m0(z, pos, edge_index, t, batch)
            nd, _, _ = m_default(z, pos, edge_index, t, batch)
        assert torch.allclose(n0, nd)
        assert set(m0.state_dict()) == set(m_default.state_dict())

    def test_refine_layers_learn(self, tiny_inputs):
        """Real training signal (MSE to a nonzero noise target) must reach
        the refine layers.

        NOTE: d(noise²)/dW = 2·noise·∂noise/∂W vanishes while the zero-init
        head's output is exactly 0 — an MSE against a *nonzero target*
        (as in real training, where the target is the sampled noise)
        gives 2(noise − target)·∂noise/∂W ≠ 0. The loss must therefore be
        computed against a nonzero target, not noise² (which is minimized
        at the zero output).
        """
        pos, z, batch, t, edge_index = tiny_inputs
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16,
                                 coord_refine_layers=1)
        refine_params = list(m.refine_layers.parameters())
        assert refine_params, "refine layers must exist"
        noise, _, _ = m(z, pos, edge_index, t, batch)
        target = torch.randn_like(noise)  # nonzero, like real diffusion noise
        torch.nn.functional.mse_loss(noise, target).backward()
        grads = [p.grad for p in refine_params if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)

    def test_refine_changes_output_after_perturbation(self, tiny_inputs):
        """Manually perturb a refine coord weight: identity broken, output
        differs from the un-refined encode pass."""
        pos, z, batch, t, edge_index = tiny_inputs
        m = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                 num_layers=2, time_dim=16,
                                 coord_refine_layers=1).eval()
        with torch.no_grad():
            # Weight of the refine coord update: nonzero now.
            m.refine_layers[0].coord_mlp[-1].weight.add_(0.5)
            noise_full, _, _ = m(z, pos, edge_index, t, batch)
            _, updated_pos = m.encode(z, pos, edge_index, t, batch)
        assert not torch.allclose(noise_full, updated_pos - pos, atol=1e-4)


# ---------------------------------------------------------------------------
# Integration: Tier 2 config round-trip through resume guards
# ---------------------------------------------------------------------------

class TestTier2ResumeGuardKeys:
    def test_guard_covers_tier2_flags(self):
        import inspect

        from train import train_diffusion

        src = inspect.getsource(train_diffusion)
        for key in ("self_condition", "coord_refine_layers", "knn_schedule"):
            assert key in src, f"resume guard must check model.{key}"
