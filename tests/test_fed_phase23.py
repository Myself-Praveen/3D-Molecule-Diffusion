"""Unit tests for Phase 2 & 3 components (federated + multi-objective).

Acceptance criteria from .idea/03_implementation_plan.md:
- Partitioning: stratification (IID), class concentration (non-IID),
  exact cover of the index set, JSON round-trip.
- FedAvg: weighted average matches hand computation (Eq. 8).
- FedProx: proximal term is zero at w == w_global and positive otherwise.
- Personalization: personal heads excluded from aggregation payload.
- Multi-objective losses: differentiable, zero when no violation / identical
  molecules, positive otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.fed.partition import (
    molecular_formula,
    partition_iid,
    partition_niid,
    save_partition,
    load_partition,
)
from src.fed.trainer import (
    LocalTrainer,
    PERSONAL_PREFIXES,
    merge_global_personal,
    proximal_term,
    split_global_personal,
    weighted_fedavg,
)
from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.egnn import EquivariantGenerator
from src.objectives import (
    diversity_regularizer,
    distance_adjacency_probs,
    molecule_pooling,
    soft_valence_penalty,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_cfg():
    return {
        "num_types": 10, "node_dim": 16, "edge_dim": 16,
        "num_layers": 2, "time_dim": 16,
    }


@pytest.fixture
def trainer_cfg():
    return {
        "model": {"num_types": 10},
        "training": {
            "lr": 1e-3, "weight_decay": 0.0, "kNN": 4,
            "epochs": 1, "type_loss_weight": 0.5,
            "valence_loss_weight": 0.0, "diversity_loss_weight": 0.0,
        },
    }


@pytest.fixture
def tiny_batch():
    """Batch of 2 molecules as a PyG-like namespace."""
    from torch_geometric.data import Batch, Data

    d1 = Data(
        pos=torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.0, 0]]),
        z=torch.tensor([6, 6, 8]),
    )
    d2 = Data(
        pos=torch.tensor([[5.0, 5, 5], [6.0, 5, 5]]),
        z=torch.tensor([6, 8]),
    )
    return Batch.from_data_list([d1, d2])


def make_trainer(model_cfg, trainer_cfg):
    model = EquivariantGenerator(**model_cfg)
    coord_ddpm = CenteredDDPM(num_steps=100, device="cpu")
    type_ddpm = TypeDDPM(num_steps=100, device="cpu")
    return LocalTrainer(model, coord_ddpm, type_ddpm, trainer_cfg, "cpu")


# ---------------------------------------------------------------------------
# Partitioning (Step 2.1)
# ---------------------------------------------------------------------------

class TestFormulaLabeling:
    def test_hill_notation(self):
        z = torch.tensor([6, 6, 8, 1, 1, 1])
        assert molecular_formula(z) == "C2H3O"

    def test_carbon_only(self):
        assert molecular_formula(torch.tensor([6])) == "C"


class TestPartitioning:
    @pytest.fixture
    def labels(self):
        rng = np.random.default_rng(0)
        classes = ["C2H6", "C3H8", "CH4"]
        return [classes[i] for i in rng.integers(0, 3, size=300)]

    def test_exact_cover(self, labels):
        parts = partition_iid(labels, num_clients=4, seed=42)
        all_idx = sorted(i for idxs in parts.values() for i in idxs)
        assert all_idx == list(range(len(labels)))
        assert set(parts) == {0, 1, 2, 3}

    def test_iid_stratified(self, labels):
        parts = partition_iid(labels, num_clients=4, seed=42)
        # All clients should have similar sizes and see every class.
        # Round-robin stratification bounds imbalance by the class count.
        sizes = [len(parts[c]) for c in range(4)]
        assert max(sizes) - min(sizes) <= len({"C2H6", "C3H8", "CH4"})
        for c in range(4):
            client_classes = {labels[i] for i in parts[c]}
            assert client_classes == {"C2H6", "C3H8", "CH4"}

    def test_niid_concentrates_classes(self, labels):
        parts = partition_niid(labels, num_clients=5, seed=7, dirichlet_alpha=0.3)
        all_idx = sorted(i for idxs in parts.values() for i in idxs)
        assert all_idx == list(range(len(labels)))
        # With small alpha, some client should be dominated by one class.
        class_ids = {"C2H6": 0, "C3H8": 1, "CH4": 2}
        dominant_frac = []
        for c in range(5):
            if not parts[c]:
                continue
            ids = [class_ids[labels[i]] for i in parts[c]]
            counts = torch.bincount(torch.tensor(ids), minlength=3)
            dominant_frac.append((counts.max() / len(parts[c])).item())
        assert max(dominant_frac) > 0.6

    def test_json_round_trip(self, tmp_path, labels):
        parts = partition_niid(labels, 3, seed=1)
        path = tmp_path / "K3_niid.json"
        save_partition(parts, path)
        loaded = load_partition(path)
        assert {c: sorted(v) for c, v in loaded.items()} == {
            c: sorted(v) for c, v in parts.items()
        }


# ---------------------------------------------------------------------------
# Weighted FedAvg (Step 2.2, Eq. 8)
# ---------------------------------------------------------------------------

class TestWeightedFedAvg:
    def test_weighted_average(self):
        s1 = {"w": torch.tensor([1.0])}
        s2 = {"w": torch.tensor([3.0])}
        agg = weighted_fedavg([(s1, 1), (s2, 3)])
        # (1*1 + 3*3)/4 = 2.5
        assert torch.allclose(agg["w"], torch.tensor([2.5]))

    def test_equal_weights(self):
        s1 = {"w": torch.tensor([0.0])}
        s2 = {"w": torch.tensor([10.0])}
        agg = weighted_fedavg([(s1, 5), (s2, 5)])
        assert torch.allclose(agg["w"], torch.tensor([5.0]))


# ---------------------------------------------------------------------------
# FedProx (Step 3.1)
# ---------------------------------------------------------------------------

class TestFedProx:
    def test_zero_at_global(self, model_cfg):
        model = EquivariantGenerator(**model_cfg)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        assert proximal_term(model, state, mu=0.1).item() == pytest.approx(0.0)

    def test_positive_when_drifted_and_grows_with_mu(self, model_cfg):
        model = EquivariantGenerator(**model_cfg)
        state = {
            k: v.clone() + 0.01 * torch.randn_like(v)
            for k, v in model.state_dict().items()
            if not k.startswith(PERSONAL_PREFIXES)
        }
        t1 = proximal_term(model, state, mu=0.01).item()
        t2 = proximal_term(model, state, mu=1.0).item()
        assert t1 > 0.0
        assert t2 > t1

    def test_personal_heads_excluded(self, model_cfg):
        model = EquivariantGenerator(**model_cfg)
        # Corrupt only personal heads in the "global" snapshot; with large mu
        # the term must still be ~0 because heads are excluded.
        state = {}
        for k, v in model.state_dict().items():
            if k.startswith(PERSONAL_PREFIXES):
                state[k] = v + 100.0
            else:
                state[k] = v.clone()
        assert proximal_term(model, state, mu=100.0).item() < 1e-6


# ---------------------------------------------------------------------------
# Personalization (Step 3.2B)
# ---------------------------------------------------------------------------

class TestPersonalization:
    def test_split_merge_round_trip(self, model_cfg):
        model = EquivariantGenerator(**model_cfg)
        full = {k: v.clone() for k, v in model.state_dict().items()}
        global_p, personal_p = split_global_personal(full)
        assert set(personal_p) and set(global_p)
        assert not any(k.startswith(PERSONAL_PREFIXES) for k in global_p)
        merged = merge_global_personal(global_p, personal_p)
        assert merged.keys() == full.keys()

    def test_aggregation_skips_personal_keys(self, model_cfg):
        model = EquivariantGenerator(**model_cfg)
        full_a = {k: v.clone() for k, v in model.state_dict().items()}
        full_b = {k: v.clone() + 1.0 for k, v in model.state_dict().items()}
        g_a, p_a = split_global_personal(full_a)
        g_b, p_b = split_global_personal(full_b)

        agg = weighted_fedavg([(g_a, 1), (g_b, 1)])
        assert not any(k.startswith(PERSONAL_PREFIXES) for k in agg)
        # Backbone keys are averaged; personal keys untouched.
        assert torch.allclose(p_a["type_head.weight"],
                              full_a["type_head.weight"])

    def test_fit_to_global_preserves_heads(self, model_cfg, trainer_cfg):
        trainer = make_trainer(model_cfg, trainer_cfg)
        original_head = trainer.model.state_dict()["type_head.weight"].clone()
        other = make_trainer(model_cfg, trainer_cfg)
        trainer.fit_to_global(other.get_parameters(exclude_personal=False))
        after = trainer.model.state_dict()["type_head.weight"]
        assert torch.equal(original_head, after)


# ---------------------------------------------------------------------------
# Multi-objective losses (Step 3.3)
# ---------------------------------------------------------------------------

class TestObjectives:
    def test_adjacency_probs_range(self):
        pos = torch.tensor([[0.0, 0, 0], [0.8, 0, 0], [4.0, 0, 0]])
        pairs = torch.tensor([[0, 0, 1], [1, 2, 2]])
        p = distance_adjacency_probs(pos, pairs)
        assert ((p >= 0) & (p <= 1)).all()
        assert p[0] > 0.9      # 0.8 Å → bonded
        assert p[2] < 0.1      # 4.0 Å → not bonded

    def test_valence_penalty_differentiable_and_ordered(self):
        n = 3
        pos = torch.tensor([[0.0, 0, 0], [0.9, 0, 0], [1.9, 0, 0]])
        pairs = torch.combinations(torch.arange(n)).T
        probs_high = torch.full((n, 10), 0.05)
        probs_high[:, 1] = 0.55  # mostly H (valence 1): over-bonding violation
        loss_h = soft_valence_penalty(probs_high.requires_grad_(True), pos, pairs,
                                      torch.zeros(n, dtype=torch.long), 10)
        probs_low = torch.full((n, 10), 0.02)
        probs_low[:, 2] = 0.82   # mostly C (valence 4): no violation
        loss_c = soft_valence_penalty(probs_low, pos, pairs,
                                      torch.zeros(n, dtype=torch.long), 10)
        assert loss_h.item() > loss_c.item() or loss_c.item() == 0.0
        assert loss_h.requires_grad

    def test_diversity_maximal_for_identical_embeddings(self):
        h = torch.randn(3, 8)
        # Two molecules with identical pooled embeddings → sim 1 everywhere.
        h_same = torch.cat([h, h])
        b = torch.tensor([0, 0, 0, 1, 1, 1])
        div_identical = diversity_regularizer(h_same, b)
        assert div_identical.item() == pytest.approx(1.0, abs=1e-5)

    def test_diversity_zero_for_orthogonal(self):
        h = torch.cat([
            torch.tensor([[1.0, 0, 0, 0]]).repeat(3, 1),
            torch.tensor([[0, 1.0, 0, 0]]).repeat(3, 1),
        ])
        batch = torch.tensor([0, 0, 0, 1, 1, 1])
        assert diversity_regularizer(h, batch).item() < 1e-6

    def test_diversity_penalizes_similarity(self):
        b6 = torch.tensor([0, 0, 0, 1, 1, 1])
        similar = diversity_regularizer(
            torch.cat([torch.randn(3, 4)] * 2), b6)
        diverse = diversity_regularizer(
            torch.cat([torch.randn(3, 4), torch.randn(3, 4) + 10]), b6)
        assert similar > diverse

    def test_molecule_pooling_shapes(self):
        h = torch.randn(5, 4)
        batch = torch.tensor([0, 0, 1, 1, 1])
        pooled = molecule_pooling(h, batch)
        assert pooled.shape == (2, 4)
        assert torch.allclose(pooled[0], h[:2].mean(dim=0))


# ---------------------------------------------------------------------------
# End-to-end local training round
# ---------------------------------------------------------------------------

class TestLocalTrainingRound:
    def test_train_one_epoch_reduces_nothing_crash(self, model_cfg, trainer_cfg, tiny_batch):
        trainer = make_trainer(model_cfg, trainer_cfg)
        loader = torch.utils.data.DataLoader([tiny_batch], batch_size=None)
        metrics = trainer.train(loader, local_epochs=1, mu=0.1)
        assert "loss" in metrics and metrics["loss"] > 0

    def test_multi_objective_terms_fire(self, model_cfg, trainer_cfg, tiny_batch):
        cfg = {
            **trainer_cfg,
            "training": {
                **trainer_cfg["training"],
                "valence_loss_weight": 0.1,
                "diversity_loss_weight": 0.1,
            },
        }
        trainer = make_trainer(model_cfg, cfg)
        loader = torch.utils.data.DataLoader([tiny_batch], batch_size=None)
        before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        trainer.train(loader, local_epochs=1)
        changed = any(
            not torch.equal(before[k], v)
            for k, v in trainer.model.state_dict().items()
        )
        assert changed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
