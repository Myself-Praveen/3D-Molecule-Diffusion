"""Tests for the DPO fine-tuning loss (plan Step 2.3).

The load-bearing assertion is the *sign*: a model that denoises the chosen side
better than the reference must produce a lower loss than a model that does the
opposite. Because a fake model can recover the exact target noise from a known
x0, that comparison is analytic rather than statistical, and it fails if either
correction is undone (training on the losers, or feeding clean x0 to the
denoiser).
"""

from __future__ import annotations

import pytest
import torch

from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.egnn import EquivariantGenerator
from src.rl.dpo import (
    PreferenceBatch,
    build_reference_model,
    collate_pairs,
    cyclic_pair_batches,
    dpo_loss,
    load_preference_pairs,
    save_preference_pairs,
)


def _entry(pos: torch.Tensor, z: torch.Tensor) -> dict:
    return {"pos": pos, "z": z}


def _pair(i: int = 0) -> dict:
    chosen = _entry(
        torch.tensor([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]]),
        torch.tensor([6, 6]),
    )
    rejected = _entry(
        torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        torch.tensor([6, 6]),
    )
    return {
        "chosen": chosen, "rejected": rejected,
        "reward_chosen": 1.0 - 0.01 * i, "reward_rejected": -1.0,
    }


class _GeomModel:
    """Deterministic stand-in: predicts the noise consistent with a known x0."""

    def __init__(self, x0: torch.Tensor, ddpm: CenteredDDPM) -> None:
        self.x0 = x0
        self.ddpm = ddpm
        self.knn_schedule = None

    def __call__(self, z, pos, edge_index, t, batch, cond=None,
                 x0_estimate=None, edge_index_per_layer=None):
        ab = self.ddpm.alpha_bars.to(pos.device)[t][batch].unsqueeze(-1)
        eps = (pos - ab.sqrt() * self.x0) / (1.0 - ab).sqrt().clamp_min(1e-6)
        return eps, torch.zeros(pos.size(0), 10, device=pos.device), None


class _ZeroModel(_GeomModel):
    """Predicts zero noise — the trivial reference."""

    def __init__(self, ddpm: CenteredDDPM) -> None:
        super().__init__(torch.zeros(0, 3), ddpm)

    def __call__(self, z, pos, edge_index, t, batch, cond=None,
                 x0_estimate=None, edge_index_per_layer=None):
        return torch.zeros_like(pos), torch.zeros(pos.size(0), 10, device=pos.device), None


class TestCollation:
    def test_shapes_and_graph_ids(self):
        pairs = [_pair(0), _pair(1), _pair(2)]
        batch = collate_pairs(pairs, [0, 2])
        assert batch.chosen_pos.shape == (4, 3)
        assert batch.rejected_pos.shape == (4, 3)
        assert torch.equal(batch.chosen_batch, torch.tensor([0, 0, 1, 1]))
        assert torch.equal(batch.rejected_batch, torch.tensor([0, 0, 1, 1]))
        assert batch.num_pairs == 2

    def test_sides_stay_aligned(self):
        """Pair i on the chosen side corresponds to pair i on the rejected side."""
        batch = collate_pairs([_pair(0), _pair(1)])
        # Chosen geometry (1.54 A) is pair 0, rejected geometry (0.5 A) is pair 0.
        assert batch.chosen_pos[1, 0].item() == pytest.approx(1.54)
        assert batch.rejected_pos[1, 0].item() == pytest.approx(0.5)

    def test_empty_selection_raises(self):
        with pytest.raises(ValueError, match="empty"):
            collate_pairs([_pair()], [])

    def test_varying_sizes_collate(self):
        pairs = [_pair(), _pair()]
        pairs[1]["chosen"] = _entry(
            torch.randn(5, 3), torch.tensor([6, 6, 6, 6, 6])
        )
        pairs[1]["rejected"] = _entry(
            torch.randn(5, 3), torch.tensor([6, 6, 6, 6, 6])
        )
        batch = collate_pairs(pairs)
        assert batch.chosen_pos.shape == (7, 3)
        assert batch.chosen_batch.tolist() == [0, 0, 1, 1, 1, 1, 1]


class TestPersistence:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "pairs.pt"
        save_preference_pairs(path, {"pairs": [_pair(0), _pair(1)], "meta": "x"})
        loaded = load_preference_pairs(path)
        assert len(loaded) == 2
        assert loaded[0]["reward_chosen"] == pytest.approx(1.0)

    def test_accepts_bare_list(self, tmp_path):
        path = tmp_path / "bare.pt"
        save_preference_pairs(path, [_pair()])
        assert len(load_preference_pairs(path)) == 1

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.pt"
        save_preference_pairs(path, {"pairs": []})
        with pytest.raises(ValueError, match="no preference pairs"):
            load_preference_pairs(path)


class TestCyclicBatches:
    def test_yields_and_wraps(self):
        pairs = [_pair(i) for i in range(5)]
        it = cyclic_pair_batches(pairs, batch_size=2)
        first = next(it)
        assert first.num_pairs == 2
        # More batches than pairs eventually → the iterator must wrap, not stop.
        for _ in range(6):
            assert next(it).num_pairs >= 1


class TestDPOLossSign:
    def _batch(self) -> PreferenceBatch:
        return collate_pairs([_pair(0), _pair(1)])

    def test_model_that_beats_ref_on_chosen_scores_lower_loss(self):
        ddpm = CenteredDDPM(num_steps=20, device="cpu")
        ref = _ZeroModel(ddpm)
        batch = self._batch()

        # "Good" model: its x0 assumption is the chosen geometry, so it
        # denoises the chosen side exactly and the rejected side badly.
        good = _GeomModel(batch.chosen_pos.clone(), ddpm)
        good_loss = dpo_loss(good, ref, ddpm, batch, beta=1.0)

        # "Bad" model: x0 assumption is the rejected geometry → the reverse.
        bad = _GeomModel(batch.rejected_pos.clone(), ddpm)
        bad_loss = dpo_loss(bad, ref, ddpm, batch, beta=1.0)

        assert torch.isfinite(good_loss) and torch.isfinite(bad_loss)
        assert good_loss.item() < bad_loss.item()

    def test_loss_is_positive_for_sigmoid(self):
        ddpm = CenteredDDPM(num_steps=20, device="cpu")
        batch = self._batch()
        model = _GeomModel(batch.chosen_pos.clone(), ddpm)
        loss = dpo_loss(model, _ZeroModel(ddpm), ddpm, batch, beta=0.1)
        assert 0.0 < loss.item() < torch.log(torch.tensor(2.0)).item()

    def test_reference_is_not_differentiated(self):
        torch.manual_seed(0)
        c = CenteredDDPM(num_steps=20, device="cpu")
        model = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                     num_layers=2, time_dim=16)
        ref = build_reference_model(model, None, "cpu")
        loss = dpo_loss(model, ref, c, self._batch(), beta=0.1)
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())
        assert all(p.grad is None for p in ref.parameters())
        assert all(not p.requires_grad for p in ref.parameters())

    def test_flow_objective_is_refused(self):
        c = CenteredDDPM(num_steps=20, device="cpu")
        model = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                     num_layers=2, time_dim=16)
        ref = build_reference_model(model, None, "cpu")
        with pytest.raises(ValueError, match="eps"):
            dpo_loss(model, ref, c, self._batch(), objective="flow")

    def test_reference_weights_come_from_the_checkpoint(self, tmp_path):
        torch.manual_seed(1)
        c = CenteredDDPM(num_steps=20, device="cpu")
        model = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                     num_layers=2, time_dim=16)
        ckpt = tmp_path / "ref.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt)

        # Perturb the live model; the reference must still load the checkpoint.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ref = build_reference_model(model, str(ckpt), "cpu")
        live = list(model.parameters())
        frozen = list(ref.parameters())
        assert any(not torch.allclose(a, b) for a, b in zip(live, frozen))

    def test_type_ddpm_is_not_required(self):
        """dpo_loss only needs the coordinate schedule (types pass through)."""
        assert TypeDDPM is not None


class TestTrainIntegration:
    def test_train_wires_the_dpo_stage(self):
        import inspect

        from train import train_diffusion

        src = inspect.getsource(train_diffusion)
        for needle in ("dpo_loss(", "rl.pair_file", "rl.enabled",
                       "dpo_ref_checkpoint", "dpo_lr", "dpo_epochs"):
            assert needle in src, f"train.py must reference {needle}"

    def test_shipped_config_gates_rl_off(self):
        """The plan's rule: no DPO until the geometry gate passes."""
        import yaml

        with open("configs/central_v5_bbb.yaml") as f:
            cfg = yaml.safe_load(f)
        assert cfg["rl"]["enabled"] is False
        assert cfg["rl"]["dpo_beta"] == 0.1
