"""Phase 2 tests: property conditioning, classifier-free guidance, label dropout.

Uses tiny synthetic batches (no dataset download) so the suite stays fast.
Gates mirror implementation.md section 2.5.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.fed.trainer import LocalTrainer, init_model_from_state
from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.egnn import EquivariantGenerator
from src.sampling import sample_molecules


def _toy_batch():
    torch.manual_seed(0)
    pos = torch.randn(11, 3)
    z = torch.randint(0, 10, (11,))
    n = pos.size(0)
    row, col = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    edge = torch.stack([row.reshape(-1), col.reshape(-1)])
    edge = edge[:, row.reshape(-1) != col.reshape(-1)]
    t = torch.tensor([10, 50])
    batch = torch.tensor([0] * 6 + [1] * 5)
    cond = {"label": torch.tensor([1, 0]),
            "properties": torch.tensor([[0.5, 1.0, -0.3, 0.2],
                                        [-0.5, 0.0, 0.4, -0.1]])}
    return z, pos, edge, t, batch, cond


def _bbb_like_batch():
    """Synthetic BBB-style batch: raw atomic numbers + property attributes."""
    torch.manual_seed(1)
    items = []
    for i in range(8):
        n = int(torch.randint(5, 10, (1,)).item())
        items.append(Data(
            pos=torch.randn(n, 3),
            z=torch.tensor([6 if j % 3 else (8 if j % 3 == 1 else 1)
                            for j in range(n)]),
            y=torch.tensor([i % 2]),
            qed=torch.tensor([0.5]),
            logp=torch.tensor([2.0]),
            tpsa=torch.tensor([60.0]),
            mw=torch.tensor([300.0]),
            smiles="C",
        ))
    return PyGDataLoader(items, batch_size=8)


class TestConditionedEGNN:
    def test_unconditional_backward_compat(self):
        z, pos, edge, t, batch, _ = _toy_batch()
        model = EquivariantGenerator(cond_dim=0)
        assert model.cond_dim == 0
        assert not hasattr(model, "cond_embed")
        n, logits, h = model(z, pos, edge, t, batch)
        assert n.shape == pos.shape and logits.shape == (11, 10)
        assert torch.isfinite(n).all()

    def test_conditional_forward(self):
        z, pos, edge, t, batch, cond = _toy_batch()
        model = EquivariantGenerator(cond_dim=32)
        n, logits, _ = model(z, pos, edge, t, batch, cond=cond)
        assert torch.isfinite(n).all() and torch.isfinite(logits).all()
        _, logits_u, _ = model(z, pos, edge, t, batch, cond=None)
        assert not torch.equal(logits, logits_u)

    def test_label_dropout_path(self):
        z, pos, edge, t, batch, _ = _toy_batch()
        model = EquivariantGenerator(cond_dim=32)
        model.eval()
        with torch.no_grad():
            a, _, _ = model(z, pos, edge, t, batch, cond=None)
            b, _, _ = model(z, pos, edge, t, batch, cond=None)
        assert torch.equal(a, b)

    def test_coord_head_has_gradient(self):
        z, pos, edge, t, batch, cond = _toy_batch()
        model = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                     num_layers=2, time_dim=8, cond_dim=16)
        n, logits, _ = model(z, pos, edge, t, batch, cond=cond)
        # NOTE: plain n.square().mean() is exactly 0 at init by the Rec-A2
        # zero-init design, so train against a random target instead.
        loss = (torch.nn.functional.mse_loss(n, torch.randn_like(n))
                + torch.nn.functional.cross_entropy(logits, z))
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())

    def test_odd_cond_dim_rejected(self):
        import pytest as _pytest
        with _pytest.raises(ValueError):
            EquivariantGenerator(cond_dim=7)


class TestGuidance:
    def _sampler(self):
        device = "cpu"
        model = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                     num_layers=2, time_dim=8, cond_dim=16)
        return (model, CenteredDDPM(num_steps=20, device=device),
                TypeDDPM(num_steps=20, device=device))

    def test_guidance_scale_zero(self):
        _, pos, _, _, _, cond = _toy_batch()
        model, c_ddpm, t_ddpm = self._sampler()
        torch.manual_seed(11)
        p1, z1 = sample_molecules(model, c_ddpm, t_ddpm, torch.tensor([4, 5]),
                                  device="cpu", ddim_steps=4,
                                  cond=cond, guidance_scale=0.0)
        torch.manual_seed(11)
        p2, z2 = sample_molecules(model, c_ddpm, t_ddpm, torch.tensor([4, 5]),
                                  device="cpu", ddim_steps=4, cond=cond)
        assert torch.equal(p1, p2) and torch.equal(z1, z2)

    def test_guidance_steers(self):
        z, pos, edge, t, batch, cond = _toy_batch()
        model, c_ddpm, t_ddpm = self._sampler()
        # Warm up: at init the Rec-A2 zero-init makes cond/uncond noise
        # identical, so take a few steps for guidance to have leverage.
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(5):
            opt.zero_grad()
            n, logits, _ = model(z, pos, edge, t, batch, cond=cond)
            (torch.nn.functional.mse_loss(n, torch.randn_like(n))
             + torch.nn.functional.cross_entropy(logits, z)).backward()
            opt.step()
        torch.manual_seed(11)
        p1, _ = sample_molecules(model, c_ddpm, t_ddpm, torch.tensor([4, 5]),
                                 device="cpu", ddim_steps=4, cond=cond,
                                 guidance_scale=0.0)
        torch.manual_seed(11)
        p3, _ = sample_molecules(model, c_ddpm, t_ddpm, torch.tensor([4, 5]),
                                 device="cpu", ddim_steps=4, cond=cond,
                                 guidance_scale=2.0)
        assert not torch.equal(p1, p3)


def _trainer_cfg(**overrides):
    cfg = {"model": {"num_types": 10, "node_dim": 16, "edge_dim": 16,
                     "num_layers": 2, "time_dim": 8,
                     "cond_dim": 16, "num_cond_classes": 2},
           "diffusion": {"num_steps": 20},
           "training": {"lr": 1e-3, "kNN": 4, "type_loss_weight": 0.5},
           "conditioning": {"enabled": True, "label_dropout": 0.1}}
    cfg.update(overrides)
    return cfg


class TestTrainerConditioning:
    def test_extract_cond(self):
        loader = _bbb_like_batch()
        cfg = _trainer_cfg()
        tr = LocalTrainer(init_model_from_state(cfg["model"], None, "cpu"),
                          CenteredDDPM(num_steps=20, device="cpu"),
                          TypeDDPM(num_steps=20, device="cpu"), cfg, "cpu")
        cond = tr._extract_cond(next(iter(loader)))
        assert cond is not None
        assert cond["label"].shape == (8,)
        assert cond["properties"].shape == (8, 4)
        assert set(cond["label"].tolist()) <= {0, 1}

    def test_disabled_is_none(self):
        loader = _bbb_like_batch()
        cfg = _trainer_cfg()
        del cfg["conditioning"]
        tr = LocalTrainer(init_model_from_state(cfg["model"], None, "cpu"),
                          CenteredDDPM(num_steps=20, device="cpu"),
                          TypeDDPM(num_steps=20, device="cpu"), cfg, "cpu")
        assert tr._extract_cond(next(iter(loader))) is None

    def test_conditioned_train_step(self):
        loader = _bbb_like_batch()
        cfg = _trainer_cfg()
        tr = LocalTrainer(init_model_from_state(cfg["model"], None, "cpu"),
                          CenteredDDPM(num_steps=20, device="cpu"),
                          TypeDDPM(num_steps=20, device="cpu"), cfg, "cpu")
        out = tr.train(loader, local_epochs=1)
        assert out["loss"] > 0 and out["num_batches"] == 1

    def test_full_dropout_trains(self):
        loader = _bbb_like_batch()
        cfg = _trainer_cfg(conditioning={"enabled": True, "label_dropout": 1.0})
        tr = LocalTrainer(init_model_from_state(cfg["model"], None, "cpu"),
                          CenteredDDPM(num_steps=20, device="cpu"),
                          TypeDDPM(num_steps=20, device="cpu"), cfg, "cpu")
        assert tr.train(loader, local_epochs=1)["loss"] > 0
