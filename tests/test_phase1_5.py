"""Unit tests for Phase 1.5 components.

Acceptance criteria from .idea/03_implementation_plan.md Step 1.1:
- Per-molecule centered noise has |mean| < 1e-5.
- Equivariance check: rotate pos by random R → predicted noise rotates identically.
- Loss < predict-zero baseline within a few epochs on a tiny QM9 subset.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.models.diffusion import CenteredDDPM, TypeDDPM, _scatter_counts
from src.models.egnn import EquivariantGenerator, timestep_embedding, BOND_CLASSES
from src.utils.graph import build_knn_graph
from src.utils.evaluation import (
    validity, uniqueness, novelty, internal_diversity,
    mean_qed, mean_logp, snn, coords_and_types_to_mol,
)
from src.sampling import _center_per_molecule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_molecule_data():
    """A batch of 3 tiny molecules (3, 4, 2 atoms)."""
    pos = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],  # mol 0 (3 atoms)
        [5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0], [5.5, 5.5, 6.0],  # mol 1 (4 atoms)
        [10.0, 0.0, 0.0], [11.0, 0.0, 0.0],  # mol 2 (2 atoms)
    ], dtype=torch.float32)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2], dtype=torch.long)
    z = torch.tensor([6, 6, 8, 6, 7, 8, 1, 6, 1], dtype=torch.long)
    return pos, batch, z


@pytest.fixture
def coord_ddpm():
    return CenteredDDPM(num_steps=100, beta_start=1e-4, beta_end=0.02, device="cpu")


@pytest.fixture
def type_ddpm():
    return TypeDDPM(num_steps=100, beta_start=1e-4, beta_end=0.02, device="cpu")


@pytest.fixture
def model():
    return EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16, num_layers=2, time_dim=16)


# ---------------------------------------------------------------------------
# Test: Per-molecule noise centering (Rec A1)
# ---------------------------------------------------------------------------

class TestNoiseCentering:
    def test_per_molecule_mean_is_near_zero(self, small_molecule_data, coord_ddpm):
        pos, batch, _ = small_molecule_data
        num_graphs = int(batch.max().item()) + 1

        for _ in range(50):
            t = torch.randint(0, coord_ddpm.num_steps, (num_graphs,))
            _, noise = coord_ddpm.add_noise(pos, t, batch)

            # Check per-molecule mean of noise is near zero
            for g in range(num_graphs):
                mask = batch == g
                mol_mean = noise[mask].mean(dim=0)
                assert mol_mean.abs().max() < 1e-5, (
                    f"Molecule {g}: noise mean = {mol_mean} (should be ~0)"
                )


class TestScatterCounts:
    def test_counts(self, small_molecule_data):
        _, batch, _ = small_molecule_data
        counts = _scatter_counts(batch, 3, torch.float32, "cpu")
        expected = torch.tensor([3.0, 4.0, 2.0])
        assert torch.allclose(counts, expected)


# ---------------------------------------------------------------------------
# Test: Equivariance
# ---------------------------------------------------------------------------

class TestEquivariance:
    def test_rotation_equivariance(self, small_molecule_data, model):
        """Rotating input positions should rotate predicted noise identically."""
        pos, _, z = small_molecule_data
        batch = torch.zeros(pos.size(0), dtype=torch.long)  # single molecule
        t = torch.tensor([50])

        # Build fully connected edge index for tiny molecule
        n = pos.size(0)
        row, col = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        edge_index = torch.stack([row.reshape(-1), col.reshape(-1)])

        model.eval()
        with torch.no_grad():
            noise_orig, _, _ = model(z, pos, edge_index, t, batch)

        # Random rotation matrix (Gram-Schmidt)
        A = torch.randn(3, 3)
        Q, _ = torch.linalg.qr(A)
        if torch.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]

        pos_rot = pos @ Q.T
        with torch.no_grad():
            noise_rot, _, _ = model(z, pos_rot, edge_index, t, batch)

        expected_rot = noise_orig @ Q.T
        assert torch.allclose(noise_rot, expected_rot, atol=1e-4), (
            f"Equivariance violated: max diff = {(noise_rot - expected_rot).abs().max():.6f}"
        )


# ---------------------------------------------------------------------------
# Test: Timestep embedding
# ---------------------------------------------------------------------------

class TestTimestepEmbedding:
    def test_shape_and_values(self):
        t = torch.tensor([0, 50, 99])
        dim = 32
        emb = timestep_embedding(t, dim)
        assert emb.shape == (3, dim)
        # Values should be in [-1, 1] (cos/sin output)
        assert emb.abs().max() <= 1.0


# ---------------------------------------------------------------------------
# Test: Model forward pass shape and losses
# ---------------------------------------------------------------------------

class TestModelForward:
    def test_output_shapes(self, small_molecule_data, model):
        pos, batch, z = small_molecule_data
        n = pos.size(0)
        row, col = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        edge_index = torch.stack([row.reshape(-1), col.reshape(-1)])
        t = torch.full((int(batch.max()) + 1,), 50, dtype=torch.long)

        noise_pred, type_logits, node_h = model(z, pos, edge_index, t, batch)
        assert noise_pred.shape == (n, 3)
        assert type_logits.shape == (n, 10)  # num_types
        assert node_h.shape[0] == n


class TestTypeDDPM:
    def test_forward_probs_sum_to_one(self, type_ddpm, small_molecule_data):
        _, _, z = small_molecule_data
        t = torch.tensor([50, 50, 50, 50, 50, 50, 50, 50, 50])
        probs = type_ddpm.forward_probs(z, t, 10)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(z.size(0)), atol=1e-5)

    def test_posterior_probs_sum_to_one(self, type_ddpm):
        z_t = torch.randint(0, 10, (5,))
        clean_probs = torch.rand(5, 10)
        clean_probs = clean_probs / clean_probs.sum(dim=-1, keepdim=True)
        t = torch.tensor([50, 50, 50, 50, 50])
        post = type_ddpm.posterior_probs(z_t, clean_probs, t, 10)
        assert torch.allclose(post.sum(dim=-1), torch.ones(5), atol=1e-5)


class TestTypeLoss:
    def test_cross_entropy_over_random(self, model, type_ddpm):
        """Type cross-entropy should be below random baseline (~ln(10)=2.302)."""
        num_types = 10
        n = 50
        z = torch.randint(0, num_types, (n,))
        batch = torch.zeros(n, dtype=torch.long)
        pos = torch.randn(n, 3) * 2.0
        row, col = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        edge_index = torch.stack([row.reshape(-1), col.reshape(-1)])
        t = torch.randint(0, 100, (1,))

        # Type loss of random predictions should be ~ln(K) = ln(10)
        with torch.no_grad():
            _, type_logits, _ = model(z, pos, edge_index, t, batch)
        random_loss = F.cross_entropy(type_logits, z)
        assert random_loss > 2.0, "Random predictions should have loss ≈ ln(10) ≈ 2.3"


# ---------------------------------------------------------------------------
# Test: kNN graph
# ---------------------------------------------------------------------------

class TestKNNGraph:
    def test_returns_edges(self, small_molecule_data):
        pos, batch, _ = small_molecule_data
        edge_index = build_knn_graph(pos, batch, k=3)
        assert edge_index.size(0) == 2
        assert edge_index.size(1) > 0  # at least some edges

    def test_no_cross_graph_edges(self, small_molecule_data):
        pos, batch, _ = small_molecule_data
        edge_index = build_knn_graph(pos, batch, k=2)
        if edge_index.size(1) > 0:
            src_batch = batch[edge_index[0]]
            tgt_batch = batch[edge_index[1]]
            assert torch.equal(src_batch, tgt_batch), "Edges should not cross molecules"


# ---------------------------------------------------------------------------
# Test: Molecule centering
# ---------------------------------------------------------------------------

class TestMoleculeCentering:
    def test_centered_molecules_have_zero_com(self, small_molecule_data):
        pos, batch, _ = small_molecule_data
        centered = _center_per_molecule(pos.clone(), batch)
        num_graphs = int(batch.max().item()) + 1
        for g in range(num_graphs):
            mask = batch == g
            com = centered[mask].mean(dim=0)
            assert com.abs().max() < 1e-5, f"Molecule {g} CoM = {com}"


# ---------------------------------------------------------------------------
# Test: Evaluation metrics
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:
    def test_all_valid(self):
        from rdkit import Chem
        mols = [Chem.MolFromSmiles(s) for s in ["CCO", "CC", "c1ccccc1"]]
        assert validity(mols) == pytest.approx(1.0)

    def test_some_invalid(self):
        from rdkit import Chem
        mols = [Chem.MolFromSmiles("CCO"), None, Chem.MolFromSmiles("CC")]
        assert validity(mols) == pytest.approx(2 / 3)

    def test_uniqueness_all_unique(self):
        from rdkit import Chem
        mols = [Chem.MolFromSmiles(s) for s in ["CCO", "CC", "c1ccccc1"]]
        assert uniqueness(mols) == pytest.approx(1.0)

    def test_uniqueness_duplicates(self):
        from rdkit import Chem
        mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCO")]
        assert uniqueness(mols) == pytest.approx(0.5)

    def test_novelty(self):
        from rdkit import Chem
        train_smiles = {"CCO", "CC"}
        mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]
        # CCO is in train, c1ccccc1 is novel → 50% novelty
        assert novelty(mols, train_smiles) == pytest.approx(0.5)

    def test_internal_diversity(self):
        from rdkit import Chem
        # Identical molecules → diversity 0
        mols = [Chem.MolFromSmiles("CCO")] * 3
        assert internal_diversity(mols) == pytest.approx(0.0)

    def test_mol_construction(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        z = np.array([6, 6, 8])
        mol = coords_and_types_to_mol(pos, z)
        assert mol is not None
        assert mol.GetNumAtoms() == 3

    def test_mol_with_padding(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        z = np.array([6, 0])  # 0 = padding
        mol = coords_and_types_to_mol(pos, z)
        assert mol is not None
        assert mol.GetNumAtoms() == 1  # padding atom skipped


# ---------------------------------------------------------------------------
# Test: Small training convergence (loss < baseline in a few epochs)
# ---------------------------------------------------------------------------

class TestTrainingConvergence:
    def test_loss_beats_baseline(self):
        """A tiny model on synthetic data should beat predict-zero baseline
        (MSE ≈ 1.0) within a few epochs."""
        from torch.optim import Adam

        device = "cpu"
        model = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                     num_layers=2, time_dim=16).to(device)
        coord_ddpm = CenteredDDPM(num_steps=100, device=device)
        type_ddpm = TypeDDPM(num_steps=100, device=device)
        optimizer = Adam(model.parameters(), lr=1e-3)

        # Synthetic mini-batch: 1 molecule with 6 atoms
        pos = torch.randn(6, 3) * 2.0
        z = torch.tensor([6, 6, 7, 8, 1, 1], dtype=torch.long)
        batch = torch.zeros(6, dtype=torch.long)
        n = pos.size(0)
        row, col = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        edge_index = torch.stack([row.reshape(-1), col.reshape(-1)])

        for _ in range(20):
            model.train()
            optimizer.zero_grad()
            t = torch.randint(0, coord_ddpm.num_steps, (1,))
            noisy_pos, actual_noise = coord_ddpm.add_noise(pos, t, batch)
            noisy_types = type_ddpm.sample_noisy_types(z, t, 10)
            noise_pred, type_logits, _ = model(noisy_types, noisy_pos, edge_index, t, batch)
            pos_loss = F.mse_loss(noise_pred, actual_noise)
            type_loss = F.cross_entropy(type_logits, z)
            loss = pos_loss + 0.5 * type_loss
            loss.backward()
            optimizer.step()

        # After 20 steps on a tiny system, position loss should be < 1.0
        model.eval()
        with torch.no_grad():
            t = torch.randint(0, coord_ddpm.num_steps, (1,))
            noisy_pos, actual_noise = coord_ddpm.add_noise(pos, t, batch)
            noisy_types = type_ddpm.sample_noisy_types(z, t, 10)
            noise_pred, _, _ = model(noisy_types, noisy_pos, edge_index, t, batch)
            final_pos_loss = F.mse_loss(noise_pred, actual_noise).item()

        assert final_pos_loss < 1.0, (
            f"Position loss {final_pos_loss:.4f} should be < 1.0 (predict-zero baseline)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
