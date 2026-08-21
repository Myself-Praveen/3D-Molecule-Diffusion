"""Centralized Phase 1 training loop for the 3D coordinate denoiser."""

from __future__ import annotations

import torch
from torch.optim import Adam

from src.dataset import load_qm9_3d
from src.models.diffusion import CenteredDDPM
from src.models.egnn import EquivariantGenerator


def build_knn_graph(
    pos: torch.Tensor, batch: torch.Tensor, k: int = 4
) -> torch.Tensor:
    """Build per-molecule k-NN edges without optional PyG binary backends."""
    edge_parts = []
    for graph_id in batch.unique(sorted=True):
        node_ids = torch.where(batch == graph_id)[0]
        if node_ids.numel() < 2:
            continue
        graph_k = min(k, node_ids.numel() - 1)
        distances = torch.cdist(pos[node_ids], pos[node_ids])
        distances.fill_diagonal_(float("inf"))
        neighbors = distances.topk(graph_k, largest=False).indices
        targets = torch.arange(
            node_ids.numel(), device=pos.device
        ).repeat_interleave(graph_k)
        sources = neighbors.reshape(-1)
        edge_parts.append(torch.stack((node_ids[sources], node_ids[targets])))
    if not edge_parts:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)
    return torch.cat(edge_parts, dim=1)


def train_diffusion(epochs: int = 50) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    loader = load_qm9_3d(batch_size=32)
    model = EquivariantGenerator().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    diffusion = CenteredDDPM(device=device)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            t = torch.randint(0, diffusion.num_steps, (batch.num_graphs,), device=device)
            noisy_pos, actual_noise = diffusion.add_noise(batch.pos, t, batch.batch)
            edge_index = build_knn_graph(noisy_pos, batch.batch, k=4)
            predicted_noise = model(batch.z, noisy_pos, edge_index, t)
            loss = torch.nn.functional.mse_loss(predicted_noise, actual_noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(loader):.4f}")


if __name__ == "__main__":
    train_diffusion()