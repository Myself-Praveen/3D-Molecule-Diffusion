"""E(3)-equivariant coordinate denoiser for molecular graphs."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing


class EGNNLayer(MessagePassing):
    """One message-passing layer with equivariant coordinate updates."""

    def __init__(self, node_dim: int, edge_dim: int) -> None:
        super().__init__(aggr="mean")
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

    def forward(
        self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        radial = coord_diff.square().sum(dim=-1, keepdim=True)
        msg = self.edge_mlp(torch.cat((h[row], h[col], radial), dim=-1))

        trans = coord_diff * self.coord_mlp(msg)
        pos_update = torch.zeros_like(pos)
        pos_update.index_add_(0, row, trans)
        updated_pos = pos + pos_update

        aggregated = torch.zeros(
            h.size(0), msg.size(-1), dtype=msg.dtype, device=msg.device
        )
        aggregated.index_add_(0, col, msg)
        counts = torch.bincount(col, minlength=h.size(0)).clamp_min(1).unsqueeze(-1)
        aggregated = aggregated / counts
        updated_h = h + self.node_mlp(torch.cat((h, aggregated), dim=-1))
        return updated_h, updated_pos


class EquivariantGenerator(nn.Module):
    """Predict coordinate noise from atom types and noisy 3D coordinates."""

    def __init__(
        self, node_dim: int = 16, edge_dim: int = 32, num_layers: int = 4
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(10, node_dim)
        self.layers = nn.ModuleList(
            [EGNNLayer(node_dim, edge_dim) for _ in range(num_layers)]
        )

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        del t  # Reserved for timestep conditioning in the next model phase.
        h = self.embed(z)
        initial_pos = pos
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
        return pos - initial_pos