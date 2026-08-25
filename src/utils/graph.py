"""Graph construction helpers (pure PyTorch, no optional binary backends)."""

from __future__ import annotations

import torch


def build_knn_graph(
    pos: torch.Tensor, batch: torch.Tensor, k: int = 4
) -> torch.Tensor:
    """Vectorized per-molecule k-NN edge construction.

    Replaces the Phase-1 per-graph Python loop with a single masked
    ``torch.cdist`` over all atoms; cross-graph and self pairs are masked to
    infinity before the top-k. Edges whose k-neighborhood falls outside a
    small molecule are dropped via the finite-distance mask.

    Returns ``edge_index`` of shape (2, E), directed (both directions appear
    as separate entries when mutual neighbors exist).
    """
    num_nodes = pos.size(0)
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)

    dist = torch.cdist(pos, pos)
    same_graph = batch.unsqueeze(0) == batch.unsqueeze(1)
    dist = dist.masked_fill(~same_graph, float("inf"))
    dist.fill_diagonal_(float("inf"))

    graph_k = min(k, num_nodes - 1)
    _, neighbors = dist.topk(graph_k, dim=1, largest=False)

    sources = (
        torch.arange(num_nodes, device=pos.device)
        .unsqueeze(1)
        .expand(-1, graph_k)
        .reshape(-1)
    )
    targets = neighbors.reshape(-1)
    finite = torch.isfinite(dist[sources, targets])
    edge_index = torch.stack((sources[finite], targets[finite]), dim=0)
    return edge_index
