"""Multi-objective generation losses (Step 3.3 of .idea/03_implementation_plan.md).

Diffusion-compatible adaptation of GraphGANFed's proposed future-work
objective ``L_gen = −λ1 log D − λ2 Validity − λ3 Uniqueness``:

- λ1's adversarial-realism role is subsumed by the diffusion NLL itself.
- ``soft_valence_penalty``  → differentiable validity surrogate (weight λ2).
- ``diversity_regularizer`` → batch-level uniqueness surrogate (weight λ3).

Both penalties are computed on differentiable model outputs (predicted type
probabilities and coordinates) so they can be added to the training loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Standard neutral valences for QM9 elements, indexed by atomic number.
VALENCE: dict[int, int] = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}

# Typical single-bond covalent radii sum (Angstrom), used for adjacency probs.
_BOND_CUTOFF = 1.8
_SOFTNESS = 0.25


def distance_adjacency_probs(pos: torch.Tensor, pair_index: torch.Tensor) -> torch.Tensor:
    """Differentiable probability that each candidate pair forms a bond.

    ``p = sigmoid((cutoff − d) / softness)`` — smooth step around the cutoff.
    """
    row, col = pair_index
    dist = (pos[row] - pos[col]).norm(dim=-1)
    return torch.sigmoid((_BOND_CUTOFF - dist) / _SOFTNESS)


def full_pair_index(batch: torch.Tensor) -> torch.Tensor:
    """All unordered atom pairs (i<j) within each molecule as (2, P) index."""
    pairs = []
    for g in torch.unique(batch):
        idx = torch.where(batch == g)[0]
        n = idx.numel()
        if n < 2:
            continue
        r, c = torch.meshgrid(idx, idx, indexing="ij")
        mask = r < c
        pairs.append(torch.stack([r[mask], c[mask]], dim=0))
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long, device=batch.device)
    return torch.cat(pairs, dim=1)


def soft_valence_penalty(
    type_probs: torch.Tensor,
    pos: torch.Tensor,
    pair_index: torch.Tensor,
    atomic_numbers: torch.Tensor,
    num_types: int,
    type_to_z: dict[int, int] | None = None,
    batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expected valence-violation penalty over predicted bonds.

    For every atom, the expected number of bonds is the sum of adjacency
    probabilities to its candidate partners; the expected valence demand is
    approximated by scaling bond count by an expected bond order (≈1.2 for
    organic molecules). The penalty is ReLU(expected_use − max_valence)²,
    averaged over atoms. Fully differentiable w.r.t. type probabilities and
    coordinates.

    NOTE: when ``batch`` is given, adjacency is counted over ALL intra-
    molecular pairs (not just the kNN ``pair_index``). kNN-capped counting
    cannot discriminate dense clumps (every atom trivially has k neighbors)
    from real molecules — all-pairs counting separates them (~2-3 vs ~10
    neighbors within 2.0 A).
    """
    if type_to_z is None:
        # Default QM9 type-index -> atomic-number map (index 0 is padding '*').
        type_to_z = {0: 0, 1: 1, 2: 6, 3: 7, 4: 8, 5: 9, 6: 15, 7: 16, 8: 17, 9: 35}

    max_valence = torch.tensor(
        [VALENCE.get(type_to_z.get(k, 0), 0) for k in range(num_types)],
        dtype=type_probs.dtype, device=type_probs.device,
    )
    # Expected maximum valence under the predicted categorical distribution.
    atom_valence = (type_probs * max_valence.unsqueeze(0)).sum(dim=-1)

    if batch is not None:
        pair_index = full_pair_index(batch.to(pos.device))
    adj_p = distance_adjacency_probs(pos, pair_index)

    # Expected number of bonds per atom (each pair counted from both ends).
    bond_count = torch.zeros_like(atom_valence)
    row, col = pair_index
    bond_count.index_add_(0, row, adj_p)
    bond_count.index_add_(0, col, adj_p)
    bond_count = bond_count / 2.0

    # Expected bond order ~ 1.2 accounts for some double/triple character.
    expected_use = 1.2 * bond_count
    violation = F.relu(expected_use - atom_valence)
    return violation.square().mean()


def molecule_pooling(h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Mean-pool atom embeddings into one vector per molecule."""
    num_graphs = int(batch.max().item()) + 1
    sums = torch.zeros(
        num_graphs, h.size(1), dtype=h.dtype, device=h.device
    )
    sums.index_add_(0, batch, h)
    counts = (
        torch.bincount(batch, minlength=num_graphs).clamp_min(1).to(h.dtype)
    )
    return sums / counts.unsqueeze(-1)


def diversity_regularizer(h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Penalize pairwise cosine similarity of pooled molecular embeddings.

    Acts as a differentiable uniqueness surrogate within a generated batch.
    """
    pooled = F.normalize(molecule_pooling(h, batch), dim=-1)
    sim = pooled @ pooled.T
    num_graphs = sim.size(0)
    if num_graphs < 2:
        return torch.zeros((), dtype=h.dtype, device=h.device)
    off_diag = sim - torch.diag(torch.diagonal(sim))
    denom = num_graphs * (num_graphs - 1)
    return off_diag.abs().sum() / denom
