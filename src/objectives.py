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


def x0_from_prediction(
    pred: torch.Tensor,
    noisy_pos: torch.Tensor,
    t: torch.Tensor,
    batch: torch.Tensor,
    alpha_bars: torch.Tensor | None = None,
    objective: str = "eps",
) -> torch.Tensor:
    """Predicted clean coordinates from either prediction target.

    - ``objective="eps"`` (DDPM noise prediction):
      ``x0 = (x_t - sqrt(1-ab) * eps) / sqrt(ab)``.
    - ``objective="flow"`` (Tier 3.2 velocity prediction on the linear path
      ``x_u = (1-u) x0 + u eps``): ``x0 = x_u - u * v`` — well-conditioned at
      every u, no sqrt(ab) division (t maps to u via t/(T-1)).
    """
    if objective == "flow":
        from src.models.flow import flow_x0_pred

        # t -> u via the same convention the trainers use (t = 0..T-1 maps
        # to u = 0..1); alpha_bars carries T when available.
        T = len(alpha_bars) if alpha_bars is not None else 1000
        u = t.to(noisy_pos.device).float() / max(T - 1, 1)
        return flow_x0_pred(pred, noisy_pos, u, batch=batch)
    ab = alpha_bars.to(noisy_pos.device)[t][batch]
    s1 = (1.0 - ab).sqrt().unsqueeze(-1)
    s0 = ab.sqrt().unsqueeze(-1).clamp_min(1e-3)
    return (noisy_pos - s1 * pred) / s0


def x0_valence_penalty(
    noise_pred: torch.Tensor,
    noisy_pos: torch.Tensor,
    type_logits: torch.Tensor,
    t: torch.Tensor,
    batch: torch.Tensor,
    z: torch.Tensor,
    num_types: int,
    type_to_z: dict[int, int] | None,
    alpha_bars: torch.Tensor,
    lambda_weight: float = 1.0,
    tau: int = 200,
    x0_clamp: float = 5.0,
    objective: str = "eps",
) -> torch.Tensor:
    """λ₂ validity pressure that actually trains geometry.

    Two corrections over the naive formulation:
    1. The penalty is evaluated on the model's denoised prediction
       ``x0 = (x_t - √(1-ā)·ε̂)/√ā`` (which depends on θ via ``noise_pred``),
       NOT on ``noisy_pos`` (which is θ-independent data — penalizing it can
       only corrupt the type head, observed as CE 0.76→1.22).
    2. Only molecules with ``t < tau`` contribute: at high noise x0 carries
       160×-amplified error and the penalty is meaningless; fine bonds are
       decided at low noise. Type probabilities are detached so the model
       cannot dodge by predicting carbon everywhere.
    ``objective="flow"`` recovers x0 from a velocity prediction (Tier 3.2)
    instead of a noise prediction. Returns 0 when no graph passes the gate.
    """
    if lambda_weight <= 0.0:
        return torch.zeros((), dtype=noise_pred.dtype, device=noise_pred.device)
    gate = (t < tau)
    if not bool(gate.any()):
        return torch.zeros((), dtype=noise_pred.dtype, device=noise_pred.device)
    with torch.no_grad():
        keep = gate[batch]
    x0_pred = x0_from_prediction(
        noise_pred, noisy_pos, t, batch,
        alpha_bars=alpha_bars, objective=objective,
    ).clamp(-x0_clamp, x0_clamp)
    x0_pred = x0_pred[keep]
    probs = torch.softmax(type_logits, dim=-1).detach()[keep]
    _, remap = torch.unique(batch[keep], return_inverse=True)
    return lambda_weight * soft_valence_penalty(
        probs, x0_pred, None, z[keep],
        num_types=num_types, type_to_z=type_to_z, batch=remap,
    )


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
