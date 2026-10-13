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
    renorm_live: bool = False,
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
    ``renorm_live`` switches the demand side to the §13.14 live-class
    renormalization (see ``soft_valence_penalty``); default ``False`` keeps the
    original behaviour for every existing arm.
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
        renorm_live=renorm_live,
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
    renorm_live: bool = False,
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

    ``renorm_live`` (ablation, progress.md §13.14): the demand side is diluted
    by classes that cannot be a real atom for this dataset (raw-Z indexing
    makes types 2-5, i.e. Z = 2-5, dead — ``max_valence`` 0 — while they still
    carry softmax mass). Renormalizing over the live classes only keeps the
    model's own predicted distribution (it still cannot dodge by predicting
    carbon everywhere) but removes the dilution: measured on real BBBP
    geometry the penalty becomes ~0.01 at every type-head confidence instead
    of growing to 0.33 at CE 1.82. Default ``False`` = original behaviour.
    """
    if type_to_z is None:
        # Default QM9 type-index -> atomic-number map (index 0 is padding '*').
        type_to_z = {0: 0, 1: 1, 2: 6, 3: 7, 4: 8, 5: 9, 6: 15, 7: 16, 8: 17, 9: 35}

    max_valence = torch.tensor(
        [VALENCE.get(type_to_z.get(k, 0), 0) for k in range(num_types)],
        dtype=type_probs.dtype, device=type_probs.device,
    )
    demand_probs = type_probs
    if renorm_live:
        live = (max_valence > 0).to(type_probs.dtype).unsqueeze(0)
        demand_probs = type_probs * live
        demand_probs = demand_probs / demand_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
    # Expected maximum valence under the predicted categorical distribution.
    atom_valence = (demand_probs * max_valence.unsqueeze(0)).sum(dim=-1)

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


def _kabsch_rotation(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Best-fit rotation ``R`` minimising ``||pred @ R − target||`` (3×3).

    Both inputs must already be centred — centring *is* the translation half of
    the rigid alignment. Degenerate cases are guarded: a reflection is excluded
    by flipping the sign of the smallest singular value, and an exactly
    singular covariance keeps the non-reflected branch instead of yielding a
    non-orthogonal matrix.
    """
    cov = pred.transpose(0, 1) @ target                    # 3×3
    u, _, vh = torch.linalg.svd(cov)
    # maximise tr(R^T cov) ⇒ R = U·diag(1,1,det(U V^T))·V^T (Kabsch), where the
    # determinant sign forces a proper rotation rather than a reflection.
    d = torch.sign(torch.linalg.det(u @ vh))
    d = torch.where(d.abs() < 1e-6, torch.ones_like(d), d)
    ones = torch.ones_like(d)
    return u @ torch.diag(torch.stack([ones, ones, d])) @ vh


def kabsch_aligned_mse(
    pred: torch.Tensor, target: torch.Tensor, batch: torch.Tensor,
) -> torch.Tensor:
    """Per-molecule coordinate MSE after optimal rigid-body alignment.

    Rationale (progress.md §15 item 10): the plain per-atom loss also charges
    the model for the *orientation* of the target, but at low noise that
    orientation is only weakly determined by ``x_t`` — most of it lives in the
    particular noise realisation. The cheapest way to reduce such a loss is
    therefore to predict a rotationally *averaged* (i.e. looser) geometry
    rather than the true one. Aligning each molecule first removes that
    unlearnable component, leaving a loss on the shape alone — which is what
    the connectivity ruler actually measures.
    """
    total = pred.new_zeros(())
    n_atoms = 0
    for g in torch.unique(batch):
        idx = torch.where(batch == g)[0]
        p, q = pred[idx], target[idx]
        p = p - p.mean(dim=0, keepdim=True)
        q = q - q.mean(dim=0, keepdim=True)
        # Degenerate/diatomic fragments cannot define a 3-D rotation.
        if p.shape[0] >= 3:
            p = p @ _kabsch_rotation(p, q)
        total = total + (p - q).square().sum()
        n_atoms += int(p.shape[0])
    return total / max(n_atoms, 1)


def pairwise_distance_loss(
    pred: torch.Tensor, target: torch.Tensor, pair_index: torch.Tensor,
) -> torch.Tensor:
    """MSE between interatomic distances of the prediction and the target.

    Invariant to rotation and translation by construction, so no alignment is
    needed, and it states the quantity the connectivity metrics score
    (bond-length windows) explicitly in the objective instead of leaving it
    implicit in a per-atom L2.
    """
    d_pred = (pred[pair_index[0]] - pred[pair_index[1]]).norm(dim=-1)
    d_true = (target[pair_index[0]] - target[pair_index[1]]).norm(dim=-1)
    return F.mse_loss(d_pred, d_true)


def x0_geometry_consistency(
    noise_pred: torch.Tensor,
    noisy_pos: torch.Tensor,
    t: torch.Tensor,
    batch: torch.Tensor,
    target_pos: torch.Tensor,
    alpha_bars: torch.Tensor,
    lambda_kabsch: float = 1.0,
    lambda_pairdist: float = 1.0,
    tau: int = 200,
    objective: str = "eps",
) -> torch.Tensor:
    """Gated shape-aware terms on the denoised ``x0`` (Task 2).

    Both terms follow ``x0_valence_penalty``'s contract: they act on the
    model's prediction (so geometry, not the type head, absorbs them) and are
    gated to ``t < tau`` because x0 error is amplified at high noise while fine
    geometry is decided at low noise. Molecules whose x0 estimate is non-finite
    are dropped rather than propagated — the same rule the samplers apply.
    Weights default to 1.0 but the caller only invokes this when the gated
    ``training.geofix`` flag is on, so every existing arm is unchanged.
    """
    zero = torch.zeros((), dtype=noise_pred.dtype, device=noise_pred.device)
    if lambda_kabsch <= 0.0 and lambda_pairdist <= 0.0:
        return zero
    gate = (t < tau)
    if not bool(gate.any()):
        return zero
    with torch.no_grad():
        keep = gate[batch]
    x0_pred = x0_from_prediction(
        noise_pred, noisy_pos, t, batch,
        alpha_bars=alpha_bars, objective=objective,
    )[keep]
    target = target_pos[keep]
    sub_batch = batch[keep]
    _, remap = torch.unique(sub_batch, return_inverse=True)

    # Drop whole molecules containing any non-finite x0 entry.
    atom_ok = torch.isfinite(x0_pred).all(dim=-1)
    bad = torch.zeros(int(remap.max().item()) + 1, dtype=torch.bool,
                      device=x0_pred.device)
    bad.index_add_(0, remap, ~atom_ok)
    good = ~bad[remap]
    if not bool(good.all()):
        if not bool(good.any()):
            return zero
        x0_pred, target, sub_batch = x0_pred[good], target[good], sub_batch[good]
        _, remap = torch.unique(sub_batch, return_inverse=True)

    loss = zero
    if lambda_kabsch > 0.0:
        loss = loss + lambda_kabsch * kabsch_aligned_mse(x0_pred, target, remap)
    if lambda_pairdist > 0.0:
        pair_index = full_pair_index(remap)
        if pair_index.numel() > 0:
            loss = loss + lambda_pairdist * pairwise_distance_loss(
                x0_pred, target, pair_index)
    return loss


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
