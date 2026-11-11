"""Tier 1 training enhancements (docs/recommendation.md, Strategies 1.1-1.5).

All pieces are opt-in via config keys so every existing checkpoint/config
keeps working unchanged:

- ``EMA`` (1.1): shadow copy of the weights, decay 0.9999 — the standard
  diffusion-training trick (DDPM, EDM, Stable Diffusion). Saved alongside the
  raw weights; generation should load the EMA copy.
- ``min_snr_weight`` (1.2): min-SNR-γ per-timestep loss weighting (Hang et
  al., CVPR 2023) so low-noise timesteps — where bond-length precision is
  decided — are not drowned out by the easy high-noise steps.
- ``random_rotation_matrix`` (1.5): SO(3) augmentation. The EGNN coordinate
  path is equivariant, but the type head is invariant and benefits from
  seeing rotated inputs; augmentation also combats overfitting.
- ``build_warmup_cosine_scheduler`` (1.4): linear LR warmup before the cosine
  decay, preventing early gradient spikes in freshly-initialized attention
  gates.
"""

from __future__ import annotations

import copy
import math

import torch


# ---------------------------------------------------------------------------
# Strategy 1.1: Exponential Moving Average of weights
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model parameters (and buffers).

    ``update()`` after each ``optimizer.step()``:
        θ_EMA ← decay · θ_EMA + (1 − decay) · θ

    Buffers (e.g. BatchNorm running stats — none in the EGNN today, but
    future-proof) are copied rather than averaged: they are not gradients.
    The decay ramps up linearly over ``warmup`` updates from 0.9 so early
    training — when weights move fast — is not averaged into mush, following
    the EDM heuristic.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        warmup: int = 1000,
    ) -> None:
        self.decay = float(decay)
        self.warmup = int(warmup)
        self.num_updates = 0
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        d = self.decay
        if self.warmup > 0:
            d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        """Load EMA weights into ``model`` (for evaluation/sampling)."""
        model.load_state_dict(
            {k: v.to(next(model.parameters()).device) for k, v in self.shadow.items()}
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state.items()}


# ---------------------------------------------------------------------------
# Strategy 1.2: min-SNR-γ timestep weighting
# ---------------------------------------------------------------------------

def min_snr_weight(
    t: torch.Tensor,
    alpha_bars: torch.Tensor,
    batch: torch.Tensor | None = None,
    gamma: float = 5.0,
) -> torch.Tensor:
    """Per-timestep min-SNR-γ loss weights (Hang et al., CVPR 2023).

    ``w(t) = min(SNR(t), γ) / SNR(t)`` with ``SNR = ᾱ / (1 − ᾱ)``.

    With ε-prediction the plain MSE implicitly weights the x0-reconstruction
    error by SNR(t), so low-noise timesteps dominate the gradient by orders
    of magnitude. min-SNR-γ caps that effective weight at γ: w ≤ 1 always,
    w ≈ 1 for SNR ≤ γ (mid/high noise), and w = γ/SNR < 1 for low noise.
    The result is a balanced timestep distribution instead of one dominated
    by the easiest (lowest-noise) steps. Weights are additionally normalized
    to mean 1 over the batch so the overall loss scale — and the pos/type
    balance — stays comparable to unweighted training:

    - ``t``: (B,) per-molecule timesteps (or (N,) per-atom when ``batch``
      is None).
    - ``batch``: per-atom graph index; when given, ``t`` is per-molecule and
      the returned weights are per-atom (broadcast through ``batch``).
    """
    if batch is not None:
        t = t.to(alpha_bars.device)[batch]
    else:
        t = t.to(alpha_bars.device)
    ab_t = alpha_bars[t]
    snr = ab_t / (1.0 - ab_t).clamp_min(1e-8)
    w = (snr.clamp(max=gamma) / snr).clamp(0.0, 1.0)
    return w / w.mean().clamp_min(1e-8)


# ---------------------------------------------------------------------------
# Strategy 1.5: random SO(3) rotation augmentation
# ---------------------------------------------------------------------------

def random_rotation_matrix(
    batch_size: int = 1, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Uniformly random SO(3) rotation matrices, one per molecule.

    Uses the QR-decomposition trick: a Gaussian matrix's Q factor is Haar-
    distributed on O(3); the determinant is fixed to +1 to land on SO(3).
    Returns ``(batch_size, 3, 3)``.
    """
    m = torch.randn(batch_size, 3, 3, device=device)
    q, r = torch.linalg.qr(m)
    # Fix sign so det(Q) = +1 (QR is only unique up to column signs).
    d = torch.diagonal(r, dim1=-2, dim2=-1)
    q = q * d.sign().unsqueeze(-2)
    det = torch.det(q)
    q[det < 0, :, 0] *= -1.0
    return q


def rotate_batch(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Apply one random rotation per molecule to ``pos`` (Strategy 1.5)."""
    num_graphs = int(batch.max().item()) + 1
    R = random_rotation_matrix(num_graphs, device=pos.device)
    return torch.bmm(pos.unsqueeze(-2), R[batch].transpose(-1, -2)).squeeze(-2)


# ---------------------------------------------------------------------------
# Strategy 1.4: warmup + cosine LR schedule
# ---------------------------------------------------------------------------

def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 0,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Linear warmup then cosine decay (Strategy 1.4 option 1).

    ``warmup_epochs=0`` returns ``None`` (caller keeps the plain cosine
    schedule — exact legacy behavior). ``total_epochs`` is the *total*
    training horizon, matching ``CosineAnnealingLR(T_max=total_epochs)``.
    """
    if warmup_epochs <= 0:
        return None
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup_epochs = min(warmup_epochs, max(total_epochs - 1, 1))
    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / max(warmup_epochs, 1),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_epochs, 1),
    )
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
