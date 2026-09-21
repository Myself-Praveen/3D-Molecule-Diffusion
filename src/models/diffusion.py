"""Centered DDPM noise schedules for molecular coordinates and atom types.

Phase 1.5 (see .idea/03_implementation_plan.md):
- ``CenteredDDPM``: Gaussian diffusion over Cartesian coordinates with
  mandatory per-molecule center-of-mass re-centering.
- ``TypeDDPM``: EDM-style categorical diffusion over discrete atom types
  (Hoogeboom et al., "Equivariant Diffusion for Molecule Generation in 3D").

Strategy 2 (validity_80_plan.md): both DDPMs support the cosine noise
schedule (Nichol & Dhariwal, 2021) alongside the original linear one via the
``schedule`` constructor argument. The cosine schedule preserves more signal
at intermediate timesteps, leaving the denoiser more "time budget" for the
fine bond-length precision that validity requires.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def _cosine_betas(
    num_steps: int,
    s: float = 0.008,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Cosine-schedule betas (Nichol & Dhariwal, 2021).

    ᾱ_t = f(t) / f(0),   f(t) = cos²((t/T + s) / (1 + s) · π/2)

    Betas are derived as ``1 − ᾱ_t / ᾱ_{t−1}`` and clipped to [1e-5, 0.999]
    as in the improved-DDPM reference implementation. Computed in float64 —
    the first ratios involve near-cancelling quantities near t = 0.
    """
    steps = torch.arange(num_steps + 1, dtype=torch.float64, device=device)
    f = torch.cos((steps / num_steps + s) / (1.0 + s) * math.pi / 2.0) ** 2
    # ᾱ straight from the reference formula (no lossy betas round-trip).
    # The vector spans t = 0..T; training/sampling use t = 0..T-1, so the
    # returned ᾱ is [:-1]. Max-clamped just below 1 only to keep the
    # degenerate ᾱ_0 = 1 from being exact (sampler divides by 1 - ᾱ); the
    # T-end stays unclamped so ᾱ_T-1 keeps its true (tiny) value.
    alpha_bars = (f / f[0]).clamp(1e-8, 0.99995)
    betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
    return alpha_bars[:-1].float(), betas.clamp(1e-5, 0.999).float()


def _build_schedule(
    num_steps: int,
    beta_start: float,
    beta_end: float,
    schedule: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(betas, alphas, alpha_bars)`` for ``"linear"`` or ``"cosine"``.

    ``beta_start``/``beta_end`` parameterize the linear schedule and are
    ignored (kept for config compatibility) when ``schedule="cosine"``.
    """
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
    elif schedule == "cosine":
        alpha_bars, betas = _cosine_betas(num_steps, device=device)
        alphas = 1.0 - betas
    else:
        raise ValueError(
            f"Unknown diffusion schedule={schedule!r} "
            f"(expected 'linear' or 'cosine')"
        )
    return betas, alphas, alpha_bars


def _scatter_counts(
    batch: torch.Tensor, size: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Number of atoms per molecule, sized exactly to the number of graphs."""
    counts = torch.zeros(size, dtype=dtype, device=device)
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=dtype))
    return counts.clamp_min(1.0)


class CenteredDDPM:
    """Gaussian DDPM over 3D coordinates that stays zero-centered per molecule."""

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = "cuda",
        schedule: str = "linear",
    ) -> None:
        self.num_steps = num_steps
        self.device = torch.device(device)
        self.betas, self.alphas, self.alpha_bars = _build_schedule(
            num_steps, beta_start, beta_end, schedule, self.device
        )

    def add_noise(
        self,
        pos: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add centered Gaussian noise, supporting one timestep per molecule.

        Returns ``(noisy_pos, noise)`` where ``noise`` has zero mean inside
        every individual molecule (translation-invariant targets).
        """
        if batch is None:
            batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        num_graphs = int(batch.max().item()) + 1

        alpha_bar = self.alpha_bars[t].to(device=pos.device)
        atom_alpha_bar = alpha_bar[batch]

        noise = torch.randn_like(pos)
        # Re-center noise per molecule: subtract its scatter-mean.
        sums = torch.zeros_like(pos).index_add(0, batch, noise)
        counts = _scatter_counts(batch, num_graphs, pos.dtype, pos.device)
        noise = noise - sums[batch] / counts[batch].unsqueeze(-1)

        noisy_pos = atom_alpha_bar.sqrt().unsqueeze(-1) * pos
        noisy_pos = noisy_pos + (1.0 - atom_alpha_bar).sqrt().unsqueeze(-1) * noise
        return noisy_pos, noise

    # Alias matching common DDPM notation.
    q_sample = add_noise


class TypeDDPM:
    """Categorical forward/reverse process for discrete atom types.

    Forward (EDM-style; alpha-bar schedule shared with coordinates):
        q(z_t | z_0) = Cat( alpha_bar_t * onehot(z_0) + (1 - alpha_bar_t) / K )
    """

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = "cuda",
        schedule: str = "linear",
    ) -> None:
        self.num_steps = num_steps
        self.device = torch.device(device)
        self.betas, self.alphas, self.alpha_bars = _build_schedule(
            num_steps, beta_start, beta_end, schedule, self.device
        )

    def _atom_batch(
        self, z: torch.Tensor, t: torch.Tensor, batch: torch.Tensor | None
    ) -> torch.Tensor:
        """Per-atom timestep index, broadcasting per-molecule ``t``."""
        if batch is None:
            if t.numel() != z.size(0):
                batch = torch.zeros(
                    z.size(0), dtype=torch.long, device=t.device
                )
            else:
                batch = torch.arange(z.size(0), device=t.device)
        return batch.to(t.device)

    def forward_probs(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        num_classes: int,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Continuous noised type distributions ``q(z_t | z_0)`` of shape (N, K).

        ``t`` holds one timestep per molecule; ``batch`` maps atoms to
        molecules (defaults to all-atoms-in-one-molecule when len(t) != N).
        """
        batch = self._atom_batch(z, t, batch)
        alpha_bar = self.alpha_bars[t.to(self.device)].to(device=z.device)[batch]
        z0 = F.one_hot(z.long(), num_classes).float()
        return alpha_bar.unsqueeze(-1) * z0 + (
            1.0 - alpha_bar
        ).unsqueeze(-1) / num_classes

    def sample_noisy_types(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        num_classes: int,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw discrete noised type indices from ``q(z_t | z_0)``."""
        probs = self.forward_probs(z, t, num_classes, batch)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def posterior_probs(
        self,
        z_t: torch.Tensor,
        clean_probs: torch.Tensor,
        t: torch.Tensor,
        num_classes: int,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Categorical posterior ``q(z_{t-1} | z_t, z0_hat)`` for every atom.

        ``clean_probs`` is the model's predicted distribution over clean types;
        ``t``/``batch`` follow the same convention as :meth:`forward_probs`.
        """
        batch = self._atom_batch(z_t, t, batch)
        t_atom = t.to(self.device)[batch]
        alpha_t = self.alpha_bars[t_atom]                      # (N,)
        alpha_prev = self.alpha_bars[(t_atom - 1).clamp_min(0)]
        alpha_prev = torch.where(t_atom > 0, alpha_prev, torch.ones_like(alpha_prev))

        # Sanitize model output: extreme logits from a confident model can
        # otherwise yield inf/NaN rows that crash multinomial sampling.
        clean = clean_probs.to(self.device).float().view(-1, num_classes)
        clean = clean.clamp(0.0, 1.0)
        clean = clean / clean.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # M[n, v, j] = q(z_t = j | z_{t-1} = v) for the molecule of atom n.
        eye = torch.eye(num_classes, device=self.device)
        trans = alpha_t.view(-1, 1, 1) * eye.unsqueeze(0) + (
            1.0 - alpha_t
        ).view(-1, 1, 1) / num_classes
        prev = alpha_prev.view(-1, 1) * clean + (1.0 - alpha_prev).view(-1, 1) / num_classes

        # Gather transition row of the currently observed z_t per atom.
        batch_idx = torch.arange(z_t.size(0), device=self.device)
        trans_obs = trans[batch_idx, :, z_t.to(self.device).clamp(0, num_classes - 1)]
        post = trans_obs * prev                                   # (N, K)
        post = post.clamp_min(0.0)
        return post / post.sum(dim=-1, keepdim=True).clamp_min(1e-12)
