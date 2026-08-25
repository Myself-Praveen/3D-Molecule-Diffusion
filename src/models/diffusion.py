"""Centered DDPM noise schedules for molecular coordinates and atom types.

Phase 1.5 (see .idea/03_implementation_plan.md):
- ``CenteredDDPM``: linear-schedule Gaussian diffusion over Cartesian
  coordinates with mandatory per-molecule center-of-mass re-centering.
- ``TypeDDPM``: EDM-style categorical diffusion over discrete atom types
  (Hoogeboom et al., "Equivariant Diffusion for Molecule Generation in 3D").
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
    ) -> None:
        self.num_steps = num_steps
        self.device = torch.device(device)
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

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

    Forward (EDM-style, linear alpha-bar schedule shared with coordinates):
        q(z_t | z_0) = Cat( alpha_bar_t * onehot(z_0) + (1 - alpha_bar_t) / K )
    """

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str | torch.device = "cuda",
    ) -> None:
        self.num_steps = num_steps
        self.device = torch.device(device)
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

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

        # M[n, v, j] = q(z_t = j | z_{t-1} = v) for the molecule of atom n.
        eye = torch.eye(num_classes, device=self.device)
        trans = alpha_t.view(-1, 1, 1) * eye.unsqueeze(0) + (
            1.0 - alpha_t
        ).view(-1, 1, 1) / num_classes
        prev = alpha_prev.view(-1, 1) * clean_probs.to(self.device).view(
            -1, num_classes
        ) + (1.0 - alpha_prev).view(-1, 1) / num_classes

        # Gather transition row of the currently observed z_t per atom.
        batch_idx = torch.arange(z_t.size(0), device=self.device)
        trans_obs = trans[batch_idx, :, z_t.to(self.device)]      # (N, K)
        post = trans_obs * prev                                   # (N, K)
        return post / post.sum(dim=-1, keepdim=True).clamp_min(1e-12)
