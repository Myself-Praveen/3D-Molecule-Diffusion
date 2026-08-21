"""Centered DDPM noise schedule for molecular coordinates."""

from __future__ import annotations

import torch


class CenteredDDPM:
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
        """Add centered Gaussian noise, supporting one timestep per molecule."""
        alpha_bar = self.alpha_bars[t].to(device=pos.device)
        if batch is None:
            batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        atom_alpha_bar = alpha_bar[batch]
        noise = torch.randn_like(pos)
        noise = noise - torch.zeros_like(pos).index_add(
            0, batch, noise
        )[batch] / torch.bincount(batch, minlength=alpha_bar.numel()).clamp_min(1)[batch].unsqueeze(-1)
        noisy_pos = atom_alpha_bar.sqrt().unsqueeze(-1) * pos
        noisy_pos = noisy_pos + (1.0 - atom_alpha_bar).sqrt().unsqueeze(-1) * noise
        return noisy_pos, noise