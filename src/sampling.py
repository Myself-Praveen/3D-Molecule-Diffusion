"""Reverse-time samplers: ancestral DDPM and DDIM (Step 1.2 of the plan)."""

from __future__ import annotations

import torch

from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.utils.graph import build_knn_graph


def _center_per_molecule(pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    num_graphs = int(batch.max().item()) + 1
    sums = torch.zeros_like(pos).index_add(0, batch, pos)
    counts = (
        torch.bincount(batch, minlength=num_graphs)
        .clamp_min(1)
        .to(pos.dtype)
        .unsqueeze(-1)
    )
    return pos - sums[batch] / counts[batch]


@torch.no_grad()
def sample_molecules(
    model,
    coord_ddpm: CenteredDDPM,
    type_ddpm: TypeDDPM,
    atom_counts: torch.Tensor,
    device: str | torch.device = "cuda",
    ddim_steps: int | None = None,
    eta: float = 0.0,
    x0_clamp: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate centered 3D coordinates and atom types for a batch of molecules.

    Args:
        model: trained ``EquivariantGenerator`` denoiser.
        coord_ddpm / type_ddpm: matching noise schedules.
        atom_counts: 1-D tensor with the number of atoms per generated
            molecule (sampled from the training size histogram).
        ddim_steps: if given, use DDIM with this many evenly spaced steps;
            otherwise full ancestral DDPM sampling.
        eta: DDIM stochasticity (0 = deterministic).
        x0_clamp: bound on the predicted clean coordinates per component
            (Angstrom). Prevents the well-known x0-prediction blow-up when the
            denoiser is undertrained; EDM-style static clamping.

    Returns:
        ``(pos, z)`` — centered coordinates (N, 3) and long type indices (N,),
        ordered by molecule.
    """
    device = torch.device(device)
    model.eval()

    counts = [int(c) for c in atom_counts]
    num_graphs = len(counts)
    batch = torch.repeat_interleave(
        torch.arange(num_graphs, device=device),
        torch.tensor(counts, device=device),
    )
    total_atoms = sum(counts)

    # x_T ~ N(0, I), re-centered per molecule so CoM stays at the origin.
    pos = _center_per_molecule(torch.randn(total_atoms, 3, device=device), batch)
    # z_T ~ uniform categorical prior over atom types.
    z = torch.randint(0, model.num_types, (total_atoms,), device=device)

    if ddim_steps is None:
        timestep_iter = list(range(coord_ddpm.num_steps - 1, -1, -1))
    else:
        grid = torch.linspace(0, coord_ddpm.num_steps - 1, ddim_steps + 1).long()
        timestep_iter = grid.flip(0).tolist()[:-1]

    def clamp_x0(x0: torch.Tensor) -> torch.Tensor:
        return x0.clamp(-x0_clamp, x0_clamp)

    for step_idx, t_cur in enumerate(timestep_iter):
        t = torch.full((num_graphs,), t_cur, dtype=torch.long, device=device)
        edge_index = build_knn_graph(pos, batch, k=4)

        noise_pred, type_logits, _ = model(z, pos, edge_index, t, batch)
        alpha_t = coord_ddpm.alphas[t_cur]
        alpha_bar_t = coord_ddpm.alpha_bars[t_cur]

        # --- Coordinate update ---
        x0_pred = clamp_x0(
            (pos - (1.0 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
        )
        if ddim_steps is None:
            # Ancestral DDPM step in x0-prediction form (Ho et al., 2020):
            # mu = c1*x0 + c2*x_t with c1+c2 coefficients from the posterior.
            beta_t = coord_ddpm.betas[t_cur]
            if t_cur > 0:
                alpha_bar_prev = coord_ddpm.alpha_bars[t_cur - 1]
                coef_x0 = (
                    alpha_bar_prev.sqrt() * beta_t / (1.0 - alpha_bar_t)
                )
                coef_xt = (
                    alpha_t.sqrt() * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                )
                mean = coef_x0 * x0_pred + coef_xt * pos
                var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                pos = mean + var.sqrt() * torch.randn_like(pos)
            else:
                pos = x0_pred
        else:
            prev_grid = timestep_iter[step_idx + 1] if step_idx + 1 < len(timestep_iter) else -1
            alpha_bar_prev = (
                coord_ddpm.alpha_bars[prev_grid] if prev_grid >= 0 else torch.tensor(1.0, device=device)
            )
            sigma = eta * ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)).sqrt() * (
                1.0 - alpha_bar_t / alpha_bar_prev
            ).sqrt()
            direction = (1.0 - alpha_bar_prev - sigma.square()).clamp_min(0).sqrt()
            noise = torch.randn_like(pos) if t_cur > 0 else torch.zeros_like(pos)
            pos = alpha_bar_prev.sqrt() * x0_pred + direction * noise_pred + sigma * noise

        pos = _center_per_molecule(pos, batch)

        # --- Atom-type update via categorical posterior q(z_{t-1} | z_t, p0) ---
        probs = type_ddpm.posterior_probs(
            z, torch.softmax(type_logits, dim=-1), t, model.num_types, batch,
        ).to(device)
        z = torch.multinomial(probs, num_samples=1).squeeze(-1)

    pos = _center_per_molecule(pos, batch)
    return pos.cpu(), z.cpu()
