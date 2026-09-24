"""Reverse-time samplers: ancestral DDPM and DDIM (Step 1.2 of the plan)."""

from __future__ import annotations

import torch

from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.flow import flow_time_embedding, flow_x0_pred
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


def ddim_timestep_grid(
    num_steps: int, ddim_steps: int, step_schedule: str = "linear"
) -> list[int]:
    """Reverse-order DDIM timestep grid over ``[0, num_steps-1]``.

    Strategy 6 (validity_80_plan.md): ``"quadratic"`` concentrates timesteps
    at low noise — where bond-length precision is decided — by spacing the
    *noise level* linearly and squaring it. ``"linear"`` is the classic
    evenly-spaced grid (backward compatible). The final element is always 0,
    so the last step lands exactly on the clean data.
    """
    if step_schedule == "quadratic":
        t_norm = torch.linspace(0, 1, ddim_steps + 1)
        grid = (t_norm ** 2 * (num_steps - 1)).long()
        # Deduplicate (squaring compresses high-noise steps) and order descending.
        grid = torch.unique_consecutive(grid)
        return grid.flip(0).tolist()
    if step_schedule == "linear":
        grid = torch.linspace(0, num_steps - 1, ddim_steps + 1).long()
        return grid.flip(0).tolist()
    raise ValueError(
        f"Unknown step_schedule={step_schedule!r} "
        f"(expected 'linear' or 'quadratic')"
    )


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
    cond: dict[str, torch.Tensor] | None = None,
    guidance_scale: float = 0.0,
    step_schedule: str = "linear",
    use_self_conditioning: bool | None = None,
    objective: str | None = None,
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
        cond: optional Phase 2 property conditioning dict
            ``{"label": (B,), "properties": (B, 4)}`` targeting e.g. BBB+.
            ``None`` samples unconditionally (backward compatible).
        guidance_scale: classifier-free guidance strength ``w`` (Ho &
            Salimans, 2022). ``0.0`` disables guidance (standard conditional
            pass). Requires a model trained with label dropout (see
            ``LocalTrainer``) and a conditional model (``cond_dim > 0``).
        step_schedule: DDIM grid spacing — ``"linear"`` (evenly spaced,
            backward compatible) or ``"quadratic"`` (denser at low noise
            where bond-length precision is decided; no retraining needed).
        use_self_conditioning: Tier 2.1 — feed the previous step's x0
            estimate back into the model. ``None`` (default) auto-detects:
            on iff the model was trained with ``self_condition=True``.
        objective: prediction target the model was trained with — ``"eps"``
            (DDPM noise; DDPM/DDIM samplers) or ``"flow"`` (Tier 3.2
            velocity field; Euler ODE integration). ``None`` (default)
            auto-detects from ``model.objective``.

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
        timestep_iter = ddim_timestep_grid(
            coord_ddpm.num_steps, ddim_steps, step_schedule
        )[:-1]

    def clamp_x0(x0: torch.Tensor) -> torch.Tensor:
        return x0.clamp(-x0_clamp, x0_clamp)

    # Tier 2.1: auto-enable self-conditioning for models trained with it.
    self_cond = (
        bool(getattr(model, "self_condition", False))
        if use_self_conditioning is None
        else bool(use_self_conditioning) and bool(getattr(model, "self_condition", False))
    )
    x0_estimate = None

    # Tier 2.3: per-layer kNN graphs when the model carries a schedule.
    knn_schedule = getattr(model, "knn_schedule", None)

    # Tier 3.2: prediction target — auto-detected from the model when None.
    obj = (objective or getattr(model, "objective", "eps")).lower()
    if obj not in ("eps", "flow"):
        raise ValueError(f"Unknown objective={obj!r} (expected 'eps' or 'flow')")

    def _update_types(type_logits: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """One categorical posterior step q(z_{t-1} | z_t, p0) for all atoms."""
        probs = type_ddpm.posterior_probs(
            z, torch.softmax(type_logits, dim=-1), t, model.num_types, batch,
        ).to(device)
        # Bulletproof multinomial input: any non-finite/negative entry (e.g.
        # from extreme logits on rare noisy inputs) falls back to uniform for
        # that atom instead of crashing the whole generation run.
        bad_rows = (~torch.isfinite(probs)).any(dim=-1) | (probs.sum(dim=-1) <= 0)
        if bad_rows.any():
            uniform = torch.full_like(probs, 1.0 / probs.size(-1))
            probs = torch.where(bad_rows.view(-1, 1), uniform, probs)
        probs = probs.clamp_min(0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    if obj == "flow":
        # Velocity-field ODE integration (Euler, downward in u):
        #   x_{u-dt} = x_u - dt * v̂,   dt = 1 / num_ode_steps.
        # Straight OT paths make this exact for constant v̂ along a path; the
        # final step lands on the (clamped) clean-coordinate prediction.
        num_ode_steps = ddim_steps or coord_ddpm.num_steps
        dt = 1.0 / num_ode_steps
        T = coord_ddpm.num_steps
        for i in range(num_ode_steps):
            u_cur = 1.0 - i * dt
            t_cur = int(flow_time_embedding(u_cur, T).item())
            t = torch.full((num_graphs,), t_cur, dtype=torch.long, device=device)
            edge_index = build_knn_graph(pos, batch, k=4)
            edge_index_per_layer = None
            if knn_schedule:
                edge_index_per_layer = [
                    build_knn_graph(pos, batch, k=k_layer) for k_layer in knn_schedule
                ]

            if guidance_scale > 0.0 and cond is not None:
                v_c, logits_c, _ = model(
                    z, pos, edge_index, t, batch, cond=cond,
                    x0_estimate=x0_estimate,
                    edge_index_per_layer=edge_index_per_layer)
                v_u, logits_u, _ = model(
                    z, pos, edge_index, t, batch, cond=None,
                    x0_estimate=x0_estimate,
                    edge_index_per_layer=edge_index_per_layer)
                v_pred = v_u + guidance_scale * (v_c - v_u)
                type_logits = logits_u + guidance_scale * (logits_c - logits_u)
            else:
                v_pred, type_logits, _ = model(
                    z, pos, edge_index, t, batch, cond=cond,
                    x0_estimate=x0_estimate,
                    edge_index_per_layer=edge_index_per_layer,
                )

            # Numerical guard (same rationale as the eps path).
            type_logits = type_logits.clamp(-15.0, 15.0)

            x0_pred = flow_x0_pred(v_pred, pos, u_cur, batch=batch, clamp=x0_clamp)
            # Tier 2.1: refined x0 becomes the next step's self-conditioning
            # input, exactly as on the DDIM path.
            if self_cond:
                x0_estimate = x0_pred
            if i + 1 < num_ode_steps:
                pos = pos - dt * v_pred
            else:
                pos = x0_pred
            pos = _center_per_molecule(pos, batch)
            z = _update_types(type_logits, t)

        pos = _center_per_molecule(pos, batch)
        return pos.cpu(), z.cpu()

    for step_idx, t_cur in enumerate(timestep_iter):
        t = torch.full((num_graphs,), t_cur, dtype=torch.long, device=device)
        edge_index = build_knn_graph(pos, batch, k=4)

        edge_index_per_layer = None
        if knn_schedule:
            edge_index_per_layer = [
                build_knn_graph(pos, batch, k=k_layer) for k_layer in knn_schedule
            ]

        if guidance_scale > 0.0 and cond is not None:
            # Classifier-free guidance (Ho & Salimans, 2022):
            # eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
            noise_c, logits_c, _ = model(
                z, pos, edge_index, t, batch, cond=cond,
                x0_estimate=x0_estimate, edge_index_per_layer=edge_index_per_layer)
            noise_u, logits_u, _ = model(
                z, pos, edge_index, t, batch, cond=None,
                x0_estimate=x0_estimate, edge_index_per_layer=edge_index_per_layer)
            noise_pred = noise_u + guidance_scale * (noise_c - noise_u)
            type_logits = logits_u + guidance_scale * (logits_c - logits_u)
        else:
            noise_pred, type_logits, _ = model(
                z, pos, edge_index, t, batch, cond=cond,
                x0_estimate=x0_estimate,
                edge_index_per_layer=edge_index_per_layer,
            )

        # Numerical guard: a confident model emits large logits whose softmax
        # saturates; clamping keeps the categorical posterior finite.
        type_logits = type_logits.clamp(-15.0, 15.0)
        alpha_t = coord_ddpm.alphas[t_cur]
        alpha_bar_t = coord_ddpm.alpha_bars[t_cur]

        # --- Coordinate update ---
        x0_pred = clamp_x0(
            (pos - (1.0 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
        )
        # Tier 2.1: the refined x0 becomes the next step's self-conditioning
        # input (Analog Bits iteration refinement).
        if self_cond:
            x0_estimate = x0_pred
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
        z = _update_types(type_logits, t)

    pos = _center_per_molecule(pos, batch)
    return pos.cpu(), z.cpu()
