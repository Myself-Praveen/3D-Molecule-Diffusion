"""Reverse-time samplers: ancestral DDPM and DDIM (Step 1.2 of the plan)."""

from __future__ import annotations

import torch

from src.guidance import (
    DEFAULT_K_REP,
    DEFAULT_SIGMA_SCALE,
    DEFAULT_VALENCE_COVALENT_FACTOR,
    DEFAULT_VALENCE_K_VAL,
    DEFAULT_VALENCE_MAX_STEP,
    DEFAULT_VALENCE_TAU,
    steric_guidance,
    steric_guidance_weight,
    valence_guidance,
    valence_guidance_weight,
)
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
    type_temperature: float = 1.0,
    steric_weight: float = 0.0,
    steric_start_frac: float = 0.7,
    steric_sigma_scale: float = DEFAULT_SIGMA_SCALE,
    steric_k_rep: float = DEFAULT_K_REP,
    steric_r_clash: float | None = None,
    steric_max_step: float = 0.5,
    valence_weight: float = 0.0,
    valence_start_frac: float = 0.7,
    valence_tau: float = DEFAULT_VALENCE_TAU,
    valence_covalent_factor: float = DEFAULT_VALENCE_COVALENT_FACTOR,
    valence_k_val: float = DEFAULT_VALENCE_K_VAL,        valence_max_step: float = DEFAULT_VALENCE_MAX_STEP,
        valence_enabled: bool = False,
        coord_temperature: float = 0.0,
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
        type_temperature: softmax temperature on the type logits before the
            categorical posterior (recommendation.md, "Improving Scaffold
            Diversity" #2). ``>1`` flattens the distribution — more diverse
            atom types/scaffolds at some validity cost; ``<1`` sharpens it.
            ``1.0`` is the legacy behavior.
        steric_weight: test-time steric (hard-sphere overlap) guidance
            strength ``w`` applied to the **clean** x̂₀ estimate — ``x̂₀ ← x̂₀ −
            w·∇E_steric(x̂₀)`` (see ``src/guidance.py``). ``w`` is the *peak*
            of a linear ramp that starts at ``steric_start_frac`` of the
            reverse trajectory and reaches ``steric_weight`` at the final
            step (``steric_guidance_weight``); ``0.0`` (default) is a
            bit-exact no-op, so every existing recipe and checkpoint is
            unchanged.
        steric_start_frac: fraction of the reverse trajectory elapsed before
            the ramp starts (0.7 = the last 30% of steps, where bond geometry
            is decided). Indexed by ``step_idx`` counting UP, not by ``t_cur``
            counting down. Ignored when ``steric_weight == 0``.
        steric_sigma_scale: contact distance as a fraction of the pair's
            covalent-radii sum (0.75 leaves every real bond untouched; see
            ``src/guidance.py``).
        steric_k_rep: overall strength of the repulsive energy (multiplies
            both the contact-distance and the ``steric_r_clash`` term).
        steric_r_clash: optional absolute severe-overlap radius (A), the
            plan's 1.1 A mode — an extra element-independent term on top of
            the per-pair contact distance. ``None`` (default) uses the
            per-pair term alone.
        steric_max_step: per-atom cap (Å) on one guidance displacement.
        coord_temperature: coordinate-space analogue of ``type_temperature``
            (plan Intervention 2): after each reverse step, add Gaussian
            noise scaled by the *remaining* noise level of the step the
            trajectory just landed on, ``sqrt(1 - ᾱ_new)``. ``0.0`` (default)
            is a bit-exact no-op. Decoupled from ``eta``: ``eta`` is the
            DDIM variance term the ancestral update derives from the
            schedule, while this is an explicit extra perturbation.

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

    def _guide(
        x0_pred: torch.Tensor,
        step_idx: int,
        num_steps: int,
        atom_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Steric + valence guidance on the clean estimate (no-op when both w <= 0).

        Part A is hard-sphere steric guidance; Part B is continuous hypervalency
        guidance. Both ramp over the last fraction of the sampling steps so they
        only act when bond geometry is being settled.
        """
        types_to_use = atom_types if atom_types is not None else z

        w_s = steric_guidance_weight(
            step_idx, num_steps,
            ramp_start_frac=steric_start_frac, max_weight=steric_weight,
        )
        if w_s > 0.0:
            x0_pred = steric_guidance(
                x0_pred, batch, atom_types=types_to_use, weight=w_s,
                sigma_scale=steric_sigma_scale, k_rep=steric_k_rep,
                r_clash=steric_r_clash,
                max_step=steric_max_step, clamp=x0_clamp,
            )

        w_v = valence_guidance_weight(
            step_idx, num_steps,
            ramp_start_frac=valence_start_frac, max_weight=valence_weight,
        )
        if w_v > 0.0:
            x0_pred = valence_guidance(
                x0_pred, batch, atom_types=types_to_use, weight=w_v,
                tau=valence_tau, covalent_factor=valence_covalent_factor,
                k_val=valence_k_val, max_step=valence_max_step,
                clamp=x0_clamp,
            )

        return x0_pred

    def _update_types(type_logits: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """One categorical posterior step q(z_{t-1} | z_t, p0) for all atoms."""
        # Temperature scaling (recommendation.md, scaffold diversity):
        # flattening the predicted distribution widens the type/scaffold
        # distribution sampled at every step.
        probs = type_ddpm.posterior_probs(
            z,
            torch.softmax(type_logits / type_temperature, dim=-1),
            t, model.num_types, batch,
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
            # Robustness: an undertrained/OOD-drifted model can emit non-finite
            # predictions mid-trajectory (observed: huge self-conditioning
            # input at high noise -> exploding velocity -> float32 overflow
            # two steps later). Substituting the predict-zero step keeps the
            # run alive — degraded quality beats a NaN-poisoned evaluation.
            if not torch.isfinite(v_pred).all():
                v_pred = torch.zeros_like(v_pred)

            x0_pred = flow_x0_pred(v_pred, pos, u_cur, batch=batch, clamp=x0_clamp)
            guided = _guide(x0_pred, i, num_ode_steps, atom_types=z)
            if guided is not x0_pred:
                # On the OT path x_u ≈ x0 at low u, so the clean-space
                # correction is applied to the current point as well; the
                # step is already capped by ``steric_max_step``.
                pos = pos + (guided - x0_pred)
                x0_pred = guided
            # Tier 2.1: refined x0 becomes the next step's self-conditioning
            # input, exactly as on the DDIM path.
            if self_cond:
                x0_estimate = x0_pred
            if i + 1 < num_ode_steps:
                pos = pos - dt * v_pred
            else:
                pos = x0_pred
            # Coordinate temperature: the OT path's noise level is u itself.
            if coord_temperature > 0.0:
                u_new = max(u_cur - dt, 0.0)
                pos = pos + (
                    coord_temperature * u_new * torch.randn_like(pos)
                )
            pos = _center_per_molecule(pos, batch)
            z = _update_types(type_logits, t)

        pos = _center_per_molecule(pos, batch)
        # Radius normalization (plan Intervention 3): the whole reverse process
        # ran in size-normalized space; return Ångström.
        pos = coord_ddpm.denormalize(pos, batch)
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
        # Robustness: same non-finite-prediction guard as the flow path —
        # an undertrained model can spike mid-trajectory (e.g. responding
        # OOD to the clamped high-noise self-conditioning input); the
        # predict-zero substitution degrades one step instead of poisoning
        # the whole trajectory via the x0/pos update.
        if not torch.isfinite(noise_pred).all():
            noise_pred = torch.zeros_like(noise_pred)
        alpha_t = coord_ddpm.alphas[t_cur]
        alpha_bar_t = coord_ddpm.alpha_bars[t_cur]

        # --- Coordinate update ---
        x0_pred = clamp_x0(
            (pos - (1.0 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()
        )
        # Steric guidance acts on the CLEAN estimate: at t > 0 the coordinates
        # still carry diffusion noise, so a "clash" there is partly the noise
        # draw. The guided x0 is what the reverse step below consumes, so the
        # DPS correction reaches the trajectory through the x0 term (and, at
        # the low-noise steps this guidance is gated to, sqrt(ᾱ_prev) ≈ 1).
        x0_pred = _guide(x0_pred, step_idx, len(timestep_iter), atom_types=z)
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
                ab_new = alpha_bar_prev
            else:
                pos = x0_pred
                ab_new = torch.ones((), device=device)
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
            ab_new = alpha_bar_prev

        # Coordinate temperature (plan Intervention 2): an explicit extra
        # perturbation scaled by the noise level the trajectory has just
        # landed on. ``t_cur > 0`` keeps the final clean sample untouched.
        if coord_temperature > 0.0 and t_cur > 0:
            pos = pos + (
                coord_temperature * (1.0 - ab_new).clamp_min(0.0).sqrt()
                * torch.randn_like(pos)
            )

        pos = _center_per_molecule(pos, batch)

        # --- Atom-type update via categorical posterior q(z_{t-1} | z_t, p0) ---
        z = _update_types(type_logits, t)

    pos = _center_per_molecule(pos, batch)
    # Radius normalization (plan Intervention 3): back from the model's
    # size-normalized space to Ångström.
    pos = coord_ddpm.denormalize(pos, batch)
    return pos.cpu(), z.cpu()
