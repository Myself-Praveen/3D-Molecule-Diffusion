"""Flow matching (Tier 3.2, docs/recommendation.md): velocity prediction.

Conditional flow matching (Lipman et al., 2023) replaces the diffusion
forward/reverse process with straight optimal-transport paths between the
data distribution and a Gaussian. With the linear (OT) path

    x(u) = (1 - u) * x0 + u * eps,    u in [0, 1],  eps ~ N(0, I)

the recommendation doc's velocity target ``v = alpha'_t * x0 + sigma'_t * eps``
(with alpha_t = 1 - u, sigma_t = u) reduces to the path-constant

    v = eps - x0,

so the model regresses a constant velocity along each path and sampling is
plain Euler integration downward in u (``x_{u-delta} = x_u - delta * v̂``),
with the final step landing on the model's clean-coordinate prediction
``x0_hat = x_u - u * v_hat`` — exact at every u, with no sqrt-alpha-bar
division (the DDPM x0 recovery amplifies error by 1/sqrt(ab) at low noise).

The same ``EquivariantGenerator`` is used; it predicts velocity instead of
noise when ``model.objective == "flow"`` (set by the trainer from
``diffusion.objective`` in the config). The DDPM/DDIM noise-prediction
objective (``"eps"``) remains the default and fallback.
"""

from __future__ import annotations

import torch


def flow_time_embedding(u: torch.Tensor | float, num_steps: int) -> torch.Tensor:
    """Map flow times ``u in [0, 1]`` to DDPM timestep indices.

    The sinusoidal time embedding and the categorical atom-type chain keep
    the DDPM convention (t = 0 clean, t = T-1 pure noise), so the flow
    objective shares the model's time conditioning and the type posterior
    with the noise-prediction objective: ``t_idx = floor(u * (T-1))``.
    """
    if not torch.is_tensor(u):
        u = torch.as_tensor(u, dtype=torch.float32)
    return (u * (num_steps - 1)).long().clamp(0, num_steps - 1)


def flow_interpolate(
    x0: torch.Tensor,
    u: torch.Tensor | float,
    batch: torch.Tensor | None = None,
    eps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One linear-path training pair: returns ``(x_u, v)``.

    ``x_u = (1 - u) * x0 + u * eps_c`` with ``eps_c`` re-centered per
    molecule — the same center-of-mass discipline as
    ``CenteredDDPM.add_noise`` — and target velocity ``v = eps_c - x0``.

    ``u`` is per-molecule (B,) or a scalar; ``eps`` overrides the internal
    noise draw (tests / reproducibility). The returned velocity is exact on
    the linear path: ``x_u == x0 + u * v`` for every u.
    """
    if batch is None:
        batch = torch.zeros(x0.size(0), dtype=torch.long, device=x0.device)
    batch = batch.to(x0.device)
    num_graphs = int(batch.max().item()) + 1

    if not torch.is_tensor(u):
        u = torch.full((num_graphs,), float(u), dtype=x0.dtype, device=x0.device)
    elif u.numel() == 1:
        # Scalar-like tensor: broadcast over all molecules.
        u = u.reshape(1).expand(num_graphs)
    u_atom = u.to(device=x0.device, dtype=x0.dtype)[batch].unsqueeze(-1)  # (N, 1)

    if eps is None:
        eps = torch.randn_like(x0)
    else:
        eps = eps.to(device=x0.device, dtype=x0.dtype)
    # Re-center noise per molecule: subtract its scatter-mean.
    sums = torch.zeros_like(eps).index_add(0, batch, eps)
    counts = (
        torch.bincount(batch, minlength=num_graphs)
        .clamp_min(1)
        .to(eps.dtype)
        .unsqueeze(-1)
    )
    eps = eps - sums[batch] / counts[batch]

    x_u = (1.0 - u_atom) * x0 + u_atom * eps
    v = eps - x0
    return x_u, v


def flow_x0_pred(
    v_pred: torch.Tensor,
    x_u: torch.Tensor,
    u: torch.Tensor | float,
    batch: torch.Tensor | None = None,
    clamp: float | None = None,
) -> torch.Tensor:
    """Recover the clean-coordinate prediction from a velocity prediction.

    On the linear path ``x_u = x0 + u * v`` holds identically, so
    ``x0_hat = x_u - u * v_hat`` is well-conditioned at every u (unlike the
    DDPM recovery ``(x_t - sqrt(1-ab) * eps_hat) / sqrt(ab)``, which blows up
    as ab -> 0). ``clamp`` bounds each component (EDM-style static clamping).
    """
    if batch is None:
        batch = torch.zeros(x_u.size(0), dtype=torch.long, device=x_u.device)
    batch = batch.to(x_u.device)
    num_graphs = int(batch.max().item()) + 1
    if not torch.is_tensor(u):
        u = torch.full((num_graphs,), float(u), dtype=x_u.dtype, device=x_u.device)
    elif u.numel() == 1:
        u = u.reshape(1).expand(num_graphs)
    u_atom = u.to(device=x_u.device, dtype=x_u.dtype)[batch].unsqueeze(-1)

    x0 = x_u - u_atom * v_pred
    if clamp is not None:
        x0 = x0.clamp(-clamp, clamp)
    return x0
