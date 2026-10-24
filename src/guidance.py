"""Test-time geometric guidance for the reverse-diffusion sampler.

Why this module exists
----------------------
``progress.md`` §15 (items 3/4/13) excludes the training budget, the λ₂
demand side and explicit shape supervision as causes of the BBBP "atom soup":
every one of those arms plateaued at ConnectedValidity ≈ 1–1.5 % with
29–33 bonds per molecule against ~19 for real BBBP compounds. What remains
is the objective↔sampler link — and one thing neither training nor sampling
has ever contained is a **steric constraint**: nothing pushes two atoms that
overlap apart, so the denoiser's local equilibrium can be a dense clump.

This module supplies that constraint as *test-time* guidance, so it can be
applied to an existing checkpoint with zero retraining:

    x̂₀ ← x̂₀ − w · ∇_{x̂₀} E_steric(x̂₀)

Three design points matter and are each the correction of a flaw in the
original plan (the pasted diagnostic's "Intervention 1"):

1. **The potential must not repel real bonds.** The plan's clash mask
   ``r_clash < d < r_bond`` with ``r_clash = 1.2 Å`` and ``r_bond = 1.8 Å``
   actually repels every real covalent bond (C–C 1.54, C–N 1.47, C–O 1.43,
   C=C 1.34, C=O 1.22 Å all sit inside that window) while applying *zero*
   force to a genuine overlap at 0.5–0.9 Å. The energy here is instead a
   per-element-pair hard-sphere overlap,

       E = ½ · k · Σ_{i≠j} relu(σ_ij − d_ij)²,   σ_ij = s · (r_i^cov + r_j^cov)

   with ``s = 0.75``. For carbon σ = 1.14 Å, so a real C–C bond (1.54 Å) and
   a real C–H bond (1.09 Å vs σ = 0.81 Å) contribute exactly zero, while an
   atomic overlap does — the chemically correct asymmetry (repulsion below
   the contact distance, freedom above it). Tightening or loosening ``s`` is
   the single knob: larger ``s`` starts repelling compressed real bonds.

2. **The guidance must act on the clean estimate x̂₀, not on x_t.** At
   ``t > 0`` the coordinates still carry the diffusion noise, so a pair can
   be "overlapping" purely because of that draw. The sampler therefore
   denoises first (DDPM/DDIM posterior or the flow-matching x0 recovery),
   applies the energy gradient in *clean* coordinate space, and only then
   performs the reverse step — Tweedie-style guidance, as used by DPS.

3. **The energy needs no model forward pass.** It is a pure function of the
   coordinates, so the plan's "recompute ``noise_pred`` with grad" wrapper is
   dropped: one ``autograd.grad`` on the energy is the whole cost.

The module is also import-light on purpose: the covalent-radii table lives
here (and is re-exported by ``src/utils/evaluation.py`` so the guidance and
the ruler that scores its output cannot disagree about an element), which
keeps RDKit off the sampler's import path.
"""

from __future__ import annotations

import torch

# Cordero et al. (2008) covalent radii in Å — the same table the eval bond
# builder uses (src/utils/evaluation.py re-exports this dict).  Elements
# outside the table (metals, noble gases, the padding class 0) fall back to
# carbon: the datasets this project trains on contain no such atoms, and a
# fallback keeps the energy finite instead of raising mid-generation.
COVALENT_RADII: dict[int, float] = {
    1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20, 53: 1.39,
}
_FALLBACK_RADIUS: float = 0.76  # carbon

# Default contact scale: σ_ij = 0.75 · (r_i + r_j).
#   0.75 · (C + C) = 1.14 Å < every real C–C bond (1.34–1.54 Å)
#   0.75 · (C + H) = 0.81 Å < every real C–H bond (1.06–1.10 Å)
#   0.75 · (C + O) = 1.07 Å < C–O 1.43 Å and C=O 1.21 Å
# so true bonds are untouched while a collapsed overlap is pushed apart.
DEFAULT_SIGMA_SCALE: float = 0.75
DEFAULT_K_REP: float = 1.0


def atom_radii(
    atom_types: torch.Tensor | None,
    num_atoms: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Covalent radius per atom, indexed by atomic number.

    ``atom_types`` are the sampler's type indices, which this project keeps
    equal to raw atomic numbers (``progress.md`` §7 Discovery B), so no
    remapping table is involved. ``None`` (types not yet resolved) returns
    the carbon fallback for every atom.
    """
    if atom_types is None:
        return torch.full((num_atoms,), _FALLBACK_RADIUS, device=device, dtype=dtype)
    types = atom_types.to(device=device).long().reshape(-1)
    if types.numel() != num_atoms:
        raise ValueError(
            f"atom_types has {types.numel()} entries but pos has {num_atoms} atoms"
        )
    # Vectorised lookup: build the sorted (Z, radius) table once per call.
    zs = torch.tensor(sorted(COVALENT_RADII), device=device, dtype=torch.long)
    rs = torch.tensor(
        [COVALENT_RADII[int(z)] for z in zs.tolist()], device=device, dtype=dtype
    )
    idx = torch.searchsorted(zs, types.clamp_min(0))
    # searchsorted can land past the end for exotic Z (e.g. 53 > table max)
    # ; clamp the index and then correct the hits back to the fallback.
    idx = idx.clamp_max(zs.numel() - 1)
    radii = rs[idx]
    matched = zs[idx] == types
    return torch.where(matched, radii, torch.full_like(radii, _FALLBACK_RADIUS))


def steric_energy(
    pos: torch.Tensor,
    batch: torch.Tensor,
    atom_types: torch.Tensor | None = None,
    sigma_scale: float = DEFAULT_SIGMA_SCALE,
    k_rep: float = DEFAULT_K_REP,
) -> torch.Tensor:
    """Hard-sphere overlap energy of a batch of molecules (Å² units).

    Args:
        pos: ``(N, 3)`` coordinates — use the *clean* estimate x̂₀.
        batch: ``(N,)`` molecule index per atom.
        atom_types: ``(N,)`` atomic numbers (``None`` → carbon radii).
        sigma_scale: contact distance as a fraction of the radii sum. Below
            ~0.6 the term is nearly inert; above ~0.9 it starts repelling
            compressed real bonds (see the module docstring).
        k_rep: overall strength.

    Returns:
        Scalar tensor, differentiable w.r.t. ``pos``. Each molecule is
        evaluated on its own pairwise block (no cross-graph pairs), so the
        per-step memory cost is that of the largest molecule, not of the
        batch.

    Notes:
        Distances use ``sqrt(d² + 1e-12)`` so the gradient is finite at
        exactly coincident atoms (where it is zero — a measure-zero
        degenerate state; the soft sphere has no preferred escape direction).
    """
    if pos.numel() == 0:
        return pos.sum() * 0.0
    if batch.numel() != pos.size(0):
        raise ValueError(
            f"batch has {batch.numel()} entries but pos has {pos.size(0)} atoms"
        )
    batch = batch.to(device=pos.device)
    radii = atom_radii(atom_types, pos.size(0), pos.device, pos.dtype)

    eye_cache: dict[int, torch.Tensor] = {}
    total = pos.sum() * 0.0  # keeps dtype/device and the autograd path alive
    num_graphs = int(batch.max().item()) + 1
    for g in range(num_graphs):
        members = (batch == g).nonzero(as_tuple=False).squeeze(-1)
        n = int(members.numel())
        if n < 2:
            continue
        p = pos[members]
        r = radii[members]
        d = torch.sqrt(torch.cdist(p, p).square() + 1e-12)
        sigma = sigma_scale * (r.unsqueeze(0) + r.unsqueeze(1))
        clash = (sigma - d).clamp_min(0.0)
        if n not in eye_cache:
            eye_cache[n] = torch.eye(n, device=pos.device, dtype=pos.dtype)
        off_diagonal = 1.0 - eye_cache[n]
        total = total + 0.5 * k_rep * (clash * off_diagonal).square().sum()
    return total


def steric_guidance(
    x0: torch.Tensor,
    batch: torch.Tensor,
    atom_types: torch.Tensor | None = None,
    weight: float = 0.0,
    sigma_scale: float = DEFAULT_SIGMA_SCALE,
    k_rep: float = DEFAULT_K_REP,
    max_step: float = 0.5,
    clamp: float | None = None,
) -> torch.Tensor:
    """One guided step in clean-coordinate space: ``x̂₀ − w·∇E(x̂₀)``.

    Args:
        x0: clean-coordinate estimate from the denoiser.
        batch: molecule index per atom.
        atom_types: atomic numbers per atom (``None`` → carbon radii).
        weight: guidance strength ``w``; ``<= 0`` is a no-op and returns
            ``x0`` unchanged (so the flag can be left at 0 in every existing
            recipe without perturbing it).
        sigma_scale / k_rep: forwarded to :func:`steric_energy`.
        max_step: per-atom displacement cap in Å. The energy gradient grows
            linearly with the overlap depth, so a badly collapsed molecule
            can produce a large first step; capping it keeps the trajectory
            stable instead of letting one step undo the denoising.
        clamp: optional per-component bound applied after the step (the
            sampler's ``x0_clamp``), keeping the guided estimate inside the
            same EDM-style box as the unguided one.

    Returns:
        Guided clean-coordinate estimate (same shape as ``x0``).
    """
    if weight <= 0.0 or x0.size(0) < 2:
        return x0
    # The sampler runs under torch.no_grad(); re-enable autograd locally for
    # the energy gradient only (no model forward pass is needed — the energy
    # is a pure function of the coordinates).
    with torch.enable_grad():
        x = x0.detach().requires_grad_(True)
        energy = steric_energy(
            x, batch, atom_types=atom_types,
            sigma_scale=sigma_scale, k_rep=k_rep,
        )
        (grad,) = torch.autograd.grad(energy, x, allow_unused=True)
    if grad is None:
        return x0
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    step = (-weight * grad).clamp(-max_step, max_step)
    guided = x0 + step
    if clamp is not None:
        guided = guided.clamp(-clamp, clamp)
    return guided
