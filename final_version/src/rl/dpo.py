"""DPO fine-tuning for the diffusion denoiser (plan Step 2.3).

Three corrections from the plan are load-bearing here, and each is pinned by a
test:

1. **Sign.** The implicit reward is ``MSE_ref − MSE_model`` — higher when the
   fine-tuned model denoises *better* than the frozen reference — and the loss
   is ``−log σ(β (r_chosen − r_rejected))``, i.e. the model **maximizes** the
   difference. (The buggy version maximized ``MSE_model − MSE_ref``, which
   trains the model to fit the losers.)
2. **Noisy inputs.** The denoiser expects ``x_t``, so each side is noised with
   ``coord_ddpm.add_noise`` at its own random timestep *before* the forward
   pass. Feeding clean ``x_0`` would evaluate a model on inputs it never saw
   during training and the implicit reward would be meaningless.
3. **Frozen reference.** ``ref_model`` runs under ``torch.no_grad()`` and is
   never optimized — the anchor that makes the reward an *implicit* one.

Both sides share the same reference, so the difference cancels any per-example
difficulty offset that comes from the molecule itself rather than from the
model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F

from src.utils.graph import build_knn_graph


@dataclass
class PreferenceBatch:
    """Tensors for one DPO minibatch (chosen and rejected, already collated).

    ``*_batch`` maps each atom to its molecule index *within that side*, so the
    two sides carry independent (and different) graph counts.
    """

    chosen_pos: torch.Tensor
    chosen_z: torch.Tensor
    chosen_batch: torch.Tensor
    rejected_pos: torch.Tensor
    rejected_z: torch.Tensor
    rejected_batch: torch.Tensor

    @property
    def num_pairs(self) -> int:
        return int(self.chosen_batch.max().item()) + 1 if self.chosen_batch.numel() else 0


def save_preference_pairs(path: str | Path, payload: dict) -> None:
    """Persist pairs (and whatever metadata the generating script recorded)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_preference_pairs(path: str | Path) -> list[dict]:
    """Load the pair list produced by ``scripts/build_rl_pairs.py``."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    pairs = payload["pairs"] if isinstance(payload, dict) else payload
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"no preference pairs found in {path}")
    return pairs


def collate_pairs(
    pairs: Sequence[dict],
    indices: Sequence[int] | None = None,
    device: str | torch.device = "cpu",
) -> PreferenceBatch:
    """Concatenate selected pairs into per-side atom batches.

    Every pair contributes exactly one graph per side, and graph ids are
    assigned in the order the pairs appear, so ``chosen_batch`` and
    ``rejected_batch`` stay aligned pair-by-pair (pair ``i`` is graph ``i``).
    """
    if indices is None:
        indices = range(len(pairs))
    idx = list(indices)
    if not idx:
        raise ValueError("cannot collate an empty set of preference pairs")

    def _side(key: str):
        pos, z, batch, offset = [], [], [], 0
        for graph_id, i in enumerate(idx):
            entry = pairs[i][key]
            n = int(entry["pos"].shape[0])
            pos.append(entry["pos"])
            z.append(entry["z"])
            batch.append(torch.full((n,), graph_id, dtype=torch.long))
            offset += n
        return (
            torch.cat(pos, dim=0),
            torch.cat(z, dim=0).long(),
            torch.cat(batch, dim=0),
        )

    c_pos, c_z, c_batch = _side("chosen")
    r_pos, r_z, r_batch = _side("rejected")
    return PreferenceBatch(
        chosen_pos=c_pos.to(device), chosen_z=c_z.to(device),
        chosen_batch=c_batch.to(device),
        rejected_pos=r_pos.to(device), rejected_z=r_z.to(device),
        rejected_batch=r_batch.to(device),
    )


def cyclic_pair_batches(
    pairs: Sequence[dict], batch_size: int, device: str | torch.device = "cpu"
) -> Iterator[PreferenceBatch]:
    """Infinite iterator over shuffled minibatches of pairs (per epoch reuse)."""
    order = list(range(len(pairs)))
    while True:
        random.shuffle(order)
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            if chunk:
                yield collate_pairs(pairs, chunk, device=device)


def _per_graph_mean(values: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Mean of per-atom ``values`` inside each graph, shape (num_graphs,)."""
    num_graphs = int(batch.max().item()) + 1
    sums = torch.zeros(num_graphs, dtype=values.dtype, device=values.device)
    counts = torch.zeros(num_graphs, dtype=values.dtype, device=values.device)
    sums.index_add_(0, batch, values)
    counts.index_add_(0, batch, torch.ones_like(values))
    return sums / counts.clamp_min(1.0)


def dpo_loss(
    model,
    ref_model,
    coord_ddpm,
    batch: PreferenceBatch,
    beta: float = 0.1,
    objective: str = "eps",
    knn: int = 4,
) -> torch.Tensor:
    """Corrected DPO loss over one minibatch of (chosen, rejected) pairs.

    ``reward = MSE_ref − MSE_model`` per graph; ``loss = −log σ(β Δreward)``
    with ``Δreward = reward_chosen − reward_rejected``. Returns a scalar tensor
    with gradient only w.r.t. ``model``.
    """
    if objective == "flow":
        raise ValueError(
            "DPO is implemented for the eps (noise-prediction) objective; "
            "the flow arm needs an analogous velocity-target rewrite first"
        )
    device = batch.chosen_pos.device

    def _denoise(pos, z, gbatch):
        num_graphs = int(gbatch.max().item()) + 1
        t = torch.randint(0, coord_ddpm.num_steps, (num_graphs,), device=device)
        noisy_pos, target = coord_ddpm.add_noise(pos, t, gbatch)
        edges = build_knn_graph(noisy_pos, gbatch, k=knn)
        # Mirror training: multi-scale models get their per-layer graphs too.
        schedule = getattr(model, "knn_schedule", None)
        edges_per_layer = (
            [build_knn_graph(noisy_pos, gbatch, k=k) for k in schedule]
            if schedule else None
        )
        pred = model(
            z, noisy_pos, edges, t, gbatch, edge_index_per_layer=edges_per_layer
        )[0]
        with torch.no_grad():
            ref_pred = ref_model(
                z, noisy_pos, edges, t, gbatch,
                edge_index_per_layer=edges_per_layer,
            )[0]
        atom_mse = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        with torch.no_grad():
            atom_mse_ref = F.mse_loss(ref_pred, target, reduction="none").mean(dim=-1)
        # Implicit reward: higher when the fine-tuned model beats the reference.
        reward = _per_graph_mean(atom_mse_ref, gbatch) - _per_graph_mean(atom_mse, gbatch)
        return reward

    reward_chosen = _denoise(batch.chosen_pos, batch.chosen_z, batch.chosen_batch)
    reward_rejected = _denoise(
        batch.rejected_pos, batch.rejected_z, batch.rejected_batch
    )
    # Model maximizes (reward_chosen - reward_rejected).
    return -F.logsigmoid(beta * (reward_chosen - reward_rejected)).mean()


def build_reference_model(model, ref_checkpoint: str | None, device) -> torch.nn.Module:
    """Frozen copy of ``model`` weights (the DPO reference policy).

    Loads ``ref_checkpoint`` when given (the plan: the pre-RL ``best.pt``) and
    falls back to the *current* weights otherwise. Parameters are frozen and
    ``eval()`` is set so dropout/self-conditioning behave as at evaluation.
    """
    import copy

    ref = copy.deepcopy(model)
    if ref_checkpoint:
        state = torch.load(ref_checkpoint, map_location=device, weights_only=False)
        state = state.get("model_state_dict", state)
        ref.load_state_dict(state)
    ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref
