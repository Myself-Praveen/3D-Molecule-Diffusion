"""Shared local training/evaluation loops for federated clients.

Refactored out of ``train.py`` so that both the centralized trainer and the
Flower clients (Step 2.2) use identical logic. Supports:

- Joint position + atom-type diffusion loss  L = L_pos + λ_type·L_type.
- Multi-objective terms: soft valence penalty (λ₂) and diversity regularizer
  (λ₃) from ``src/objectives.py`` (Step 3.3).
- FedProx proximal term  μ/2 · ‖w − w_global‖²  added client-side (Step 3.1).
- Personal-head training (Step 3.2B): parameters whose names contain any of
  ``personal_prefixes`` are excluded from global aggregation.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.egnn import EquivariantGenerator
from src.objectives import diversity_regularizer, x0_valence_penalty
from src.training_utils import min_snr_weight, rotate_batch
from src.utils.graph import build_knn_graph

# FedPer-style personal parameters: kept local, never aggregated.
PERSONAL_PREFIXES = ("type_head", "bond_head")

# Type-index -> atomic-number map used by the valence penalty.
# NOTE (coord-fix era): model indices ARE raw atomic numbers (PyG QM9 z is
# {1:H, 6:C, 7:N, 8:O, 9:F}), so this is the identity over supported Z.
TYPE_TO_Z = {i: i for i in range(10)}

# Phase 2: model indices are raw atomic numbers, so supported Z (QM9 + BBB
# common atoms H,C,N,O,F) map to themselves. Heavier BBB atoms (S,P,halogens
# with Z >= num_types) would crash one_hot and fall back to carbon with a
# warning; full-fidelity BBB configs should use a larger num_types (Phase 5).
_SUPPORTED_Z = (1, 6, 7, 8, 9)
_CARBON_Z = 6
_exotic_warned = False


def split_global_personal(
    state_dict: dict[str, torch.Tensor],
    prefixes: tuple[str, ...] = PERSONAL_PREFIXES,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split a state dict into (global backbone, personal head) parameters."""
    global_params, personal_params = {}, {}
    for name, tensor in state_dict.items():
        if name.startswith(prefixes):
            personal_params[name] = tensor
        else:
            global_params[name] = tensor
    return global_params, personal_params


def merge_global_personal(
    global_params: dict[str, torch.Tensor],
    personal_params: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    merged = dict(global_params)
    merged.update(personal_params)
    return merged


def proximal_term(
    model: torch.nn.Module, global_state: dict[str, torch.Tensor], mu: float
) -> torch.Tensor:
    """FedProx penalty μ/2 · ‖w − w_global‖² over non-personal parameters."""
    if mu <= 0.0:
        return torch.zeros((), device=next(model.parameters()).device)
    term = torch.zeros((), device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name.startswith(PERSONAL_PREFIXES):
            continue  # personal heads drift freely by design
        gname = name.replace("module.", "")
        if gname in global_state:
            term = term + (param - global_state[gname]).square().sum()
    return 0.5 * mu * term


class LocalTrainer:
    """Runs local epochs of diffusion training on one client's data shard."""

    def __init__(
        self,
        model: torch.nn.Module,
        coord_ddpm,
        type_ddpm,
        config: dict[str, Any],
        device: str | torch.device,
    ) -> None:
        self.model = model
        self.coord_ddpm = coord_ddpm
        self.type_ddpm = type_ddpm
        self.cfg = config
        self.device = torch.device(device)

    def _extract_cond(self, batch_data) -> dict[str, torch.Tensor] | None:
        """Build the Phase 2 conditioning dict from a BBB batch.

        Returns ``None`` when conditioning is disabled (QM9 unconditional
        path). Missing attributes degrade gracefully to zeros so unconditional
        configs never crash.
        """
        if not self.cfg.get("conditioning", {}).get("enabled", False):
            return None
        n_graphs = int(batch_data.num_graphs)
        device = self.device

        def _graph_attr(name: str) -> torch.Tensor:
            attr = getattr(batch_data, name, None)
            if attr is None:
                return torch.zeros(n_graphs, 1, device=device)
            return attr.to(device).reshape(n_graphs, -1)

        labels = _graph_attr("y").reshape(-1)[:n_graphs]
        if labels.numel() < n_graphs:
            labels = torch.zeros(n_graphs, device=device)
        props = torch.cat(
            [_graph_attr(k) for k in ("qed", "logp", "tpsa", "mw")], dim=-1,
        )
        if props.shape != (n_graphs, 4):
            props = torch.zeros(n_graphs, 4, device=device)
        return {"label": labels.long(), "properties": props.float()}

    def _type_indices(self, batch_data) -> torch.Tensor:
        """Atom-type indices for the model (identity: indices ARE atomic numbers).

        Both QM9 and BBB batches store raw Z ({1:H, 6:C, 7:N, 8:O, 9:F});
        values pass through unchanged. Heavier BBB atoms (S/P/halogens,
        Z >= num_types) cannot be indexed and fall back to carbon with a
        one-time warning — full-fidelity BBB configs should raise num_types
        (Phase 5) instead.
        """
        global _exotic_warned
        z = batch_data.z.long()
        num_types = int(self.cfg["model"]["num_types"])
        exotic = z >= num_types
        if bool(exotic.any()):
            if not _exotic_warned:
                logger.warning(
                    "Mapping %d exotic atoms (Z>=%d) to carbon; "
                    "consider num_types>=54 for full fidelity",
                    int(exotic.sum()), num_types,
                )
                _exotic_warned = True
            z = torch.where(exotic, torch.full_like(z, _CARBON_Z), z)
        return z

    def _batch_loss(
        self,
        batch_data,
        optimizer: torch.optim.Optimizer | None,
        global_state: dict[str, torch.Tensor] | None,
        mu: float,
        lambda_type: float,
        lambda_valence: float,
        lambda_diversity: float,
        grad_clip: float = 0.0,
    ) -> tuple[torch.Tensor, float, float]:
        batch_data = batch_data.to(self.device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        t = torch.randint(
            0, self.coord_ddpm.num_steps,
            (batch_data.num_graphs,), device=self.device,
        )
        # Strategy 1.5: random per-molecule SO(3) rotation augmentation
        # (train-time only; evaluation keeps the true geometry).
        aug_rot = float(self.cfg["training"].get("rotation_augment_prob", 0.0))
        if optimizer is not None and aug_rot > 0.0 and torch.rand(1).item() < aug_rot:
            batch_data.pos = rotate_batch(batch_data.pos, batch_data.batch)
        # Phase 2: property conditioning with label dropout for
        # classifier-free guidance (10% unconditional by default).
        cond = self._extract_cond(batch_data)
        if cond is not None and self.model.training:
            dropout_prob = float(
                self.cfg.get("conditioning", {}).get("label_dropout", 0.1)
            )
            if dropout_prob > 0.0 and torch.rand(1).item() < dropout_prob:
                cond = None
        # Atom-type indices are raw atomic numbers; exotic Z falls back to C.
        z_idx = self._type_indices(batch_data)
        noisy_pos, actual_noise = self.coord_ddpm.add_noise(
            batch_data.pos, t, batch_data.batch,
        )
        noisy_types = self.type_ddpm.sample_noisy_types(
            z_idx, t, self.cfg["model"]["num_types"], batch_data.batch,
        )
        edge_index = build_knn_graph(
            noisy_pos, batch_data.batch, k=self.cfg["training"]["kNN"],
        )
        # Tier 2.3: per-layer graphs when the model carries a kNN schedule.
        edge_index_per_layer = None
        if getattr(self.model, "knn_schedule", None):
            edge_index_per_layer = [
                build_knn_graph(noisy_pos, batch_data.batch, k=k_layer)
                for k_layer in self.model.knn_schedule
            ]
        # Tier 2.1: self-conditioning with 50% dropout (training only).
        x0_estimate = None
        if (
            getattr(self.model, "self_condition", False)
            and optimizer is not None
            and torch.rand(1).item() >= float(
                self.cfg["training"].get("self_cond_dropout", 0.5))
        ):
            ab = self.coord_ddpm.alpha_bars.to(noisy_pos.device)[t][batch_data.batch]
            x0_estimate = (
                noisy_pos
                - (1.0 - ab).sqrt().unsqueeze(-1) * torch.zeros_like(noisy_pos)
            ) / ab.sqrt().unsqueeze(-1).clamp_min(1e-3)
        noise_pred, type_logits, node_h = self.model(
            noisy_types, noisy_pos, edge_index, t, batch_data.batch, cond=cond,
            x0_estimate=x0_estimate,
            edge_index_per_layer=edge_index_per_layer,
        )

        # Strategy 1.2: optional min-SNR-γ timestep weighting on the
        # coordinate loss (same default-off contract as train.py).
        pos_loss_raw = F.mse_loss(noise_pred, actual_noise, reduction="none").mean(dim=-1)
        snr_gamma = float(self.cfg["training"].get("min_snr_gamma", 0.0))
        if snr_gamma > 0.0:
            w = min_snr_weight(
                t, self.coord_ddpm.alpha_bars,
                batch=batch_data.batch, gamma=snr_gamma,
            )
            pos_loss = (w * pos_loss_raw).mean()
        else:
            pos_loss = pos_loss_raw.mean()
        type_loss = F.cross_entropy(type_logits, z_idx)
        loss = pos_loss + lambda_type * type_loss

        # Step 3.3 multi-objective terms (λ₂ evaluated on denoised x0 so
        # geometry — not types — absorbs the pressure; see x0_valence_penalty).
        if lambda_valence > 0.0:
            loss = loss + x0_valence_penalty(
                noise_pred, noisy_pos, type_logits, t, batch_data.batch,
                batch_data.z,
                num_types=self.cfg["model"]["num_types"],
                type_to_z=TYPE_TO_Z,
                alpha_bars=self.coord_ddpm.alpha_bars,
                lambda_weight=lambda_valence,
                tau=int(self.cfg["training"].get("valence_tau", 200)),
            )
        if lambda_diversity > 0.0:
            loss = loss + lambda_diversity * diversity_regularizer(
                node_h, batch_data.batch,
            )

        # Step 3.1 FedProx client-side proximal term.
        if global_state is not None and mu > 0.0:
            loss = loss + proximal_term(self.model, global_state, mu)

        if optimizer is not None:
            loss.backward()
            bad = [n for n, p in self.model.named_parameters()
                   if p.grad is not None and not torch.isfinite(p.grad).all()]
            if bad:
                optimizer.zero_grad(set_to_none=True)
                return loss.detach(), float("nan"), float("nan")
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), grad_clip)
            optimizer.step()
        return loss.detach(), pos_loss.detach().item(), type_loss.detach().item()

    def train(
        self,
        loader: torch.utils.data.DataLoader,
        global_state: dict[str, torch.Tensor] | None = None,
        local_epochs: int | None = None,
        mu: float = 0.0,
        lr: float | None = None,
    ) -> dict[str, float]:
        """Run local epochs; returns mean losses over batches."""
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr if lr is not None else self.cfg["training"]["lr"],
            weight_decay=self.cfg["training"].get("weight_decay", 1e-5),
        )
        epochs = local_epochs or self.cfg["training"]["epochs"]
        lambda_type = self.cfg["training"].get("type_loss_weight", 0.5)
        lambda_valence = self.cfg["training"].get("valence_loss_weight", 0.0)
        lambda_diversity = self.cfg["training"].get("diversity_loss_weight", 0.0)
        grad_clip = float(self.cfg["training"].get("grad_clip_norm", 0.0))

        total_loss = total_pos = total_type = 0.0
        steps = 0
        for _ in range(epochs):
            for batch_data in loader:
                loss, pos_l, type_l = self._batch_loss(
                    batch_data, optimizer, global_state, mu,
                    lambda_type, lambda_valence, lambda_diversity,
                    grad_clip,
                )
                if pos_l != pos_l:  # NaN: batch skipped by grad guard
                    continue
                total_loss += loss.item()
                total_pos += pos_l
                total_type += type_l
                steps += 1
        n = max(steps, 1)
        return {
            "loss": total_loss / n,
            "pos_loss": total_pos / n,
            "type_loss": total_type / n,
            "num_batches": steps,
        }

    @torch.no_grad()
    def evaluate(self, loader: torch.utils.data.DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = total_pos = total_type = 0.0
        steps = 0
        lambda_type = self.cfg["training"].get("type_loss_weight", 0.5)
        for batch_data in loader:
            _, pos_l, type_l = self._batch_loss(
                batch_data, None, None, 0.0, lambda_type, 0.0, 0.0,
            )
            total_pos += pos_l
            total_type += type_l
            total_loss += pos_l + lambda_type * type_l
            steps += 1
        n = max(steps, 1)
        return {
            "loss": total_loss / n,
            "pos_loss": total_pos / n,
            "type_loss": total_type / n,
        }

    # -- Weight transfer helpers -------------------------------------------

    def get_parameters(self, exclude_personal: bool = True) -> dict[str, torch.Tensor]:
        """Return weights to send to the server.

        With personalization enabled, personal heads stay on-device.
        """
        state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        if exclude_personal:
            state, _ = split_global_personal(state)
        return state

    def set_parameters(self, params: dict[str, torch.Tensor]) -> None:
        """Load server-provided weights; keep local personal heads intact."""
        current = self.model.state_dict()
        current.update({k: v.to(current[k].device) for k, v in params.items()})
        self.model.load_state_dict(current)

    def init_personal_heads(self) -> dict[str, torch.Tensor]:
        """Snapshot personal-head params at round 0 so they persist locally."""
        full = self.model.state_dict()
        return {k: v.clone() for k, v in split_global_personal(full)[1].items()}

    def fit_to_global(self, global_state: dict[str, torch.Tensor]) -> None:
        """Replace all non-personal parameters with the aggregated global ones."""
        merged = merge_global_personal(
            {k: v.clone() for k, v in global_state.items()},
            split_global_personal(self.model.state_dict())[1],
        )
        self.model.load_state_dict(merged)


def init_model_from_state(
    model_cfg: dict[str, Any],
    state: dict[str, torch.Tensor] | None,
    device: str | torch.device,
) -> EquivariantGenerator:
    """Build an ``EquivariantGenerator`` from config, optionally loading weights."""
    model = EquivariantGenerator(
        num_types=model_cfg["num_types"],
        node_dim=model_cfg["node_dim"],
        edge_dim=model_cfg["edge_dim"],
        num_layers=model_cfg["num_layers"],
        time_dim=model_cfg["time_dim"],
        cond_dim=int(model_cfg.get("cond_dim", 0)),
        num_cond_classes=int(model_cfg.get("num_cond_classes", 2)),
        use_attention=bool(model_cfg.get("use_attention", False)),
        # Tier 2 flags — default off, matching train.py's construction.
        self_condition=bool(model_cfg.get("self_condition", False)),
        coord_refine_layers=int(model_cfg.get("coord_refine_layers", 0)),
        knn_schedule=model_cfg.get("knn_schedule") or None,
    ).to(device)
    if state is not None:
        model.load_state_dict(state)
    return model


def weighted_fedavg(
    client_states: list[tuple[dict[str, torch.Tensor], int]],
) -> dict[str, torch.Tensor]:
    """Weighted parameter aggregation (Eq. 8 of the paper).

    Each key is averaged with weight ``n_i / Σ n_j``. Keys missing from a
    client (e.g. personal heads) are skipped gracefully.
    """
    total = sum(n for _, n in client_states)
    aggregated: dict[str, torch.Tensor] = {}
    for key in client_states[0][0]:
        acc = None
        weight_acc = 0.0
        for state, n in client_states:
            if key not in state:
                continue
            contribution = state[key].to(torch.float64) * (n / total)
            acc = contribution if acc is None else acc + contribution
            weight_acc += n / total
        if acc is not None and weight_acc > 0:
            aggregated[key] = (acc / weight_acc).to(torch.float32)
    return aggregated
