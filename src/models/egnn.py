"""E(3)-equivariant denoiser for molecular graphs (Phase 1.5 EGNN v2 + Phase 2).

Phase 2 (implementation.md): optional property conditioning for BBB-targeted
generation. When ``cond_dim > 0`` the model accepts ``cond`` dicts
``{"label": (B,) long, "properties": (B, 4) float}`` and adds the projected
conditioning vector to the timestep embedding (EDM/Imagen-style additive
conditioning). With ``cond_dim=0`` (default) the model is exactly the
unconditional Phase 1.5 denoiser (backward compatible).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

BOND_CLASSES = 5  # none, single, double, triple, aromatic


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal diffusion-timestep embedding of shape (len(t), dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    args = t.to(torch.float32).unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class EGNNLayer(MessagePassing):
    """One message-passing layer with equivariant coordinate updates.

    Strategy 4 (validity_80_plan.md): with ``use_attention=True``, a sigmoid
    gate learned from the edge message re-weights each edge contribution
    before both the coordinate and node updates, letting the model
    distinguish bonded from merely-close neighbors (ring closure, steric
    contacts). Default off — identical to the pre-attention layer.
    """

    def __init__(
        self, node_dim: int, edge_dim: int, time_dim: int,
        use_attention: bool = False,
    ) -> None:
        super().__init__(aggr="mean")
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1 + time_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim + time_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )
        # Strategy 4: per-edge sigmoid attention gate over the message.
        self.attn_mlp = (
            nn.Sequential(nn.Linear(edge_dim, 1), nn.Sigmoid())
            if use_attention else None
        )
        # Zero-init the final coordinate update: the untrained network is the
        # identity map, so predicted noise starts near zero (Rec A2).
        nn.init.zeros_(self.coord_mlp[-1].weight)
        nn.init.zeros_(self.coord_mlp[-1].bias)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        radial = coord_diff.square().sum(dim=-1, keepdim=True)
        msg_input = torch.cat((h[row], h[col], radial, t_emb[row]), dim=-1)
        msg = self.edge_mlp(msg_input)
        if self.attn_mlp is not None:
            msg = msg * self.attn_mlp(msg)

        trans = coord_diff * self.coord_mlp(msg)
        pos_update = torch.zeros_like(pos)
        pos_update.index_add_(0, row, trans)
        updated_pos = pos + pos_update

        aggregated = torch.zeros(
            h.size(0), msg.size(-1), dtype=msg.dtype, device=msg.device
        )
        aggregated.index_add_(0, col, msg)
        counts = (
            torch.bincount(col, minlength=h.size(0)).clamp_min(1).unsqueeze(-1)
        )
        aggregated = aggregated / counts
        node_input = torch.cat((h, aggregated, t_emb), dim=-1)
        updated_h = h + self.node_mlp(node_input)
        return updated_h, updated_pos


class EquivariantGenerator(nn.Module):
    """Predict coordinate noise and atom-type logits from noisy molecular state.

    Returns ``(noise_pred, type_logits, node_h)``. ``node_h`` are the final
    hidden atom embeddings, consumed by the bond head during chemistry
    reconstruction (Step 1.4 of the implementation plan).
    """

    def __init__(
        self,
        num_types: int = 10,
        node_dim: int = 64,
        edge_dim: int = 64,
        num_layers: int = 4,
        time_dim: int = 32,
        cond_dim: int = 0,
        num_cond_classes: int = 2,
        use_attention: bool = False,
        self_condition: bool = False,
        coord_refine_layers: int = 0,
        knn_schedule: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_types = num_types
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self.num_cond_classes = num_cond_classes
        # Tier 2.1 (docs/recommendation.md): self-conditioning — feed the
        # previous step's x0 estimate back as an extra per-atom feature.
        self.self_condition = bool(self_condition)
        if self.self_condition:
            self.x0_proj = nn.Linear(3, node_dim)
        # Tier 2.4: lightweight post-hoc coordinate refinement EGNN over the
        # predicted clean coordinates x0. 0 = off (legacy behavior).
        self.coord_refine_layers = int(coord_refine_layers)
        if self.coord_refine_layers > 0:
            self.refine_layers = nn.ModuleList(
                [
                    EGNNLayer(node_dim, edge_dim, time_dim,
                              use_attention=use_attention)
                    for _ in range(self.coord_refine_layers)
                ]
            )
        # Tier 2.3: per-layer kNN schedule for multi-scale message passing
        # (e.g. [4, 4, 8, 8, 16]). Empty/None = the caller's fixed graph is
        # reused for every layer (legacy behavior).
        self.knn_schedule = list(knn_schedule) if knn_schedule else None
        self.embed = nn.Embedding(num_types, node_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # Phase 2: property conditioning projector. Input is
        # [class_embedding(num_cond_classes) + continuous_props(4)].
        if cond_dim > 0:
            if cond_dim % 2 != 0:
                raise ValueError(f"cond_dim must be even, got {cond_dim}")
            self.cond_embed = nn.Embedding(num_cond_classes, cond_dim // 2)
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim // 2 + 4, cond_dim),  # 4 = QED, LogP, tPSA, MW
                nn.SiLU(),
                nn.Linear(cond_dim, time_dim),  # project to time_dim
            )
        self.layers = nn.ModuleList(
            [
                EGNNLayer(node_dim, edge_dim, time_dim, use_attention=use_attention)
                for _ in range(num_layers)
            ]
        )
        self.type_head = nn.Linear(node_dim, num_types)
        self.bond_head = nn.Sequential(
            nn.Linear(node_dim * 2 + 1, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, BOND_CLASSES),
        )

    @property
    def output_dim(self) -> int:
        return self.layers[0].node_mlp[-1].out_features

    def _cond_embedding(
        self,
        t_emb: torch.Tensor,
        cond: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """Add property conditioning to the per-graph timestep embedding.

        ``t_emb`` is (B, time_dim); ``cond`` holds ``label`` (B,) and
        ``properties`` (B, 4). ``cond=None`` (or ``cond_dim=0``) is the
        unconditional path used for classifier-free guidance training.
        """
        if self.cond_dim == 0 or cond is None:
            return t_emb
        labels = cond["label"].reshape(-1).long().to(t_emb.device)
        labels = labels.clamp(0, self.num_cond_classes - 1)
        props = cond["properties"].to(dtype=t_emb.dtype, device=t_emb.device)
        if props.dim() == 3:
            props = props.squeeze(-2)  # tolerate (B, 1, 4) batched layout
        class_emb = self.cond_embed(labels)  # (B, cond_dim//2)
        cond_emb = self.cond_proj(torch.cat([class_emb, props], dim=-1))
        return t_emb + cond_emb

    def encode(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor | None = None,
        cond: dict[str, torch.Tensor] | None = None,
        x0_estimate: torch.Tensor | None = None,
        edge_index_per_layer: list[torch.Tensor] | None = None,
        return_t_emb: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run message passing; returns (final atom embeddings, updated pos).

        ``x0_estimate``: Tier 2.1 self-conditioning input — the previous
        denoising step's clean-coordinate estimate, projected and added to
        the atom features. ``None`` (training with dropout, or feature off)
        is the plain path.

        ``edge_index_per_layer``: Tier 2.3 multi-scale graphs — one kNN graph
        per layer. ``None`` reuses ``edge_index`` for every layer.
        """
        if batch is None:
            batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        t_emb = timestep_embedding(t, self.time_dim)
        t_emb = self._cond_embedding(self.time_proj(t_emb), cond)[batch]
        h = self.embed(z.long())
        if self.self_condition and x0_estimate is not None:
            h = h + self.x0_proj(x0_estimate)
        for i, layer in enumerate(self.layers):
            ei = (
                edge_index_per_layer[i]
                if edge_index_per_layer is not None else edge_index
            )
            h, pos = layer(h, pos, ei, t_emb)
        return (h, pos, t_emb) if return_t_emb else (h, pos)

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        t: torch.Tensor,
        batch: torch.Tensor | None = None,
        cond: dict[str, torch.Tensor] | None = None,
        x0_estimate: torch.Tensor | None = None,
        edge_index_per_layer: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch is None:
            batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        # NOTE: clone is load-bearing — without it initial_pos aliases pos
        # and noise_pred collapses to exactly 0 with no gradient path
        # (the pre-Phase-2 coordinate head never learned).
        initial_pos = pos.clone()
        h, updated_pos, t_emb = self.encode(
            z, pos, edge_index, t, batch, cond=cond,
            x0_estimate=x0_estimate,
            edge_index_per_layer=edge_index_per_layer,
            return_t_emb=True,
        )
        noise_pred = updated_pos - initial_pos
        # Tier 2.4: refine the predicted clean coordinates with a lightweight
        # second EGNN pass before converting back to noise space. The refine
        # layers are zero-initialized (via EGNNLayer), so at init the
        # refinement is the identity — safe to bolt onto any training run.
        if self.coord_refine_layers > 0:
            r_pos = initial_pos + noise_pred
            ei_last = (
                edge_index_per_layer[-1]
                if edge_index_per_layer is not None else edge_index
            )
            for layer in self.refine_layers:
                h_r, r_pos = layer(h, r_pos, ei_last, t_emb)
            noise_pred = r_pos - initial_pos
        return noise_pred, self.type_head(h), h

    def predict_bond_logits(
        self, h: torch.Tensor, pos: torch.Tensor, pair_index: torch.Tensor
    ) -> torch.Tensor:
        """Bond-type logits for candidate atom pairs ``pair_index`` (2, P).

        Pair features: [h_i, h_j, squared distance] — rotation invariant.
        """
        row, col = pair_index
        dist = (pos[row] - pos[col]).square().sum(dim=-1, keepdim=True)
        pair_feats = torch.cat((h[row], h[col], dist), dim=-1)
        return self.bond_head(pair_feats)
