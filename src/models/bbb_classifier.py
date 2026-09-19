"""BBB permeability classifier oracle (Phase 3 of implementation.md).

A lightweight 3-layer GCN trained on BBBP to score generated molecules::

    atom_embedding(z) -> GCN x3 -> global_mean_pool -> MLP -> logit

Atom types reuse the Phase 2 10-type vocabulary (``Z_TO_INDEX``), so the
oracle and the diffusion model share one atom representation. Edges are
distance-gated 2D topology (cutoff 2.5 A), consistent with
``src/utils/evaluation.py`` reconstruction — the oracle scores 2D topology,
not 3D conformations, which is exactly what drug-discovery BBB oracles do.
"""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool

# Frozen atom vocabulary of the trained oracle (dense 0-9 indices).
# NOTE: this intentionally differs from the diffusion model's raw-Z index
# convention — the oracle was trained end-to-end with this mapping and is
# self-consistent (train and inference share it). Do not "fix" to identity
# without retraining models/bbb_oracle.pt.
_ORACLE_Z_TO_INDEX = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4,
                      15: 5, 16: 6, 17: 7, 35: 8, 53: 9}
Z_TO_INDEX = dict(_ORACLE_Z_TO_INDEX)  # kept for backward-compatible imports



class BBBClassifier(nn.Module):
    """3-layer GCN predicting BBB permeability (logit; sigmoid at inference)."""

    def __init__(
        self,
        num_types: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_types, hidden_dim)
        self.convs = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-graph BBB+ logits of shape (B,)."""
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        h = self.embed(z.long())
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.dropout(torch.relu(norm(conv(h, edge_index))))
        return self.mlp(global_mean_pool(h, batch)).squeeze(-1)

    # ------------------------------------------------------------- inference

    @staticmethod
    def data_to_input(data: Data) -> Data | None:
        """Phase 1-style 3D ``Data`` -> classifier input.

        Atom indices via ``Z_TO_INDEX`` (exotics -> C) and distance-gated
        topology from ``data.pos`` — byte-identical to the training pipeline,
        so generated molecules (which already carry 3D coords) are scored
        with zero distribution shift. Prefer this over ``mol_to_data``.
        """
        try:
            if data.pos.numel() == 0 or data.z.numel() == 0:
                return None
            z = data.z.long().clamp(0, 63)
            lut = torch.tensor(
                [Z_TO_INDEX.get(i, 1) for i in range(64)], dtype=torch.long,
            )
            return Data(
                z=lut[z],
                edge_index=bbb_topology_edges(data.pos),
            )
        except Exception:
            return None

    @staticmethod
    def mol_to_data(mol: Chem.Mol) -> Data | None:
        """Convert an RDKit Mol to classifier input.

        Mirrors training: explicit Hs + ETKDG conformer + distance-gated
        edges. Returns ``None`` when 3D embedding fails (caller scores 0.0).
        """
        try:
            from rdkit.Chem import AllChem

            mol_h = Chem.AddHs(Chem.RemoveHs(mol))
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol_h, params) != 0:
                return None
            conf = mol_h.GetConformer()
            pos = torch.tensor(
                [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                  conf.GetAtomPosition(i).z]
                 for i in range(mol_h.GetNumAtoms())],
                dtype=torch.float32,
            )
            z = torch.tensor(
                [a.GetAtomicNum() for a in mol_h.GetAtoms()], dtype=torch.long,
            )
            return BBBClassifier.data_to_input(
                Data(pos=pos, z=z)
            )
        except Exception:
            return None

    @torch.no_grad()
    def predict_mol(self, mol: Chem.Mol | None) -> float:
        """Score a single RDKit Mol; returns BBB+ probability in [0, 1]."""
        if mol is None:
            return 0.0
        self.eval()
        try:
            data = self.mol_to_data(mol)
            if data is None or data.z.numel() == 0:
                return 0.0
            device = next(self.parameters()).device
            logit = self(
                data.z.to(device), data.edge_index.to(device),
                torch.zeros(data.z.numel(), dtype=torch.long, device=device),
            )
            return float(torch.sigmoid(logit).item())
        except Exception:
            return 0.0

    @torch.no_grad()
    def predict_batch(self, mols: list[Chem.Mol | None]) -> list[float]:
        """Score a batch; ``None`` entries yield 0.0 (never crash)."""
        self.eval()
        try:
            items = [self.mol_to_data(m) if m is not None else None for m in mols]
            valid_idx = [i for i, d in enumerate(items) if d is not None]
            out = [0.0] * len(mols)
            if not valid_idx:
                return out
            device = next(self.parameters()).device
            batch = Batch.from_data_list([items[i] for i in valid_idx]).to(device)
            logits = self(batch.z, batch.edge_index, batch.batch)
            probs = torch.sigmoid(logits).tolist()
            for i, p in zip(valid_idx, probs):
                out[i] = float(p)
            return out
        except Exception:
            return [0.0] * len(mols)


def bbb_topology_edges(
    pos: np.ndarray | torch.Tensor,
    cutoff: float = 2.5,
) -> torch.Tensor:
    """Distance-gated edge index (2, E) for classifier training on 3D Data.

    Phase 1 BBB ``Data`` objects carry 3D coords but no bond table; the
    oracle trains on distance-derived topology, matching how it will score
    generated molecules later.
    """
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    n = pos.shape[0]
    if n == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    rows, cols = np.where((d < cutoff) & (d > 1e-6))
    if len(rows) == 0:
        rows = cols = np.arange(n)
    return torch.tensor([rows, cols], dtype=torch.long)
