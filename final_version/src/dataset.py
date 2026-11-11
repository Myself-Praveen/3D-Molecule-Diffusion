"""QM9 loading and lightweight preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from torch_geometric.datasets import QM9


class QM9Preprocessor(BaseTransform):
    """Normalize the fields used by 3D graph models without changing geometry."""

    def forward(self, data: Data) -> Data:
        data.pos = data.pos.to(dtype=torch.float32)
        data.z = data.z.to(dtype=torch.long)
        return data


def load_qm9(root: str | Path = "data", transform: Any | None = None) -> QM9:
    """Download (when needed) and return the QM9 dataset."""
    selected_transform = transform if transform is not None else QM9Preprocessor()
    return QM9(root=str(Path(root)), transform=selected_transform)


def create_qm9_dataloader(
    root: str | Path = "data",
    batch_size: int = 32,
    shuffle: bool = True,
    **loader_kwargs: Any,
) -> DataLoader:
    """Create a PyG DataLoader for QM9 graphs."""
    return DataLoader(
        load_qm9(root), batch_size=batch_size, shuffle=shuffle, **loader_kwargs
    )


def load_qm9_3d(
    root: str | Path = "data", batch_size: int = 32, shuffle: bool = True, **loader_kwargs: Any
) -> DataLoader:
    """Return the QM9 3D graphs used by the centralized diffusion trainer."""
    return create_qm9_dataloader(
        root=root, batch_size=batch_size, shuffle=shuffle, **loader_kwargs
    )
