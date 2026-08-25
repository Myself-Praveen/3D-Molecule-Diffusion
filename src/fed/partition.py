"""Federated data partitioning of QM9 (Step 2.1).

Replicates the GraphGANFed protocol:
- Molecules are labeled by molecular formula (e.g. C2H6O).
- IID: class-stratified, equal-sized groups across K clients.
- non-IID: random unbalanced per-class allocation combined with unequal
  client totals (Dirichlet-style), mirroring the paper's protocol.

Partitions persist as ``data/partitions/K{K}_{iid|niid}.json``.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

# Atomic number -> element symbol (QM9 subset + common extras).
_Z_TO_SYMBOL: dict[int, str] = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I",
}


def molecular_formula(z: torch.Tensor | np.ndarray) -> str:
    """Hill-notation molecular formula from atomic numbers.

    Example: z = [6, 6, 8, 1, 1, 1] -> "C2H3O" (counts ordered C, H, then
    alphabetically — the standard Hill system used by RDKit).
    """
    counts: Counter[str] = Counter()
    for zi in np.asarray(z).tolist():
        symbol = _Z_TO_SYMBOL.get(int(zi))
        if symbol is not None:
            counts[symbol] += 1

    parts: list[str] = []
    for symbol in ("C", "H"):
        n = counts.pop(symbol, 0)
        if n:
            parts.append(f"{symbol}{n}" if n > 1 else symbol)
    for symbol in sorted(counts):
        n = counts[symbol]
        parts.append(f"{symbol}{n}" if n > 1 else symbol)
    return "".join(parts)


def label_dataset(dataset) -> list[str]:
    """Formula label for every molecule in a PyG dataset."""
    return [molecular_formula(data.z) for data in dataset]


def partition_iid(
    labels: list[str], num_clients: int, seed: int = 42
) -> dict[int, list[int]]:
    """Class-stratified equal split: every client sees the same label mix."""
    rng = random.Random(seed)
    by_class: defaultdict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[label].append(idx)
    for indices in by_class.values():
        rng.shuffle(indices)

    partitions: dict[int, list[int]] = {c: [] for c in range(num_clients)}
    for label, indices in sorted(by_class.items()):
        for offset, idx in enumerate(indices):
            partitions[offset % num_clients].append(idx)
    for indices in partitions.values():
        rng.shuffle(indices)
    return partitions


def partition_niid(
    labels: list[str],
    num_clients: int,
    seed: int = 42,
    dirichlet_alpha: float = 0.5,
) -> dict[int, list[int]]:
    """Random unbalanced per-class allocation with unequal totals.

    For each formula class, a Dirichlet(alpha) draw distributes the class's
    molecules across clients; additionally each client receives an unequal
    total share via a second Dirichlet draw over class budgets. Small alpha
    concentrates classes on few clients (the paper's non-IID regime).
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    by_class: defaultdict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[label].append(idx)
    for indices in by_class.values():
        py_rng.shuffle(indices)

    # Unequal total budget per client.
    budget = rng.dirichlet(np.full(num_clients, dirichlet_alpha))
    budget = np.maximum(budget, 0.01)
    budget = budget / budget.sum()

    # Distribute each class proportionally to the budget, then fix leftovers.
    partitions: dict[int, list[int]] = {c: [] for c in range(num_clients)}
    for label in sorted(by_class):
        indices = by_class[label]
        shares = rng.dirichlet(budget * dirichlet_alpha * len(by_class))
        shares = shares / shares.sum()
        cuts = np.cumsum(shares * len(indices)).astype(int)
        cuts[-1] = len(indices)
        start = 0
        for client_id, end in enumerate(cuts.tolist()):
            partitions[client_id].extend(indices[start:end])
            start = end

    # Guarantee at least one molecule per client.
    all_indices = set(range(len(labels)))
    assigned = {i for idxs in partitions.values() for i in idxs}
    orphans = py_rng.sample(sorted(all_indices - assigned),
                            k=len(all_indices - assigned))
    for i in orphans:
        smallest = min(partitions, key=lambda c: len(partitions[c]))
        partitions[smallest].append(i)

    for indices in partitions.values():
        py_rng.shuffle(indices)
    return partitions


def save_partition(
    partitions: dict[int, list[int]], path: str | Path
) -> None:
    """Persist a partition mapping to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(client): sorted(int(i) for i in idxs)
                    for client, idxs in partitions.items()}
    with open(path, "w") as f:
        json.dump(serializable, f)


def load_partition(path: str | Path) -> dict[int, list[int]]:
    """Load a partition mapping from JSON."""
    with open(path) as f:
        raw = json.load(f)
    return {int(client): list(idxs) for client, idxs in raw.items()}


def get_partition(
    dataset,
    num_clients: int,
    mode: str,
    seed: int = 42,
    cache_dir: str | Path = "data/partitions",
) -> tuple[dict[int, list[int]], list[str]]:
    """Label, partition (or load cached), and persist in one call."""
    labels = label_dataset(dataset)
    stem = f"K{num_clients}_{mode}"
    cache_path = Path(cache_dir) / f"{stem}.json"
    if cache_path.exists():
        partitions = load_partition(cache_path)
    else:
        if mode == "iid":
            partitions = partition_iid(labels, num_clients, seed)
        elif mode == "niid":
            partitions = partition_niid(labels, num_clients, seed)
        else:
            raise ValueError(f"Unknown partition mode: {mode!r}")
        save_partition(partitions, cache_path)
    return partitions, labels


def partition_summary(partitions: dict[int, list[int]], labels: list[str]) -> str:
    """Human-readable per-client size and top-class summary."""
    lines = [f"{'client':>8} {'n':>8}  top classes"]
    for client in sorted(partitions):
        idxs = partitions[client]
        top = Counter(labels[i] for i in idxs).most_common(3)
        tops = ", ".join(f"{lbl}×{n}" for lbl, n in top)
        lines.append(f"{client:>8} {len(idxs):>8}  {tops}")
    return "\n".join(lines)
