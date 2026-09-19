"""Train the BBB permeability oracle (Phase 3 of implementation.md).

Usage:
    .venv/bin/python scripts/train_bbb_classifier.py --data bbbp --epochs 100
    .venv/bin/python scripts/train_bbb_classifier.py --data b3db --epochs 100 \\
        --output models/bbb_oracle_b3db.pt

Loads the Phase 1 cached 3D data (no SMILES reconversion), trains the
3-layer GCN with binary cross-entropy, early-stops on validation AUROC,
and saves the best checkpoint. Target: AUROC >= 0.88 on the scaffold-split
test set (literature ~0.92).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from src.dataset_bbb import get_dataset_info, load_b3db, load_bbbp
from src.fed.trainer import Z_TO_INDEX
from src.models.bbb_classifier import BBBClassifier, bbb_topology_edges

_LOADERS = {"bbbp": (load_bbbp, "data/bbbp"), "b3db": (load_b3db, "data/b3db")}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_clf_data(d: Data) -> Data:
    """Phase 1 3D Data -> classifier Data (type indices + topology edges)."""
    z = d.z.long().clamp(0, 63)
    lut = torch.tensor(
        [Z_TO_INDEX.get(i, 1) for i in range(64)], dtype=torch.long,
    )
    return Data(
        z=lut[z],
        edge_index=bbb_topology_edges(d.pos),
        y=d.y.long().view(-1),
    )


def _prepare(root: str, name: str):
    load_fn, default_root = _LOADERS[name]
    train, val, test = load_fn(root=root or default_root)
    info = get_dataset_info(name)
    to_loader = lambda ds, shuffle: PyGDataLoader(
        [_to_clf_data(d) for d in ds], batch_size=64, shuffle=shuffle,
    )
    return to_loader(train, True), to_loader(val, False), to_loader(test, False), info


def _probs_and_labels(model, loader, device):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            probs.extend(torch.sigmoid(model(b.z, b.edge_index, b.batch)).tolist())
            labels.extend(b.y.view(-1).tolist())
    return np.array(probs), np.array(labels)


def _report(probs, labels, prefix=""):
    pred = (probs > 0.5).astype(int)
    msg = (f"{prefix}AUROC={roc_auc_score(labels, probs):.4f} "
           f"acc={accuracy_score(labels, pred):.4f} "
           f"prec={precision_score(labels, pred, zero_division=0):.4f} "
           f"rec={recall_score(labels, pred, zero_division=0):.4f} "
           f"F1={f1_score(labels, pred, zero_division=0):.4f}")
    print(msg)
    return msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BBB oracle classifier")
    parser.add_argument("--data", choices=["bbbp", "b3db"], default="bbbp")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="models/bbb_oracle.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training BBB oracle on {args.data} ({device})")

    train_loader, val_loader, test_loader, info = _prepare(args.root, args.data)
    n_pos = sum(b.y.view(-1).sum().item() for b in train_loader)
    n_tot = sum(b.y.numel() for b in train_loader)
    pos_weight = torch.tensor([(n_tot - n_pos) / max(n_pos, 1)], device=device)
    print(f"train molecules: {n_tot} ({100 * n_pos / n_tot:.1f}% BBB+)")

    model = BBBClassifier(hidden_dim=args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_auroc, best_state, stale = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for b in train_loader:
            b = b.to(device)
            optimizer.zero_grad()
            logits = model(b.z, b.edge_index, b.batch)
            loss = F.binary_cross_entropy_with_logits(
                logits, b.y.float().view(-1), pos_weight=pos_weight,
            )
            loss.backward()
            optimizer.step()
            total += loss.item()
        probs, labels = _probs_and_labels(model, val_loader, device)
        auroc = roc_auc_score(labels, probs)
        print(f"Epoch {epoch:3d} train_loss={total / len(train_loader):.4f} "
              f"val_AUROC={auroc:.4f}")
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    print(f"\nBest val AUROC: {best_auroc:.4f}")
    test_probs, test_labels = _probs_and_labels(model, test_loader, device)
    test_msg = _report(test_probs, test_labels, prefix="TEST ")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "hidden_dim": args.hidden,
        "dataset": args.data,
        "val_auroc": best_auroc,
        "test_report": test_msg,
        "property_stats": info.get("property_stats", {}),
    }, out)
    print(f"Saved oracle -> {out}")


if __name__ == "__main__":
    main()
