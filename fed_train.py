"""Federated training entry point (Phases 2 & 3 of .idea/03_implementation_plan.md).

Runs the simulated GraphGANFed protocol over QM9:
  python fed_train.py --config configs/fed_iid.yaml
  python fed_train.py --config configs/fed_niid.yaml

The config controls:
- IID vs non-IID formula-based partitioning (Step 2.1)
- K clients, local epochs E, rounds (Step 2.3 grid: K x {iid, niid})
- FedProx μ (Step 3.1), personal heads (Step 3.2B),
  multi-objective λ₂/λ₃ weights (Step 3.3)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader

from src.dataset import load_qm9
from src.fed.partition import get_partition, partition_summary
from src.fed.server import FederatedServer
from src.fed.trainer import LocalTrainer, init_model_from_state
from src.models.diffusion import CenteredDDPM, TypeDDPM


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_client_loaders(dataset, partitions, batch_size: int) -> list:
    loaders = []
    for client_id in sorted(partitions):
        subset = torch.utils.data.Subset(dataset, partitions[client_id])
        loaders.append(DataLoader(
            subset, batch_size=batch_size, shuffle=True,
        ))
    return loaders


def run_federated(config_path: str = "configs/fed_iid.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Federated training on: {device}")

    # ---- Data & partitioning ---------------------------------------------
    dataset = load_qm9(root=cfg["data"]["root"])
    max_mols = int(cfg["data"].get("max_molecules", 0))
    indices = list(range(len(dataset)))
    if max_mols > 0:
        g = torch.Generator().manual_seed(cfg.get("seed", 42))
        indices = torch.randperm(len(dataset), generator=g)[:max_mols].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    partitions, labels = get_partition(
        dataset,
        num_clients=int(cfg["fed"]["num_clients"]),
        mode=cfg["fed"]["mode"],
        seed=cfg.get("seed", 42),
        cache_dir=cfg.get("partition_cache", "data/partitions"),
    )
    print(partition_summary(partitions, labels))

    # Shared global validation set (positional indices into `dataset`).
    n_total = len(indices)
    n_val = max(1, int(n_total * float(cfg["data"].get("val_frac", 0.1))))
    val_positions = list(range(n_total - n_val, n_total))
    val_positions_set = set(val_positions)
    train_indices_per_client = {
        c: [i for i in idxs if i not in val_positions_set]
        for c, idxs in partitions.items()
    }
    val_set = torch.utils.data.Subset(dataset, val_positions)
    val_loader = DataLoader(
        val_set, batch_size=cfg["training"]["batch_size"], shuffle=False,
    )
    client_loaders = build_client_loaders(
        dataset,
        train_indices_per_client,
        int(cfg["training"]["batch_size"]),
    )

    # ---- Models -----------------------------------------------------------
    diff_cfg, model_cfg = cfg["diffusion"], cfg["model"]
    coord_ddpm = CenteredDDPM(device=device, **{
        k: diff_cfg[k] for k in ("num_steps", "beta_start", "beta_end")
    })
    type_ddpm = TypeDDPM(device=device, **{
        k: diff_cfg[k] for k in ("num_steps", "beta_start", "beta_end")
    })

    trainers = [
        LocalTrainer(
            init_model_from_state(model_cfg, None, device),
            coord_ddpm, type_ddpm, cfg, device,
        )
        for _ in range(int(cfg["fed"]["num_clients"]))
    ]
    for trainer in trainers:
        trainer.set_parameters(trainers[0].get_parameters(exclude_personal=False))

    server = FederatedServer(
        trainers=trainers,
        train_loaders=client_loaders,
        val_loader=val_loader,
        coord_ddpm=coord_ddpm,
        type_ddpm=type_ddpm,
        config=cfg,
        output_dir=cfg.get("output_dir", "outputs/fed"),
    )

    print(
        f"K={cfg['fed']['num_clients']} mode={cfg['fed']['mode']} "
        f"E={cfg['fed']['local_epochs']} rounds={cfg['fed']['rounds']} "
        f"mu={cfg['fed'].get('proximal_mu', 0.0)} "
        f"personal_heads={cfg['fed'].get('personal_heads', False)}"
    )
    server.fit()

    # ---- Final per-client personalized evaluation --------------------------
    if server.personalized:
        print("\n--- Per-client validation loss (personalized heads) ---")
        for i, trainer in enumerate(trainers):
            metrics = trainer.evaluate(val_loader)
            print(f"client {i}: loss={metrics['loss']:.4f} "
                  f"pos={metrics['pos_loss']:.4f} type={metrics['type_loss']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated diffusion training")
    parser.add_argument("--config", type=str, default="configs/fed_iid.yaml")
    args = parser.parse_args()
    run_federated(args.config)
