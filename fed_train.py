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
import json
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


def run_federated(
    config_path: str = "configs/fed_iid.yaml",
    resume: bool = False,
    run_rounds: int | None = None,
) -> None:
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

    # ---- Resume handling (explicit --resume only) ---------------------------
    output_dir = Path(cfg.get("output_dir", "outputs/fed"))
    last_path = output_dir / "last_global.pt"
    hist_path = output_dir / "history.json"
    start_round = 1
    if resume:
        if last_path.exists() and hist_path.exists():
            print(f"Resuming from {last_path}")
            rstate = torch.load(last_path, map_location="cpu", weights_only=False)
            # Config-mismatch guard: arch + partition must match
            for k in ("num_types", "node_dim", "edge_dim", "num_layers", "time_dim"):
                if rstate.get("config", {}).get("model", {}).get(k) != cfg["model"].get(k):
                    raise ValueError(
                        f"Resume config mismatch for model.{k}: "
                        f"checkpoint={rstate.get('config', {}).get('model', {}).get(k)} "
                        f"vs current={cfg['model'].get(k)}. Refusing to resume."
                    )
            if rstate.get("config", {}).get("fed", {}).get("num_clients") != cfg["fed"].get("num_clients"):
                raise ValueError("Resume config mismatch for fed.num_clients. Refusing to resume.")
            for trainer in trainers:
                trainer.set_parameters(
                    {k: v.to(trainer.device) for k, v in rstate["global_state"].items()}
                )
            # Restore personal heads if present
            if rstate.get("personal_states") is not None:
                for trainer, pstate in zip(trainers, rstate["personal_states"]):
                    if pstate:
                        cur = trainer.model.state_dict()
                        cur.update({k: v.to(cur[k].device) for k, v in pstate.items()})
                        trainer.model.load_state_dict(cur)
            with open(hist_path) as f:
                server.history = json.load(f)
            start_round = int(rstate.get("round", len(server.history))) + 1
            print(f"  → round {start_round}, history_len={len(server.history)}")
        else:
            print(f"  ! --resume given but {last_path} or {hist_path} missing; starting fresh")
    elif last_path.exists() or hist_path.exists():
        print(f"  ! {output_dir} has prior state but --resume not given; starting fresh "
              f"(history.json will be overwritten)")

    print(
        f"K={cfg['fed']['num_clients']} mode={cfg['fed']['mode']} "
        f"E={cfg['fed']['local_epochs']} rounds={cfg['fed']['rounds']} "
        f"mu={cfg['fed'].get('proximal_mu', 0.0)} "
        f"personal_heads={cfg['fed'].get('personal_heads', False)}"
    )
    server.fit(start_round=start_round, run_rounds=run_rounds)

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
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from output_dir/last_global.pt + history.json (explicit flag required)",
    )
    parser.add_argument(
        "--run_rounds", type=int, default=None,
        help="Max rounds for this invocation (1-hour chunking, e.g. 5). None = to end.",
    )
    args = parser.parse_args()
    run_federated(args.config, resume=args.resume, run_rounds=args.run_rounds)
