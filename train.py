"""Centralized Phase 1.5 training loop for the joint position + type diffusion.

Reads its configuration from ``configs/central.yaml`` (override via CLI arg).
Key changes vs Phase 1:
- Vectorized kNN graph construction (Rec A1 in .idea/02_recommendations.md).
- Joint loss  L = L_pos + λ_type · L_type  for coordinate + atom-type heads.
- Predict-zero baseline loss logged alongside MSE each epoch.
- Train / val / test 80:10:10 split with early-stopping on val loss.
- Periodic checkpointing of the best validation model.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import load_qm9
from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.egnn import EquivariantGenerator
from src.utils.graph import build_knn_graph
from torch_geometric.loader import DataLoader as PyGDataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic save: write to tmp then rename so a kill mid-write never corrupts last.pt
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_diffusion(
    config_path: str = "configs/central.yaml",
    resume: bool = False,
    run_epochs: int | None = None,
) -> None:
    # Load configuration ----------------------------------------------------
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    diff_cfg = cfg["diffusion"]
    train_cfg = cfg["training"]
    ckpt_cfg = cfg["checkpoint"]

    # Data ------------------------------------------------------------------
    full_dataset = load_qm9(root=data_cfg["root"])
    # Optional subsampling for fast CPU smoke runs (mirrors fed_train.py).
    # data.max_molecules: 0 or absent = use full QM9.
    max_mols = int(data_cfg.get("max_molecules", 0))
    if max_mols > 0 and max_mols < len(full_dataset):
        g_sub = torch.Generator().manual_seed(cfg.get("seed", 42))
        sub_idx = torch.randperm(len(full_dataset), generator=g_sub)[:max_mols].tolist()
        full_dataset = torch.utils.data.Subset(full_dataset, sub_idx)
    n = len(full_dataset)
    n_train = int(n * data_cfg["train_frac"])
    n_val = int(n * data_cfg["val_frac"])
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
    )

    train_loader = PyGDataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
    )
    val_loader = PyGDataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
    )
    test_loader = PyGDataLoader(
        test_ds, batch_size=train_cfg["batch_size"], shuffle=False,
    )

    print(
        f"Dataset split — train: {len(train_ds)}, val: {len(val_ds)}, "
        f"test: {len(test_ds)}"
    )

    # Models ----------------------------------------------------------------
    coord_ddpm = CenteredDDPM(
        num_steps=diff_cfg["num_steps"],
        beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"],
        device=device,
    )
    type_ddpm = TypeDDPM(
        num_steps=diff_cfg["num_steps"],
        beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"],
        device=device,
    )
    model = EquivariantGenerator(
        num_types=model_cfg["num_types"],
        node_dim=model_cfg["node_dim"],
        edge_dim=model_cfg["edge_dim"],
        num_layers=model_cfg["num_layers"],
        time_dim=model_cfg["time_dim"],
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    total_epochs = int(train_cfg["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    ckpt_dir = Path(ckpt_cfg["dir"])
    last_path = ckpt_dir / "last.pt"

    lambda_type = train_cfg.get("type_loss_weight", 0.5)

    # ---- Resume handling (explicit --resume only) ---------------------------
    start_epoch = 1
    best_val_loss = float("inf")
    patience = train_cfg.get("patience", 10)
    epochs_no_improve = 0
    if resume:
        if last_path.exists():
            print(f"Resuming from {last_path}")
            resume_state = torch.load(last_path, map_location=device, weights_only=False)
            # Config-mismatch guard: model arch must match
            for k in ("num_types", "node_dim", "edge_dim", "num_layers", "time_dim"):
                if resume_state.get("config", {}).get("model", {}).get(k) != model_cfg.get(k):
                    raise ValueError(
                        f"Resume config mismatch for model.{k}: "
                        f"checkpoint={resume_state.get('config', {}).get('model', {}).get(k)} "
                        f"vs current={model_cfg.get(k)}. Refusing to resume."
                    )
            model.load_state_dict(resume_state["model_state_dict"])
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            if "scheduler_state_dict" in resume_state:
                scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            best_val_loss = float(resume_state.get("best_val_loss", resume_state.get("val_loss", float("inf"))))
            epochs_no_improve = int(resume_state.get("epochs_no_improve", 0))
            start_epoch = int(resume_state.get("epoch", 0)) + 1
            # Restore RNG for exact data/noise sequence continuation
            try:
                random.setstate(resume_state["rng_random"])
                torch.set_rng_state(resume_state["rng_torch"].cpu())
            except Exception as e:
                print(f"  ! Could not restore RNG states ({e}); continuing anyway")
            print(f"  → epoch {start_epoch}, best_val={best_val_loss:.4f}, "
                  f"no_improve={epochs_no_improve}")
        else:
            print(f"  ! --resume given but {last_path} not found; starting fresh")
    elif last_path.exists():
        print(f"  ! {last_path} exists but --resume not given; starting fresh "
              f"(will overwrite last.pt/best.pt as training progresses)")

    # Cap this invocation to run_epochs (1-hour chunking). None = run to total_epochs.
    if run_epochs is not None and run_epochs > 0:
        end_epoch = min(total_epochs, start_epoch + run_epochs - 1)
    else:
        end_epoch = total_epochs
    if start_epoch > total_epochs:
        print(f"Training already complete (start_epoch={start_epoch} > total={total_epochs})")
        return

    # Training loop ---------------------------------------------------------
    print(f"Running epochs {start_epoch}..{end_epoch} (total {total_epochs})")
    for epoch in range(start_epoch, end_epoch + 1):
        # ---- Train ----
        model.train()
        total_pos_loss = 0.0
        total_type_loss = 0.0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad(set_to_none=True)

            t = torch.randint(
                0, coord_ddpm.num_steps,
                (batch_data.num_graphs,),
                device=device,
            )

            # Coordinate diffusion
            noisy_pos, actual_noise = coord_ddpm.add_noise(
                batch_data.pos, t, batch_data.batch,
            )

            # Atom-type diffusion (EDM-style categorical)
            noisy_types = type_ddpm.sample_noisy_types(
                batch_data.z, t, model_cfg["num_types"],
            )

            edge_index = build_knn_graph(noisy_pos, batch_data.batch, k=train_cfg["kNN"])

            noise_pred, type_logits, _ = model(
                noisy_types, noisy_pos, edge_index, t, batch_data.batch,
            )

            # Losses
            pos_loss = F.mse_loss(noise_pred, actual_noise)
            type_loss = F.cross_entropy(type_logits, batch_data.z.long())
            loss = pos_loss + lambda_type * type_loss

            loss.backward()
            optimizer.step()
            total_pos_loss += pos_loss.item()
            total_type_loss += type_loss.item()

        scheduler.step()
        avg_pos_loss = total_pos_loss / len(train_loader)
        avg_type_loss = total_type_loss / len(train_loader)

        # ---- Predict-zero baseline (MSE of 1.0 is expected for N(0,1) targets) ----
        baseline_mse = 1.0  # constant: E[||noise - 0||^2] = 1 for unit-variance noise

        # ---- Validate ----
        model.eval()
        val_pos_loss = 0.0
        val_type_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                t = torch.randint(
                    0, coord_ddpm.num_steps,
                    (batch_data.num_graphs,),
                    device=device,
                )
                noisy_pos, actual_noise = coord_ddpm.add_noise(
                    batch_data.pos, t, batch_data.batch,
                )
                noisy_types = type_ddpm.sample_noisy_types(
                    batch_data.z, t, model_cfg["num_types"],
                )
                edge_index = build_knn_graph(noisy_pos, batch_data.batch, k=train_cfg["kNN"])
                noise_pred, type_logits, _ = model(
                    noisy_types, noisy_pos, edge_index, t, batch_data.batch,
                )
                val_pos_loss += F.mse_loss(noise_pred, actual_noise).item()
                val_type_loss += F.cross_entropy(type_logits, batch_data.z.long()).item()

        avg_val_pos = val_pos_loss / max(len(val_loader), 1)
        avg_val_type = val_type_loss / max(len(val_loader), 1)
        avg_val_loss = avg_val_pos + lambda_type * avg_val_type

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:4d} | "
            f"train_pos={avg_pos_loss:.4f}  train_type={avg_type_loss:.4f}  "
            f"baseline={baseline_mse:.4f} | "
            f"val_pos={avg_val_pos:.4f}  val_type={avg_val_type:.4f}  val={avg_val_loss:.4f} | "
            f"lr={lr_now:.2e}"
        )

        # ---- Checkpointing (update best/counter FIRST, then save last.pt once) ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_loss": best_val_loss,
                    "epochs_no_improve": epochs_no_improve,
                    "rng_random": random.getstate(),
                    "rng_torch": torch.get_rng_state(),
                    "config": cfg,
                },
                ckpt_dir / "best.pt",
            )
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # last.pt every epoch = resume point (max 1 epoch lost on kill)
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "val_loss": avg_val_loss,
                "epochs_no_improve": epochs_no_improve,
                "rng_random": random.getstate(),
                "rng_torch": torch.get_rng_state(),
                "config": cfg,
            },
            ckpt_dir / "last.pt",
        )

        if epoch % ckpt_cfg.get("save_every", 5) == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_loss": avg_val_loss,
                    "epochs_no_improve": epochs_no_improve,
                    "rng_random": random.getstate(),
                    "rng_torch": torch.get_rng_state(),
                    "config": cfg,
                },
                ckpt_dir / f"epoch_{epoch}.pt",
            )

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    completed = (epoch >= total_epochs) or (epochs_no_improve >= patience)
    if not completed:
        print(f"\nChunk done at epoch {epoch}/{total_epochs}. "
              f"Resume with: python train.py --config {config_path} --resume --run_epochs N")
        return

    # ---- Final test evaluation ----
    print("\n--- Test evaluation ---")
    ckpt_path = ckpt_dir / "best.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
    model.eval()
    test_pos_loss = 0.0
    test_type_loss = 0.0
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            t = torch.randint(
                0, coord_ddpm.num_steps,
                (batch_data.num_graphs,),
                device=device,
            )
            noisy_pos, actual_noise = coord_ddpm.add_noise(
                batch_data.pos, t, batch_data.batch,
            )
            noisy_types = type_ddpm.sample_noisy_types(
                batch_data.z, t, model_cfg["num_types"],
            )
            edge_index = build_knn_graph(noisy_pos, batch_data.batch, k=train_cfg["kNN"])
            noise_pred, type_logits, _ = model(
                noisy_types, noisy_pos, edge_index, t, batch_data.batch,
            )
            test_pos_loss += F.mse_loss(noise_pred, actual_noise).item()
            test_type_loss += F.cross_entropy(type_logits, batch_data.z.long()).item()

    avg_test_pos = test_pos_loss / max(len(test_loader), 1)
    avg_test_type = test_type_loss / max(len(test_loader), 1)
    avg_test_loss = avg_test_pos + lambda_type * avg_test_type
    print(
        f"Test pos_loss={avg_test_pos:.4f}  test_type_loss={avg_test_type:.4f}  "
        f"test_loss={avg_test_loss:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Centralized Phase 1.5 training")
    parser.add_argument(
        "--config", type=str, default="configs/central.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint dir/last.pt (explicit flag required)",
    )
    parser.add_argument(
        "--run_epochs", type=int, default=None,
        help="Max epochs for this invocation (1-hour chunking, e.g. 5). None = to end.",
    )
    args = parser.parse_args()
    train_diffusion(args.config, resume=args.resume, run_epochs=args.run_epochs)
