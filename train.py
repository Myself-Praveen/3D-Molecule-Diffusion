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
from src.fed.trainer import TYPE_TO_Z
from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.egnn import EquivariantGenerator
from src.models.flow import flow_interpolate
from src.objectives import diversity_regularizer, full_pair_index, x0_valence_penalty
from src.training_utils import EMA, build_warmup_cosine_scheduler, min_snr_weight, rotate_batch
from src.utils.graph import build_knn_graph
from torch_geometric.loader import DataLoader as PyGDataLoader


def _bond_pair_labels(
    pair_index: torch.Tensor,
    true_edge_index: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """Binary bond labels (1 = bonded, 0 = none) for candidate pairs.

    Tier 2.2 supervision: QM9 stores connectivity in ``data.edge_index``
    (undirected duplicates included). A candidate pair (i, j) is positive
    iff that edge appears in the ground truth. Distances at ``noisy_pos``
    decide nothing here — labels are graph-structural.
    """
    true = set()
    ei = true_edge_index.tolist()
    for a, b in zip(ei[0], ei[1]):
        true.add((a, b))
        true.add((b, a))
    rows = pair_index[0].tolist()
    cols = pair_index[1].tolist()
    labels = torch.tensor(
        [1 if (a, b) in true else 0 for a, b in zip(rows, cols)],
        dtype=torch.long, device=pair_index.device,
    )
    return labels


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


@torch.no_grad()
def _quick_validity(
    model, coord_ddpm, type_ddpm, device,
    num_samples: int = 16, ddim_steps: int = 20,
) -> float:
    """Sample a few molecules and return validity % (geometry health probe)."""
    from src.sampling import sample_molecules
    from src.utils.evaluation import coords_and_types_to_mol, validity

    model.eval()
    counts = torch.full((num_samples,), 18, dtype=torch.long)
    try:
        pos, z = sample_molecules(
            model, coord_ddpm, type_ddpm, counts,
            device=device, ddim_steps=ddim_steps,
        )
    except Exception:
        return 0.0
    mols = []
    off = 0
    for _ in range(num_samples):
        mols.append(coords_and_types_to_mol(
            pos[off:off + 18].numpy(), z[off:off + 18].numpy().astype(int),
        ))
        off += 18
    return float(validity(mols)) * 100.0


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
        schedule=diff_cfg.get("schedule", "linear"),
    )
    type_ddpm = TypeDDPM(
        num_steps=diff_cfg["num_steps"],
        beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"],
        device=device,
        schedule=diff_cfg.get("schedule", "linear"),
    )
    model = EquivariantGenerator(
        num_types=model_cfg["num_types"],
        node_dim=model_cfg["node_dim"],
        edge_dim=model_cfg["edge_dim"],
        num_layers=model_cfg["num_layers"],
        time_dim=model_cfg["time_dim"],
        use_attention=model_cfg.get("use_attention", False),
        # Tier 2 (docs/recommendation.md) opt-ins — default off everywhere.
        self_condition=model_cfg.get("self_condition", False),
        coord_refine_layers=int(model_cfg.get("coord_refine_layers", 0)),
        knn_schedule=model_cfg.get("knn_schedule") or None,
    ).to(device)
    # Tier 3.2: prediction-target semantics ("eps" DDPM noise [default] or
    # "flow" velocity on the linear OT path). Same architecture either way.
    objective = str(diff_cfg.get("objective", "eps")).lower()
    if objective not in ("eps", "flow"):
        raise ValueError(
            f"Unknown diffusion.objective={objective!r} (expected 'eps' or 'flow')"
        )
    model.objective = objective

    optimizer = Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    total_epochs = int(train_cfg["epochs"])
    # Strategy 1.4: warmup+cosine when warmup_epochs > 0, else plain cosine.
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_epochs,
        warmup_epochs=int(train_cfg.get("warmup_epochs", 0)),
    )
    if scheduler is None:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    # Tier 1 opt-ins (docs/recommendation.md)
    ema_decay = float(train_cfg.get("ema_decay", 0.0))
    ema = EMA(model, decay=ema_decay, warmup=int(train_cfg.get("ema_warmup_steps", 1000))) \
        if ema_decay > 0.0 else None
    snr_gamma = float(train_cfg.get("min_snr_gamma", 0.0))
    if snr_gamma > 0.0 and objective == "flow":
        print("  ! min_snr_gamma ignored with diffusion.objective=flow "
              "(SNR weighting is defined over the DDPM schedule only)")
    aug_rot = float(train_cfg.get("rotation_augment_prob", 0.0))
    # Tier 2 opt-ins
    sc_dropout = float(train_cfg.get("self_cond_dropout", 0.5))  # Chen et al. 50%
    use_multiscale = model.knn_schedule is not None

    ckpt_dir = Path(ckpt_cfg["dir"])
    last_path = ckpt_dir / "last.pt"

    lambda_type = train_cfg.get("type_loss_weight", 0.5)
    lambda_valence = float(train_cfg.get("valence_loss_weight", 0.0))
    lambda_diversity = float(train_cfg.get("diversity_loss_weight", 0.0))
    # Tier 2.2: bond-head supervision weight (0 = off; heads then stay at
    # init and eval keeps inferring bonds from distances as before).
    lambda_bond = float(train_cfg.get("bond_loss_weight", 0.0))
    # 0 = off (legacy behavior). >0 clips global grad norm each step — needed for
    # escalated-capacity models where a rare pathological batch can explode grads
    # and NaN-poison the weights (observed at 128-dim/6-layer, absent at 64/4).
    grad_clip = float(train_cfg.get("grad_clip_norm", 0.0))

    # ---- Resume handling (explicit --resume only) ---------------------------
    start_epoch = 1
    best_val_loss = float("inf")
    patience = train_cfg.get("patience", 10)
    epochs_no_improve = 0
    if resume:
        if last_path.exists():
            print(f"Resuming from {last_path}")
            resume_state = torch.load(last_path, map_location=device, weights_only=False)
            # Config-mismatch guard: model arch must match (Tier 2 flags
            # change the parameter set, so they are guarded too).
            for k in ("num_types", "node_dim", "edge_dim", "num_layers", "time_dim",
                      "self_condition", "coord_refine_layers", "knn_schedule"):
                if resume_state.get("config", {}).get("model", {}).get(k) != model_cfg.get(k):
                    raise ValueError(
                        f"Resume config mismatch for model.{k}: "
                        f"checkpoint={resume_state.get('config', {}).get('model', {}).get(k)} "
                        f"vs current={model_cfg.get(k)}. Refusing to resume."
                    )
            # Noise schedule must also match: alpha_bars are baked into every
            # trained timestep, so a linear↔cosine switch invalidates weights.
            resume_sched = resume_state.get("config", {}).get("diffusion", {}).get("schedule", "linear")
            if resume_sched != diff_cfg.get("schedule", "linear"):
                raise ValueError(
                    f"Resume config mismatch for diffusion.schedule: "
                    f"checkpoint={resume_sched} vs current={diff_cfg.get('schedule', 'linear')}. "
                    f"Refusing to resume."
                )
            # Tier 3.2: the prediction target is baked into the weights just
            # like the schedule — an eps↔flow switch invalidates training.
            resume_obj = resume_state.get("config", {}).get("diffusion", {}).get("objective", "eps")
            if resume_obj != objective:
                raise ValueError(
                    f"Resume config mismatch for diffusion.objective: "
                    f"checkpoint={resume_obj} vs current={objective}. "
                    f"Refusing to resume."
                )
            model.load_state_dict(resume_state["model_state_dict"])
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            if "scheduler_state_dict" in resume_state:
                scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            if ema is not None and "ema_state_dict" in resume_state:
                ema.load_state_dict(resume_state["ema_state_dict"])
            elif ema is not None:
                ema = EMA(model, decay=ema_decay,
                          warmup=int(train_cfg.get("ema_warmup_steps", 1000)))
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

            # Strategy 1.5: random per-molecule SO(3) rotation augmentation.
            if aug_rot > 0.0 and torch.rand(1).item() < aug_rot:
                batch_data.pos = rotate_batch(batch_data.pos, batch_data.batch)

            # Coordinate corruption: DDPM noise (default) or Tier 3.2
            # flow-matching linear path per molecule. Flow ties u to the same
            # t that drives the time embedding and type chain, so conditioning
            # always matches the geometry corruption level (and the sampler's
            # u->t grid).
            if objective == "flow":
                u = t.float() / max(coord_ddpm.num_steps - 1, 1)
                noisy_pos, v_target = flow_interpolate(
                    batch_data.pos, u, batch_data.batch,
                )
            else:
                noisy_pos, actual_noise = coord_ddpm.add_noise(
                    batch_data.pos, t, batch_data.batch,
                )

            # Atom-type diffusion (EDM-style categorical). batch= is required:
            # without it _atom_batch broadcasts one timestep per ATOM, so the
            # type head would train on (noise-level, conditioned-t) mismatches.
            noisy_types = type_ddpm.sample_noisy_types(
                batch_data.z, t, model_cfg["num_types"],
                batch=batch_data.batch,
            )

            edge_index = build_knn_graph(noisy_pos, batch_data.batch, k=train_cfg["kNN"])

            # Tier 2.3: one kNN graph per layer from the kNN schedule.
            edge_index_per_layer = None
            if use_multiscale:
                edge_index_per_layer = [
                    build_knn_graph(noisy_pos, batch_data.batch, k=k_layer)
                    for k_layer in model.knn_schedule
                ]

            # Tier 2.1: self-conditioning — 50% of steps feed the previous
            # x0 estimate (here approximated with the current noise level:
            # the first prediction of a step sees the *noisy* geometry),
            # the rest see None, exactly like the Analog Bits schedule.
            x0_estimate = None
            if model.self_condition and torch.rand(1).item() >= sc_dropout:
                if objective == "flow":
                    # Zero-velocity draft x0 = x_u - u*0 = x_u: the exact
                    # analog of the eps-objective draft below.
                    x0_estimate = noisy_pos
                else:
                    ab = coord_ddpm.alpha_bars.to(noisy_pos.device)[t][batch_data.batch]
                    x0_estimate = (
                        noisy_pos
                        - (1.0 - ab).sqrt().unsqueeze(-1) * torch.zeros_like(noisy_pos)
                    ) / ab.sqrt().unsqueeze(-1).clamp_min(1e-3)

            noise_pred, type_logits, node_h = model(
                noisy_types, noisy_pos, edge_index, t, batch_data.batch,
                x0_estimate=x0_estimate,
                edge_index_per_layer=edge_index_per_layer,
            )

            # Losses. Coordinate target: DDPM noise (eps) or flow velocity
            # v = eps - x0 (Tier 3.2). Strategy 1.2: optional min-SNR-γ
            # weighting (eps objective only — it is defined over the SNR of
            # the DDPM schedule).
            if objective == "flow":
                pos_loss_raw = F.mse_loss(noise_pred, v_target, reduction="none").mean(dim=-1)
                pos_loss = pos_loss_raw.mean()
            else:
                pos_loss_raw = F.mse_loss(noise_pred, actual_noise, reduction="none").mean(dim=-1)
                if snr_gamma > 0.0:
                    w = min_snr_weight(t, coord_ddpm.alpha_bars,
                                       batch=batch_data.batch, gamma=snr_gamma)
                    pos_loss = (w * pos_loss_raw).mean()
                else:
                    pos_loss = pos_loss_raw.mean()
            type_loss = F.cross_entropy(type_logits, batch_data.z.long())
            loss = pos_loss + lambda_type * type_loss

            # Tier 2.2: bond-head supervision against QM9 ground-truth bonds
            # (data.edge_index). Low-noise gated (t < valence_tau): at high
            # noise the head would only learn to predict "no bond", since
            # x0 is meaningless there. node_h is detached — the bond head is
            # an auxiliary classifier; back-propagating it into the denoiser
            # trunk would fight the diffusion objective.
            if lambda_bond > 0.0 and bool((t < int(train_cfg.get("valence_tau", 200))).any()):
                pair_index = full_pair_index(batch_data.batch)
                if pair_index.numel() > 0:
                    bond_logits = model.predict_bond_logits(
                        node_h.detach(), noisy_pos, pair_index)
                    bond_targets = _bond_pair_labels(
                        pair_index, batch_data.edge_index, batch_data.batch)
                    # Per-PAIR gate: a pair is in the loss iff BOTH its atoms
                    # belong to a low-noise (t < tau) molecule.
                    gate_pair = (
                        (t < int(train_cfg.get("valence_tau", 200)))[batch_data.batch]
                    )[pair_index[0]] & (
                        (t < int(train_cfg.get("valence_tau", 200)))[batch_data.batch]
                    )[pair_index[1]]
                    if bool(gate_pair.any()):
                        bond_loss = F.cross_entropy(
                            bond_logits[gate_pair], bond_targets[gate_pair])
                        loss = loss + lambda_bond * bond_loss

            # λ₂ validity pressure evaluated on the denoised x0 prediction
            # (low-noise gated) so geometry — not types — absorbs it.
            # See x0_valence_penalty: penalizing noisy_pos only corrupts the
            # type head (observed CE 0.76→1.22) since x_t is θ-independent.
            if lambda_valence > 0.0:
                loss = loss + x0_valence_penalty(
                    noise_pred, noisy_pos, type_logits, t, batch_data.batch,
                    batch_data.z,
                    num_types=model_cfg["num_types"],
                    type_to_z=TYPE_TO_Z,
                    alpha_bars=coord_ddpm.alpha_bars,
                    lambda_weight=lambda_valence,
                    tau=int(train_cfg.get("valence_tau", 200)),
                    objective=objective,
                )
            if lambda_diversity > 0.0:
                loss = loss + lambda_diversity * diversity_regularizer(
                    node_h, batch_data.batch,
                )

            loss.backward()

            # NaN/Inf guard: skip a batch whose gradients exploded instead of
            # letting one pathological batch poison the weights for the rest
            # of the run (all losses after such a step print NaN).
            bad_grad_params = [
                name for name, p in model.named_parameters()
                if p.grad is not None and not torch.isfinite(p.grad).all()
            ]
            if bad_grad_params:
                shown = ", ".join(bad_grad_params[:5])
                more = "…" if len(bad_grad_params) > 5 else ""
                print(f"  ! non-finite grads in [{shown}{more}] — batch skipped")
                optimizer.zero_grad(set_to_none=True)
                continue
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
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
                if objective == "flow":
                    u = t.float() / max(coord_ddpm.num_steps - 1, 1)
                    noisy_pos, v_target = flow_interpolate(
                        batch_data.pos, u, batch_data.batch)
                    noisy_types = type_ddpm.sample_noisy_types(
                        batch_data.z, t, model_cfg["num_types"],
                        batch=batch_data.batch,
                    )
                    edge_index = build_knn_graph(
                        noisy_pos, batch_data.batch, k=train_cfg["kNN"])
                    v_pred, type_logits, _ = model(
                        noisy_types, noisy_pos, edge_index, t, batch_data.batch,
                    )
                    val_pos_loss += F.mse_loss(v_pred, v_target).item()
                else:
                    noisy_pos, actual_noise = coord_ddpm.add_noise(
                        batch_data.pos, t, batch_data.batch,
                    )
                    noisy_types = type_ddpm.sample_noisy_types(
                        batch_data.z, t, model_cfg["num_types"],
                        batch=batch_data.batch,
                    )
                    edge_index = build_knn_graph(
                        noisy_pos, batch_data.batch, k=train_cfg["kNN"])
                    noise_pred, type_logits, _ = model(
                        noisy_types, noisy_pos, edge_index, t, batch_data.batch,
                    )
                    val_pos_loss += F.mse_loss(noise_pred, actual_noise).item()
                val_type_loss += F.cross_entropy(type_logits, batch_data.z.long()).item()

        avg_val_pos = val_pos_loss / max(len(val_loader), 1)
        avg_val_type = val_type_loss / max(len(val_loader), 1)
        avg_val_loss = avg_val_pos + lambda_type * avg_val_type

        # ---- Periodic quick validity probe (tracks geometry quality live) ----
        gen_cfg = cfg.get("generation", {})
        gen_every = int(gen_cfg.get("eval_every", 0))
        quick_valid = None
        if gen_every > 0 and (epoch % gen_every == 0 or epoch == end_epoch):
            quick_valid = _quick_validity(
                model, coord_ddpm, type_ddpm, device,
                num_samples=int(gen_cfg.get("num_samples", 16)),
                ddim_steps=int(gen_cfg.get("ddim_steps", 20)),
            )

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:4d} | "
            f"train_pos={avg_pos_loss:.4f}  train_type={avg_type_loss:.4f}  "
            f"baseline={baseline_mse:.4f} | "
            f"val_pos={avg_val_pos:.4f}  val_type={avg_val_type:.4f}  val={avg_val_loss:.4f} | "
            f"lr={lr_now:.2e}"
            + (f" | quick_valid={quick_valid:.1f}%" if quick_valid is not None else "")
        )

        # ---- Checkpointing (update best/counter FIRST, then save last.pt once) ----
        # Best-model tracking: with EMA enabled, validate/score the EMA weights
        # (they are what generation will load); otherwise the raw model.
        if ema is not None:
            eval_model_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "ema_state_dict": ema.state_dict() if ema is not None else None,
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

        if ema is not None:
            model.load_state_dict(eval_model_backup)

        # last.pt every epoch = resume point (max 1 epoch lost on kill)
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_state_dict": ema.state_dict() if ema is not None else None,
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
                    "ema_state_dict": ema.state_dict() if ema is not None else None,
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
            if objective == "flow":
                u = t.float() / max(coord_ddpm.num_steps - 1, 1)
                noisy_pos, v_target = flow_interpolate(
                    batch_data.pos, u, batch_data.batch)
                noisy_types = type_ddpm.sample_noisy_types(
                    batch_data.z, t, model_cfg["num_types"],
                    batch=batch_data.batch,
                )
                edge_index = build_knn_graph(
                    noisy_pos, batch_data.batch, k=train_cfg["kNN"])
                v_pred, type_logits, _ = model(
                    noisy_types, noisy_pos, edge_index, t, batch_data.batch,
                )
                test_pos_loss += F.mse_loss(v_pred, v_target).item()
            else:
                noisy_pos, actual_noise = coord_ddpm.add_noise(
                    batch_data.pos, t, batch_data.batch,
                )
                noisy_types = type_ddpm.sample_noisy_types(
                    batch_data.z, t, model_cfg["num_types"],
                    batch=batch_data.batch,
                )
                edge_index = build_knn_graph(
                    noisy_pos, batch_data.batch, k=train_cfg["kNN"])
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
