"""Flow-arm triage diagnostics (progress.md §13.6 open question).

The V4-flow arm early-stopped with train_pos ≈ 2.11, which looked close to
the trivial zero-velocity predictor. train.py logs a hardcoded
``baseline=1.0`` (the predict-zero MSE for the *eps* objective) — that is
NOT the flow objective's trivial loss. This script computes the exact flow
trivial baseline and measures, per noise level ``u`` and per split, whether
the trained velocity field actually beats it.

Diagnostics per (checkpoint, split, u):
  - ``mse``      — E||v_hat - v||^2 (per atom-dim, the training objective)
  - ``trivial``  — E||v||^2 (the v_hat = 0 predictor; constant in u)
  - ``skill``    — 1 - mse/trivial  (>0 = model beats trivial; 0 = learned
                   nothing; <0 = predictions actively harmful)
  - ``cos``      — mean cosine similarity between v_hat and v per atom
  - ``mag``      — mean ||v_hat|| / mean ||v|| (1 = calibrated magnitude;
                   <<1 = collapsed toward the trivial predictor)
  - ``x0_mse`` / ``x0_trivial`` — one-step clean-coordinate recovery
                   ||x_u - u*v_hat - x0||^2 vs the v=0 draft ||x_u - x0||^2

Two self-conditioning variants are reported (training feeds a zero-velocity
draft ``x0_estimate = x_u`` on 50% of steps; sampling feeds the previous
step's x0 estimate): ``sc=none`` and ``sc=draft``.

Usage:
    .venv/bin/python scripts/flow_triage.py \
        --config configs/central_v4_flow.yaml \
        --checkpoints checkpoints/central_v4_flow/best.pt,checkpoints/central_v4_flow/last.pt \
        --out outputs/flow_triage.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
import torch
import yaml
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import load_qm9
from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.egnn import EquivariantGenerator
from src.models.flow import flow_interpolate, flow_time_embedding
from src.utils.graph import build_knn_graph


def build_model(ckpt_cfg: dict, device: torch.device) -> EquivariantGenerator:
    """Mirror generate_and_eval.py's checkpoint -> model construction."""
    m = ckpt_cfg["model"]
    model = EquivariantGenerator(
        num_types=m["num_types"],
        node_dim=m["node_dim"],
        edge_dim=m["edge_dim"],
        num_layers=m["num_layers"],
        time_dim=m["time_dim"],
        cond_dim=int(m.get("cond_dim", 0)),
        num_cond_classes=int(m.get("num_cond_classes", 2)),
        use_attention=m.get("use_attention", False),
        self_condition=m.get("self_condition", False),
        coord_refine_layers=int(m.get("coord_refine_layers", 0)),
        knn_schedule=m.get("knn_schedule") or None,
    ).to(device)
    model.objective = "flow"
    return model


def trivial_baseline(x0: torch.Tensor, batch: torch.Tensor) -> float:
    """Exact E||eps_c - x0||^2 per atom-dim for the v_hat = 0 predictor.

    eps_c is standard noise re-centered per molecule, so per atom-dim
    E[(eps_c - x0)^2] = (1 - 1/n_g) + x0_dim^2 (cross term vanishes);
    averaged over the 3 dims: (1 - 1/n_g) + ||x0||^2 / 3. This matches the
    training loss scale (F.mse_loss(...).mean(dim=-1) -> .mean()).
    """
    num_graphs = int(batch.max().item()) + 1
    counts = torch.bincount(batch, minlength=num_graphs).clamp_min(1).float()
    per_atom = 1.0 - 1.0 / counts[batch]
    return float((per_atom + x0.square().sum(dim=-1) / 3.0).mean().item())


def per_atom_cos(v_pred: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(v_pred, dim=-1)
    b = torch.nn.functional.normalize(v, dim=-1)
    return (a * b).sum(dim=-1)


def main() -> None:
    p = argparse.ArgumentParser(description="Flow-arm velocity-field triage")
    p.add_argument("--config", default="configs/central_v4_flow.yaml")
    p.add_argument("--checkpoints",
                   default="checkpoints/central_v4_flow/best.pt,"
                           "checkpoints/central_v4_flow/last.pt")
    p.add_argument("--n_mols", type=int, default=256,
                   help="molecules per split for the probe")
    p.add_argument("--u_points", default="0.05,0.25,0.5,0.75,0.95")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="outputs/flow_triage.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg, diff_cfg = cfg["data"], cfg["diffusion"]
    T = int(diff_cfg["num_steps"])

    # --- Splits: mirror train.py (random_split, seed 42) ---------------
    full_dataset = load_qm9(root=data_cfg["root"])
    n = len(full_dataset)
    n_train = int(n * data_cfg["train_frac"])
    n_val = int(n * data_cfg["val_frac"])
    n_test = n - n_train - n_val
    train_ds, val_ds, _ = torch.utils.data.random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(int(cfg.get("seed", 42))),
    )
    splits = {}
    for name, ds in (("train", train_ds), ("val", val_ds)):
        idx = list(range(min(args.n_mols, len(ds))))
        loader = PyGDataLoader(ds[idx], batch_size=len(idx), shuffle=False)
        splits[name] = next(iter(loader))
    print(f"Probe: {args.n_mols} molecules/split, T={T}, "
          f"u in [{args.u_points}]")

    coord_ddpm = CenteredDDPM(
        num_steps=T, beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"], device=device,
        schedule=diff_cfg.get("schedule", "linear"),
    )
    type_ddpm = TypeDDPM(
        num_steps=T, beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"], device=device,
        schedule=diff_cfg.get("schedule", "linear"),
    )

    u_points = [float(u) for u in args.u_points.split(",")]
    results: dict = {"u_points": u_points, "checkpoints": {}}

    # --- Trivial baseline (checkpoint-independent) ---------------------
    baseline = {}
    for name, batch in splits.items():
        baseline[name] = trivial_baseline(batch.pos, batch.batch)
    print("\nTrivial zero-velocity baseline  E||eps_c - x0||^2  "
          "(train_pos must beat THIS; logged baseline=1.0 is eps-only):")
    for name, v in baseline.items():
        print(f"  {name:>5}: {v:.4f}")
    results["trivial_baseline"] = baseline

    for ckpt_path in args.checkpoints.split(","):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ckpt_cfg = ckpt.get("config", cfg)
        model = build_model(ckpt_cfg, device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        ckpt_key = str(Path(ckpt_path))
        epoch = ckpt.get("epoch", "?")
        print(f"\n=== {ckpt_key} (epoch={epoch}) ===")
        entry = {"epoch": epoch, "splits": {}}

        for split_name, batch in splits.items():
            batch = batch.to(device)
            x0 = batch.pos
            b = batch.batch
            n_mols = int(b.max().item()) + 1
            k = int(cfg["training"]["kNN"])
            rows = []
            # One fixed noise draw shared across u for comparability.
            g = torch.Generator().manual_seed(args.seed)
            eps_shared = torch.randn(x0.shape, generator=g)
            for u in u_points:
                torch.manual_seed(args.seed + int(u * 1000))  # type-chain noise
                x_u, v = flow_interpolate(x0, u, b, eps=eps_shared.clone())
                t_idx = int(flow_time_embedding(u, T).item())
                t = torch.full((n_mols,), t_idx, dtype=torch.long, device=device)
                noisy_types = type_ddpm.sample_noisy_types(
                    batch.z.long(), t, model.num_types, batch=b)
                edge_index = build_knn_graph(x_u, b, k=k)
                per_layer = None
                if model.knn_schedule:
                    per_layer = [build_knn_graph(x_u, b, k=kk)
                                 for kk in model.knn_schedule]

                row = {"u": u}
                for sc in ("none", "draft"):
                    x0_est = x_u if sc == "draft" else None
                    with torch.no_grad():
                        v_pred, _, _ = model(
                            noisy_types, x_u, edge_index, t, b,
                            cond=None, x0_estimate=x0_est,
                            edge_index_per_layer=per_layer)
                    mse = float(((v_pred - v) ** 2).mean().item())
                    trivial = float((v ** 2).mean().item())
                    cos = float(per_atom_cos(v_pred, v).mean().item())
                    mag = float(v_pred.norm(dim=-1).mean().item()
                                / v.norm(dim=-1).mean().item())
                    x0_hat = x_u - u * v_pred
                    row[sc] = {
                        "mse": mse,
                        "trivial": trivial,
                        "skill": 1.0 - mse / trivial,
                        "cos": cos,
                        "mag": mag,
                        "x0_mse": float(((x0_hat - x0) ** 2).mean().item()),
                        "x0_trivial": float(((x_u - x0) ** 2).mean().item()),
                    }
                rows.append(row)
            entry["splits"][split_name] = rows

            # --- table --------------------------------------------------
            print(f"  [{split_name}] baseline(trivial)="
                  f"{baseline[split_name]:.4f}")
            print(f"  {'u':>5} | {'sc':>5} | {'mse':>7} | {'skill':>7} | "
                  f"{'cos':>6} | {'mag':>5} | {'x0_mse':>7} | "
                  f"{'x0_triv':>7}")
            for row in rows:
                for sc in ("none", "draft"):
                    r = row[sc]
                    print(f"  {row['u']:>5.2f} | {sc:>5} | {r['mse']:>7.4f} | "
                          f"{r['skill']:>+7.4f} | {r['cos']:>6.3f} | "
                          f"{r['mag']:>5.3f} | {r['x0_mse']:>7.4f} | "
                          f"{r['x0_trivial']:>7.4f}")
        results["checkpoints"][ckpt_key] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
