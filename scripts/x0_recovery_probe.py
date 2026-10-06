"""One-step x0 recovery probe: is the denoiser better than doing nothing?

For each timestep ``t`` a clean batch is noised once, the model predicts eps,
and x0 is recovered in closed form::

    x0_hat = (x_t - sqrt(1 - abar_t) * eps_hat) / sqrt(abar_t)

That estimate is compared against the trivial *eps = 0* predictor::

    x0_trivial = x_t / sqrt(abar_t)

A trained eps model must beat the trivial predictor at every t (at high t the
trivial predictor is wildly wrong, since x_t is almost pure noise). Losing to it
at low t means the model adds noise instead of removing it — the signature of a
model that is undertrained rather than misconfigured.

Reported per t: x0 MSE for the model and for the trivial predictor, plus the
median nearest-neighbour distance of x0_hat (geometry tightness; training data
sits near ~1.1 A for these molecules).

Usage:
    .venv/bin/python scripts/x0_recovery_probe.py
    .venv/bin/python scripts/x0_recovery_probe.py --n-mols 32 \
        --timesteps 20,100,500

The probe passes ``cond=None`` (the unconditional path) for both models so the
geometry comparison is apples-to-apples; conditioned training includes an
unconditional dropout batch, so this stays in-distribution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.diffusion import CenteredDDPM, TypeDDPM  # noqa: E402
from src.models.egnn import EquivariantGenerator  # noqa: E402
from src.utils.graph import build_knn_graph  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# (label, config, checkpoint, dataset)
DEFAULT_PROBES = [
    ("QM9 V4-eps (control)", "configs/central_v4_eps.yaml",
     "checkpoints/central_v4_eps/best.pt", "qm9"),
    ("BBBP central", "configs/central_bbb.yaml",
     "checkpoints/bbb/best.pt", "bbbp"),
]


def _load_batch(dataset: str, cfg: dict, n_mols: int, seed: int = 0):
    """A single fixed batch of ``n_mols`` real molecules from the train split."""
    from torch_geometric.loader import DataLoader

    if dataset == "bbbp":
        from src.dataset_bbb import load_bbbp

        full, _, _ = load_bbbp(root=cfg["data"]["root"])
    elif dataset == "b3db":
        from src.dataset_bbb import load_b3db

        full, _, _ = load_b3db(root=cfg["data"]["root"])
    else:
        from src.dataset import load_qm9

        full = load_qm9(root=cfg["data"]["root"])

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(full), generator=g)[:n_mols].tolist()
    return next(iter(DataLoader([full[i] for i in idx],
                                batch_size=n_mols, shuffle=False)))


def _nn_p50(pos: torch.Tensor, batch: torch.Tensor) -> float:
    """Median nearest-neighbour distance, per molecule."""
    out = []
    for i in range(int(batch.max()) + 1):
        pts = pos[batch == i].numpy()
        if len(pts) < 2:
            continue
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        out.append(d.min(axis=1))
    return float(np.median(np.concatenate(out))) if out else float("nan")


def _build_model(ckpt: dict, cfg: dict, device):
    m = (ckpt.get("config") or cfg)["model"]
    model = EquivariantGenerator(
        num_types=m["num_types"], node_dim=m["node_dim"],
        edge_dim=m["edge_dim"], num_layers=m["num_layers"],
        time_dim=m["time_dim"],
        cond_dim=int(m.get("cond_dim", 0)),
        num_cond_classes=int(m.get("num_cond_classes", 2)),
        use_attention=m.get("use_attention", False),
        self_condition=m.get("self_condition", False),
        coord_refine_layers=int(m.get("coord_refine_layers", 0)),
        knn_schedule=m.get("knn_schedule") or None,
    )
    model.objective = str(
        (ckpt.get("config") or cfg)["diffusion"].get("objective", "eps")
    ).lower()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, (ckpt.get("config") or cfg)


def probe(tag: str, cfg_path: str, ckpt_path: str, dataset: str,
          n_mols: int, timesteps: list[int]) -> None:
    device = torch.device("cpu")
    cfg = yaml.safe_load(open(REPO / cfg_path))
    if not (REPO / ckpt_path).exists():
        print(f"== {tag}: SKIP (missing {ckpt_path})")
        return
    ckpt = torch.load(REPO / ckpt_path, map_location=device,
                      weights_only=False)
    model, cfg = _build_model(ckpt, cfg, device)
    dd = cfg["diffusion"]
    coord_ddpm = CenteredDDPM(
        num_steps=dd["num_steps"], beta_start=dd["beta_start"],
        beta_end=dd["beta_end"], device=device,
        schedule=dd.get("schedule", "linear"),
    )
    type_ddpm = TypeDDPM(
        num_steps=dd["num_steps"], beta_start=dd["beta_start"],
        beta_end=dd["beta_end"], device=device,
        schedule=dd.get("schedule", "linear"),
    )

    batch = _load_batch(dataset, cfg, n_mols)
    pos0, b = batch.pos, batch.batch
    z0 = batch.z.long()
    from train import sanitize_type_indices  # noqa: E402

    z0 = sanitize_type_indices(z0, cfg["model"]["num_types"])
    n_graphs = int(b.max()) + 1

    print(f"== {tag}  data NN_p50={_nn_p50(pos0, b):.2f}  "
          f"n_mols={n_graphs} ckpt={ckpt_path}")
    print(f"   {'t':>5} {'x0_mse':>10} {'trivial-eps0':>14} {'x0hat_NN_p50':>13}"
          f"  verdict")
    for t_val in timesteps:
        torch.manual_seed(123)
        t = torch.full((n_graphs,), t_val, dtype=torch.long)
        noisy, _ = coord_ddpm.add_noise(pos0, t, b)
        nt = type_ddpm.sample_noisy_types(z0, t, cfg["model"]["num_types"],
                                          batch=b)
        ei = build_knn_graph(noisy, b, k=6)
        per = None
        if model.knn_schedule:
            per = [build_knn_graph(noisy, b, k=k) for k in model.knn_schedule]
        with torch.no_grad():
            eps_hat, _, _ = model(nt, noisy, ei, t, b, cond=None,
                                  edge_index_per_layer=per)

        ab = coord_ddpm.alpha_bars.to(pos0.device)[t][b]
        ab_sqrt = ab.sqrt().unsqueeze(-1).clamp_min(1e-3)
        x0_hat = ((noisy - (1 - ab).sqrt().unsqueeze(-1) * eps_hat) / ab_sqrt)
        x0_hat = x0_hat.clamp(-10, 10)
        x0_triv = noisy / ab_sqrt

        mse = float(((x0_hat - pos0) ** 2).mean())
        triv = float(((x0_triv - pos0) ** 2).mean())
        print(f"   {t_val:>5} {mse:>10.4f} {triv:>14.4f} "
              f"{_nn_p50(x0_hat, b):>13.2f}  "
              f"{'WORSE than trivial' if mse > triv else 'beats trivial'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-mols", type=int, default=64)
    p.add_argument("--timesteps", default="20,50,100,200,500,900",
                   help="Comma-separated timesteps to probe")
    p.add_argument("--only", default=None,
                   help="Only run probes whose label contains this substring")
    p.add_argument("--qm9-checkpoint", default=None,
                   help="Override the QM9 control checkpoint (e.g. a training "
                        "checkpoint for tracking the trend)")
    p.add_argument("--bbbp-checkpoint", default=None,
                   help="Override the BBBP checkpoint (e.g. "
                        "checkpoints/bbb_longprobe/epoch_200.pt)")
    args = p.parse_args()

    overrides = {"qm9": args.qm9_checkpoint, "bbbp": args.bbbp_checkpoint,
                 "b3db": args.bbbp_checkpoint}
    timesteps = [int(s) for s in args.timesteps.split(",") if s.strip()]
    for tag, cfg_path, ckpt_path, dataset in DEFAULT_PROBES:
        if args.only and args.only.lower() not in tag.lower():
            continue
        ckpt_path = overrides.get(dataset) or ckpt_path
        probe(tag, cfg_path, ckpt_path, dataset, args.n_mols, timesteps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
