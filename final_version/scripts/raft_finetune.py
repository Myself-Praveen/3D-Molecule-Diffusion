"""RAFT fallback: filter an elite subset, then supervised fine-tune on it
(plan Step 1.5).

The plan's position is that if test-time steric guidance lands only in the
4--6% ConnV band, do not go to RL yet: rejection-sample a large pool, keep the
molecules that already pass the geometry gate, and spend one cheap supervised
pass teaching the denoiser to reproduce *those*. Preference optimization on top
of atom soup teaches the model to reward soup (the plan's own rule), so the
elite pass comes first.

Elite criteria (per molecule, measured with the project's own ruler):

* the molecule built and sanitized (it exists in the SDF, so it did),
* **connected** — one fragment,
* bond count inside a window around the real-BBBP reference. The plan writes
  ``15 <= bonds <= 25`` for a heavy-atom count of ~19; this project's ruler
  counts H too, where real BBBP sits at ``REFERENCE_BONDS_PER_MOL = 26.88``.
  Both spellings are supported: the default window is the plan's ±35% *relative*
  tolerance applied to the reference for the active ruler, and ``--bond_min`` /
  ``--bond_max`` override it explicitly.

Fine-tuning is plain supervised diffusion training (MSE on the noise target) on
the elite coordinates only — no RL, no reward — and the output is written in
``train.py``'s checkpoint format so ``generate_and_eval.py`` consumes it
unchanged.

Usage:
    .venv/bin/python scripts/raft_finetune.py \
        --checkpoint checkpoints/central_v5_bbb/best.pt \
        --config configs/central_v5_bbb.yaml \
        --candidates outputs/eval_v5_bbb_steric_w10/molecules.sdf \
        --out_dir checkpoints/central_v5_bbb_raft \
        --epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from rdkit import Chem

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from generate_and_eval import _load_model_weights  # noqa: E402
from src.models.diffusion import CenteredDDPM  # noqa: E402
from src.models.egnn import EquivariantGenerator  # noqa: E402
from src.rl.rewards import REFERENCE_BONDS_PER_MOL  # noqa: E402
from src.utils.graph import build_knn_graph  # noqa: E402
from train import sanitize_type_indices  # noqa: E402

# The plan's relative tolerance around its own target: 15-25 around 19 is
# roughly -21%/+32%; 0.35 is the symmetric version of the same intent.
DEFAULT_REL_TOL = 0.35


def bond_window(
    rel_tol: float = DEFAULT_REL_TOL,
    bond_min: float | None = None,
    bond_max: float | None = None,
    reference_bonds: float = REFERENCE_BONDS_PER_MOL,
) -> tuple[float, float]:
    """``(low, high)`` bond-count window for elite selection."""
    if bond_min is not None and bond_max is not None:
        return float(bond_min), float(bond_max)
    return reference_bonds * (1.0 - rel_tol), reference_bonds * (1.0 + rel_tol)


def select_elites(
    mols: list[Chem.Mol | None],
    rel_tol: float = DEFAULT_REL_TOL,
    bond_min: float | None = None,
    bond_max: float | None = None,
    reference_bonds: float = REFERENCE_BONDS_PER_MOL,
    require_connected: bool = True,
) -> tuple[list[Chem.Mol], dict[str, float]]:
    """Keep valid, connected, bond-count-in-window molecules.

    Returns ``(elites, stats)``; ``stats`` records how many candidates failed
    each criterion so the filter can never quietly discard everything.
    """
    low, high = bond_window(rel_tol, bond_min, bond_max, reference_bonds)
    total = 0
    invalid = 0
    fragmented = 0
    out_of_window = 0
    elites: list[Chem.Mol] = []
    for mol in mols:
        if mol is None:
            invalid += 1
            continue
        total += 1
        frags = len(Chem.GetMolFrags(mol))
        if require_connected and frags != 1:
            fragmented += 1
            continue
        bonds = mol.GetNumBonds()
        if not (low <= bonds <= high):
            out_of_window += 1
            continue
        elites.append(mol)
    stats = {
        "n_candidates": total,
        "n_invalid": invalid,
        "n_fragmented": fragmented,
        "n_out_of_window": out_of_window,
        "n_elites": len(elites),
        "bond_window_low": low,
        "bond_window_high": high,
        "elite_fraction": (len(elites) / max(total, 1)),
    }
    return elites, stats


def _tensorize(mols: list[Chem.Mol], num_types: int):
    """Concatenate elite molecules into ``(pos, z, batch)`` training tensors."""
    positions, types, batch = [], [], []
    for graph_id, mol in enumerate(mols):
        conf = mol.GetConformer()
        n = mol.GetNumAtoms()
        positions.append(torch.tensor(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
              conf.GetAtomPosition(i).z] for i in range(n)],
            dtype=torch.float32,
        ))
        z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()],
                         dtype=torch.long)
        types.append(z)
        batch.append(torch.full((n,), graph_id, dtype=torch.long))
    pos = torch.cat(positions, dim=0)
    z = sanitize_type_indices(torch.cat(types, dim=0), num_types)
    return pos, z, torch.cat(batch, dim=0)


def finetune(
    model,
    coord_ddpm: CenteredDDPM,
    elites: list[Chem.Mol],
    *,
    num_types: int,
    epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 32,
    knn: int = 8,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> dict:
    """Supervised MSE-noise fine-tuning on the elite subset (plan Step 1.5.4).

    Returns the training history; the caller writes the checkpoint.
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos, z, batch = _tensorize(elites, num_types)
    n_graphs = int(batch.max().item()) + 1

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n_graphs)
        losses = []
        for start in range(0, n_graphs, batch_size):
            keep_graphs = order[start:start + batch_size]
            # Mask out graphs outside the chunk (small datasets => one chunk).
            mask = torch.isin(batch, keep_graphs)
            if not bool(mask.any()):
                continue
            # Remap surviving graph ids to a dense 0..k-1 range.
            sub_pos, sub_z, sub_batch = pos[mask], z[mask], batch[mask]
            _, dense = torch.unique(sub_batch, return_inverse=True)
            optimizer.zero_grad(set_to_none=True)
            t = torch.randint(0, coord_ddpm.num_steps,
                              (int(dense.max()) + 1,), device=device)
            noisy, target = coord_ddpm.add_noise(sub_pos.to(device), t, dense.to(device))
            edges = build_knn_graph(noisy, dense.to(device), k=knn)
            pred = model(sub_z.to(device), noisy, edges, t, dense.to(device))[0]
            loss = F.mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch,
                        "loss": float(np.mean(losses)) if losses else float("nan")})
        if epoch == 1 or epoch % max(epochs // 5, 1) == 0:
            print(f"  raft epoch {epoch}/{epochs} loss={history[-1]['loss']:.4f}")
    model.eval()
    return {"history": history, "n_elites": len(elites)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/central_v5_bbb.yaml")
    parser.add_argument("--candidates", required=True,
                        help="SDF of candidate molecules (an eval's molecules.sdf)")
    parser.add_argument("--out_dir", required=True,
                        help="Checkpoint directory (best.pt written here)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--rel_tol", type=float, default=DEFAULT_REL_TOL,
                        help="Relative half-width of the bond window around the "
                             "reference (0.35 ~ the plan's 15-25 around 19)")
    parser.add_argument("--bond_min", type=float, default=None)
    parser.add_argument("--bond_max", type=float, default=None)
    parser.add_argument("--allow_fragments", action="store_true",
                        help="Skip the connectedness criterion (not recommended)")
    parser.add_argument("--min_elites", type=int, default=5,
                        help="Refuse to fine-tune on fewer elites than this")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stats_out", default=None,
                        help="Optional JSON path for the filter statistics")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # --- 1. filter the elite subset -------------------------------------
    mols: list[Chem.Mol | None] = [
        m for m in Chem.SDMolSupplier(args.candidates, removeHs=False,
                                      sanitize=True)
    ]
    elites, stats = select_elites(
        mols, rel_tol=args.rel_tol, bond_min=args.bond_min,
        bond_max=args.bond_max, require_connected=not args.allow_fragments,
    )
    print(f"elite filter: {stats}")
    if len(elites) < args.min_elites:
        print(f"FAIL: only {len(elites)} elites (< {args.min_elites}). "
              f"Per the plan's decision table, do NOT proceed to RL on this "
              f"pool — raise samples or relax the window.")
        return 2

    # --- 2. supervised fine-tune ----------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config", cfg)
    model_cfg = ckpt_cfg["model"]
    model = EquivariantGenerator(
        num_types=model_cfg["num_types"], node_dim=model_cfg["node_dim"],
        edge_dim=model_cfg["edge_dim"], num_layers=model_cfg["num_layers"],
        time_dim=model_cfg["time_dim"],
        cond_dim=int(model_cfg.get("cond_dim", 0)),
        num_cond_classes=int(model_cfg.get("num_cond_classes", 2)),
        use_attention=model_cfg.get("use_attention", False),
        self_condition=model_cfg.get("self_condition", False),
        coord_refine_layers=int(model_cfg.get("coord_refine_layers", 0)),
        knn_schedule=model_cfg.get("knn_schedule") or None,
    ).to(device)
    model.objective = str(ckpt_cfg["diffusion"].get("objective", "eps")).lower()
    model.load_state_dict(_load_model_weights(ckpt, model))
    coord_ddpm = CenteredDDPM.from_config(ckpt_cfg["diffusion"], device=device)

    print(f"fine-tuning on {len(elites)} elites "
          f"({args.epochs} epochs, lr {args.lr:g})")
    result = finetune(
        model, coord_ddpm, elites,
        num_types=int(model_cfg["num_types"]),
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        knn=args.knn, seed=args.seed, device=device,
    )

    # --- 3. save in train.py's format -----------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best.pt"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "best_val_loss": result["history"][-1]["loss"] if result["history"] else None,
        "config": ckpt_cfg,
        "raft": {"source_checkpoint": args.checkpoint,
                 "candidates": args.candidates,
                 "stats": stats,
                 "seed": args.seed},
    }, out_path)
    print(f"saved -> {out_path}")

    if args.stats_out:
        import json

        stats_path = Path(args.stats_out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump({"filter": stats, "train": result["history"]}, f, indent=2)
        print(f"saved stats -> {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
