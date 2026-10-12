"""ConnV-LDAG evaluation of a BBB full checkpoint (parameterized).

Usage:
    .venv/bin/python eval_connv.py                          # epoch_2800.pt
    .venv/bin/python eval_connv.py --ckpt checkpoints/bbb_full/epoch_3500.pt
    .venv/bin/python eval_connv.py --ckpt checkpoints/bbb_full/best.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdchem

REPO = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")

BBB_PROPS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}  # atomic number -> symbol


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/bbb_full/epoch_2800.pt",
                   help="path to checkpoint under repo root")
    p.add_argument("--n-samples", type=int, default=256)
    p.add_argument("--ddim-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _build_model(ckpt: dict):
    from src.models.egnn import EquivariantGenerator

    m = ckpt["config"]["model"]
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
    )
    model.objective = str(ckpt["config"]["diffusion"].get("objective", "eps")).lower()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["config"]


def _sample(n_samples: int, model, cfg: dict, seed: int, ddim_steps: int):
    from src.models.diffusion import CenteredDDPM, TypeDDPM
    from src.sampling import sample_molecules

    dd = cfg["diffusion"]
    coord_ddpm = CenteredDDPM(
        num_steps=dd["num_steps"], beta_start=dd["beta_start"],
        beta_end=dd["beta_end"], device=DEVICE,
        schedule=dd.get("schedule", "linear"),
    )
    type_ddpm = TypeDDPM(
        num_steps=dd["num_steps"], beta_start=dd["beta_start"],
        beta_end=dd["beta_end"], device=DEVICE,
        schedule=dd.get("schedule", "linear"),
    )

    g = torch.Generator().manual_seed(seed)
    atom_counts = torch.full((n_samples,), 20, dtype=torch.long)
    return sample_molecules(
        model,
        coord_ddpm,
        type_ddpm,
        atom_counts,
        device=DEVICE,
        ddim_steps=ddim_steps,
        guidance_scale=cfg.get("conditioning", {}).get("guidance_scale", 2.0),
    )


def _to_rdkit_mol(pos: torch.Tensor, z: torch.Tensor, n_per_mol: int = 20):
    """Convert a flat (N,3)/(N,) sample into RDKit mols (no optimization)."""
    mols = []
    n_total = pos.size(0)
    n_mols = n_total // n_per_mol
    for i in range(n_mols):
        start = i * n_per_mol
        end = start + n_per_mol
        coords = pos[start:end].numpy()
        atom_types = z[start:end].tolist()

        mol = Chem.RWMol()
        conf = Chem.Conformer(n_per_mol)
        for j in range(n_per_mol):
            anum = BBB_PROPS.get(int(atom_types[j]), None)
            if anum is None:
                anum = "C"  # unknown type -> placeholder carbon (keep geometry alive)
            at = rdchem.Atom(anum)
            mol.AddAtom(at)
            conf.SetAtomPosition(j, coords[j].tolist())
        mol.AddConformer(conf)
        Chem.SanitizeMol(mol)
        mols.append(mol)
    return mols


def _connv_metrics(obs_pos_list):
    """Geometry metrics over a list of (N_i, 3) arrays.

    Comparable to the trivial all-at-origin baseline.
    """
    if len(obs_pos_list) < 2:
        return {
            "coverage": float("nan"),
            "uniqueness": float("nan"),
            "nn_p50_to_origin": float("nan"),
            "mean_atom_dist_to_origin": float("nan"),
            "n_conformers": len(obs_pos_list),
        }

    per_conf_med = np.array([
        np.median(np.linalg.norm(pts, axis=1)) for pts in obs_pos_list
    ])
    nn_p50_to_origin = float(np.mean(per_conf_med))
    mean_atom_dist_to_origin = float(np.mean([
        np.mean(np.linalg.norm(pts, axis=1)) for pts in obs_pos_list
    ]))

    # uniqueness: median pairwise min-atom-dist across conformers
    uniq_d = []
    for i in range(len(obs_pos_list)):
        for j in range(i + 1, len(obs_pos_list)):
            a = obs_pos_list[i]
            b = obs_pos_list[j]
            d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
            uniq_d.append(np.min(d))
    uniqueness = float(np.median(uniq_d)) if uniq_d else float("nan")

    coverage = float(np.mean(per_conf_med < 2.0))

    return {
        "coverage": coverage,
        "uniqueness": uniqueness,
        "nn_p50_to_origin": nn_p50_to_origin,
        "mean_atom_dist_to_origin": mean_atom_dist_to_origin,
        "n_conformers": len(obs_pos_list),
    }


def main() -> int:
    args = _parse_args()
    ckpt_path = REPO / args.ckpt
    n_samples = args.n_samples
    ddim_steps = args.ddim_steps
    seed = args.seed

    print(f"== ConnV-LDAG eval of {ckpt_path.name}  (seed={seed}, n={n_samples})")
    t0 = time.time()

    if not ckpt_path.exists():
        print(f"ERROR: checkpoint missing: {ckpt_path}")
        return 2

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    print(f"   checkpoint epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}"
          f"  best_val={ckpt['best_val_loss']:.4f}")
    print(f"   loading model...")
    model, cfg = _build_model(ckpt)

    print(f"   sampling {n_samples} molecules (DDIM {ddim_steps} steps)...")
    pos, z = _sample(n_samples, model, cfg, seed, ddim_steps)
    mols = _to_rdkit_mol(pos, z, n_per_mol=20)
    print(f"   {len(mols)} valid RDKit mols from {n_samples} samples")

    print("   using raw sampled coords for geometry metric "
          "(obabel not available)")
    obs_pos_list = [pos[i * 20:(i + 1) * 20].numpy()
                    for i in range(len(mols))]
    metrics = _connv_metrics(obs_pos_list)

    # Trivial baseline: all-at-origin (worst-case geometry)
    print(f"   computing trivial (all-at-origin) baseline...")
    trivial_obs = [np.zeros((20, 3)) for _ in range(n_samples)]
    trivial_metrics = _connv_metrics(trivial_obs)

    elapsed = time.time() - t0
    print()
    print(f"== ConnV-LDAG results ({ckpt_path.name}, elapsed {elapsed:.1f}s)")
    print(f"   {'metric':<28} {ckpt_path.name:>16} {'trivial baseline':>16}")
    print(f"   {'-'*60}")
    for key in ["coverage", "uniqueness", "nn_p50_to_origin",
                "mean_atom_dist_to_origin", "n_conformers"]:
        v = metrics.get(key, float("nan"))
        t = trivial_metrics.get(key, float("nan"))
        if key == "n_conformers":
            print(f"   {key:<28} {int(v):>16} {int(t):>16}")
        else:
            print(f"   {key:<28} {v:>16.4f} {t:>16.4f}")

    print()
    if metrics["mean_atom_dist_to_origin"] < trivial_metrics["mean_atom_dist_to_origin"]:
        print("   >> beats trivial: geometry is tighter than all-at-origin")
    else:
        print("   >> does NOT beat trivial: still worse than all-at-origin")
    print(f"   >> uniqueness (median min-atom-dist across pairs): "
          f"{metrics['uniqueness']:.3f} A")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
