"""End-to-end generation and MOSES-style evaluation (Step 1.4 acceptance).

Usage:
    python generate_and_eval.py --checkpoint checkpoints/best.pt \
                                --num_samples 1000 \
                                --ddim_steps 50 \
                                --config configs/central.yaml
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import yaml
from rdkit import Chem

from src.dataset import load_qm9
from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.egnn import EquivariantGenerator
from src.sampling import sample_molecules
from src.utils.evaluation import coords_and_types_to_mol, evaluate, print_metrics

# NOTE (coord-fix era): the diffusion model indexes atom types by RAW ATOMIC
# NUMBER (PyG QM9 ``z`` is {1:H, 6:C, 7:N, 8:O, 9:F}; classes 0,2-5 are dead).
# The sampled index IS the atomic number — do NOT remap through a table
# (the old QM9_ATOMIC_NUMBERS lookup shifted every element and fabricated
# the pre-fix validity numbers).


def _load_model_weights(ckpt: dict, model) -> dict:
    """Extract loadable weights from central or federated checkpoints.

    Central ``train.py`` checkpoints store ``model_state_dict`` (full model).
    Federated ``fed_train.py`` checkpoints store ``global_state``: the full
    model when ``personal_heads=false``, or the backbone only (type/bond
    heads stay at init) when personalization is enabled.
    """
    if "model_state_dict" in ckpt:
        print("Checkpoint format: central (model_state_dict)")
        return ckpt["model_state_dict"]
    if "global_state" in ckpt:
        state = ckpt["global_state"]
        model_keys = set(model.state_dict())
        missing = sorted(model_keys - set(state))
        if missing:
            print(f"Checkpoint format: federated backbone-only; "
                  f"{len(missing)} head params at init: {missing}")
        else:
            print("Checkpoint format: federated full (global_state)")
        merged = dict(model.state_dict())
        merged.update(state)
        return merged
    raise KeyError(
        f"Unknown checkpoint format: keys {sorted(ckpt)} "
        f"(expected 'model_state_dict' or 'global_state')"
    )


def _build_atom_count_histogram(dataset) -> list[int]:
    """Compute per-molecule atom counts across the full QM9 dataset."""
    counts = []
    for data in dataset:
        counts.append(int(data.z.size(0)))
    return counts


def _sample_atom_counts(histogram: list[int], n: int, rng: np.random.Generator) -> list[int]:
    """Sample n atom counts from the empirical histogram."""
    return rng.choice(histogram, size=n, replace=True).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate molecules and evaluate")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a training checkpoint (.pt)")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of molecules to generate")
    parser.add_argument("--ddim_steps", type=int, default=None,
                        help="DDIM steps (None = ancestral DDPM)")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM stochasticity (0 = deterministic)")
    parser.add_argument("--config", type=str, default="configs/central.yaml",
                        help="Config YAML used during training")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Generate in batches of this size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cuda / cpu)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="If given, save metrics.json, smiles.txt and "
                             "molecules.sdf there (created if needed)")
    parser.add_argument("--bbb_oracle", type=str, default=None,
                        help="Path to trained BBB oracle (.pt); enables BBB%% metric")
    parser.add_argument("--relax", action="store_true",
                        help="MMFF94/UFF-relax each valid molecule and report a second "
                             "(relaxed) metric table; raw table always reported too")
    parser.add_argument("--step_schedule", type=str, default="linear",
                        choices=["linear", "quadratic"],
                        help="DDIM grid spacing (quadratic densifies low-noise steps)")
    parser.add_argument("--type_temperature", type=float, default=1.0,
                        help="Softmax temperature on type logits (>1 = more "
                             "diverse scaffolds, recommendation.md ScaffDiv #2)")
    parser.add_argument("--min_fragment_atoms", type=int, default=1,
                        help="Connectivity post-processing: drop provisional "
                             "distance-graph fragments smaller than this before "
                             "bond assignment (1 = off, 2 = drop isolated atoms)")
    parser.add_argument("--min_qed", type=float, default=0.0,
                        help="Rejection filter: replace molecules with QED below "
                             "this by invalid (0 = off; try 0.5 for drug-likeness)")
    parser.add_argument("--guidance_scale", type=float, default=2.0,
                        help="Classifier-free guidance strength w (Phase 6; "
                             "ignored unless the checkpoint has conditioning "
                             "enabled and --target_bbb is given)")
    parser.add_argument("--target_bbb", type=int, default=1, choices=[0, 1],
                        help="Target BBB class for guided generation: "
                             "0=BBB-, 1=BBB+")
    args = parser.parse_args()

    # Config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config", cfg)

    model_cfg = ckpt_cfg["model"]
    diff_cfg = ckpt_cfg["diffusion"]

    model = EquivariantGenerator(
        num_types=model_cfg["num_types"],
        node_dim=model_cfg["node_dim"],
        edge_dim=model_cfg["edge_dim"],
        num_layers=model_cfg["num_layers"],
        time_dim=model_cfg["time_dim"],
        # Phase 6: must mirror training to load conditioned checkpoints.
        cond_dim=int(model_cfg.get("cond_dim", 0)),
        num_cond_classes=int(model_cfg.get("num_cond_classes", 2)),
        use_attention=model_cfg.get("use_attention", False),
        # Tier 2 flags must mirror training to load Tier 2 checkpoints.
        self_condition=model_cfg.get("self_condition", False),
        coord_refine_layers=int(model_cfg.get("coord_refine_layers", 0)),
        knn_schedule=model_cfg.get("knn_schedule") or None,
    ).to(device)
    # Tier 3.2: prediction target the checkpoint was trained with ("eps" DDPM
    # noise [default] or "flow" velocity). The sampler auto-detects from this.
    objective = str(diff_cfg.get("objective", "eps")).lower()
    if objective not in ("eps", "flow"):
        raise ValueError(
            f"Unknown diffusion.objective={objective!r} (expected 'eps' or 'flow')"
        )
    model.objective = objective
    model.load_state_dict(_load_model_weights(ckpt, model))

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

    # Load training set for novelty / SNN baselines (Phase 6 dispatch:
    # BBB datasets use their scaffold-split train portion).
    data_name = str(ckpt_cfg["data"].get("dataset", "qm9")).lower()
    if data_name == "bbbp":
        from src.dataset_bbb import load_bbbp

        full_dataset, _, _ = load_bbbp(root=ckpt_cfg["data"]["root"])
    elif data_name == "b3db":
        from src.dataset_bbb import load_b3db

        full_dataset, _, _ = load_b3db(root=ckpt_cfg["data"]["root"])
    else:
        full_dataset = load_qm9(root=ckpt_cfg["data"]["root"])
    train_smiles: set[str] = set()
    train_mols: list[Chem.Mol] = []
    prop_sum = torch.zeros(4)
    prop_n = 0
    for data in full_dataset:
        mol = _data_to_rdkit_mol(data)
        if mol is not None:
            s = Chem.MolToSmiles(mol)
            train_smiles.add(s)
            train_mols.append(mol)
        # Raw QED/LogP/TPSA/MW — same keys/units ``extract_cond`` feeds the
        # model at training time (src/fed/trainer.py). Accumulated here so
        # guided generation can condition on the training "average molecule".
        props = [getattr(data, k, None) for k in ("qed", "logp", "tpsa", "mw")]
        if all(p is not None for p in props):
            prop_sum += torch.tensor(
                [float(p.view(-1)[0]) for p in props], dtype=torch.float32
            )
            prop_n += 1
    properties_mean = (prop_sum / prop_n) if prop_n else None

    print(f"Training set: {len(train_smiles)} unique SMILES across {len(train_mols)} molecules")

    # Build atom-count histogram and sample
    rng = np.random.default_rng(args.seed)
    histogram = _build_atom_count_histogram(full_dataset)

    all_mols: list[Chem.Mol | None] = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    cond_enabled = bool(
        (ckpt_cfg.get("conditioning") or {}).get("enabled", False)
    )
    if cond_enabled:
        print(f"Conditioning enabled — guiding toward BBB class "
              f"{args.target_bbb} with w={args.guidance_scale}")
        # Property conditioning must use the training-set raw means (the
        # distribution ``extract_cond`` saw); zeros would be out-of-distribution.
        if properties_mean is None:
            print("  WARNING: no training property means found — "
                  "falling back to zeros (out-of-distribution)")
        else:
            print("  Conditioned on training-mean profile (raw units): "
                  f"qed={properties_mean[0]:.3f} logp={properties_mean[1]:.3f} "
                  f"tpsa={properties_mean[2]:.3f} mw={properties_mean[3]:.3f}")

    for b_idx in range(num_batches):
        n_batch = min(args.batch_size, args.num_samples - b_idx * args.batch_size)
        atom_counts = _sample_atom_counts(histogram, n_batch, rng)
        total_atoms = sum(atom_counts)

        print(
            f"Batch {b_idx + 1}/{num_batches}: generating {n_batch} molecules "
            f"({total_atoms} atoms)"
        )

        # Phase 6: classifier-free guidance toward the target BBB class.
        # Only active when the checkpoint was trained with conditioning.
        cond = None
        if cond_enabled:
            cond = build_target_cond(
                args.target_bbb, n_batch, device,
                properties_mean=properties_mean,
            )

        pos, z = sample_molecules(
            model, coord_ddpm, type_ddpm,
            torch.tensor(atom_counts, dtype=torch.long),
            device=device,
            ddim_steps=args.ddim_steps,
            eta=args.eta,
            step_schedule=args.step_schedule,
            type_temperature=args.type_temperature,
            cond=cond,
            guidance_scale=args.guidance_scale if cond is not None else 0.0,
        )

        # Convert each generated molecule to RDKit
        offset = 0
        for count in atom_counts:
            mol_pos = pos[offset:offset + count].numpy()
            mol_z = z[offset:offset + count].numpy()
            # Model type indices ARE raw atomic numbers (see note above).
            atomic_numbers = np.array([int(t) for t in mol_z])
            mol = coords_and_types_to_mol(
                mol_pos, atomic_numbers,
                min_fragment_atoms=args.min_fragment_atoms,
            )
            all_mols.append(mol)
            offset += count

    # Rejection sampling on QED (recommendation.md, "Improving QED" #3):
    # report the raw table first, then filter and evaluate the survivors.
    if args.min_qed > 0.0:
        from src.utils.evaluation import apply_qed_filter

        all_mols, n_qed_dropped = apply_qed_filter(all_mols, args.min_qed)
        print(f"QED filter (>= {args.min_qed:.2f}): dropped {n_qed_dropped} "
              f"of {n_qed_dropped + sum(m is not None for m in all_mols)} molecules")

    # Evaluate — BBB oracle resolution: explicit --bbb_oracle flag first,
    # else the checkpoint config's bbb_classifier.checkpoint, else the
    # --config YAML's. Sweep/ablation runners never pass the flag, so this
    # keeps BBB% populated for configs that declare an oracle (Phase 5
    # metric lists); a missing declared file warns instead of crashing.
    oracle_path = args.bbb_oracle
    if not oracle_path:
        for source in (ckpt_cfg, cfg):
            candidate = (source.get("bbb_classifier") or {}).get("checkpoint")
            if candidate:
                oracle_path = str(candidate)
                break
    bbb_classifier = None
    if oracle_path and not Path(oracle_path).exists():
        if args.bbb_oracle:
            raise FileNotFoundError(f"--bbb_oracle not found: {oracle_path}")
        print(f"WARNING: config declares bbb_classifier.checkpoint="
              f"{oracle_path} but it does not exist — BBB% disabled")
        oracle_path = None
    if oracle_path:
        from src.models.bbb_classifier import BBBClassifier

        bbb_ckpt = torch.load(oracle_path, map_location=device,
                              weights_only=False)
        bbb_classifier = BBBClassifier(
            hidden_dim=int(bbb_ckpt.get("hidden_dim", 128))).to(device)
        bbb_classifier.load_state_dict(bbb_ckpt["model_state_dict"])
        bbb_classifier.eval()
        print(f"BBB oracle loaded from {oracle_path} "
              f"(val AUROC {bbb_ckpt.get('val_auroc', float('nan')):.4f})")
    metrics = evaluate(all_mols, train_smiles, train_mols,
                       bbb_classifier=bbb_classifier)
    print("\n--- RAW (as generated) ---")
    print_metrics(metrics)

    # Print as table for easy copy-paste. Includes the Phase 4 BBB metrics
    # (BBB%, ScaffDiv, ScaffCov, Lipinski%, Veber%, CNS_MPO) so Table I rows
    # can be pasted straight from the eval log.
    print("\nLaTeX row:")
    metric_keys = list(metrics.keys())
    vals = [f"{metrics[k]:.4f}" for k in metric_keys]
    latex_row = "  & ".join(vals) + " \\\\"
    print(latex_row)

    relaxed = None
    if args.relax:
        from src.utils.evaluation import relax_molecule

        relaxed = [relax_molecule(m) for m in all_mols]
        metrics_relaxed = evaluate(relaxed, train_smiles, train_mols,
                                   bbb_classifier=bbb_classifier)
        print("\n--- RELAXED (MMFF94/UFF post-hoc; report alongside raw) ---")
        print_metrics(metrics_relaxed)
        print("\nLaTeX row (relaxed):")
        print("  & ".join(f"{metrics_relaxed[k]:.4f}" for k in metric_keys) + " \\\\")
    else:
        metrics_relaxed = None

    if args.output_dir:
        _save_eval_outputs(
            args.output_dir, all_mols, metrics, latex_row,
            checkpoint=args.checkpoint, num_samples=args.num_samples,
            ddim_steps=args.ddim_steps, eta=args.eta, seed=args.seed,
            step_schedule=args.step_schedule, relax=args.relax,
            relaxed_mols=relaxed, relaxed_metrics=metrics_relaxed,
        )


def build_target_cond(
    target_bbb: int, num_graphs: int, device,
    properties_mean: torch.Tensor | Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Phase 6: conditioning dict for BBB-targeted generation.

    ``label`` is the target class for every graph in the batch. ``properties``
    is the training-set mean QED/LogP/TPSA/MW profile — the "average molecule"
    the model actually saw. ``extract_cond`` feeds *raw* property values
    during training (src/fed/trainer.py), so ``properties_mean`` must be in
    those same raw units; passing zeros (the previous behaviour, and the
    fallback when no mean is supplied) is out-of-distribution and leaves
    guidance driven by the class label alone.
    """
    if target_bbb not in (0, 1):
        raise ValueError(f"target_bbb must be 0 or 1, got {target_bbb!r}")

    if properties_mean is None:
        properties = torch.zeros(num_graphs, 4, device=device)
    else:
        mean = torch.as_tensor(
            properties_mean, dtype=torch.float32, device=device
        ).reshape(-1)
        if mean.numel() != 4:
            raise ValueError(
                f"properties_mean must have 4 entries, got {mean.numel()}"
            )
        properties = mean.unsqueeze(0).expand(num_graphs, 4).contiguous()

    return {
        "label": torch.full((num_graphs,), int(target_bbb),
                            dtype=torch.long, device=device),
        "properties": properties,
    }


def _save_eval_outputs(
    output_dir: str,
    mols: list,
    metrics: dict,
    latex_row: str,
    relaxed_mols: list | None = None,
    relaxed_metrics: dict | None = None,
    **run_cfg,
) -> None:
    """Persist eval results: metrics.json, smiles.txt, molecules.sdf (3D)."""
    import json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    valid_smiles: list[str] = []
    for m in mols:
        if m is None:
            continue
        try:
            valid_smiles.append(Chem.MolToSmiles(m))
        except Exception:
            continue

    payload = {
        "metrics": metrics,
        "latex_row": latex_row,
        "num_valid": len(valid_smiles),
        "num_total": len(mols),
        "run": run_cfg,
    }
    if relaxed_metrics is not None:
        payload["metrics_relaxed"] = relaxed_metrics
    with open(out / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(out / "smiles.txt", "w") as f:
        f.write("\n".join(valid_smiles) + ("\n" if valid_smiles else ""))

    sdf_path = out / "molecules.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    # Kekulization fails on fragment-soup outputs; keep it off so the SDF
    # faithfully records every valid molecule (validity itself is measured
    # in-eval by connectivity(), not by SDF round-tripping).
    writer.SetKekulize(False)
    saved = 0
    for mol, smi in zip(
        [m for m in mols if m is not None], valid_smiles,
    ):
        try:
            mol.SetProp("_Name", smi)
            writer.write(mol)
            saved += 1
        except Exception:
            continue
    writer.close()
    print(f"\nSaved {saved} molecules -> {sdf_path}")
    print(f"Saved metrics -> {out / 'metrics.json'}, smiles -> {out / 'smiles.txt'}")
    if relaxed_mols is not None:
        r_path = out / "molecules_relaxed.sdf"
        r_writer = Chem.SDWriter(str(r_path))
        r_writer.SetKekulize(False)
        r_saved = 0
        for mol in relaxed_mols:
            if mol is None:
                continue
            try:
                mol.SetProp("_Name", Chem.MolToSmiles(mol))
                r_writer.write(mol)
                r_saved += 1
            except Exception:
                continue
        r_writer.close()
        print(f"Saved {r_saved} relaxed molecules -> {r_path}")


def _data_to_rdkit_mol(data) -> Chem.Mol | None:
    """Convert a QM9 PyG Data sample to an RDKit Mol (best-effort)."""
    try:
        from rdkit.Chem import rdDetermineBonds
        mol = Chem.RWMol()
        for z_val in data.z.tolist():
            symbol = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
                       15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"
                       }.get(int(z_val), "C")
            mol.AddAtom(Chem.Atom(symbol))
        # Attempt to infer bonds from 3D distances
        pos = data.pos.numpy()
        n_atoms = mol.GetNumAtoms()
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = float(np.linalg.norm(pos[i] - pos[j]))
                if dist < 2.0:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


if __name__ == "__main__":
    main()
