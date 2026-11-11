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
from src.utils.evaluation import (
    MIN_REPORTABLE_N,
    coords_and_types_to_mol,
    connectivity,
    evaluate,
    geometry_only_metrics,
    latex_row,
    print_metrics,
)

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


def _reference_mols_and_smiles(full_dataset) -> tuple[list, set[str], dict]:
    """Canonical reference set for Novelty / SNN / ScaffCov (fix.md F1).

    Bond orders come from the dataset SMILES. Coordinate perception is only
    the fallback, and it uses the eval's own covalent-radii builder — the old
    hard-coded ``dist < 2.0`` rule destroyed 98.8 % of BBBP (20 of 1628
    molecules survived), so every Novelty/SNN/ScaffCov number in the record
    was measured against 1.2 % of the training data.
    """
    mols: list = []
    smiles: set[str] = set()
    n_with_smiles = 0
    n_smiles_parsed = 0
    for data in full_dataset:
        smi = getattr(data, "smiles", None)
        mol = None
        if smi:
            n_with_smiles += 1
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                n_smiles_parsed += 1
                # Hydrogen-inclusive, like the generated molecules: the
                # model denoises every atom including H, so a reference that
                # counted heavy atoms only would put MeanAtoms at 24 instead
                # of the measured 45.9.
                try:
                    mol = Chem.AddHs(mol)
                except Exception:
                    pass
        if mol is None:
            mol = coords_and_types_to_mol(
                data.pos.numpy(), data.z.numpy().astype(int))
        if mol is None:
            continue
        mols.append(mol)
        try:
            smiles.add(Chem.MolToSmiles(mol))
        except Exception:
            continue
    info = {
        "requested": len(full_dataset),
        "n": len(mols),
        "coverage": (len(mols) / len(full_dataset)) if len(full_dataset) else 0.0,
        "n_dataset_smiles": n_with_smiles,
        "n_smiles_parsed": n_smiles_parsed,
        "source": ("dataset_smiles" if n_smiles_parsed
                   else "coordinate_perception"),
    }
    return mols, smiles, info


def _load_oracle(args, ckpt_cfg: dict, cfg: dict, device):
    """Resolve + load the BBB oracle. Returns ``(classifier, meta)``.

    Explicit ``--bbb_oracle`` wins, else the checkpoint config's
    ``bbb_classifier.checkpoint``, else the ``--config`` YAML's. The meta
    block records the path and validation AUROC so a BBB% column can be
    traced to the model that produced it (fix.md F9).
    """
    oracle_path = args.bbb_oracle
    if not oracle_path:
        for source in (ckpt_cfg, cfg):
            candidate = (source.get("bbb_classifier") or {}).get("checkpoint")
            if candidate:
                oracle_path = str(candidate)
                break
    if oracle_path and not Path(oracle_path).exists():
        if args.bbb_oracle:
            raise FileNotFoundError(f"--bbb_oracle not found: {oracle_path}")
        print(f"WARNING: config declares bbb_classifier.checkpoint="
              f"{oracle_path} but it does not exist — BBB% disabled")
        oracle_path = None
    if not oracle_path:
        return None, None
    from src.models.bbb_classifier import BBBClassifier

    bbb_ckpt = torch.load(oracle_path, map_location=device, weights_only=False)
    classifier = BBBClassifier(
        hidden_dim=int(bbb_ckpt.get("hidden_dim", 128))).to(device)
    classifier.load_state_dict(bbb_ckpt["model_state_dict"])
    classifier.eval()
    auroc = bbb_ckpt.get("val_auroc")
    print(f"BBB oracle loaded from {oracle_path} "
          f"(val AUROC {auroc if auroc is not None else float('nan'):.4f})")
    return classifier, {
        "path": oracle_path,
        "val_auroc": float(auroc) if auroc is not None else None,
    }


def _run_cfg(args, seed: int) -> dict:
    """The exact protocol that produced a row, recorded in ``run``."""
    return {
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "ddim_steps": args.ddim_steps,
        "eta": args.eta,
        "seed": seed,
        "step_schedule": args.step_schedule,
        "relax": args.relax,
        "type_temperature": args.type_temperature,
        "steric_weight": args.steric_weight,
        "steric_start_frac": args.steric_start_frac,
        "steric_sigma_scale": args.steric_sigma_scale,
        "steric_k_rep": args.steric_k_rep,
        "steric_r_clash": args.steric_r_clash,
        "valence_weight": args.valence_weight,
        "valence_start_frac": args.valence_start_frac,
        "valence_tau": args.valence_tau,
        "valence_covalent_factor": args.valence_covalent_factor,
        "valence_k_val": args.valence_k_val,
        "valence_max_step": args.valence_max_step,
        "connectivity_weight": args.connectivity_weight,
        "connectivity_start_frac": args.connectivity_start_frac,
        "connectivity_window_pad": args.connectivity_window_pad,
        "connectivity_k_att": args.connectivity_k_att,
        "connectivity_max_step": args.connectivity_max_step,
        "coord_temperature": args.coord_temperature,
        "keep_largest_fragment": args.keep_largest_fragment,
        "min_fragment_atoms": args.min_fragment_atoms,
        "bond_tolerance": args.bond_tolerance,
        "max_bonds_ratio": args.max_bonds_ratio,
        "perceive_bonds": args.perceive_bonds,
        "min_qed": args.min_qed,
        "veber_filter": args.veber_filter,
    }


def _metrics_meta(args, seed: int, reference: dict | None,
                  oracle_meta: dict | None) -> dict:
    """Everything a reader needs to decide whether two rows are comparable."""
    try:
        import rdkit as _rdkit
        rdkit_version = _rdkit.__version__
    except Exception:  # pragma: no cover - rdkit is a hard dependency
        rdkit_version = None
    git_sha = None
    try:
        import subprocess

        git_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=Path(__file__).parent,
        ).stdout.strip() or None
    except Exception:  # pragma: no cover - git may be absent
        git_sha = None
    post_hoc: list[str] = []
    if args.keep_largest_fragment:
        post_hoc.append("keep_largest_fragment")
    if args.min_fragment_atoms > 1:
        post_hoc.append(f"min_fragment_atoms={args.min_fragment_atoms}")
    if args.max_bonds_ratio > 0.0:
        post_hoc.append(f"max_bonds_ratio={args.max_bonds_ratio}")
    if args.min_qed > 0.0:
        post_hoc.append(f"min_qed={args.min_qed}")
    if args.veber_filter:
        post_hoc.append("veber_filter")
    return {
        "oracle": oracle_meta,
        "reference": reference,
        "relaxed_metrics_are": "geometry-only" if args.relax else None,
        "post_hoc": post_hoc,
        "protocol": {
            "ddim_steps": args.ddim_steps, "eta": args.eta,
            "step_schedule": args.step_schedule, "relax": args.relax,
            "bond_tolerance": args.bond_tolerance,
            "min_fragment_atoms": args.min_fragment_atoms,
            "max_bonds_ratio": args.max_bonds_ratio,
            "perceive_bonds": args.perceive_bonds,
            "seed": seed,
        },
        "versions": {"torch": torch.__version__, "rdkit": rdkit_version,
                     "git": git_sha},
    }


def _generate_batches(args, model, coord_ddpm, type_ddpm, histogram, device,
                      cond_enabled: bool, properties_mean, seed: int):
    """Generate ``--num_samples`` molecules for one seed.

    Returns ``(mols, stats)``: ``stats[i]`` is the atom accounting of sample
    ``i`` (fix.md F3), so post-hoc pruning is visible in the payload.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    all_mols: list = []
    stats: list[dict] = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
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
            steric_weight=args.steric_weight,
            steric_start_frac=args.steric_start_frac,
            steric_sigma_scale=args.steric_sigma_scale,
            steric_k_rep=args.steric_k_rep,
            steric_r_clash=args.steric_r_clash,
            valence_weight=args.valence_weight,
            valence_start_frac=args.valence_start_frac,
            valence_tau=args.valence_tau,
            valence_covalent_factor=args.valence_covalent_factor,
            valence_k_val=args.valence_k_val,
            valence_max_step=args.valence_max_step,
            connectivity_weight=args.connectivity_weight,
            connectivity_start_frac=args.connectivity_start_frac,
            connectivity_window_pad=args.connectivity_window_pad,
            connectivity_k_att=args.connectivity_k_att,
            connectivity_max_step=args.connectivity_max_step,
            coord_temperature=args.coord_temperature,
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
            atom_stats: dict = {}
            mol = coords_and_types_to_mol(
                mol_pos, atomic_numbers,
                min_fragment_atoms=args.min_fragment_atoms,
                bond_tolerance=args.bond_tolerance,
                max_bonds_ratio=args.max_bonds_ratio,
                perceive_orders=args.perceive_bonds,
                keep_largest_fragment=args.keep_largest_fragment,
                stats=atom_stats,
            )
            all_mols.append(mol)
            stats.append(atom_stats)
            offset += count
    return all_mols, stats


def _evaluate_seed(args, *, seed: int, model, coord_ddpm, type_ddpm, histogram,
                   device, train_smiles: set[str], train_mols: list,
                   reference: dict | None, bbb_classifier,
                   oracle_meta: dict | None, full_dataset,
                   cond_enabled: bool, properties_mean):
    """Generate + score one seed. Returns ``(payload, mols, relaxed_mols)``."""
    print(f"\n===== seed {seed} =====")
    all_mols, stats = _generate_batches(
        args, model, coord_ddpm, type_ddpm, histogram, device,
        cond_enabled, properties_mean, seed,
    )

    # Rejection sampling on QED (recommendation.md, "Improving QED" #3):
    # report the raw table first, then filter and evaluate the survivors.
    if args.min_qed > 0.0:
        from src.utils.evaluation import apply_qed_filter

        all_mols, n_qed_dropped = apply_qed_filter(all_mols, args.min_qed)
        print(f"QED filter (>= {args.min_qed:.2f}): dropped {n_qed_dropped} "
              f"of {n_qed_dropped + sum(m is not None for m in all_mols)} molecules")

    # Veber filter (plan workstream C2): drop Veber-failing molecules, but
    # keep the PRE-filter pass rate as Veber_filtered% — with the filter on
    # the table's Veber% is 100 by construction and the honest number must
    # stay visible next to it (same raw-vs-filtered rationale as --min_qed).
    veber_before = None
    if args.veber_filter:
        from src.utils.evaluation import apply_veber_filter, veber_pass_rate

        veber_before = veber_pass_rate(all_mols)
        all_mols, n_veber_dropped = apply_veber_filter(all_mols)
        print(f"Veber filter (TPSA <= 140, rot <= 10): dropped "
              f"{n_veber_dropped} of "
              f"{n_veber_dropped + sum(m is not None for m in all_mols)} "
              f"molecules (pre-filter pass rate {veber_before * 100.0:.1f}%)")

    metrics = evaluate(all_mols, train_smiles, train_mols,
                       bbb_classifier=bbb_classifier,
                       stats=stats, reference=reference)
    if args.veber_filter:
        # Pre-filter pass fraction — computed before apply_veber_filter and
        # reported alongside the (trivially 100) filtered Veber%.
        metrics["Veber_filtered%"] = veber_before * 100.0

    # Lab-readiness gate (stage 7): computed on the RAW samples — the gate
    # answers "would we order this batch", and a lab orders the molecules as
    # generated, not an idealized post-hoc relaxation. Gate metric keys are
    # disjoint from evaluate()'s, so the update below cannot clobber a MOSES
    # number; the keys land in metrics.json like any other.
    gate = None
    gate_relaxed = None
    gate_ref = None
    if args.lab_gate:
        from src.utils.lab_gate import print_gate, run_gate

        gate_ref = _gate_reference_mols(full_dataset, train_smiles)
        gate = run_gate(all_mols, gate_ref)
        metrics.update(gate["metrics"])
        print_gate(gate)

    print("\n--- RAW (as generated) ---")
    print_metrics(metrics)

    # LaTeX row is RENDERED from the canonical order, never stored (fix.md
    # F8: a frozen row went stale the moment a metric was patched).
    print("\nLaTeX row (canonical order):")
    print(latex_row(metrics))

    # fix.md F7: relaxation moves atoms, not bonds. Only geometry-dependent
    # quantities can change, so only those are reported a second time.
    relaxed = None
    relaxed_metrics = None
    if args.relax:
        from src.utils.evaluation import relax_molecule

        relaxed = [relax_molecule(m) for m in all_mols]
        relaxed_metrics = geometry_only_metrics(relaxed)
        print("\n--- RELAXED (geometry only; graph metrics cannot change) ---")
        for key, val in relaxed_metrics.items():
            print(f"{key:<22} {'—' if val is None else f'{val:.4f}':>10}")
        if gate is not None:
            print("\n--- lab-readiness gate on RELAXED samples ---")
            gate_relaxed = run_gate(relaxed, gate_ref)
            print_gate(gate_relaxed)

    payload = {
        "metrics_schema": 2,
        "metrics": metrics,
        "reportable": int(metrics.get("n_valid") or 0) >= MIN_REPORTABLE_N,
        "num_valid": int(metrics.get("n_valid") or 0),
        "num_total": int(metrics.get("n_total") or 0),
        "metrics_meta": _metrics_meta(args, seed, reference, oracle_meta),
        "run": _run_cfg(args, seed),
    }
    if relaxed_metrics is not None:
        payload["metrics_relaxed"] = relaxed_metrics
    if gate is not None:
        payload["lab_gate"] = gate
    if gate_relaxed is not None:
        payload["lab_gate_relaxed"] = gate_relaxed
    return payload, all_mols, relaxed


def _aggregate_payloads(payloads: list[dict], seeds: list[int]) -> dict:
    """Seed-aggregated row: median [min, max] per metric (fix.md F6)."""
    import statistics

    first = payloads[0]
    median: dict = {}
    lo: dict = {}
    hi: dict = {}
    for key, val in first["metrics"].items():
        vals = [p["metrics"].get(key) for p in payloads]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool)
               for v in vals):
            nums = [float(v) for v in vals]
            median[key] = statistics.median(nums)
            lo[key] = min(nums)
            hi[key] = max(nums)
        else:
            median[key] = val
    meta = dict(first.get("metrics_meta") or {})
    meta["aggregation"] = "median [min, max] over seeds"
    return {
        "metrics_schema": 2,
        "metrics": median,
        "metrics_min": lo,
        "metrics_max": hi,
        "seeds": list(seeds),
        "reportable": all(p.get("reportable") for p in payloads),
        "num_valid": int(statistics.median(
            [p["num_valid"] for p in payloads])),
        "num_total": first["num_total"],
        "metrics_meta": meta,
        "run": {**first["run"], "seeds": list(seeds)},
        "per_seed": {f"seed_{s}": p["metrics"]
                     for s, p in zip(seeds, payloads)},
    }


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
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for BOTH the atom-count histogram rng and "
                             "torch's sampling noise (same seed => identical "
                             "molecules; enables the plan's seed medians)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Replicate the WHOLE generation+eval per seed and "
                             "report median [min, max] (fix.md F6). A "
                             "single-seed row must be labelled as such; "
                             "--seed still sets the first seed when this is "
                             "omitted)")
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
    parser.add_argument("--coord_temperature", type=float, default=0.0,
                        help="Coordinate analogue of --type_temperature (plan "
                             "Intervention 2): extra Gaussian perturbation per "
                             "reverse step, scaled by the remaining noise level "
                             "(0 = off, bit-exact legacy sampler)")
    parser.add_argument("--keep_largest_fragment", action="store_true",
                        help="Spatial-clustering fallback (plan Intervention 2): "
                             "keep only the largest distance-graph component per "
                             "molecule. Post-hoc correction, not a generation fix")
    parser.add_argument("--steric_enabled", action="store_true",
                        help="Turn on test-time steric guidance at "
                             "--steric_max_weight (plan T1-T4 test matrix; "
                             "equivalent to --steric_weight > 0)")
    parser.add_argument("--steric_weight", type=float, default=0.0,
                        help="Peak steric guidance weight w on the clean x0 "
                             "estimate (0 = off, bit-exact legacy sampler; "
                             "overrides --steric_max_weight when > 0)")
    parser.add_argument("--steric_max_weight", type=float, default=1.0,
                        help="Peak ramp weight used with --steric_enabled "
                             "(plan T3 primary setting)")
    parser.add_argument("--steric_start_frac", "--steric_ramp_start",
                        type=float, default=0.7,
                        help="Fraction of the reverse trajectory elapsed when "
                             "the guidance ramp starts (0.7 = last 30%%; "
                             "plan --steric_ramp_start)")
    parser.add_argument("--steric_sigma_scale", type=float, default=0.75,
                        help="Contact distance as a fraction of the covalent-"
                             "radii sum; 0.75 leaves every real bond untouched")
    parser.add_argument("--steric_k_rep", type=float, default=1.0,
                        help="Overall strength of the repulsive energy")
    parser.add_argument("--steric_r_clash", type=float, default=None,
                        help="Absolute severe-overlap radius in A (plan mode: "
                             "1.1) added on top of the per-pair contact term")
    parser.add_argument("--min_fragment_atoms", type=int, default=1,
                        help="Connectivity post-processing: drop provisional "
                             "distance-graph fragments smaller than this before "
                             "bond assignment (1 = off, 2 = drop isolated atoms)")
    parser.add_argument("--bond_tolerance", type=float, default=None,
                        help="Bond-perception chain A2: tighten-cap (Å) on the "
                             "per-pair covalent-radius tolerance, min(base, cap) "
                             "(default: legacy Table I protocol; 0.35 = BBBP-era "
                             "setting; tighten-only, never loosens)")
    parser.add_argument("--max_bonds_ratio", type=float, default=0.0,
                        help="Bond chain A3: reject molecules whose bond count "
                             "exceeds ratio * 0.95 * n_atoms — over-bonded soup "
                             "counts as invalid (0 = off, legacy; plan: 1.4)")
    parser.add_argument("--perceive_bonds", action="store_true",
                        help="Bond chain A4: after building the single-bond "
                             "topology, attempt rdDetermineBonds bond-order "
                             "perception (fallback keeps single bonds; "
                             "Validity/ConnV invariant by construction)")
    parser.add_argument("--min_qed", type=float, default=0.0,
                        help="Rejection filter: replace molecules with QED below "
                             "this by invalid (0 = off; try 0.5 for drug-likeness)")
    parser.add_argument("--veber_filter", action="store_true",
                        help="Rejection filter C2: drop molecules failing Veber "
                             "(TPSA <= 140, rotatable bonds <= 10); the PRE-filter "
                             "pass rate is still reported as Veber_filtered%% so "
                             "the filtered table's Veber%% (100 by construction) "
                             "cannot launder the raw one")
    parser.add_argument("--guidance_scale", type=float, default=2.0,
                        help="Classifier-free guidance strength w (Phase 6; "
                             "ignored unless the checkpoint has conditioning "
                             "enabled and --target_bbb is given)")
    parser.add_argument("--target_bbb", type=int, default=1, choices=[0, 1],
                        help="Target BBB class for guided generation: "
                             "0=BBB-, 1=BBB+")
    parser.add_argument("--lab_gate", action="store_true",
                        help="Run the six-stage lab-readiness gate (perception, "
                             "geometry, synthesis, applicability domain, ADMET, "
                             "retrosynthesis) on the RAW samples: merges gate "
                             "metrics into the metric table / metrics.json and "
                             "stores the PASS-FAIL verdict as lab_gate")
    parser.add_argument("--valence_enabled", action="store_true",
                        help="Turn on test-time physical valence guidance at "
                             "--valence_weight (contract bbb-valence-dpo-v2.1)")
    parser.add_argument("--valence_weight", type=float, default=0.0,
                        help="Peak valence guidance weight w on the clean x0 "
                             "estimate (0 = off; overridden to 0.5 when "
                             "--valence_enabled and --valence_weight <= 0)")
    parser.add_argument("--valence_start_frac", type=float, default=0.7,
                        help="Fraction of the reverse trajectory elapsed when "
                             "the valence ramp starts (0.7 = last 30%%)")
    parser.add_argument("--valence_tau", type=float, default=0.05,
                        help="Sigmoid sharpness for soft bond perception in Å "
                             "(default 0.05; contract: 0.05)")
    parser.add_argument("--valence_covalent_factor", type=float, default=1.10,
                        help="Multiplier on the covalent-radius sum for the "
                             "soft bond envelope (default 1.10; contract: 1.10)")
    parser.add_argument("--valence_k_val", type=float, default=1.0,
                        help="Valence penalty force constant (default 1.0)")
    parser.add_argument("--valence_max_step", type=float, default=0.15,
                        help="Max per-atom displacement per guided step in Å "
                             "(default 0.15)")
    parser.add_argument("--connectivity_enabled", action="store_true",
                        help="Rescue plan R4: turn on attractive connectivity "
                             "guidance at --connectivity_max_weight (the one "
                             "sampler term that can pull a fragment gap shut)")
    parser.add_argument("--connectivity_weight", type=float, default=0.0,
                        help="Peak attractive guidance weight w on the clean "
                             "x0 estimate (0 = off; overrides "
                             "--connectivity_max_weight when > 0)")
    parser.add_argument("--connectivity_max_weight", type=float, default=0.5,
                        help="Peak ramp weight used with --connectivity_enabled")
    parser.add_argument("--connectivity_start_frac", type=float, default=0.7,
                        help="Fraction of the reverse trajectory elapsed when "
                             "the attractive ramp starts (same contract as the "
                             "steric/valence ramps; 0.7 = last 30%%)")
    parser.add_argument("--connectivity_window_pad", type=float, default=0.5,
                        help="Extra reach (Å) beyond the eval bond window; the "
                             "plan measured inter-fragment gaps 0.1-0.5 Å "
                             "outside it (default: 0.5)")
    parser.add_argument("--connectivity_k_att", type=float, default=1.0,
                        help="Force constant of the attractive energy")
    parser.add_argument("--connectivity_max_step", type=float, default=0.15,
                        help="Max per-atom attractive displacement in Å per "
                             "guided step (default: 0.15)")
    args = parser.parse_args()

    # Plan Step 1.2 test matrix: --steric_enabled is the convenience switch
    # (peak weight = --steric_max_weight) while an explicit --steric_weight
    # always wins, so both spellings of the same experiment agree.
    args.steric_weight = max(float(args.steric_weight), 0.0)
    if args.steric_weight <= 0.0 and args.steric_enabled:
        args.steric_weight = float(args.steric_max_weight)

    args.valence_weight = max(float(args.valence_weight), 0.0)
    if args.valence_weight <= 0.0 and args.valence_enabled:
        args.valence_weight = 0.5

    args.connectivity_weight = max(float(args.connectivity_weight), 0.0)
    if args.connectivity_weight <= 0.0 and args.connectivity_enabled:
        args.connectivity_weight = float(args.connectivity_max_weight)

    # Seed torch's GLOBAL generator — the sampler's randn noise draws come
    # from it, while --seed previously only reached the numpy atom-count rng.
    # Without this, every run generated different molecules regardless of
    # --seed, which fabricated the false "reproducibility band" recorded in
    # progress.md §15 item 11 and made the plan's seed-median requirement
    # unrunnable. Same seed + same flags => byte-identical sampling.
    torch.manual_seed(args.seed)

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

    # Schedule + coordinate normalization come from the checkpoint's own
    # diffusion block (``from_config``), so a radius-normalized checkpoint is
    # sampled in the same coordinate space it was trained in.
    coord_ddpm = CenteredDDPM.from_config(diff_cfg, device=device)
    if coord_ddpm.radius_norm != "none":
        print(f"Coordinate normalization: {coord_ddpm.radius_norm} "
              f"(R(N) = {coord_ddpm.radius_coeff:.3f} * N^(1/3) Å)")
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
    train_mols, train_smiles, ref_info = _reference_mols_and_smiles(full_dataset)
    if ref_info["coverage"] < 0.9:
        msg = (f"Reference set covers only "
               f"{100.0 * ref_info['coverage']:.1f}% of "
               f"{ref_info['requested']} dataset molecules "
               f"({ref_info['n']} usable): Novelty/SNN/ScaffCov would be "
               f"measured against a fragment of the data (fix.md F1)")
        if ref_info["n_dataset_smiles"] > 0:
            raise SystemExit("ERROR: " + msg)
        print("WARNING: " + msg)

    # Raw QED/LogP/TPSA/MW — same keys/units ``extract_cond`` feeds the model
    # at training time (src/fed/trainer.py). Accumulated separately so guided
    # generation can condition on the training "average molecule" profile.
    prop_sum = torch.zeros(4)
    prop_n = 0
    for data in full_dataset:
        props = [getattr(data, k, None) for k in ("qed", "logp", "tpsa", "mw")]
        if all(p is not None for p in props):
            prop_sum += torch.tensor(
                [float(p.view(-1)[0]) for p in props], dtype=torch.float32
            )
            prop_n += 1
    properties_mean = (prop_sum / prop_n) if prop_n else None

    ref_conn = connectivity(train_mols)
    reference = {
        **ref_info,
        "dataset": data_name,
        "BondsPerAtom": ref_conn.get("BondsPerAtom"),
        "BondsPerMol": ref_conn.get("BondsPerMol"),
        "MeanAtoms": ref_conn.get("MeanAtoms"),
        "FragmentsPerMol": ref_conn.get("FragmentsPerMol"),
    }
    print(f"Reference set: {ref_info['n']} / {ref_info['requested']} molecules "
          f"({100.0 * ref_info['coverage']:.1f}%, "
          f"source={ref_info['source']}), "
          f"{len(train_smiles)} unique SMILES")
    print("Reference geometry: "
          + ("bonds/atom "
             f"{reference['BondsPerAtom']:.3f}, atoms "
             f"{reference['MeanAtoms']:.1f}, fragments "
             f"{reference['FragmentsPerMol']:.2f}"
             if reference["BondsPerAtom"] is not None
             else "not measurable"))

    bbb_classifier, oracle_meta = _load_oracle(args, ckpt_cfg, cfg, device)

    histogram = _build_atom_count_histogram(full_dataset)

    cond_enabled = bool(
        (ckpt_cfg.get("conditioning") or {}).get("enabled", False)
    )
    if cond_enabled:
        print(f"Conditioning enabled — guiding toward BBB class "
              f"{args.target_bbb} with w={args.guidance_scale}")
        if properties_mean is None:
            print("  WARNING: no training property means found — "
                  "falling back to zeros (out-of-distribution)")
        else:
            print("  Conditioned on training-mean profile (raw units): "
                  f"qed={properties_mean[0]:.3f} logp={properties_mean[1]:.3f} "
                  f"tpsa={properties_mean[2]:.3f} mw={properties_mean[3]:.3f}")

    seeds = [int(s) for s in (args.seeds or [args.seed])]
    results: list[dict] = []
    first_mols: list = []
    first_relaxed = None
    for seed in seeds:
        payload, mols, relaxed = _evaluate_seed(
            args, seed=seed, model=model, coord_ddpm=coord_ddpm,
            type_ddpm=type_ddpm, histogram=histogram, device=device,
            train_smiles=train_smiles, train_mols=train_mols,
            reference=reference, bbb_classifier=bbb_classifier,
            oracle_meta=oracle_meta, full_dataset=full_dataset,
            cond_enabled=cond_enabled, properties_mean=properties_mean,
        )
        results.append(payload)
        if len(results) == 1:
            first_mols, first_relaxed = mols, relaxed
        if args.output_dir and len(seeds) > 1:
            _save_eval_outputs(
                str(Path(args.output_dir) / f"seed_{seed}"), mols, payload,
                relaxed_mols=relaxed,
            )

    if args.output_dir:
        final = (results[0] if len(results) == 1
                 else _aggregate_payloads(results, seeds))
        _save_eval_outputs(args.output_dir, first_mols, final,
                           relaxed_mols=first_relaxed)
        if len(results) > 1:
            print(f"Aggregate (median over seeds {seeds}) written to "
                  f"{Path(args.output_dir) / 'metrics.json'}")


def _gate_reference_mols(full_dataset, train_smiles: set[str]):
    """Correct-bond-order reference molecules for the gate's stage 4.

    Prefers the dataset's original ``data.smiles`` (BBBP/B3DB carry it):
    re-parsing SMILES restores proper bond orders, so the Morgan
    fingerprints driving the applicability-domain check compare real
    chemistry. The fallback path (QM9, no SMILES column) re-parses the
    canonical SMILES already collected during eval — also correct bond
    orders, unlike the distance-built single-bond graphs themselves.
    """
    refs: list[Chem.Mol] = []
    for data in full_dataset:
        smi = getattr(data, "smiles", None)
        if smi:
            m = Chem.MolFromSmiles(str(smi))
            if m is not None:
                refs.append(m)
    if not refs:
        for smi in sorted(train_smiles):
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                refs.append(m)
    return refs


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
    payload: dict,
    relaxed_mols: list | None = None,
) -> None:
    """Persist eval results: metrics.json (schema v2), smiles.txt, SDFs.

    fix.md F8: no stored ``latex_row`` — rows are rendered from
    ``METRIC_ORDER`` by whoever draws the table, so a metric patched after
    the fact can never disagree with its own row.
    """
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
    """Convert a dataset sample to an RDKit Mol with correct bond orders.

    Prefers the dataset's own SMILES (BBBP/B3DB carry it). The coordinate
    fallback uses the eval's covalent-radii builder — the old
    ``dist < 2.0`` rule bonded every close pair and mapped unknown elements
    to carbon, so only 20 of 1628 BBBP molecules survived it (fix.md F1).
    """
    smi = getattr(data, "smiles", None)
    if smi:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            return mol
    try:
        return coords_and_types_to_mol(
            data.pos.numpy(), data.z.numpy().astype(int))
    except Exception:
        return None


if __name__ == "__main__":
    main()
