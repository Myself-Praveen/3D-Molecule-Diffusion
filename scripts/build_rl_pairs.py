"""Generate preference pairs for DPO fine-tuning (plan Step 2.2 driver).

Both rollouts in a pair are drawn from the *same* torch seed, so the atom
counts and the initial noise are identical and the only difference inside a
pair is the classifier-free guidance scale (``--high_w`` vs ``--low_w``) — the
seed-matched construction the corrected plan asks for. Every sample keeps its
generated ``(pos, z)`` even when RDKit molecule construction fails, so invalid
clumps survive as the rejected side instead of being filtered away.

Usage:
    .venv/bin/python scripts/build_rl_pairs.py \
        --checkpoint checkpoints/central_v5_bbb/best.pt \
        --config configs/central_v5_bbb.yaml \
        --num_pairs 400 --high_w 4.0 --low_w 0.0 --ddim_steps 50 \
        --strategy seed_matched --out outputs/rl_pairs/pairs.pt

Strategies:
    seed_matched      same-seed rollouts at two guidance scales (default)
    valid_vs_invalid  one rollout, best valid molecules vs invalid clumps

The output is the ``rl.pair_file`` consumed by ``train.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from generate_and_eval import (  # noqa: E402
    _build_atom_count_histogram,
    _load_model_weights,
    _sample_atom_counts,
    build_target_cond,
)
from src.models.diffusion import CenteredDDPM, TypeDDPM  # noqa: E402
from src.models.egnn import EquivariantGenerator  # noqa: E402
from src.sampling import sample_molecules  # noqa: E402
from src.rl.dpo import save_preference_pairs  # noqa: E402
from src.rl.preference_pairs import (  # noqa: E402
    Sample,
    generate_seed_matched_pairs,
    generate_valid_vs_invalid_pairs,
    pair_statistics,
    sample_reward,
)
from src.rl.rewards import REFERENCE_BONDS_PER_MOL, load_bbb_oracle  # noqa: E402
from src.utils.evaluation import coords_and_types_to_mol  # noqa: E402
from src.guidance import (
    DEFAULT_VALENCE_COVALENT_FACTOR,
    DEFAULT_VALENCE_K_VAL,
    DEFAULT_VALENCE_MAX_STEP,
    DEFAULT_VALENCE_TAU,
    valence_guidance,
    valence_guidance_weight,
)  # noqa: E402


def _load(args) -> tuple[EquivariantGenerator, dict, dict, CenteredDDPM, TypeDDPM]:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config", cfg)

    model_cfg, diff_cfg = ckpt_cfg["model"], ckpt_cfg["diffusion"]
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
    objective = str(diff_cfg.get("objective", "eps")).lower()
    model.objective = objective
    model.load_state_dict(_load_model_weights(ckpt, model))
    model.eval()

    coord_ddpm = CenteredDDPM.from_config(diff_cfg, device=device)
    type_ddpm = TypeDDPM(
        num_steps=diff_cfg["num_steps"], beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"], device=device,
        schedule=diff_cfg.get("schedule", "linear"),
    )
    return model, ckpt_cfg, cfg, coord_ddpm, type_ddpm


def _properties_mean(cfg: dict, device) -> torch.Tensor | None:
    """Training-set raw QED/LogP/TPSA/MW means (the conditioning profile)."""
    data_cfg = cfg["data"]
    if str(data_cfg.get("dataset", "qm9")).lower() != "bbbp":
        return None
    from src.dataset_bbb import load_bbbp

    dataset, _, _ = load_bbbp(root=data_cfg["root"])
    acc, n = torch.zeros(4), 0
    for data in dataset:
        vals = [getattr(data, k, None) for k in ("qed", "logp", "tpsa", "mw")]
        if all(v is not None for v in vals):
            acc += torch.tensor([float(v.view(-1)[0]) for v in vals])
            n += 1
    return acc / max(n, 1) if n else None


def _samples(pos, z, offset: int, counts: list[int]) -> list[Sample]:
    out: list[Sample] = []
    for count in counts:
        p = pos[offset:offset + count]
        t = z[offset:offset + count]
        mol = coords_and_types_to_mol(
            p.numpy(), t.numpy().astype(int)
        )
        out.append(Sample(pos=p.clone(), z=t.clone(), mol=mol))
        offset += count
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/central_v5_bbb.yaml")
    parser.add_argument("--num_pairs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--step_schedule", default="quadratic",
                        choices=["linear", "quadratic"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--high_w", type=float, default=4.0)
    parser.add_argument("--low_w", type=float, default=0.0)
    parser.add_argument("--strategy", default="seed_matched",
                        choices=["seed_matched", "valid_vs_invalid"])
    parser.add_argument("--min_reward_gap", type=float, default=0.1)
    parser.add_argument("--target_bonds", type=float, default=None,
                        help="Bond-count target for the reward "
                             "(default: REFERENCE_BONDS_PER_MOL, the "
                             "H-inclusive measure for BBBP)")
    parser.add_argument("--no_oracle", action="store_true",
                        help="Skip the BBB term (no models/bbb_oracle.pt)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="outputs/rl_pairs/pairs.pt")
    parser.add_argument("--valence_weight", type=float, default=0.0,
                        help="Peak test-time valence guidance weight on the clean x0 estimate for both rollouts (0 = off; overrides --valence_enabled)")
    parser.add_argument("--valence_enabled", action="store_true",
                        help="Turn on test-time valence guidance for pair rollouts at the default peak weight 0.5")
    parser.add_argument("--valence_start_frac", type=float, default=0.7)
    parser.add_argument("--valence_tau", type=float, default=0.05)
    parser.add_argument("--valence_covalent_factor", type=float, default=1.10)
    parser.add_argument("--valence_k_val", type=float, default=1.0)
    parser.add_argument("--valence_max_step", type=float, default=0.15)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    target_bonds = (REFERENCE_BONDS_PER_MOL if args.target_bonds is None
                    else float(args.target_bonds))
    model, ckpt_cfg, cfg, coord_ddpm, type_ddpm = _load(args)
    device = next(model.parameters()).device
    cond_enabled = bool((ckpt_cfg.get("conditioning") or {}).get("enabled", False))
    oracle = None if args.no_oracle else load_bbb_oracle(
        (ckpt_cfg.get("bbb_classifier") or {}).get(
            "checkpoint", "models/bbb_oracle.pt"), device
    )
    if oracle is None:
        print("WARNING: no BBB oracle — the bbb reward term is 0 for every sample")
    properties_mean = (_properties_mean(cfg, device)
                       if cond_enabled else None)

    args.valence_weight = float(args.valence_weight)
    if args.valence_weight <= 0.0 and args.valence_enabled:
        args.valence_weight = 0.5
    args.valence_start_frac = float(args.valence_start_frac)
    args.valence_tau = float(args.valence_tau)
    args.valence_covalent_factor = float(args.valence_covalent_factor)
    args.valence_k_val = float(args.valence_k_val)
    args.valence_max_step = float(args.valence_max_step)
    rng = np.random.default_rng(args.seed)
    from src.dataset_bbb import load_bbbp

    dataset, _, _ = load_bbbp(root=cfg["data"]["root"])
    histogram = _build_atom_count_histogram(dataset)

    def _rollout(counts: list[int], w: float) -> tuple[torch.Tensor, torch.Tensor]:
        cond = None
        if cond_enabled:
            cond = build_target_cond(1, len(counts), device,
                                     properties_mean=properties_mean)
        return sample_molecules(
            model, coord_ddpm, type_ddpm,
            torch.tensor(counts, dtype=torch.long),
            device=device, ddim_steps=args.ddim_steps, eta=args.eta,
            step_schedule=args.step_schedule,
            steric_weight=0.0,
            valence_weight=args.valence_weight,
            valence_start_frac=args.valence_start_frac,
            valence_tau=args.valence_tau,
            valence_covalent_factor=args.valence_covalent_factor,
            valence_k_val=args.valence_k_val,
            valence_max_step=args.valence_max_step,
            valence_enabled=args.valence_enabled,
            cond=cond, guidance_scale=w if cond is not None else 0.0,
        )

    n = int(args.num_pairs)
    counts = _sample_atom_counts(histogram, n, rng)
    all_samples: list[Sample] = []
    high_samples: list[Sample] = []
    low_samples: list[Sample] = []
    for start in range(0, n, args.batch_size):
        chunk = counts[start:start + args.batch_size]
        # Same seed for both rollouts => identical noise and atom counts.
        torch.manual_seed(args.seed + start // args.batch_size)
        pos_h, z_h = _rollout(chunk, args.high_w)
        torch.manual_seed(args.seed + start // args.batch_size)
        pos_l, z_l = _rollout(chunk, args.low_w)
        high_samples += _samples(pos_h, z_h, 0, chunk)
        low_samples += _samples(pos_l, z_l, 0, chunk)
        print(f"  rollout {min(start + len(chunk), n)}/{n}")

    if args.strategy == "seed_matched":
        pairs = generate_seed_matched_pairs(
            high_samples, low_samples, oracle, n_pairs=n,
            min_reward_gap=args.min_reward_gap, target_bonds=target_bonds,
        )
    else:
    elif args.strategy == "valid_vs_invalid":
        pairs = generate_valid_vs_invalid_pairs(
            low_samples + high_samples, oracle, n_pairs=n,
            target_bonds=target_bonds,
        )
    else:
        raise ValueError(
            f"Unknown strategy={args.strategy!r} "
            f"(expected 'seed_matched' or 'valid_vs_invalid')"
        )

    if not pairs:
        print("No pairs satisfied the criteria — nothing written.")
        return 1

    stats = pair_statistics(pairs)
    stats["mean_sample_reward_high_w"] = float(np.mean([
        sample_reward(s, oracle, target_bonds) for s in high_samples
    ]))
    stats["mean_sample_reward_low_w"] = float(np.mean([
        sample_reward(s, oracle, target_bonds) for s in low_samples
    ]))
    stats["valence_weight"] = args.valence_weight
    stats["valence_enabled"] = bool(args.valence_enabled)
    stats["valence_start_frac"] = args.valence_start_frac
    stats["valence_tau"] = args.valence_tau
    stats["valence_covalent_factor"] = args.valence_covalent_factor
    stats["valence_k_val"] = args.valence_k_val
    stats["valence_max_step"] = args.valence_max_step
    save_preference_pairs(args.out, {
        "pairs": [
            {
                "chosen": {"pos": p.chosen.pos, "z": p.chosen.z},
                "rejected": {"pos": p.rejected.pos, "z": p.rejected.z},
                "reward_chosen": p.reward_chosen,
                "reward_rejected": p.reward_rejected,
            }
            for p in pairs
        ],
        "stats": stats,
        "meta": {
            "checkpoint": args.checkpoint,
            "strategy": args.strategy,
            "high_w": args.high_w, "low_w": args.low_w,
            "seed": args.seed, "ddim_steps": args.ddim_steps,
            "target_bonds": target_bonds,
            "oracle_used": oracle is not None,
        },
    })
    print(f"Wrote {len(pairs)} pairs -> {args.out}")
    print("stats:", {k: round(v, 4) for k, v in stats.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
