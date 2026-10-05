"""Run the ablation study (implementation.md §5.4).

Usage:
    .venv/bin/python scripts/run_ablation.py --dry-run
    .venv/bin/python scripts/run_ablation.py --only no_conditioning,no_diversity_loss
    .venv/bin/python scripts/run_ablation.py --seeds 42,123,456

Ablation YAML format (configs/sweeps/ablation.yaml):
    base_config: configs/central_bbb.yaml
    ablations:
      full_model: {}
      no_conditioning: {conditioning.enabled: false, model.cond_dim: 0}
      no_valence_penalty: {training.valence_loss_weight: 0.0}

Each arm = base_config + dotted-key overrides, trained and evaluated per
seed exactly like run_sweep.py. Results land in
outputs/ablation_<arm>/seed_<s>/results.json. Arms whose overrides target
the ``fed.*`` section are skipped when base_config has no ``fed`` section
(their own YAML comments mark them "fed runs only").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_lib import (  # noqa: E402
    REPO_ROOT, collect_results, eval_checkpoint, load_yaml, parse_seeds,
    python, run_cmd, write_run_config,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ablation study runner")
    p.add_argument("--config", default="configs/sweeps/ablation.yaml",
                   help="Ablation YAML (default: configs/sweeps/ablation.yaml)")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated seed override (default: 42)")
    p.add_argument("--only", default=None,
                   help="Comma-separated arm names to run (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print arms + commands, execute nothing")
    p.add_argument("--eval_samples", type=int, default=1000)
    p.add_argument("--ddim_steps", type=int, default=200)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--step_schedule", default="quadratic")
    p.add_argument("--relax", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()

    spec = load_yaml(args.config)
    base_path = spec.get("base_config")
    if not base_path:
        print(f"ERROR: {args.config} has no base_config")
        return 2
    base_cfg = load_yaml(base_path)
    is_fed = isinstance(base_cfg.get("fed"), dict)
    train_script = "fed_train.py" if is_fed else "train.py"
    seeds = parse_seeds(args.seeds, spec.get("seeds") or [42])
    arms: dict[str, dict] = spec.get("ablations") or {}
    if not arms:
        print(f"ERROR: {args.config} has no ablations mapping")
        return 2
    only = {a.strip() for a in args.only.split(",")} if args.only else None

    print(f"ablation base={base_path} trainer={train_script} seeds={seeds}")
    if args.dry_run:
        print("--dry-run: printing commands only\n")

    failures = skipped = planned = 0
    for arm_name, overrides in arms.items():
        if only is not None and arm_name not in only:
            continue
        if not is_fed and any(k.startswith("fed.") for k in overrides):
            print(f"[{arm_name}] skipped — fed-only overrides on a central "
                  f"base config ({base_path})")
            skipped += 1
            continue
        for seed in seeds:
            planned += 1
            run_dir = REPO_ROOT / "outputs" / f"ablation_{arm_name}" / f"seed_{seed}"
            tag = f"[{arm_name} seed={seed}]"
            print(f"{tag} overrides={overrides or '{} (baseline)'}")
            if args.dry_run:
                # No filesystem side effects in dry-run: point at the config
                # path that a real run would materialize.
                cfg_path = run_dir / "config.yaml"
            else:
                cfg_path = write_run_config(base_cfg, overrides, seed, run_dir,
                                            is_fed)
            ckpt = eval_checkpoint(cfg_path, is_fed)

            train_exit = 0
            if args.skip_train:
                print(f"{tag} --skip_train: using {ckpt}")
            else:
                train_exit = run_cmd(
                    [python(), REPO_ROOT / train_script, "--config", cfg_path],
                    run_dir / "train.log", args.dry_run)
                if train_exit != 0:
                    failures += 1
                    print(f"{tag} TRAIN FAILED (exit {train_exit}) — see "
                          f"{run_dir / 'train.log'}")

            eval_exit = 0
            eval_dir = run_dir / "eval"
            if train_exit == 0 and (args.dry_run or ckpt.exists()):
                eval_cmd = [
                    python(), REPO_ROOT / "generate_and_eval.py",
                    "--checkpoint", ckpt, "--config", cfg_path,
                    "--num_samples", args.eval_samples, "--seed", seed,
                    "--ddim_steps", args.ddim_steps, "--eta", args.eta,
                    "--step_schedule", args.step_schedule,
                    "--output_dir", eval_dir,
                ]
                if args.relax:
                    eval_cmd.append("--relax")
                eval_exit = run_cmd(eval_cmd, run_dir / "eval.log",
                                    args.dry_run)
                if eval_exit != 0:
                    failures += 1
                    print(f"{tag} EVAL FAILED (exit {eval_exit})")
            elif not args.dry_run:
                print(f"{tag} missing checkpoint {ckpt} — skipping eval")
                failures += 1

            if args.dry_run:
                continue
            results = {
                "ablation": arm_name,
                "overrides": overrides,
                "seed": seed,
                "config": str(cfg_path.relative_to(REPO_ROOT)),
                "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
                "train_exit": train_exit,
                "eval_exit": eval_exit,
                "status": "ok" if train_exit == eval_exit == 0 else "failed",
                **collect_results(run_dir, eval_dir,
                                  ["Validity", "ConnectedValidity", "BBB%"],
                                  is_fed),
            }
            (run_dir / "results.json").write_text(json.dumps(results, indent=2))
            print(f"{tag} status={results['status']} -> "
                  f"{run_dir / 'results.json'}")

    if args.dry_run:
        print(f"\ndry-run complete: {planned} run(s) planned, "
              f"{skipped} arm(s) skipped, nothing executed")
        return 0
    if failures:
        print(f"\n{failures} failure(s) — see logs above")
        return 1
    print(f"\nall {planned} run(s) completed ({skipped} arm(s) skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
