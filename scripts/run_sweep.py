"""Run a grid of experiments with multiple seeds (implementation.md §5.3).

Usage:
    .venv/bin/python scripts/run_sweep.py --sweep configs/sweeps/sweep_K.yaml
    .venv/bin/python scripts/run_sweep.py --sweep configs/sweeps/sweep_mu.yaml \\
        --seeds 42,123,456 --dry-run
    .venv/bin/python scripts/run_sweep.py --sweep configs/sweeps/sweep_lambda.yaml \\
        --eval_samples 200 --relax

Sweep YAML format:
    base_config: configs/fed_bbb_niid.yaml
    grid:
      fed.num_clients: [1, 2, 4, 7]
      fed.mode: [iid, niid]
    seeds: [42, 123, 456]
    metrics: [Validity, ConnectedValidity, BBB%, ...]

For each combination:
    1. Create a run config by overriding base_config fields (dotted keys),
       plus seed and an isolated checkpoint/output dir (sweep_lib).
    2. Run fed_train.py (or train.py for central base configs).
    3. Run generate_and_eval.py with --num_samples (default 1000).
    4. Save results to outputs/sweep_<name>/<combo>/seed_<s>/results.json
       (config.yaml, train/eval logs, eval/ metrics.json sit alongside).

--dry-run prints the full grid and exact commands without executing
anything (verification item §5.5).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_lib import (  # noqa: E402
    REPO_ROOT, collect_results, combo_slug, eval_checkpoint, expand_grid,
    load_yaml, parse_seeds, python, run_cmd, write_run_config,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grid-search experiment runner")
    p.add_argument("--sweep", required=True,
                   help="Sweep YAML (e.g. configs/sweeps/sweep_K.yaml)")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated seed override (default: YAML seeds)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the full grid + commands, execute nothing")
    p.add_argument("--eval_samples", type=int, default=1000,
                   help="generate_and_eval.py sample count (default 1000)")
    p.add_argument("--ddim_steps", type=int, default=200)
    p.add_argument("--eta", type=float, default=1.0,
                   help="DDIM eta (V4 adopted protocol: 1.0)")
    p.add_argument("--step_schedule", default="quadratic")
    p.add_argument("--relax", action="store_true",
                   help="Also report MMFF94/UFF-relaxed metrics")
    p.add_argument("--skip_train", action="store_true",
                   help="Only (re)run evaluation from existing checkpoints")
    return p


def main() -> int:
    args = build_parser().parse_args()

    sweep = load_yaml(args.sweep)
    base_path = sweep.get("base_config")
    if not base_path:
        print(f"ERROR: {args.sweep} has no base_config")
        return 2
    base_cfg = load_yaml(base_path)
    is_fed = isinstance(base_cfg.get("fed"), dict)
    train_script = "fed_train.py" if is_fed else "train.py"
    sweep_name = sweep.get("sweep") or Path(args.sweep).stem
    seeds = parse_seeds(args.seeds, sweep.get("seeds") or [42])
    combos = expand_grid(sweep.get("grid") or {})
    requested = sweep.get("metrics") or []

    print(f"sweep={sweep_name}  base={base_path}  trainer={train_script}")
    print(f"grid size={len(combos)}  seeds={seeds}  "
          f"runs={len(combos) * len(seeds)}")
    if args.dry_run:
        print("--dry-run: printing commands only\n")

    failures = 0
    for combo in combos:
        slug = combo_slug(combo)
        for seed in seeds:
            run_dir = REPO_ROOT / "outputs" / f"sweep_{sweep_name}" / slug / f"seed_{seed}"
            tag = f"[{slug} seed={seed}]"
            print(f"{tag} combo={combo}")
            if args.dry_run:
                # No filesystem side effects in dry-run: point at the config
                # path that a real run would materialize.
                cfg_path = run_dir / "config.yaml"
            else:
                cfg_path = write_run_config(base_cfg, combo, seed, run_dir,
                                            is_fed)
            ckpt = eval_checkpoint(cfg_path, is_fed)

            train_exit = 0
            if args.skip_train:
                print(f"{tag} --skip_train: using {ckpt}")
            else:
                train_cmd = [python(), REPO_ROOT / train_script,
                             "--config", cfg_path]
                train_exit = run_cmd(train_cmd, run_dir / "train.log",
                                     args.dry_run)
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
                    print(f"{tag} EVAL FAILED (exit {eval_exit}) — see "
                          f"{run_dir / 'eval.log'}")
            elif not args.dry_run:
                print(f"{tag} missing checkpoint {ckpt} — skipping eval")
                failures += 1

            if args.dry_run:
                continue
            results = {
                "sweep": sweep_name,
                "combo": combo,
                "seed": seed,
                "config": str(cfg_path.relative_to(REPO_ROOT)),
                "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
                "train_exit": train_exit,
                "eval_exit": eval_exit,
                "status": "ok" if train_exit == eval_exit == 0 else "failed",
                **collect_results(run_dir, eval_dir, requested, is_fed),
            }
            (run_dir / "results.json").write_text(json.dumps(results, indent=2))
            print(f"{tag} status={results['status']} -> "
                  f"{run_dir / 'results.json'}")

    if args.dry_run:
        print(f"\ndry-run complete: {len(combos) * len(seeds)} run(s) planned, "
              "nothing executed")
        return 0
    if failures:
        print(f"\n{failures} failure(s) — see logs above")
        return 1
    print(f"\nall {len(combos) * len(seeds)} run(s) completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
