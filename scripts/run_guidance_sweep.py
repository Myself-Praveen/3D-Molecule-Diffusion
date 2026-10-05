"""Guidance-scale sweep for conditioned BBB generation (Phase 5 / §6.4).

Runs generate_and_eval.py across a grid of classifier-free guidance weights
``w`` against the trained conditioned checkpoint, writing one directory per
weight under ``outputs/sweep_guidance/`` with ``metrics.json`` (from the
eval) plus a ``results.json`` in the shape
``scripts/generate_paper_figures.py`` expects for Fig. 7::

    {"combo": {"guidance_scale": w}, "metrics": {...}, "status": "ok"}

The BBB oracle comes from the checkpoint config's
``bbb_classifier.checkpoint`` (auto-loaded by generate_and_eval.py), so
BBB% is populated without an explicit --bbb_oracle flag.

Usage:
    .venv/bin/python scripts/run_guidance_sweep.py \
        --checkpoint checkpoints/bbb/best.pt --config configs/central_bbb.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_lib import REPO_ROOT, combo_slug, python, run_cmd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Guidance-scale sweep (BBB%)")
    p.add_argument("--checkpoint", default="checkpoints/bbb/best.pt")
    p.add_argument("--config", default="configs/central_bbb.yaml")
    p.add_argument("--scales", default="0,0.5,1,2,4,8",
                   help="Comma-separated guidance weights w")
    p.add_argument("--target_bbb", type=int, default=1, choices=[0, 1])
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--ddim_steps", type=int, default=200)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--step_schedule", default="quadratic")
    p.add_argument("--outdir", default="outputs/sweep_guidance")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    ckpt = REPO_ROOT / args.checkpoint
    if not args.dry_run and not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}")
        return 2
    outdir = REPO_ROOT / args.outdir
    scales = [float(s) for s in args.scales.replace(" ", "").split(",") if s]
    print(f"guidance sweep: w in {scales}  ckpt={args.checkpoint}  "
          f"n={args.num_samples} target_bbb={args.target_bbb}")

    failures = 0
    for w in scales:
        run_dir = outdir / combo_slug({"guidance_scale": w})
        run_dir.mkdir(parents=True, exist_ok=True)
        eval_dir = run_dir
        cmd = [
            python(), REPO_ROOT / "generate_and_eval.py",
            "--checkpoint", ckpt,
            "--config", REPO_ROOT / args.config,
            "--num_samples", str(args.num_samples),
            "--ddim_steps", str(args.ddim_steps),
            "--eta", str(args.eta),
            "--step_schedule", args.step_schedule,
            "--guidance_scale", str(w),
            "--target_bbb", str(args.target_bbb),
            "--output_dir", eval_dir,
        ]
        print(f"[w={w}] {' '.join(str(c) for c in cmd)}")
        exit_code = run_cmd(cmd, run_dir / "eval.log", args.dry_run)
        if args.dry_run:
            continue
        if exit_code != 0:
            failures += 1
            print(f"[w={w}] EVAL FAILED (exit {exit_code}) — "
                  f"see {run_dir / 'eval.log'}")
        metrics = {}
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text()).get("metrics", {})
            except (json.JSONDecodeError, OSError):
                metrics = {}
        results = {
            "sweep": "guidance",
            "combo": {"guidance_scale": w},
            "checkpoint": args.checkpoint,
            "config": args.config,
            "eval_exit": exit_code,
            "status": "ok" if exit_code == 0 else "failed",
            "metrics": metrics,
        }
        (run_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"[w={w}] status={results['status']} "
              f"BBB%={metrics.get('BBB%')} Validity={metrics.get('Validity')}")

    if args.dry_run:
        print(f"\ndry-run complete: {len(scales)} run(s) planned, "
              "nothing executed")
        return 0
    if failures:
        print(f"\n{failures} failure(s) — see logs above")
        return 1
    print(f"\nall {len(scales)} run(s) completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
