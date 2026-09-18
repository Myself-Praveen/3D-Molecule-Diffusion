"""One-command training status check.

Usage:
    .venv/bin/python scripts/check_status.py
    .venv/bin/python scripts/check_status.py --checkpoint checkpoints/best.pt

Prints central + federated progress without error-prone `python -c` one-liners
(note: `for`/`with` blocks cannot follow `;` in `python -c`, which is what
caused the SyntaxError).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def show_central(ckpt_dir: Path) -> None:
    for name in ("best.pt", "last.pt"):
        path = ckpt_dir / name
        if not path.exists():
            print(f"  {name}: MISSING")
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        info = {
            k: (round(float(v), 4) if isinstance(v, float) else v)
            for k, v in state.items()
            if k in ("epoch", "val_loss", "best_val_loss", "epochs_no_improve")
        }
        print(f"  {name}: {info}")


def show_fed(output_dir: Path) -> None:
    hist_path = output_dir / "history.json"
    last_path = output_dir / "last_global.pt"
    if not hist_path.exists():
        print("  history.json: MISSING (not started)")
        return
    history = json.loads(hist_path.read_text())
    print(f"  rounds done: {len(history)}", end="")
    if history:
        first, last = history[0], history[-1]
        print(
            f" | server_loss {first['server_loss']:.4f} -> {last['server_loss']:.4f}",
            end="",
        )
        vals = [h["val"]["loss"] for h in history if h.get("val")]
        if vals:
            print(f" | val {vals[0]:.4f} -> {vals[-1]:.4f}", end="")
    print()
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        print(f"  last_global round: {state.get('round')}")
    else:
        print("  last_global.pt: MISSING")


def main() -> None:
    parser = argparse.ArgumentParser(description="Training status check")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--fed_dirs", nargs="*", default=["outputs/fed_iid", "outputs/fed_niid"])
    args = parser.parse_args()

    print("[central]")
    show_central(Path(args.ckpt_dir))
    for fed_dir in args.fed_dirs:
        print(f"[{fed_dir}]")
        show_fed(Path(fed_dir))


if __name__ == "__main__":
    main()
