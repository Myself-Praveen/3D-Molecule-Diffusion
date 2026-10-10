"""How large is the lambda2 valence penalty on *correct* geometry?

The penalty in ``src/objectives.py`` is one-sided::

    violation = relu(1.2 * bond_count - expected_valence)     # squared, mean

so it can only ever push atoms APART. Its demand side is the model's own
predicted type distribution::

    expected_valence = sum_k p_k * VALENCE[type_to_z[k]]

With raw-Z indexing (``TYPE_TO_Z = {i: i}``, correct for both QM9 and BBBP)
``max_valence`` is ``[0, 1, 0, 0, 0, 0, 4, 3, 2, 1]`` — classes 2-5 (Z=2,3,4,5)
are *dead* but still carry softmax mass early in training, so they dilute the
carbon/nitrogen/oxygen demand toward ~1.1 bonds per atom.

This probe measures the penalty on the molecules we actually want (real BBBP
geometries) as a function of how confident the type head is, and compares the
current demand side against a renormalized-over-live-classes variant::

    p_live = p * (max_valence > 0);  p_live /= p_live.sum(-1, keepdim=True)

The train_pos MSE for the BBB run is ~0.46, so a penalty of that order is
competing with the entire coordinate reconstruction loss.

Usage:
    .venv/bin/python scripts/valence_scale_probe.py
    .venv/bin/python scripts/valence_scale_probe.py --n-mols 100 --temperature 1.0
"""

from __future__ import annotations

import argparse
import math
import statistics as st
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.fed.trainer import TYPE_TO_Z  # noqa: E402
from src.objectives import VALENCE, full_pair_index  # noqa: E402

BOND_CUTOFF = 1.8
SOFTNESS = 0.25
BOND_ORDER = 1.2


def valence_table(num_types: int) -> torch.Tensor:
    return torch.tensor(
        [VALENCE.get(TYPE_TO_Z.get(k, 0), 0) for k in range(num_types)],
        dtype=torch.float32,
    )


def penalty(pos: torch.Tensor, probs: torch.Tensor, batch: torch.Tensor,
            max_valence: torch.Tensor) -> torch.Tensor:
    """Mirror of ``soft_valence_penalty``'s math, with the demand side swappable."""
    pair_index = full_pair_index(batch)
    row, col = pair_index
    dist = (pos[row] - pos[col]).norm(dim=-1)
    adj = torch.sigmoid((BOND_CUTOFF - dist) / SOFTNESS)
    atom_valence = (probs * max_valence.unsqueeze(0)).sum(dim=-1)
    bond_count = torch.zeros_like(atom_valence)
    bond_count.index_add_(0, row, adj)
    bond_count.index_add_(0, col, adj)
    bond_count = bond_count / 2.0
    return torch.relu(BOND_ORDER * bond_count - atom_valence).square().mean()


def temperature_for_ce(ce: float, num_types: int) -> float:
    """Logit scale giving cross-entropy ``ce`` on an otherwise one-hot target."""
    p = math.exp(-ce)
    return math.log(p * (num_types - 1) / (1.0 - p))


def smoothed_onehot(z: torch.Tensor, num_types: int, logit_scale: float) -> torch.Tensor:
    oh = torch.zeros(z.numel(), num_types)
    oh[torch.arange(z.numel()), z.clamp(max=num_types - 1)] = 1.0
    if logit_scale <= 0.0:
        return torch.full_like(oh, 1.0 / num_types)
    return torch.softmax(oh * logit_scale, dim=-1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-mols", type=int, default=48)
    p.add_argument("--dataset", default="data/bbbp")
    p.add_argument("--num-types", type=int, default=10)
    p.add_argument("--train-pos-mse", type=float, default=0.46,
                   help="central_bbb train_pos MSE, for scale comparison")
    args = p.parse_args()

    from src.dataset_bbb import load_bbbp

    train, _, _ = load_bbbp(args.dataset, seed=42)
    mols = train[: args.n_mols]
    max_valence = valence_table(args.num_types)
    live = max_valence > 0
    print(f"max_valence per class : {max_valence.tolist()}")
    print(f"live classes          : {live.nonzero().flatten().tolist()}  "
          f"dead (diluting): {(~live).nonzero().flatten().tolist()}")
    print(f"molecules             : {len(mols)} real BBBP\n")

    print(f"{'type head':34s} {'current':>9s} {'renorm-live':>12s}")
    cases = [("one-hot true (floor)", None)]
    for ce in (0.82, 1.0, 1.5, 1.82):
        cases.append((f"CE={ce:.2f}  (a={temperature_for_ce(ce, args.num_types):.3f})", ce))
    cases.append(("uniform", 0.0))

    for label, ce in cases:
        cur, fix = [], []
        for d in mols:
            n = d.z.numel()
            pos = d.pos.float()
            batch = torch.zeros(n, dtype=torch.long)
            if ce is None:
                probs = smoothed_onehot(d.z.long(), args.num_types, -1.0)
                probs = torch.zeros(n, args.num_types)
                probs[torch.arange(n), d.z.long().clamp(max=args.num_types - 1)] = 1.0
            else:
                probs = smoothed_onehot(d.z.long(), args.num_types,
                                        temperature_for_ce(ce, args.num_types)
                                        if ce > 0 else 0.0)
            probs_live = probs * live
            probs_live = probs_live / probs_live.sum(-1, keepdim=True)
            cur.append(float(penalty(pos, probs, batch, max_valence)))
            fix.append(float(penalty(pos, probs_live, batch, max_valence)))
        print(f"{label:34s} {st.mean(cur):9.4f} {st.mean(fix):12.4f}")

    print(f"\nscale reference: train_pos MSE = {args.train_pos_mse:.2f}  "
          f"(lambda2 weight = 1.0 on central_bbb)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
