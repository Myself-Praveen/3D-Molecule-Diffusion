"""Preference-pair generation for DPO fine-tuning (plan Step 2.2).

Two corrections shape this module:

1. **Invalid samples are the rejected side, not garbage.** An invalid clump
   (``mol is None``) is the single most informative negative example for a model
   that produces atom soup, so a sample keeps its generated ``(pos, z)`` even
   when molecule construction failed — those coordinates are exactly what the
   reference model should be told to assign *lower* likelihood to. Dropping
   ``None`` entries (the obvious implementation) deletes the very examples the
   plan wants learned from.
2. **Pairing is seed-matched or validity-matched, never O(N²).** The plan's
   seed-matched construction pairs rollouts from the *same* initial noise under
   two guidance scales (w=4.0 vs w=0.0), so the only difference inside a pair is
   what guidance did — a much cleaner gradient than random all-pairs matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from rdkit import Chem

from src.rl.rewards import (
    INVALID_REWARD,
    composite_reward,
)


@dataclass(frozen=True)
class Sample:
    """One generated molecule *as sampled*, valid or not.

    ``pos`` is ``(N, 3)`` in Ångström and ``z`` the model's type indices (raw
    atomic numbers in this project), so an invalid sample is still fully
    encodable for a diffusion loss. ``mol`` is the RDKit product of
    ``coords_and_types_to_mol`` and is ``None`` when construction failed.
    """

    pos: torch.Tensor
    z: torch.Tensor
    mol: Chem.Mol | None

    @property
    def is_valid(self) -> bool:
        return self.mol is not None


@dataclass(frozen=True)
class PreferencePair:
    """A (chosen, rejected) pair with its composite rewards for diagnostics."""

    chosen: Sample
    rejected: Sample
    reward_chosen: float
    reward_rejected: float


def sample_reward(
    sample: Sample,
    bbb_classifier=None,
    target_bonds: float = 19.0,
    weights: dict[str, float] | None = None,
) -> float:
    """Reward of a :class:`Sample`: invalid samples get ``INVALID_REWARD``."""
    if sample.mol is None:
        return INVALID_REWARD
    return composite_reward(sample.mol, bbb_classifier, target_bonds, weights)


def generate_seed_matched_pairs(
    molecules_high_w: Sequence[Sample],
    molecules_low_w: Sequence[Sample],
    bbb_classifier=None,
    n_pairs: int = 500,
    min_reward_gap: float = 0.1,
    target_bonds: float = 19.0,
) -> list[PreferencePair]:
    """Pair rollouts from the same seed under two guidance scales.

    ``molecules_high_w[i]`` and ``molecules_low_w[i]`` must come from the same
    initial noise (same torch seed, same atom counts) with only ``w`` differing.
    The higher-reward member becomes ``chosen``; invalid members are kept when
    they are the rejected side (they carry ``INVALID_REWARD``), so an invalid
    high-``w`` clump is *not* discarded — it is paired against the valid rollout
    from the same seed.
    """
    if len(molecules_high_w) != len(molecules_low_w):
        raise ValueError(
            f"seed-matched pairing needs equal lengths, got "
            f"{len(molecules_high_w)} vs {len(molecules_low_w)}"
        )
    pairs: list[PreferencePair] = []
    for hi, lo in zip(molecules_high_w, molecules_low_w):
        r_hi = sample_reward(hi, bbb_classifier, target_bonds)
        r_lo = sample_reward(lo, bbb_classifier, target_bonds)
        if abs(r_hi - r_lo) < min_reward_gap:
            continue
        if r_hi >= r_lo:
            chosen, rejected, rc, rr = hi, lo, r_hi, r_lo
        else:
            chosen, rejected, rc, rr = lo, hi, r_lo, r_hi
        # The chosen side must be encodable; the rejected side may be invalid
        # (that is the point — it keeps its coordinates).
        if chosen.mol is None:
            continue
        pairs.append(PreferencePair(chosen, rejected, rc, rr))
        if len(pairs) >= n_pairs:
            break
    return pairs


def generate_valid_vs_invalid_pairs(
    molecules: Sequence[Sample],
    bbb_classifier=None,
    n_pairs: int = 500,
    target_bonds: float = 19.0,
) -> list[PreferencePair]:
    """Pair the best valid molecules against invalid clumps.

    Follows the plan's correction 5 exactly: everything with
    ``composite_reward > 0`` is eligible as chosen (sorted best-first), every
    sample scoring ``<= INVALID_REWARD`` is a rejected candidate, and the two
    lists are zipped up to ``n_pairs``. Without invalid samples this returns
    ``[]`` rather than silently degrading into random matching.
    """
    valid = [
        (s, sample_reward(s, bbb_classifier, target_bonds))
        for s in molecules
        if s.mol is not None
    ]
    valid = [(s, r) for s, r in valid if r > 0.0]
    invalid = [s for s in molecules if s.mol is None]

    if not valid or not invalid:
        return []
    valid.sort(key=lambda x: x[1], reverse=True)

    pairs: list[PreferencePair] = []
    for i in range(min(n_pairs, len(valid), len(invalid))):
        sample, reward = valid[i]
        pairs.append(
            PreferencePair(sample, invalid[i], reward, INVALID_REWARD)
        )
    return pairs


def pair_statistics(pairs: Sequence[PreferencePair]) -> dict[str, float]:
    """Mean rewards and the fraction whose rejected side is an invalid clump."""
    if not pairs:
        return {
            "n_pairs": 0,
            "mean_reward_chosen": 0.0,
            "mean_reward_rejected": 0.0,
            "invalid_rejected_fraction": 0.0,
        }
    return {
        "n_pairs": float(len(pairs)),
        "mean_reward_chosen": float(
            sum(p.reward_chosen for p in pairs) / len(pairs)
        ),
        "mean_reward_rejected": float(
            sum(p.reward_rejected for p in pairs) / len(pairs)
        ),
        "invalid_rejected_fraction": float(
            sum(p.rejected.mol is None for p in pairs) / len(pairs)
        ),
    }
