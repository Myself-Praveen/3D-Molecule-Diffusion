"""Tests for preference-pair generation (plan Step 2.2).

The two corrections under test: seed-matched pairing picks the better-reward
member as chosen, and invalid clumps are *kept* as the rejected side instead
of being filtered out.
"""

from __future__ import annotations

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from src.rl.preference_pairs import (
    PreferencePair,
    Sample,
    generate_seed_matched_pairs,
    generate_valid_vs_invalid_pairs,
    pair_statistics,
    sample_reward,
)
from src.rl.rewards import INVALID_REWARD


def _sample(smiles: str | None) -> Sample:
    if smiles is None:
        # An invalid clump: coordinates exist, molecule construction failed.
        return Sample(
            pos=torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]),
            z=torch.tensor([6, 6]),
            mol=None,
        )
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    mol = Chem.AddHs(mol)
    assert AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) == 0
    conf = mol.GetConformer()
    pos = torch.tensor(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
          conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())],
        dtype=torch.float32,
    )
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()])
    return Sample(pos=pos, z=z, mol=mol)


class TestSample:
    def test_validity_flag(self):
        assert _sample("CCO").is_valid
        assert not _sample(None).is_valid

    def test_reward_of_invalid_is_the_sentinel(self):
        assert sample_reward(_sample(None)) == INVALID_REWARD

    def test_reward_of_valid_is_finite_and_above_sentinel(self):
        r = sample_reward(_sample("CCO"))
        assert r > INVALID_REWARD and r == r


class TestSeedMatchedPairs:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal lengths"):
            generate_seed_matched_pairs([_sample("CCO")], [])

    def test_better_member_is_chosen(self):
        good = _sample("CCCC")
        bad = _sample("C.C.C")
        pairs = generate_seed_matched_pairs(
            [good], [bad], min_reward_gap=0.0,
        )
        assert len(pairs) == 1
        assert pairs[0].chosen is good
        assert pairs[0].rejected is bad
        assert pairs[0].reward_chosen > pairs[0].reward_rejected

    def test_invalid_low_side_is_kept_as_rejected(self):
        valid = _sample("CCO")
        invalid = _sample(None)
        pairs = generate_seed_matched_pairs(
            [valid], [invalid], min_reward_gap=0.0,
        )
        assert len(pairs) == 1
        assert pairs[0].rejected.mol is None
        assert pairs[0].rejected.pos.numel() > 0  # coordinates survive

    def test_invalid_high_side_becomes_rejected_not_chosen(self):
        invalid = _sample(None)
        valid = _sample("CCO")
        pairs = generate_seed_matched_pairs(
            [invalid], [valid], min_reward_gap=0.0,
        )
        assert len(pairs) == 1
        assert pairs[0].chosen is valid
        assert pairs[0].rejected is invalid

    def test_pair_where_both_invalid_is_dropped(self):
        pairs = generate_seed_matched_pairs(
            [_sample(None)], [_sample(None)], min_reward_gap=0.0,
        )
        assert pairs == []

    def test_small_gap_is_skipped(self):
        a, b = _sample("CCCC"), _sample("CCCCO")
        assert generate_seed_matched_pairs(
            [a], [b], min_reward_gap=1e9,
        ) == []

    def test_n_pairs_cap(self):
        samples = [_sample("CCCC") for _ in range(5)]
        others = [_sample("C") for _ in range(5)]
        pairs = generate_seed_matched_pairs(
            samples, others, n_pairs=2, min_reward_gap=0.0,
        )
        assert len(pairs) == 2
        assert all(isinstance(p, PreferencePair) for p in pairs)


class TestValidVsInvalidPairs:
    def test_no_invalid_samples_returns_empty(self):
        assert generate_valid_vs_invalid_pairs(
            [_sample("CCO"), _sample("CCCC")]
        ) == []

    def test_no_valid_samples_returns_empty(self):
        assert generate_valid_vs_invalid_pairs([_sample(None)]) == []

    def test_best_valid_paired_with_invalid(self):
        best = _sample("CCCC")
        worse = _sample("C")
        invalid = _sample(None)
        pairs = generate_valid_vs_invalid_pairs([worse, invalid, best])
        assert len(pairs) == 1
        assert pairs[0].chosen is best
        assert pairs[0].rejected is invalid
        assert pairs[0].reward_rejected == INVALID_REWARD

    def test_respects_n_pairs(self):
        valids = [_sample("CCCC") for _ in range(4)]
        invalids = [_sample(None) for _ in range(4)]
        pairs = generate_valid_vs_invalid_pairs(valids + invalids, n_pairs=3)
        assert len(pairs) == 3


class TestStatistics:
    def test_empty(self):
        stats = pair_statistics([])
        assert stats["n_pairs"] == 0

    def test_reports_invalid_fraction(self):
        pairs = generate_seed_matched_pairs(
            [_sample("CCCC"), _sample("CCO")],
            [_sample(None), _sample("CCCCO")],
            min_reward_gap=0.0,
        )
        stats = pair_statistics(pairs)
        assert stats["n_pairs"] == len(pairs) >= 1
        assert 0.0 <= stats["invalid_rejected_fraction"] <= 1.0
        assert stats["mean_reward_chosen"] > stats["mean_reward_rejected"]
