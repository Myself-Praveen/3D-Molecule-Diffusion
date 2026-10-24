"""Tests for the Phase 2 composite reward (plan Step 2.1).

Pins the plan's corrections: the oracle is loaded as a plain nn.Module (no
Lightning API), invalid molecules score exactly -1.0 and are therefore usable
as rejected pairs, and the reward is higher for a connected molecule than for
its fragmented counterpart.
"""

from __future__ import annotations

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from src.rl.rewards import (
    DEFAULT_WEIGHTS,
    INVALID_REWARD,
    REFERENCE_BONDS_PER_MOL,
    bbb_probability,
    composite_reward,
    compute_steric_clash_score,
    load_bbb_oracle,
    reward_batch,
)


def _embed(smiles: str, add_hs: bool = True) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    if add_hs:
        mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0
    assert AllChem.EmbedMolecule(mol, params) == 0, f"embed failed for {smiles}"
    return mol


def _oracle():
    oracle = load_bbb_oracle("models/bbb_oracle.pt")
    if oracle is None:
        pytest.skip("models/bbb_oracle.pt not present in this checkout")
    return oracle


class TestOracleLoader:
    def test_missing_checkpoint_returns_none(self, tmp_path):
        assert load_bbb_oracle(str(tmp_path / "nope.pt")) is None

    def test_loads_as_plain_module(self):
        oracle = _oracle()
        assert isinstance(oracle, torch.nn.Module)
        assert not oracle.training
        # Cached: the same path yields the same instance (one load per run).
        assert load_bbb_oracle("models/bbb_oracle.pt") is oracle

    def test_oracle_scores_a_real_molecule_in_unit_interval(self):
        oracle = _oracle()
        p = bbb_probability(_embed("c1ccccc1O"), oracle)
        assert 0.0 <= p <= 1.0

    def test_none_molecule_scores_zero(self):
        oracle = _oracle()
        assert bbb_probability(None, oracle) == 0.0
        assert bbb_probability(_embed("CCO"), None) == 0.0

    def test_geometry_less_molecule_falls_back_to_embedding(self):
        oracle = _oracle()
        mol = Chem.MolFromSmiles("CCO")  # no conformer -> predict_mol path
        assert 0.0 <= bbb_probability(mol, oracle) <= 1.0


class TestInvalidPenalty:
    def test_none_is_exactly_invalid_reward(self):
        assert composite_reward(None) == INVALID_REWARD == -1.0

    def test_unsanitizable_molecule_is_invalid(self):
        # A carbon with five neighbors cannot sanitize.
        rw = Chem.RWMol()
        for _ in range(6):
            rw.AddAtom(Chem.Atom(6))
        for i in range(1, 6):
            rw.AddBond(0, i, Chem.BondType.SINGLE)
        mol = rw.GetMol()
        assert composite_reward(mol) == INVALID_REWARD

    def test_valid_molecule_beats_invalid(self):
        mol = _embed("CCO")
        assert composite_reward(mol) > INVALID_REWARD


class TestRewardShape:
    def test_connected_beats_fragmented(self):
        connected = _embed("CCCC")
        fragmented = _embed("CCCC.C")
        r_conn = composite_reward(connected)
        r_frag = composite_reward(fragmented)
        assert r_conn > r_frag

    def test_clash_score_is_zero_for_a_real_molecule(self):
        assert compute_steric_clash_score(_embed("CCO")) == 0.0

    def test_clash_score_detects_an_overlap(self):
        mol = Chem.RWMol()
        for _ in range(2):
            mol.AddAtom(Chem.Atom(6))
        mol.AddBond(0, 1, Chem.BondType.SINGLE)
        m = mol.GetMol()
        conf = Chem.Conformer(2)
        conf.SetAtomPosition(0, (0.0, 0.0, 0.0))
        conf.SetAtomPosition(1, (0.3, 0.0, 0.0))  # < 0.76+0.76-0.30
        m.AddConformer(conf)
        assert compute_steric_clash_score(m) > 0.0

    def test_bond_target_shifts_the_reward(self):
        mol = _embed("c1ccccc1")  # 6 C-C + 6 C-H = 12 bonds
        on_target = composite_reward(mol, target_bonds=12.0)
        far_target = composite_reward(mol, target_bonds=40.0)
        assert on_target > far_target

    def test_plan_weights_are_the_defaults(self):
        assert DEFAULT_WEIGHTS["bbb"] == 0.4
        negative_keys = {"bonds", "clash", "fragments"}
        positive = sum(v for k, v in DEFAULT_WEIGHTS.items()
                       if k not in negative_keys)
        # bbb .4 + validity .2 + qed .15 + connectivity .15 = .9
        assert positive == pytest.approx(0.9)
        # Measured H-inclusive reference, documented for the target_bonds fix.
        assert REFERENCE_BONDS_PER_MOL > 19.0
        # Measured H-inclusive reference, documented for the target_bonds fix.
        assert REFERENCE_BONDS_PER_MOL > 19.0

    def test_reward_batch_is_aligned_and_finite(self):
        mols = [_embed("CCO"), None, _embed("CC")]
        rewards = reward_batch(mols, _oracle())
        assert len(rewards) == 3
        assert rewards[1] == INVALID_REWARD
        assert all(r == r for r in rewards)  # no NaN

    def test_explicit_weights_are_used(self):
        mol = _embed("CCCC")
        zero_bbb = composite_reward(mol, weights={**DEFAULT_WEIGHTS, "bbb": 0.0})
        full_bbb = composite_reward(mol, weights={**DEFAULT_WEIGHTS, "bbb": 1.0})
        assert full_bbb >= zero_bbb
