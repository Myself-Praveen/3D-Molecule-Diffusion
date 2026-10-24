"""Tests for scripts/diagnose_bbb_clump.py (the triage plan's Step 4)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.diagnose_bbb_clump import (
    aggregate,
    bond_window,
    molecule_statistics,
)
from src.utils.evaluation import get_bond_tolerance


def _chain(n: int, x0: float = 0.0, spacing: float = 1.54):
    pos = np.stack([np.array([x0 + spacing * i, 0.0, 0.0]) for i in range(n)])
    z = np.full(n, 6, dtype=int)
    return pos, z


class TestBondWindow:
    def test_matches_the_eval_ruler(self):
        assert bond_window(6, 6) == pytest.approx(
            0.76 + 0.76 + get_bond_tolerance(6, 6)
        )
        # Order-independent and cap-aware, exactly like the eval builder.
        assert bond_window(8, 6, 0.25) == pytest.approx(0.66 + 0.76 + 0.25)

    def test_unknown_element_is_nan(self):
        assert np.isnan(bond_window(6, 99))


class TestMoleculeStatistics:
    def test_chain_is_one_fragment(self):
        pos, z = _chain(4)
        s = molecule_statistics(pos, z)
        assert s["n_atoms"] == 4
        assert s["contacts"] == 3
        assert s["fragments"] == 1
        assert s["largest_fragment_fraction"] == 1.0
        assert s["nn_p50"] == pytest.approx(1.54)

    def test_two_separated_chains_are_two_fragments(self):
        a, z = _chain(3)
        b, _ = _chain(3, x0=20.0)
        s = molecule_statistics(np.concatenate([a, b]), np.concatenate([z, z]))
        assert s["fragments"] == 2
        assert s["largest_fragment_fraction"] == pytest.approx(0.5)
        # The 20 A gap makes the largest NN distance large.
        assert s["nn_mean"] > 1.5

    def test_clump_has_more_contacts_per_atom_than_a_chain(self):
        chain_pos, z = _chain(6)
        clump_pos = np.random.default_rng(0).normal(scale=0.4, size=(6, 3))
        chain = molecule_statistics(chain_pos, z)
        clump = molecule_statistics(clump_pos, z)
        assert clump["contact_fraction"] > chain["contact_fraction"]

    def test_disconnected_single_atoms_report_deterministically(self):
        pos = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
        s = molecule_statistics(pos, np.array([6, 6]))
        assert s["fragments"] == 2 and s["contacts"] == 0

    def test_one_atom_is_handled(self):
        s = molecule_statistics(np.zeros((1, 3)), np.array([6]))
        assert s["n_atoms"] == 1 and s["n_pairs"] == 0


class TestAggregate:
    def test_reports_the_scale_free_numbers(self):
        stats = [molecule_statistics(*_chain(5)) for _ in range(3)]
        agg = aggregate(stats)
        assert agg["n_molecules"] == 3
        assert agg["bonds_per_mol"] == pytest.approx(4.0)
        assert agg["bonds_per_atom"] == pytest.approx(0.8)
        assert agg["fragments_per_mol"] == pytest.approx(1.0)
        assert agg["single_fragment_pct"] == pytest.approx(100.0)

    def test_bonds_size_correlation_needs_variance(self):
        stats = [molecule_statistics(*_chain(n)) for n in (2, 4, 8, 12)]
        agg = aggregate(stats)
        assert agg["bonds_vs_size_pearson"] == pytest.approx(1.0, abs=1e-6)

    def test_empty_input(self):
        assert aggregate([])["n_molecules"] == 0
