"""Stage 6: template-based retrosynthesis feasibility.

Three regression tests here each pin a bug found during implementation:
(a) the template-subset zipping mislabelled reactions when a subset of
templates was requested; (b) non-canonical building-block inputs were
unreachable in the provenance walk (KeyError) because keys weren't
canonicalized; (c) the intermediate cap discarded *discovered* routes —
the cap must bound growth only, never throw away a found route.
"""

import unittest

from rdkit import Chem

from src.utils.retrosynthesis import (
    DEFAULT_BUILDING_BLOCKS,
    REACTION_TEMPLATES,
    RouteResult,
    canonical,
    feasibility_rate,
    find_route,
)


def _smiles(smi: str):
    m = Chem.MolFromSmiles(smi)
    assert m is not None, smi
    return m


class TestFindRoute(unittest.TestCase):
    def test_aspirin_one_step_ester(self):
        r = find_route("CC(=O)Oc1ccccc1C(=O)O")
        assert r.feasible, r
        assert r.steps == 1
        assert r.templates == ["ester_formation"]
        # leaves are the two building blocks (canonical SMILES)
        assert set(r.leaves) == {canonical("CC(=O)O"),
                                 canonical("OC(=O)c1ccccc1O")}

    def test_building_block_is_zero_steps(self):
        r = find_route("CCO")
        assert r.feasible and r.steps == 0
        assert r.leaves == ["CCO"]

    def test_nitrobenzene_infeasible_proven_within_bounds(self):
        # No nitration template exists, and with the default bounds no
        # pool truncation happens at max_steps=2, so this infeasibility is
        # a proven statement *within bounds*, not a capped guess.
        r = find_route("c1ccc([N+](=O)[O-])cc1")
        assert not r.feasible
        assert r.steps is None and r.leaves == []
        assert r.note == "no template path within bounds"

    def test_none_target_safe(self):
        r = find_route(None)
        assert not r.feasible and r.note

    def test_two_step_route_has_three_leaves(self):
        # salicylic + acetic -> aspirin (layer 1), then + methylamine
        # (layer 2 block x intermediate).
        r = find_route("CC(=O)Oc1ccccc1C(=O)NC")
        assert r.feasible and r.steps == 2, r
        assert set(r.templates) == {"ester_formation", "amide_formation"}
        # binary tree: 2 steps -> 3 building-block leaves
        assert len(r.leaves) == 3
        assert set(r.leaves) <= {canonical(b) for b in DEFAULT_BUILDING_BLOCKS}

    def test_max_steps_zero_only_matches_blocks(self):
        assert find_route("CCO", max_steps=0).feasible
        r = find_route("CCOC(=O)c1ccccc1O", max_steps=0)
        assert not r.feasible
        assert r.note == "no template path within bounds"

    def test_route_result_as_dict_shape(self):
        d = find_route("CCO").as_dict()
        assert set(d) == {"feasible", "steps", "leaves", "templates",
                          "intermediates", "note"}
        assert d["feasible"] is True and d["steps"] == 0


class TestTemplateSubsetAlignment(unittest.TestCase):
    """Regression (a): subset requests must not mislabel reactions."""

    def test_ester_only_subset_labels_correctly(self):
        r = find_route("CCOC(=O)c1ccccc1O", templates=["ester_formation"])
        assert r.feasible and r.steps == 1
        # the buggy zip mapped the ester rxn onto the *first* template name
        assert r.templates == ["ester_formation"], r.templates

    def test_template_names_are_unique(self):
        names = [n for n, _ in REACTION_TEMPLATES]
        assert len(names) == len(set(names))


class TestCanonicalBlockKeys(unittest.TestCase):
    """Regression (b): non-canonical block inputs must still resolve."""

    def test_non_canonical_block_input(self):
        r = find_route("CCO", building_blocks=("OCC", "OC(=O)c1ccccc1"))
        assert r.feasible and r.steps == 0
        assert r.leaves == ["CCO"]

    def test_non_canonical_block_participates_in_route(self):
        # phenol written backwards (Oc1ccccc1); target = ethyl phenyl ether
        r = find_route("CCOc1ccccc1", building_blocks=("Oc1ccccc1", "CCBr"))
        assert r.feasible and r.steps == 1, r
        assert set(r.leaves) == {canonical("Oc1ccccc1"), canonical("CCBr")}


class TestCapSemantics(unittest.TestCase):
    """Regression (c): the cap bounds growth, never discards found routes."""

    def test_found_route_survives_tiny_pool_cap(self):
        # max_intermediates=1 truncates every pairing pool, but a route
        # discovered in a truncated layer must still be reported.
        r = find_route("CC(=O)Oc1ccccc1C(=O)O", max_steps=2,
                       max_intermediates=1)
        assert r.feasible and r.steps == 1, r

    def test_truncated_pool_makes_infeasibility_conservative(self):
        # With rounds remaining and a tiny pool, an infeasible verdict must
        # say it is provisional rather than claim proven impossibility.
        r = find_route("c1ccc([N+](=O)[O-])cc1", max_steps=3,
                       max_intermediates=5)
        assert not r.feasible
        assert "conservative" in r.note, r.note


class TestFeasibilityRate(unittest.TestCase):
    def test_cohort_metrics(self):
        mols = [_smiles("CC(=O)Oc1ccccc1C(=O)O"),
                _smiles("c1ccc([N+](=O)[O-])cc1"),
                None]  # None excluded (valid-molecule denominator)
        out = feasibility_rate(mols)
        assert out["RouteFeasible%"] == 50.0
        assert out["MeanRouteSteps"] == 1.0
        assert out["AlreadyBuildingBlock%"] == 0.0

    def test_empty_cohort(self):
        assert feasibility_rate([])["RouteFeasible%"] == 0.0
        assert feasibility_rate([None])["RouteFeasible%"] == 0.0

    def test_blocks_cohort_counts_zero_step(self):
        out = feasibility_rate([_smiles("CCO"), _smiles("c1ccccc1")])
        assert out["RouteFeasible%"] == 100.0
        assert out["AlreadyBuildingBlock%"] == 100.0


if __name__ == "__main__":
    unittest.main()
