"""Stage 7: the lab-readiness gate over all six stages.

Pins the semantics that make the verdict trustworthy:
- sentinel -1.0 ("not measurable") -> criterion SKIP, never a fake pass/fail;
- any FAIL -> overall FAIL, any SKIP (without FAIL) -> INCOMPLETE;
- gate metric keys are disjoint from evaluate()'s, so --lab_gate merging
  into the eval metric dict cannot clobber a MOSES number;
- the whole verdict is JSON-serializable (metrics.json / lab_gate.json).
"""

import json
import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

from src.utils.lab_gate import (DEFAULT_CRITERIA, flatten_report,
                                lab_readiness_report, print_gate, run_gate)


def _smiles(smi: str, embed: bool = False, seed: int = 42) -> Chem.Mol | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    if embed:
        if AllChem.EmbedMolecule(m, randomSeed=seed) != 0:
            return None
    return m


# Cohort whose stage 3/5/6 verdicts are deterministically good: both entries
# are building blocks themselves (routes 100%), PAINS-clean, ESOL-soluble.
CLEAN = ["CCO", "c1ccccc1"]


class TestReportStructure(unittest.TestCase):
    def test_all_six_stages_present(self):
        mols = [_smiles(s) for s in CLEAN] + [None]
        rep = lab_readiness_report(mols, [_smiles("CCO")])
        for key in ("stage1_perception", "stage2_geometry",
                    "stage3_synthesis", "stage4_domain", "stage5_admet",
                    "stage6_routes"):
            assert key in rep, key
        assert rep["n_total"] == 3 and rep["n_valid"] == 2
        assert rep["n_3d"] == 0  # SMILES-loaded, no coordinates

    def test_flat_metrics_unique_and_disjoint(self):
        mols = [_smiles(s) for s in CLEAN]
        flat = flatten_report(lab_readiness_report(mols, [_smiles("CCO")]))
        # uniqueness within the flat dict is guaranteed by dict semantics,
        # but stage-overlap would silently drop keys — check both stages
        # actually contributed their distinctive names.
        assert "NonSingleFrac" in flat          # stage 1
        assert "MeanBondStrainPct" in flat      # stage 2
        assert "SAComplexity" in flat           # stage 3
        assert "InDomain%" in flat              # stage 4
        assert "ESOL_Soluble%" in flat          # stage 5
        assert "RouteFeasible%" in flat         # stage 6
        assert "RouteFeasibleRatio%" in flat    # stage 6 relative criterion

    def test_empty_cohort_is_safe(self):
        rep = lab_readiness_report([], None)
        assert rep["n_total"] == rep["n_valid"] == rep["n_3d"] == 0
        gate = run_gate([], None)
        assert gate["overall"] == "INCOMPLETE"
        assert all(v["status"] == "SKIP" for v in gate["verdicts"].values())


class TestSentinelAndSkip(unittest.TestCase):
    def test_no_3d_makes_perception_and_geometry_skip(self):
        mols = [_smiles(s) for s in CLEAN]
        gate = run_gate(mols, None)
        assert gate["metrics"]["Perception%"] == -1.0
        assert gate["metrics"]["MeanBondStrainPct"] == -1.0
        assert gate["verdicts"]["perception"]["status"] == "SKIP"
        assert gate["verdicts"]["geometry"]["status"] == "SKIP"

    def test_no_train_set_makes_in_domain_skip(self):
        gate = run_gate([_smiles(s) for s in CLEAN], None)
        assert gate["metrics"]["InDomain%"] == -1.0
        assert gate["verdicts"]["in_domain"]["status"] == "SKIP"

    def test_all_clean_without_prereqs_is_incomplete_not_pass(self):
        # Everything measurable passes, but the 3D-, train-set- and
        # baseline-dependent criteria cannot be measured: the verdict must
        # be INCOMPLETE — incomplete evidence never passes.
        gate = run_gate([_smiles(s) for s in CLEAN], None)
        assert gate["verdicts"]["pains"]["status"] == "PASS"
        assert gate["verdicts"]["soluble"]["status"] == "PASS"
        # routes needs the train-set baseline for its relative criterion.
        assert gate["verdicts"]["routes"]["status"] == "SKIP"
        assert gate["overall"] == "INCOMPLETE"

    def test_3d_cohort_makes_geometry_measurable(self):
        mols = [_smiles("c1ccccc1", embed=True),
                _smiles("CC(=O)Oc1ccccc1C(=O)O", embed=True)]
        assert all(m is not None for m in mols), "embedding failed"
        gate = run_gate(mols, None)
        assert gate["report"]["n_3d"] == 2
        assert gate["metrics"]["Perception%"] >= 0.0
        assert gate["metrics"]["MeanBondStrainPct"] >= 0.0
        assert gate["verdicts"]["perception"]["status"] in ("PASS", "FAIL")


class TestVerdictPropagation(unittest.TestCase):
    def test_any_fail_fails_overall_even_with_passes(self):
        criteria = dict(DEFAULT_CRITERIA)
        criteria["pains"] = {"metric": "PAINSFree%", "op": ">=",
                             "threshold": 101.0,
                             "why": "impossible on purpose"}
        gate = run_gate([_smiles(s) for s in CLEAN], None, criteria=criteria)
        assert gate["verdicts"]["pains"]["status"] == "FAIL"
        assert gate["verdicts"]["soluble"]["status"] == "PASS"
        assert gate["overall"] == "FAIL"

    def test_routes_criterion_is_relative_to_train(self):
        # Same cohort as the reference -> identical bounded search result ->
        # ratio exactly 100, regardless of what the absolute rate is.
        gate = run_gate([_smiles(s) for s in CLEAN],
                        [_smiles(s) for s in CLEAN])
        m = gate["metrics"]
        assert m["TrainRouteFeasible%"] == 100.0
        assert m["RouteFeasibleRatio%"] == 100.0
        assert gate["verdicts"]["routes"]["status"] == "PASS"

    def test_zero_train_baseline_makes_ratio_unmeasurable(self):
        # A reference set nothing routes (deep, exotic molecules) -> the
        # ratio would be division by zero: report NA, criterion SKIP.
        gate = run_gate([_smiles(s) for s in CLEAN],
                        [_smiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C")])  # caffeine
        assert gate["metrics"]["TrainRouteFeasible%"] == 0.0
        assert gate["metrics"]["RouteFeasibleRatio%"] == -1.0
        assert gate["verdicts"]["routes"]["status"] == "SKIP"

    def test_all_criteria_measurable_and_passing_is_pass(self):
        # 3D cohort + a train set makes every criterion measurable; with
        # maximally permissive overrides the verdict must reach PASS.
        mols = [_smiles("c1ccccc1", embed=True),
                _smiles("CCO", embed=True)]
        assert all(m is not None for m in mols), "embedding failed"
        train = [_smiles(s) for s in CLEAN]
        criteria = {k: dict(c,
                            threshold=(1e9 if c["op"] == "<=" else 0.0))
                    for k, c in DEFAULT_CRITERIA.items()}
        gate = run_gate(mols, train, criteria=criteria)
        assert all(v["status"] == "PASS" for v in gate["verdicts"].values()), \
            gate["verdicts"]
        assert gate["overall"] == "PASS"

    def test_unknown_op_rejected(self):
        criteria = {"x": {"metric": "PAINSFree%", "op": "==", "threshold": 1.0,
                          "why": ""}}
        with self.assertRaises(ValueError):
            run_gate([_smiles("CCO")], None, criteria=criteria)


class TestEvalMerge(unittest.TestCase):
    """The --lab_gate merge path: gate keys must never clobber evaluate()."""

    def test_gate_keys_disjoint_from_evaluate(self):
        from src.utils.evaluation import evaluate

        mols = [_smiles(s) for s in CLEAN]
        base = evaluate(mols, {"CCO"}, [_smiles("CCO")])
        gate = run_gate(mols, [_smiles("CCO")])
        overlap = set(base) & set(gate["metrics"])
        assert not overlap, f"gate would overwrite eval keys: {overlap}"
        merged = {**base, **gate["metrics"]}
        assert merged["Validity"] == base["Validity"]
        assert "RouteFeasible%" in merged

    def test_verdict_is_json_serializable(self):
        gate = run_gate([_smiles(s) for s in CLEAN] + [None],
                        [_smiles("CCO")])
        text = json.dumps(gate)
        assert json.loads(text)["overall"] == gate["overall"]

    def test_print_gate_smoke(self):
        gate = run_gate([_smiles(s) for s in CLEAN], None)
        print_gate(gate)  # must not raise


if __name__ == "__main__":
    unittest.main()
