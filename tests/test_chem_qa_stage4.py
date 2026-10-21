"""Stage 4 of the lab-readiness stack: applicability domain.

A generated molecule is only trustworthy where the model has seen similar
chemistry — and, critically, where a *learned oracle* (BBB%, the ADMET
proxies) was trained. A candidate far from the training distribution is
where those numbers become fiction regardless of how good they look.

The cross-check test matters most: MeanMaxTanimoto must agree exactly with
the MOSES suite's SNN for the same inputs. Two code paths computing the
"same" quantity differently would be a silent inconsistency in every table
that reports both.
"""

from rdkit import Chem
from rdkit.Chem import AllChem

from src.utils.chem_qa import IN_DOMAIN_THRESHOLD, applicability_domain
from src.utils.evaluation import snn


def _mols(*smiles: str) -> list[Chem.Mol]:
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        assert m is not None, f"could not parse {s!r}"
        out.append(m)
    return out


class TestApplicabilityDomain:
    def test_identical_to_training_is_fully_in_domain(self):
        train = _mols("CCO", "c1ccccc1")
        out = applicability_domain(_mols("CCO"), train)
        assert out["InDomain%"] == 100.0
        assert out["MeanMaxTanimoto"] == 1.0
        assert out["PctBeyond60"] == 100.0

    def test_unrelated_molecule_is_out_of_domain(self):
        # Train = tiny alcohols; query = a big polycyclic ether with no
        # shared substructure beyond trivial bits.
        train = _mols("CCO", "CCCO")
        query = _mols("c1ccc2c(c1)ccc1ccccc12")
        out = applicability_domain(query, train)
        assert out["InDomain%"] == 0.0
        assert out["MeanMaxTanimoto"] < IN_DOMAIN_THRESHOLD

    def test_mixed_cohort_counts_each_side(self):
        train = _mols("CCO", "c1ccccc1")
        query = _mols("CCO", "CCCCCCCCCCCCCCCC")  # one match, one not
        out = applicability_domain(query, train)
        assert out["InDomain%"] == 50.0

    def test_threshold_is_configurable(self):
        train = _mols("CCO")
        query = _mols("CCCO")  # close but not identical
        sim = applicability_domain(query, train, threshold=0.0)["MeanMaxTanimoto"]
        assert 0.0 < sim < 1.0
        low = applicability_domain(query, train, threshold=sim - 0.05)
        high = applicability_domain(query, train, threshold=sim + 0.05)
        assert low["InDomain%"] == 100.0
        assert high["InDomain%"] == 0.0

    def test_percentile_cuts_are_ordered(self):
        train = _mols("CCO", "CCCO", "CCCCO")
        query = _mols("CCO", "CCCCCCCCCCCCCCCC", "c1ccccc1")
        out = applicability_domain(query, train)
        # Tighter threshold can never include more molecules.
        assert out["PctBeyond60"] <= out["PctBeyond40"] <= 100.0

    def test_matches_snn_exactly(self):
        """MeanMaxTanimoto is the MOSES SNN quantity — the two paths must
        agree bit-for-bit or every table reporting both is inconsistent."""
        train = _mols("CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
                      "CN1CCC[C@H]1c1cccnc1")
        query = _mols("CCO", "c1ccncc1", "CCCCCCCC")
        out = applicability_domain(query, train)
        assert abs(out["MeanMaxTanimoto"] - snn(query, train)) < 1e-12

    def test_empty_inputs_return_zero_not_raise(self):
        assert applicability_domain([], _mols("CCO"))["InDomain%"] == 0.0
        assert applicability_domain(_mols("CCO"), [])["MeanMaxTanimoto"] == 0.0
        assert applicability_domain([None], _mols("CCO"))["InDomain%"] == 0.0
        assert applicability_domain([], [])["PctBeyond40"] == 0.0

    def test_none_entries_excluded_from_denominator(self):
        train = _mols("CCO")
        out = applicability_domain([None, None, Chem.MolFromSmiles("CCO")],
                                   train)
        assert out["InDomain%"] == 100.0  # only the real molecule counts
