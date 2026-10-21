"""Stage 3 of the lab-readiness stack: synthetic accessibility + PAINS.

Two questions a synthesis lab asks about a candidate that no geometry metric
covers: can it be made, and will it give a real assay readout.

The complexity score is deliberately named and documented as a *proxy*
(``synthetic_complexity_score``), not the published Ertl SA score, because
that score's dominant term comes from fragment contributions fitted against
a catalogue of purchasable compounds this repo does not ship. Substituting
guessed fragment values would yield a number that looks like a literature SA
score while being incomparable to it. These tests therefore assert the
ordering and the monotonicity of the individual terms, not absolute
literature agreement - which is exactly what a triage gate needs, and
exactly what an honest name permits.

PAINS, by contrast, IS the real published catalogue (Baell & Holloway,
shipped with RDKit), so its tests assert exact known matches.
"""

from rdkit import Chem

from src.utils.chem_qa import (
    _spiro_bridgehead_count,
    pains_free_rate,
    pains_matches,
    synthetic_complexity_score,
)


def _mol(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, f"could not parse {smiles!r}"
    return m


# Reference molecules ordered from easy to hard to make. A chemist would
# broadly agree with this ordering, which is the bar the proxy is held to.
EASY = _mol("CCO")                            # ethanol
SIMPLE_DRUG = _mol("CC(=O)Oc1ccccc1C(=O)O")   # aspirin
POLYCYCLIC = _mol("c1ccc2c(c1)ccc1ccccc12")    # phenanthrene
COMPLEX = _mol("C1CC2C3CCC4CC(O)CC4C3CC2C1")  # bridged polycyclic


class TestSyntheticComplexityScore:
    def test_range_is_1_to_10(self):
        for smiles in ("CCO", "c1ccccc1", "C1CC2C3CCC4CC(O)CC4C3CC2C1"):
            score = synthetic_complexity_score(_mol(smiles))
            assert score is not None
            assert 1.0 <= score <= 10.0

    def test_easy_molecule_scores_higher_than_complex(self):
        assert (synthetic_complexity_score(EASY)
                > synthetic_complexity_score(COMPLEX))

    def test_complexity_decreases_across_the_reference_series(self):
        scores = [synthetic_complexity_score(m)
                  for m in (EASY, SIMPLE_DRUG, POLYCYCLIC, COMPLEX)]
        assert scores == sorted(scores, reverse=True)

    def test_size_penalty_is_monotone(self):
        # Straight alkane chains: more heavy atoms, never easier.
        small = synthetic_complexity_score(_mol("CCC"))
        large = synthetic_complexity_score(_mol("CCCCCCCCCCCCCCCCCCCC"))
        assert small > large

    def test_rings_make_it_harder(self):
        assert (synthetic_complexity_score(_mol("CCCCCCCCCC"))
                > synthetic_complexity_score(_mol("c1ccccc1")))

    def test_stereocentres_make_it_harder(self):
        flat = _mol("CC(O)C(=O)O")
        chiral = _mol("C[C@H](O)C(=O)O")
        assert (synthetic_complexity_score(flat)
                > synthetic_complexity_score(chiral))

    def test_heteroatoms_make_it_harder(self):
        hydrocarbon = _mol("CCCCCCCC")
        with_hetero = _mol("CCCCCCCCO")
        assert (synthetic_complexity_score(hydrocarbon)
                > synthetic_complexity_score(with_hetero))

    def test_none_and_empty_are_safe(self):
        assert synthetic_complexity_score(None) is None
        empty = Chem.RWMol()
        assert synthetic_complexity_score(empty) is None


class TestSpiroBridgehead:
    def test_bridgehead_is_counted(self):
        # Adamantane has 4 bridgehead carbons.
        assert _spiro_bridgehead_count(_mol("C1C2CC3CC1CC(C2)C3")) >= 4

    def test_spiro_is_counted(self):
        # spiro[4.5]decane: cyclopentane + cyclohexane sharing exactly one
        # atom (ring intersection == {3}). The previous fixture
        # "C1CCCC1C2CCCCC2" parses as two DISJOINT rings joined by a bond
        # (intersection empty) - not spiro at all - so it asserted nothing.
        assert _spiro_bridgehead_count(_mol("C1CCC2(CCCCC2)C1")) >= 1

    def test_flat_molecule_has_none(self):
        assert _spiro_bridgehead_count(_mol("c1ccccc1")) == 0
        assert _spiro_bridgehead_count(_mol("CCCCC")) == 0


class TestPAINS:
    def test_rhodanine_alert_fires(self):
        # Rhodanine is the canonical PAINS-A frequent-hitter scaffold.
        hits = pains_matches(_mol("O=C1NC(=S)SC1c1ccccc1"))
        assert hits and "rhod" in hits[0].lower()

    def test_clean_drug_has_no_alert(self):
        assert pains_matches(_mol("CC(=O)Nc1ccc(O)cc1")) == []   # paracetamol
        assert pains_matches(_mol("CC(=O)Oc1ccccc1C(=O)O")) == []  # aspirin
        assert pains_matches(_mol("CN1CCC[C@H]1c1cccnc1")) == []   # nicotine

    def test_catechol_alert_fires(self):
        # Catechol (1,2-dihydroxybenzene) is a classic PAINS-B motif.
        hits = pains_matches(_mol("c1ccc(O)c(O)c1"))
        assert hits

    def test_none_is_safe(self):
        assert pains_matches(None) == []


class TestPainsFreeRate:
    def test_over_valid_only(self):
        clean = [_mol("CCO"), _mol("CC(=O)Nc1ccc(O)cc1"), None]
        assert pains_free_rate(clean) == 100.0  # None excluded from denominator

    def test_dirty_set_scores_zero(self):
        dirty = [_mol("O=C1NC(=S)SC1c1ccccc1")]
        assert pains_free_rate(dirty) == 0.0

    def test_mixed_set(self):
        mixed = [_mol("CCO"), _mol("O=C1NC(=S)SC1c1ccccc1")]
        assert pains_free_rate(mixed) == 50.0

    def test_empty_is_safe(self):
        assert pains_free_rate([]) == 0.0
        assert pains_free_rate([None]) == 0.0
