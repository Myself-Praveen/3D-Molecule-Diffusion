"""Stage 5 of the lab-readiness stack: ADMET proxies.

ESOL uses the *published* Delaney coefficients, so unlike the complexity
score (which documents that it is a local proxy) its output is on the
literature scale — the tests verify the exact formula against a
hand-computation, not just a shape.

The structural alerts are explicitly heuristic first-pass flags, not
trained predictors; their tests assert which substructures fire, i.e.
pattern correctness rather than predictive accuracy.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors as MD

from src.utils.chem_qa import admet_alerts, admet_summary, esol_log_s


def _mol(smiles: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, f"could not parse {smiles!r}"
    return m


class TestESOL:
    def test_matches_published_formula_exactly(self):
        """Delaney 2007: logS = 0.16 - 0.63 logP - 0.0062 MW + 0.062 RB + 0.066 AP."""
        m = _mol("c1ccccc1")
        logp = Descriptors.MolLogP(m)
        mw = Descriptors.MolWt(m)
        rb = MD.CalcNumRotatableBonds(m)
        ap = (sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
              / m.GetNumHeavyAtoms())
        hand = 0.16 - 0.63 * logp - 0.0062 * mw + 0.062 * rb + 0.066 * ap
        assert abs(esol_log_s(m) - hand) < 1e-9

    def test_logp_dominates_decreasing_solubility(self):
        """Longer alkane => higher logP => lower logS, strictly."""
        series = ["CCCCCCCC", "CCCCCCCCCCCC", "CCCCCCCCCCCCCCCC"]
        log_s = [esol_log_s(_mol(s)) for s in series]
        log_p = [Descriptors.MolLogP(_mol(s)) for s in series]
        assert log_p == sorted(log_p)
        assert log_s == sorted(log_s, reverse=True)

    def test_water_beats_fat(self):
        assert esol_log_s(_mol("CCO")) > esol_log_s(_mol("c1ccccc1"))

    def test_soluble_flag_threshold_is_minus4(self):
        # Long alkane chain: logS far below -4 (insoluble).
        out = admet_summary([_mol("CCCCCCCCCCCCCCCC")])
        assert out["ESOL_Soluble%"] == 0.0
        # Ethanol: well above -4.
        out = admet_summary([_mol("CCO")])
        assert out["ESOL_Soluble%"] == 100.0

    def test_none_and_empty_safe(self):
        assert esol_log_s(None) is None
        assert esol_log_s(Chem.RWMol()) is None
        assert admet_summary([])["ESOL_logS"] == 0.0
        assert admet_summary([None])["ESOL_Soluble%"] == 0.0


class TestStructuralAlerts:
    def test_basic_amine_fires_herg(self):
        hits = admet_alerts(_mol("CN1CCC(CC1)c1ccccc1"))  # N-methylphenylpiperidine
        assert "hERG_basic_amine" in hits
        assert "Pgp_amphiphile" in hits  # two alkyl arms around the N

    def test_plain_hydrocarbon_fires_nothing(self):
        assert admet_alerts(_mol("CCCCCCCC")) == []
        assert admet_alerts(_mol("c1ccccc1")) == []

    def test_imidazole_fires_cyp3a4_including_nh_tautomer(self):
        # Regression: the first pattern required TWO pyridine-type N and
        # missed imidazole written with its [nH] entirely.
        assert "CYP3A4_azole" in admet_alerts(_mol("c1cnc[nH]1"))
        assert "CYP3A4_azole" in admet_alerts(_mol("c1ccn[nH]1"))

    def test_pyridine_and_pyrrole_stay_quiet(self):
        # 6-ring azines are not 5-membered azole ligands; pyrrole's only N
        # carries H (not a heme-ligating pyridine-type N).
        assert "CYP3A4_azole" not in admet_alerts(_mol("c1ccncc1"))
        assert "CYP3A4_azole" not in admet_alerts(_mol("c1cc[nH]c1"))

    def test_thiophene_fires_cyp_soft_spot(self):
        assert "CYP3A4_thiophene" in admet_alerts(_mol("c1ccsc1"))

    def test_flat_polyaromatic_fires(self):
        assert "flat_polyaromatic" in admet_alerts(_mol("c1ccc2c(c1)ccc1ccccc12"))

    def test_aspirin_and_paracetamol_clean(self):
        # The drugs this gate should wave through.
        assert admet_alerts(_mol("CC(=O)Oc1ccccc1C(=O)O")) == []
        assert admet_alerts(_mol("CC(=O)Nc1ccc(O)cc1")) == []

    def test_none_safe(self):
        assert admet_alerts(None) == []


class TestAdmetSummary:
    def test_alert_rates_use_valid_denominator(self):
        # 1 firing molecule + Nones: rate is 100% of the ONE valid molecule,
        # not diluted by the None entries (same convention as MOSES suite).
        out = admet_summary([_mol("CN1CCC(CC1)c1ccccc1"), None, None])
        assert out["hERG_basic_amine%"] == 100.0

    def test_keys_present(self):
        out = admet_summary([_mol("CCO")])
        for key in ("ESOL_logS", "ESOL_Soluble%", "hERG_basic_amine%",
                    "CYP3A4_azole%", "CYP3A4_thiophene%",
                    "Pgp_amphiphile%", "flat_polyaromatic%"):
            assert key in out
        # The no-valid-molecules path must emit the SAME key names as the
        # normal path — a report must not crash on an all-None cohort.
        empty = admet_summary([None, None])
        for key in ("ESOL_logS", "ESOL_Soluble%", "hERG_basic_amine%",
                    "CYP3A4_azole%", "CYP3A4_thiophene%",
                    "Pgp_amphiphile%", "flat_polyaromatic%"):
            assert key in empty
