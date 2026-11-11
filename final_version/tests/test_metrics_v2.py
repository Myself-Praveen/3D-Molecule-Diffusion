"""fix.md schema v2: counts, CIs, size-normalized geometry, honest tables.

Every test here pins one of the defects fix.md documents — the ones that
changed a published conclusion rather than merely tidying a number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from rdkit import Chem

from src.utils.evaluation import (
    BONDS_PER_ATOM_REFERENCE,
    METRIC_ORDER,
    bbb_rates,
    connectivity,
    coords_and_types_to_mol,
    evaluate,
    geometry_metrics,
    geometry_only_metrics,
    ks_two_sample,
    latex_row,
    nn_outside_window_pct,
    print_metrics,
    relax_molecule,
    wilson_ci,
)


def _ring(n: int) -> Chem.Mol:
    """A cycloalkane with n carbons and n bonds (no implicit H)."""
    rw = Chem.RWMol()
    for _ in range(n):
        atom = Chem.Atom(6)
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for i in range(n):
        rw.AddBond(i, (i + 1) % n, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _chain_mol(n: int) -> Chem.Mol:
    """A built-from-coordinates n-carbon chain (gives a conformer)."""
    pos = np.zeros((n, 3))
    pos[:, 0] = np.arange(n) * 1.54
    return coords_and_types_to_mol(pos, np.full(n, 6, dtype=int))


# ---------------------------------------------------------------------------
# F6 — one uncertainty helper for every percentage
# ---------------------------------------------------------------------------

class TestWilsonCI:
    def test_edge_values(self):
        lo, hi = wilson_ci(0, 200)
        assert lo == 0.0 and 0.5 < hi < 3.0
        lo, hi = wilson_ci(200, 200)
        assert hi == 100.0 and 97.0 < lo < 99.5

    def test_symmetry(self):
        lo1, hi1 = wilson_ci(30, 200)
        lo2, hi2 = wilson_ci(170, 200)
        assert lo1 == pytest.approx(100.0 - hi2, abs=1e-9)
        assert hi1 == pytest.approx(100.0 - lo2, abs=1e-9)

    def test_undefined_is_not_zero(self):
        assert all(math.isnan(v) for v in wilson_ci(0, 0))


# ---------------------------------------------------------------------------
# F2 — bonds per atom, not absolute bonds against a QM9 number
# ---------------------------------------------------------------------------

class TestSizeNormalizedBonds:
    def test_bonds_per_atom_is_size_invariant(self):
        out = connectivity([_ring(18), _ring(46)])
        assert out["BondsPerAtom"] == pytest.approx(1.0)
        assert out["MeanAtoms"] == pytest.approx(32.0)
        assert out["BondsPerMol"] == pytest.approx(32.0)
        # The ratio is what says whether the model is under- or over-bonded.
        assert out["BondsPerAtom_rel"] == pytest.approx(
            1.0 / BONDS_PER_ATOM_REFERENCE)

    def test_reference_ratio_flips_the_diagnosis(self):
        """V6's 0.366 bonds/atom is UNDER-bonded, not 'soup' (fix.md F2)."""
        out = connectivity([_ring(46)], reference_bonds_per_atom=1.035)
        assert out["BondsPerAtom_rel"] == pytest.approx(1.0 / 1.035)


# ---------------------------------------------------------------------------
# F10 — undefined connectivity is None, not a 0 % measurement
# ---------------------------------------------------------------------------

class TestUndefinedIsNotZero:
    def test_empty(self):
        out = connectivity([])
        assert out["ConnectedValidity"] is None
        assert out["BondsPerMol"] is None
        assert out["n_total"] == 0

    def test_no_valid_molecules(self):
        out = connectivity([None, None])
        # 0 connected out of 2 samples IS a measured zero...
        assert out["ConnectedValidity"] == 0.0
        # ...but the means over valid molecules do not exist.
        assert out["BondsPerMol"] is None
        assert out["BondsPerAtom"] is None
        assert out["ConnectedFrac"] is None
        assert out["n_valid"] == 0

    def test_single_atom_has_no_neighbour(self):
        assert math.isnan(nn_outside_window_pct(np.zeros((1, 3)),
                                                np.array([6])))

    def test_unknown_element_counts_as_orphan(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        assert nn_outside_window_pct(pos, np.array([6, 99])) == 100.0


# ---------------------------------------------------------------------------
# F3 — post-hoc atom deletion is accounted for
# ---------------------------------------------------------------------------

class TestPruningAccounted:
    @staticmethod
    def _two_fragments() -> tuple[np.ndarray, np.ndarray]:
        pos = np.zeros((10, 3))
        pos[0:7, 0] = np.arange(7) * 1.54          # 7-atom chain
        pos[7:10, 0] = 25.0 + np.arange(3) * 1.54  # 3-atom satellite
        return pos, np.full(10, 6, dtype=int)

    def test_pruning_is_accounted(self):
        pos, z = self._two_fragments()
        stats: dict = {}
        mol = coords_and_types_to_mol(pos, z, keep_largest_fragment=True,
                                      stats=stats)
        assert mol is not None and mol.GetNumAtoms() == 7
        assert stats["requested_atoms"] == 10
        assert stats["kept_atoms"] == 7
        assert stats["pruned_atoms"] == 3

        metrics = evaluate([mol], set(), [], stats=[stats])
        assert metrics["AtomsRetained%"] == pytest.approx(70.0)
        assert metrics["MeanAtoms_requested"] == pytest.approx(10.0)
        assert metrics["MeanAtoms"] == pytest.approx(7.0)

    def test_no_pruning_is_reported_as_no_pruning(self):
        pos, z = self._two_fragments()
        stats: dict = {}
        mol = coords_and_types_to_mol(pos, z, stats=stats)
        assert mol is not None and mol.GetNumAtoms() == 10
        assert stats["pruned_atoms"] == 0
        assert evaluate([mol], set(), [], stats=[stats])["AtomsRetained%"] == 100.0


# ---------------------------------------------------------------------------
# F4 / F9 — both BBB denominators, no -1 sentinel
# ---------------------------------------------------------------------------

class _Oracle:
    def __init__(self, probs):
        self._probs = list(probs)

    def predict_batch(self, mols):
        return self._probs[:len(mols)]


class TestBBBDenominators:
    def test_both_denominators(self):
        mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCC")] + [None] * 8
        rates = bbb_rates(mols, _Oracle([0.9, 0.1]))
        assert rates["BBB%"] == pytest.approx(50.0)        # of valid
        assert rates["BBB_yield%"] == pytest.approx(10.0)  # of all samples
        assert rates["n_bbb"] == 1
        assert rates["n_valid"] == 2
        assert rates["n_total"] == 10

    def test_no_oracle_is_none_not_minus_one(self):
        rates = bbb_rates([Chem.MolFromSmiles("CCO")], None)
        assert rates["BBB%"] is None
        assert rates["BBB_yield%"] is None
        metrics = evaluate([Chem.MolFromSmiles("CCO")], set(), [])
        assert metrics["BBB%"] is None


# ---------------------------------------------------------------------------
# F5 — small-n means are not printed as measurements
# ---------------------------------------------------------------------------

class TestReportability:
    def test_small_n_is_not_reported_as_a_mean(self, capsys):
        print_metrics({
            "Validity": 50.0, "ConnectedValidity": 0.0,
            "QED": 0.4481, "BondsPerAtom": 1.1,
            "n_valid": 2, "n_total": 4,
        })
        out = capsys.readouterr().out
        assert "n/a (n=2)" in out      # the 2-molecule mean
        assert "0.4481" not in out
        assert "50.0000" in out        # a rate over n_total stays valid

    def test_large_n_prints_the_mean(self, capsys):
        print_metrics({"Validity": 90.0, "QED": 0.4481, "n_valid": 100,
                       "n_total": 100})
        out = capsys.readouterr().out
        assert "0.4481" in out and "n/a" not in out


# ---------------------------------------------------------------------------
# F8 — rows are rendered, never stored
# ---------------------------------------------------------------------------

class TestLatexRow:
    def test_latex_row_is_derived_not_stored(self):
        metrics = {"Validity": 50.0, "BBB%": 10.0}
        row = latex_row(metrics)
        assert row.count("&") == len(METRIC_ORDER) - 1
        assert "50.0000" in row
        metrics["Validity"] = 60.0
        assert "60.0000" in latex_row(metrics)

    def test_extra_keys_never_change_the_shape(self):
        base = latex_row({"Validity": 1.0})
        extra = latex_row({"Validity": 1.0, "lab_gate": {"overall": "PASS"},
                           "Veber_filtered%": 80.0})
        assert base.count("&") == extra.count("&")

    def test_missing_metric_renders_dash(self):
        assert "---" in latex_row({"Validity": 1.0})


# ---------------------------------------------------------------------------
# F7 — the relaxed table is geometry-only
# ---------------------------------------------------------------------------

class TestRelaxedIsGeometryOnly:
    def test_relaxed_table_cannot_change_graph_metrics(self):
        mol = _chain_mol(8)
        assert mol is not None
        relaxed = relax_molecule(mol)
        payload = geometry_only_metrics([relaxed])
        assert set(payload) == {"NN_Outside%", "NN_p50", "Rg"}
        raw = connectivity([mol])
        rel = connectivity([relaxed])
        assert rel["BondsPerMol"] == raw["BondsPerMol"]
        assert rel["ConnectedFrac"] == raw["ConnectedFrac"]


# ---------------------------------------------------------------------------
# F12 — the geometry axis lives in metrics.json
# ---------------------------------------------------------------------------

class TestGeometryInMetrics:
    def test_evaluate_carries_the_geometry_axis(self):
        mol = _chain_mol(8)
        metrics = evaluate([mol], set(), [])
        for key in ("BondsPerAtom", "BondsPerMol", "FragmentsPerMol",
                    "MeanAtoms", "NN_Outside%", "NN_p50", "Rg",
                    "n_valid", "n_total"):
            assert key in metrics, key
        assert metrics["MeanAtoms"] == pytest.approx(8.0)
        assert metrics["NN_p50"] == pytest.approx(1.54)

    def test_geometry_matches_direct_measurement(self):
        mol = _chain_mol(6)
        conf = mol.GetConformer()
        pos = np.array([[conf.GetAtomPosition(i).x,
                         conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z] for i in range(6)])
        geo = geometry_metrics([mol])
        assert geo["NN_Outside%"] == pytest.approx(
            nn_outside_window_pct(pos, np.full(6, 6)))


# ---------------------------------------------------------------------------
# F16 — size match
# ---------------------------------------------------------------------------

class TestSizeMatch:
    def test_ks_trivial_cases(self):
        assert ks_two_sample([1, 2, 3], [1, 2, 3]) == 0.0
        assert ks_two_sample([1, 2], [5, 6]) == 1.0
        assert ks_two_sample([], [1]) is None
        assert ks_two_sample([1, 2, 3, 4], [1, 2, 3, 4, 5]) > 0.0


# ---------------------------------------------------------------------------
# F1 — the reference set is the dataset, not the 1.2 % that survived a bug
# ---------------------------------------------------------------------------

def test_reference_coverage():
    from src.dataset_bbb import load_bbbp

    import generate_and_eval as G

    try:
        dataset, _, _ = load_bbbp(root="data/bbbp")
    except Exception as exc:  # pragma: no cover - data absent
        pytest.skip(f"BBBP unavailable: {exc}")
    mols, smiles, info = G._reference_mols_and_smiles(dataset)
    assert info["coverage"] > 0.95, info
    assert len(smiles) > 1000
    assert len(mols) == info["n"]


# ---------------------------------------------------------------------------
# F15 — Table I refuses mixed protocols and labels pruned rows
# ---------------------------------------------------------------------------

def _import_script(name: str):
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_row(tmp_path, name, *, ddim, retained=100.0, extra=None):
    import json

    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": {"Validity": 50.0, "AtomsRetained%": retained},
        "num_valid": 5, "num_total": 10, "reportable": False,
        "metrics_meta": {"protocol": {
            "ddim_steps": ddim, "eta": 1.0, "step_schedule": "quadratic",
            "relax": True, "keep_largest_fragment": False,
            "bond_tolerance": None, "max_bonds_ratio": 0.0,
            "perceive_bonds": False,
        }},
    }
    if extra:
        payload.update(extra)
    (d / "metrics.json").write_text(json.dumps(payload))


def test_table_rejects_mixed_protocols(tmp_path, monkeypatch):
    figs = _import_script("generate_paper_figures")
    _write_row(tmp_path, "d200", ddim=200)
    _write_row(tmp_path, "d50", ddim=50)
    monkeypatch.setattr(figs, "OUTPUTS", tmp_path)
    monkeypatch.setattr(figs, "TABLE_RUNS", [("A", "d200"), ("B", "d50")])
    with pytest.raises(SystemExit) as exc:
        figs.load_table_rows()
    assert exc.value.code == 2


def test_protocol_exception_keeps_a_row(tmp_path, monkeypatch):
    figs = _import_script("generate_paper_figures")
    _write_row(tmp_path, "d200", ddim=200)
    _write_row(tmp_path, "d50", ddim=50,
               extra={"protocol_exception": "probe-scale diagnostic"})
    monkeypatch.setattr(figs, "OUTPUTS", tmp_path)
    monkeypatch.setattr(figs, "TABLE_RUNS", [("A", "d200"), ("B", "d50")])
    rows = figs.load_table_rows()
    assert len(rows) == 2


def test_pruned_row_is_labelled(tmp_path, monkeypatch):
    figs = _import_script("generate_paper_figures")
    _write_row(tmp_path, "pruned", ddim=200, retained=60.0)
    monkeypatch.setattr(figs, "OUTPUTS", tmp_path)
    monkeypatch.setattr(figs, "TABLE_RUNS", [("R5-best", "pruned")])
    rows = figs.load_table_rows()
    (label, row), = rows.items()
    assert "fragment-pruned" in label and "60% atoms" in label
    assert row["n"] == "5/10"
    assert row["reportable"] == "no"
