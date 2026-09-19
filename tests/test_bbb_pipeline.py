"""Phase 1 tests: BBB SMILES->3D pipeline, scaffold splits, dataset loaders.

Loader tests reuse the on-disk cache (``data/bbbp`` / ``data/b3db``), so
they run in seconds after the first conversion. If the raw tables are
missing (offline machine), loader tests are skipped.
"""

from __future__ import annotations

import pytest
import torch

from src.dataset_bbb import (
    _murcko_scaffold,
    compute_property_stats,
    get_dataset_info,
    load_b3db,
    load_bbbp,
    normalize_properties,
    scaffold_split,
    smiles_to_3d_data,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2n(C)c(=O)n(C)c(=O)c12"


class TestSmilesTo3D:
    def test_valid_conversion(self):
        data = smiles_to_3d_data(ASPIRIN, 1)
        assert data is not None
        assert data.pos.shape == (21, 3)  # C9H8O4 + explicit Hs
        assert data.z.shape == (21,)
        assert data.y.tolist() == [1]
        assert data.smiles == ASPIRIN

    def test_properties_present(self):
        data = smiles_to_3d_data(CAFFEINE, 0)
        assert data is not None
        for key in ("qed", "logp", "tpsa", "mw"):
            assert key in data, key
            assert torch.isfinite(data[key]).all()
        # Caffeine is a known drug-like molecule.
        assert 0.3 < float(data.qed) < 0.8

    def test_invalid_smiles_returns_none(self):
        assert smiles_to_3d_data("not_a_smiles!!!", 1) is None

    def test_geometry_sane(self):
        data = smiles_to_3d_data(ASPIRIN, 1)
        assert data is not None
        dist = torch.cdist(data.pos.float(), data.pos.float())
        dist.fill_diagonal_(float("inf"))
        nn_mean = float(dist.min(dim=1).values.mean())
        assert 0.5 < nn_mean < 2.5, f"unphysical geometry: {nn_mean}"

    def test_normalize_shape(self):
        data = smiles_to_3d_data(ASPIRIN, 1)
        assert data is not None
        props = normalize_properties(data)
        assert props.shape == (4,)
        assert torch.isfinite(props).all()


class TestScaffoldSplit:
    SMILES = [
        "c1ccccc1", "c1ccccc1C", "c1ccccc1CC",  # benzene scaffold family
        "C1CCCCC1", "C1CCCCC1C",  # cyclohexane family
        "CCO", "CCN", "CCC",  # acyclics
        "c1ccncc1", "c1ccncc1C",
    ]

    def test_disjoint_and_covering(self):
        tr, va, te = scaffold_split(self.SMILES, [0] * len(self.SMILES), seed=42)
        assert sorted(tr + va + te) == list(range(len(self.SMILES)))

    def test_no_scaffold_leak(self):
        tr, va, te = scaffold_split(self.SMILES, [0] * len(self.SMILES), seed=42)
        s_tr = {_murcko_scaffold(self.SMILES[i]) for i in tr}
        s_va = {_murcko_scaffold(self.SMILES[i]) for i in va}
        s_te = {_murcko_scaffold(self.SMILES[i]) for i in te}
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)

    def test_known_scaffolds(self):
        assert _murcko_scaffold("c1ccccc1CC") == "c1ccccc1"
        # Salts fall back to the largest fragment's scaffold, never invalid.
        assert _murcko_scaffold("[Cl].CC(C)NCC(O)COc1cccc2ccccc12") != "__invalid__"


def _needs_bbbp_cache() -> bool:
    from pathlib import Path

    return not (Path("data/bbbp/BBBP.csv").exists())


def _needs_b3db_cache() -> bool:
    from pathlib import Path

    return not (Path("data/b3db/B3DB_classification.tsv").exists())


class TestBBBP:
    @pytest.mark.skipif(_needs_bbbp_cache(), reason="BBBP table not downloaded")
    def test_load_sizes(self):
        train, val, test = load_bbbp(root="data/bbbp")
        assert len(train) > 1000
        assert len(val) > 100
        assert len(test) > 100

    @pytest.mark.skipif(_needs_bbbp_cache(), reason="BBBP table not downloaded")
    def test_label_distribution(self):
        train, _, _ = load_bbbp(root="data/bbbp")
        frac_pos = sum(d.y.item() for d in train) / len(train)
        assert 0.6 < frac_pos < 0.9, f"BBBP ~75% BBB+: {frac_pos}"

    @pytest.mark.skipif(_needs_bbbp_cache(), reason="BBBP table not downloaded")
    def test_attrs_and_stats(self):
        train, _, _ = load_bbbp(root="data/bbbp")
        d = train[0]
        for key in ("pos", "z", "y", "qed", "logp", "tpsa", "mw", "smiles"):
            assert key in d, key
        info = get_dataset_info("bbbp")
        assert info["n_failed_3d"] / info["n_usable"] < 0.10
        assert set(info["property_stats"]) == {"qed", "logp", "tpsa", "mw"}

    @pytest.mark.skipif(_needs_bbbp_cache(), reason="BBBP table not downloaded")
    def test_scaffold_disjointness(self):
        from src.dataset_bbb import _murcko_scaffold as scaf

        train, val, test = load_bbbp(root="data/bbbp")
        s_tr = {scaf(d.smiles) for d in train}
        s_va = {scaf(d.smiles) for d in val}
        s_te = {scaf(d.smiles) for d in test}
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)


class TestB3DB:
    @pytest.mark.skipif(_needs_b3db_cache(), reason="B3DB table not downloaded")
    def test_load_sizes(self):
        train, val, test = load_b3db(root="data/b3db")
        assert len(train) > 5000
        assert len(val) > 500
        assert len(test) > 500

    @pytest.mark.skipif(_needs_b3db_cache(), reason="B3DB table not downloaded")
    def test_label_distribution(self):
        train, _, _ = load_b3db(root="data/b3db")
        frac_pos = sum(d.y.item() for d in train) / len(train)
        assert 0.5 < frac_pos < 0.8, f"B3DB majority BBB+: {frac_pos}"

    @pytest.mark.skipif(_needs_b3db_cache(), reason="B3DB table not downloaded")
    def test_attrs_and_stats(self):
        train, _, _ = load_b3db(root="data/b3db")
        d = train[0]
        for key in ("pos", "z", "y", "qed", "logp", "tpsa", "mw", "smiles"):
            assert key in d, key
        props = normalize_properties(d, get_dataset_info("b3db")["property_stats"])
        assert props.shape == (4,)
        assert torch.isfinite(props).all()

    def test_compute_stats(self):
        a = smiles_to_3d_data(ASPIRIN, 1)
        b = smiles_to_3d_data(CAFFEINE, 0)
        assert a is not None and b is not None
        stats = compute_property_stats([a, b])
        assert set(stats) == {"qed", "logp", "tpsa", "mw"}
        assert all(s["std"] > 0 for s in stats.values())
