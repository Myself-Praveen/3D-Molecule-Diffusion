"""Phase 1 + Phase 8 tests: BBB dataset pipeline and the §8.1 spec suite.

Loader tests reuse the on-disk cache (``data/bbbp`` / ``data/b3db``), so
they run in seconds after the first conversion. If the raw tables are
missing (offline machine), loader tests are skipped.

Phase 8 (implementation.md §8.1) adds:
- ``TestConditionedEGNN`` / ``TestBBBClassifier`` / ``TestExtendedMetrics``
  — the spec-named classes (the deeper equivalents live in
  tests/test_phase2_conditioning.py, tests/test_bbb_oracle.py and
  tests/test_extended_metrics.py),
- ``TestSweepRunners`` (Phase 5), ``TestEntryPoints`` (Phase 6) and
  ``TestVisualizationPipeline`` (Phase 7) — coverage for the code built by
  Phases 5–7.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
REPO = Path(__file__).resolve().parent.parent


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


# ---------------------------------------------------------------------------
# §8.1 spec-named classes (condensed — see module docstring for the deeper
# equivalents in the phase-specific test files)
# ---------------------------------------------------------------------------

def _toy_batch():
    torch.manual_seed(0)
    pos = torch.randn(11, 3)
    z = torch.randint(0, 10, (11,))
    n = pos.size(0)
    row, col = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    edge = torch.stack([row.reshape(-1), col.reshape(-1)])
    edge = edge[:, row.reshape(-1) != col.reshape(-1)]
    t = torch.tensor([10, 50])
    batch = torch.tensor([0] * 6 + [1] * 5)
    cond = {"label": torch.tensor([1, 0]),
            "properties": torch.tensor([[0.5, 1.0, -0.3, 0.2],
                                        [-0.5, 0.0, 0.4, -0.1]])}
    return z, pos, edge, t, batch, cond


class TestConditionedEGNN:
    def test_unconditional_backward_compat(self):
        from src.models.egnn import EquivariantGenerator

        z, pos, edge, t, batch, cond = _toy_batch()
        model = EquivariantGenerator(cond_dim=0).eval()
        with torch.no_grad():
            a, la, _ = model(z, pos, edge, t, batch)
            b, lb, _ = model(z, pos, edge, t, batch, cond=cond)
        # cond_dim=0: the cond kwarg must be ignored entirely.
        assert torch.equal(a, b) and torch.equal(la, lb)
        assert a.shape == pos.shape and torch.isfinite(a).all()

    def test_conditional_forward(self):
        from src.models.egnn import EquivariantGenerator

        z, pos, edge, t, batch, cond = _toy_batch()
        model = EquivariantGenerator(cond_dim=32).eval()
        with torch.no_grad():
            _, lc, _ = model(z, pos, edge, t, batch, cond=cond)
            _, lu, _ = model(z, pos, edge, t, batch, cond=None)
        assert torch.isfinite(lc).all()
        assert not torch.equal(lc, lu), "cond must change the prediction"

    def test_label_dropout(self):
        """100% label dropout == unconditional: cond=None on a conditioned
        model must match a shared-weights cond_dim=0 twin exactly."""
        from src.models.egnn import EquivariantGenerator

        z, pos, edge, t, batch, _ = _toy_batch()
        plain = EquivariantGenerator(cond_dim=0).eval()
        cond_model = EquivariantGenerator(cond_dim=32).eval()
        shared = {k: v for k, v in plain.state_dict().items()
                  if k in cond_model.state_dict()}
        cond_model.load_state_dict(shared, strict=False)
        with torch.no_grad():
            a, _, _ = plain(z, pos, edge, t, batch)
            b, _, _ = cond_model(z, pos, edge, t, batch, cond=None)
        assert torch.equal(a, b)

    def test_guidance_scale_zero(self):
        from src.models.diffusion import CenteredDDPM, TypeDDPM
        from src.models.egnn import EquivariantGenerator
        from src.sampling import sample_molecules

        _, _, _, _, _, cond = _toy_batch()
        model = EquivariantGenerator(num_types=10, node_dim=16, edge_dim=16,
                                     num_layers=2, time_dim=8, cond_dim=16)
        counts = torch.tensor([4, 5])
        kwargs = dict(ddim_steps=4, device="cpu")
        torch.manual_seed(11)
        p1, z1 = sample_molecules(
            model, CenteredDDPM(num_steps=20, device="cpu"),
            TypeDDPM(num_steps=20, device="cpu"), counts,
            cond=cond, guidance_scale=0.0, **kwargs)
        torch.manual_seed(11)
        p2, z2 = sample_molecules(
            model, CenteredDDPM(num_steps=20, device="cpu"),
            TypeDDPM(num_steps=20, device="cpu"), counts,
            cond=cond, **kwargs)
        assert torch.equal(p1, p2) and torch.equal(z1, z2)


class TestBBBClassifier:
    @staticmethod
    def _oracle():
        import torch as _torch

        from src.models.bbb_classifier import BBBClassifier

        ckpt_path = REPO / "models" / "bbb_oracle.pt"
        if not ckpt_path.exists():
            pytest.skip("models/bbb_oracle.pt not trained yet")
        ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
        clf = BBBClassifier(hidden_dim=int(ckpt.get("hidden_dim", 128)))
        clf.load_state_dict(ckpt["model_state_dict"])
        clf.eval()
        return clf

    def test_predict_mol_range(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(CAFFEINE)
        p = self._oracle().predict_mol(mol)
        assert 0.0 <= float(p) <= 1.0

    def test_known_bbb_positive(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(CAFFEINE)  # known BBB-permeable
        assert float(self._oracle().predict_mol(mol)) > 0.5


class TestExtendedMetrics:
    def test_scaffold_diversity_range(self):
        from rdkit import Chem

        from src.utils.evaluation import scaffold_diversity

        mols = [Chem.MolFromSmiles(s)
                for s in (ASPIRIN, CAFFEINE, "c1ccccc1")]
        mols.append(None)
        assert scaffold_diversity([]) == 0.0
        assert 0.0 <= scaffold_diversity(mols) <= 1.0

    def test_lipinski_known(self):
        from rdkit import Chem

        from src.utils.evaluation import lipinski_pass_rate

        aspirin = Chem.MolFromSmiles(ASPIRIN)  # marketed drug
        assert lipinski_pass_rate([aspirin]) == 1.0


# ---------------------------------------------------------------------------
# Phase 5 — sweep/ablation runners (scripts/sweep_lib.py + run_*.py)
# ---------------------------------------------------------------------------

class TestSweepRunners:
    def test_expand_grid(self):
        sys.path.insert(0, str(REPO / "scripts"))
        try:
            from sweep_lib import expand_grid
        finally:
            sys.path.pop(0)
        combos = expand_grid({"a": [1, 2], "b": ["x", "y", "z"]})
        assert len(combos) == 6
        assert combos[0] == {"a": 1, "b": "x"}
        assert expand_grid({}) == [{}]

    def test_combo_slug_and_set_dotted(self):
        sys.path.insert(0, str(REPO / "scripts"))
        try:
            from sweep_lib import combo_slug, set_dotted
        finally:
            sys.path.pop(0)
        assert combo_slug({"fed.num_clients": 4}) == "numclients4"
        assert "/" not in combo_slug({"a.b": "c/d"})
        cfg: dict = {}
        set_dotted(cfg, "a.b.c", 7)
        set_dotted(cfg, "a.b.d", 8)
        assert cfg == {"a": {"b": {"c": 7, "d": 8}}}

    def test_collect_results(self, tmp_path):
        sys.path.insert(0, str(REPO / "scripts"))
        try:
            from sweep_lib import collect_results
        finally:
            sys.path.pop(0)
        run_dir = tmp_path / "run"
        eval_dir = run_dir / "eval"
        eval_dir.mkdir(parents=True)
        (eval_dir / "metrics.json").write_text(
            json.dumps({"metrics": {"Validity": 50.0, "BBB%": -1.0}}))
        out = collect_results(run_dir, eval_dir,
                              ["Validity", "MissingKey"], is_fed=False)
        assert out["metrics"]["Validity"] == 50.0
        assert out["requested"] == {"Validity": 50.0, "MissingKey": None}
        # Fed history supplies server_loss when asked.
        (run_dir / "history.json").write_text(json.dumps(
            [{"round": 1, "val": {"loss": 2.0}},
             {"round": 2, "val": {"loss": 1.5}}]))
        out = collect_results(run_dir, eval_dir, ["server_loss"],
                              is_fed=True)
        assert out["requested"]["server_loss"] == 1.5

    @pytest.mark.parametrize("script,extra", [
        ("run_sweep.py", ["--sweep", "configs/sweeps/sweep_K.yaml"]),
        ("run_ablation.py", []),
    ])
    def test_dry_run_has_no_side_effects(self, script, extra):
        """§5.5: --dry-run prints the grid and touches no files."""
        outputs = REPO / "outputs"
        before = {p.name for p in outputs.glob("sweep_*")} \
            | {p.name for p in outputs.glob("ablation_*")}
        proc = subprocess.run(
            [sys.executable, "scripts/run_" + script.split("run_", 1)[-1],
             *extra, "--dry-run"],
            cwd=REPO, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "nothing executed" in proc.stdout
        after = {p.name for p in outputs.glob("sweep_*")} \
            | {p.name for p in outputs.glob("ablation_*")}
        assert after == before, "dry-run must not create output dirs"

    def test_ablation_skips_fed_only_arms(self):
        proc = subprocess.run(
            [sys.executable, "scripts/run_ablation.py", "--dry-run"],
            cwd=REPO, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        # ablation.yaml marks no_fedprox/no_personal_heads "fed runs only".
        assert "2 arm(s) skipped" in proc.stdout
        assert "7 run(s) planned" in proc.stdout


# ---------------------------------------------------------------------------
# Phase 6 — entry points: guidance cond, type sanitizing, label partition
# ---------------------------------------------------------------------------

class TestEntryPoints:
    def test_build_target_cond(self):
        from generate_and_eval import build_target_cond

        cond = build_target_cond(1, 3, torch.device("cpu"))
        assert cond["label"].shape == (3,) and cond["label"].dtype == torch.long
        assert bool((cond["label"] == 1).all())
        assert cond["properties"].shape == (3, 4)
        assert float(cond["properties"].abs().sum()) == 0.0
        with pytest.raises(ValueError):
            build_target_cond(2, 3, torch.device("cpu"))

    def test_sanitize_type_indices(self):
        from train import sanitize_type_indices

        z = torch.tensor([1, 6, 7, 8, 9, 16, 17, 35])
        out = sanitize_type_indices(z, num_types=10)
        assert out.tolist() == [1, 6, 7, 8, 9, 6, 6, 6]
        # In-range values are untouched (QM9 path is a no-op).
        q = torch.tensor([1, 6, 8, 9])
        assert torch.equal(sanitize_type_indices(q, 10), q)

    def test_extract_cond_disabled(self):
        from src.fed.trainer import extract_cond

        assert extract_cond({}, None, "cpu") is None
        assert extract_cond({"conditioning": {"enabled": False}},
                            None, "cpu") is None

    @staticmethod
    def _labeled(n, y_func):
        from torch_geometric.data import Data

        return [Data(pos=torch.randn(4, 3),
                     z=torch.tensor([1, 6, 7, 8]),
                     y=torch.tensor([y_func(i)])) for i in range(n)]

    def test_partition_bbb_label_skew_and_coverage(self):
        from src.fed.partition import partition_bbb_label

        ds = self._labeled(40, lambda i: i % 2)
        parts = partition_bbb_label(ds, 4, seed=42)
        assert sorted(i for v in parts.values() for i in v) \
            == list(range(len(ds)))
        ratios = [sum(int(ds[i].y.item()) for i in idxs) / len(idxs)
                  for idxs in parts.values()]
        assert max(ratios) - min(ratios) > 0.5, \
            f"label skew not extreme enough: {ratios}"

    def test_partition_bbb_label_rejects_bad_labels(self):
        from torch_geometric.data import Data

        from src.fed.partition import partition_bbb_label

        no_y = [Data(pos=torch.randn(3, 3), z=torch.tensor([1, 6, 8]))]
        with pytest.raises(ValueError, match="data.y"):
            partition_bbb_label(no_y, 2, seed=42)
        non_binary = [Data(pos=torch.randn(3, 3), z=torch.tensor([1, 6]),
                           y=torch.tensor([0.5]))]
        with pytest.raises(ValueError, match="binary"):
            partition_bbb_label(non_binary, 2, seed=42)

    def test_partition_cache_keyed_by_dataset_size(self, tmp_path):
        """QM9 and BBBP partitions for the same K/mode must not collide."""
        from torch_geometric.data import Data

        from src.fed.partition import get_partition

        def make(n):
            return [Data(pos=torch.randn(3, 3), z=torch.tensor([1, 6, 8]),
                         y=torch.tensor([i % 2])) for i in range(n)]

        p1, _ = get_partition(make(6), 2, "iid", seed=42,
                              cache_dir=tmp_path)
        p2, _ = get_partition(make(9), 2, "iid", seed=42,
                              cache_dir=tmp_path)
        assert max(max(v) for v in p1.values()) < 6
        assert max(max(v) for v in p2.values()) < 9
        assert sorted(p.name for p in tmp_path.glob("*.json")) \
            == ["K2_iid_n6.json", "K2_iid_n9.json"]


# ---------------------------------------------------------------------------
# Phase 7 — visualization pipeline (§7.2: renders with synthetic data,
# saves PDF + PNG)
# ---------------------------------------------------------------------------

class TestVisualizationPipeline:
    @staticmethod
    def _mols(n=3):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mols = []
        for smi in (ASPIRIN, CAFFEINE, "c1ccccc1")[:n]:
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            if AllChem.EmbedMolecule(mol, randomSeed=42) == 0:
                mols.append(mol)
        return mols

    @staticmethod
    def _check(paths):
        assert len(paths) == 2, paths
        for p in paths:
            assert p.exists() and p.stat().st_size > 0
        assert {p.suffix for p in paths} == {".pdf", ".png"}

    def test_all_figures_render(self, tmp_path):
        from src.utils import visualization as viz

        history = tmp_path / "history.json"
        history.write_text(json.dumps([
            {"round": 1, "server_loss": 3.0,
             "gen": {"IntDiv_p": 0.8, "ScaffDiv": 0.4}},
            {"round": 2, "server_loss": 2.0,
             "gen": {"IntDiv_p": 0.85, "ScaffDiv": 0.5}},
        ]))
        mols = self._mols()

        self._check(viz.plot_convergence_curves(
            {"run": str(history)}, tmp_path / "f1"))
        latex = viz.plot_metric_comparison_table(
            {"A": {"Validity": 50.0}, "B": {"Validity": 60.0}},
            tmp_path / "tbl")
        assert isinstance(latex, str) and "Validity" in latex
        self._check([tmp_path / "tbl.pdf", tmp_path / "tbl.png"])
        self._check(viz.plot_pareto_front(
            [{"Validity": 50.0, "Uniqueness": 99.0, "label": "l2=1"}],
            tmp_path / "f2"))
        self._check(viz.plot_mode_collapse_analysis(
            {"run": str(history)}, tmp_path / "f3"))
        self._check(viz.plot_ablation_bars(
            {"full": {"Validity": 55.0, "BBB%": 40.0}},
            tmp_path / "f4"))
        self._check(viz.plot_bbb_property_distribution(
            {"BBB+": mols, "BBB-": mols}, tmp_path / "f5"))
        self._check(viz.render_molecules_3d(mols, tmp_path / "f6"))
        self._check(viz.plot_guidance_scale_sweep(
            {0.0: 30.0, 2.0: 55.0}, tmp_path / "f7"))

    def test_empty_inputs_render_placeholder(self, tmp_path):
        """Missing sweep/ablation data must not crash the pipeline."""
        from src.utils.visualization import (
            plot_ablation_bars, plot_convergence_curves,
            plot_guidance_scale_sweep, plot_pareto_front,
        )

        self._check(plot_convergence_curves({}, tmp_path / "e1"))
        self._check(plot_pareto_front([], tmp_path / "e2"))
        self._check(plot_ablation_bars({}, tmp_path / "e4"))
        self._check(plot_guidance_scale_sweep({}, tmp_path / "e7"))
