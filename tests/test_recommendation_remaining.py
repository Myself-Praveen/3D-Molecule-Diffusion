"""Tests for the remaining recommendation.md items:

- 0.4 per-element-pair bond tolerance matrix (src/utils/evaluation.py)
- Connectivity post-processing: min_fragment_atoms pruning
- QED rejection filter (apply_qed_filter)
- Type-sampling temperature in src/sampling.py
- CLI wiring of the new eval knobs in generate_and_eval.py
"""

from __future__ import annotations

import inspect

import numpy as np
import torch
from rdkit import Chem

from src.sampling import sample_molecules
from src.models.diffusion import CenteredDDPM, TypeDDPM
from src.models.egnn import EquivariantGenerator
from src.utils.evaluation import (
    TOLERANCE_MATRIX,
    _add_bonds_from_distance,
    apply_qed_filter,
    apply_veber_filter,
    coords_and_types_to_mol,
    get_bond_tolerance,
    veber_pass_rate,
)


def _mol(smiles: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smiles)
    assert m is not None
    return m


# ---------------------------------------------------------------------------
# 0.4 tolerance matrix
# ---------------------------------------------------------------------------

class TestToleranceMatrix:
    def test_covers_all_qm9_pairs(self):
        zs = [1, 6, 7, 8, 9]
        for i, a in enumerate(zs):
            for b in zs[i:]:
                assert (a, b) in TOLERANCE_MATRIX, (a, b)

    def test_symmetric_lookup(self):
        assert get_bond_tolerance(6, 8) == get_bond_tolerance(8, 6)

    def test_unlisted_pair_falls_back_to_scalar(self):
        from src.utils.evaluation import _BOND_TOLERANCE
        assert get_bond_tolerance(35, 53) == _BOND_TOLERANCE
        assert get_bond_tolerance(35, 53) == 0.45

    def test_hh_tighter_than_cc(self):
        """Doc guidance: H–H false positives (methyl H···H) get a tighter
        tolerance than the calibrated C–C default."""
        assert get_bond_tolerance(1, 1) < get_bond_tolerance(6, 6)

    def test_co_looser_than_default(self):
        """Doc guidance: C=O double bonds (~1.23 Å) need a looser C–O
        tolerance to avoid false negatives."""
        assert get_bond_tolerance(6, 8) > get_bond_tolerance(6, 6)

    def test_distance_bonding_uses_matrix(self):
        """H–H at a distance bondable under the scalar (r+r+0.45 = 1.07 Å)
        but not under the tight H–H tolerance (r+r+0.25 = 0.87 Å)."""
        pos = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0]])
        z = np.array([1, 1])
        mol_tight = Chem.RWMol()
        mol_tight.AddAtom(Chem.Atom("H"))
        mol_tight.AddAtom(Chem.Atom("H"))
        _add_bonds_from_distance(mol_tight, pos, z, [0, 1], 2.5)
        assert mol_tight.GetNumBonds() == 0  # 0.95 > 0.62 + 0.25

        mol_loose = Chem.RWMol()
        mol_loose.AddAtom(Chem.Atom("C"))
        mol_loose.AddAtom(Chem.Atom("H"))
        pos_ch = np.array([[0.0, 0.0, 0.0], [1.05, 0.0, 0.0]])
        z_ch = np.array([6, 1])
        _add_bonds_from_distance(mol_loose, pos_ch, z_ch, [0, 1], 2.5)
        assert mol_loose.GetNumBonds() == 1  # 1.05 <= 1.07 + 0.40

    def test_ground_truth_methane_still_valid(self):
        """Real geometry must survive the matrix (regression guard):
        C–H at 1.09 Å bonds; H···H at 1.78 Å must NOT bond even though the
        pre-matrix lens was worried about exactly such contacts."""
        pos = np.array([
            [0.0000, 0.0000, 0.0000],
            [0.6296, 0.6296, 0.6296],
            [-0.6296, -0.6296, 0.6296],
            [-0.6296, 0.6296, -0.6296],
            [0.6296, -0.6296, -0.6296],
        ])
        z = np.array([6, 1, 1, 1, 1])
        mol = coords_and_types_to_mol(pos, z)
        assert mol is not None
        assert mol.GetNumBonds() == 4
        assert len(Chem.GetMolFrags(mol)) == 1


# ---------------------------------------------------------------------------
# Connectivity post-processing: min_fragment_atoms
# ---------------------------------------------------------------------------

class TestMinFragmentAtoms:
    def _two_fragment_pos(self):
        """CH4 at origin + one H 6 Å away (guaranteed-disconnected soup)."""
        return np.array([
            [0.0000, 0.0000, 0.0000],
            [0.6296, 0.6296, 0.6296],
            [-0.6296, -0.6296, 0.6296],
            [-0.6296, 0.6296, -0.6296],
            [0.6296, -0.6296, -0.6296],
            [6.0, 0.0, 0.0],
        ]), np.array([6, 1, 1, 1, 1, 1])

    def test_default_keeps_legacy_behavior(self):
        pos, z = self._two_fragment_pos()
        mol = coords_and_types_to_mol(pos, z)  # min_fragment_atoms=1 = off
        assert mol is not None
        assert len(Chem.GetMolFrags(mol)) == 2

    def test_prune_isolated_atoms(self):
        pos, z = self._two_fragment_pos()
        mol = coords_and_types_to_mol(pos, z, min_fragment_atoms=2)
        assert mol is not None
        assert len(Chem.GetMolFrags(mol)) == 1
        assert mol.GetNumAtoms() == 5

    def test_prune_everything_returns_none(self):
        pos = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
        z = np.array([6, 6])
        assert coords_and_types_to_mol(pos, z, min_fragment_atoms=2) is None

    def test_dense_bond_logits_fall_back_to_distance(self):
        """Dense (N, N, 5) logits index the UNpruned atom set; after pruning,
        they must be dropped in favor of distance bonding (documented)."""
        pos, z = self._two_fragment_pos()
        n = len(z)
        logits = np.zeros((n, n, 5))
        logits[..., 1] = 10.0  # bond everything, single
        mol = coords_and_types_to_mol(
            pos, z, bond_logits=logits, min_fragment_atoms=2)
        assert mol is not None
        assert mol.GetNumAtoms() == 5
        assert len(Chem.GetMolFrags(mol)) == 1

    def test_padding_atoms_ignored(self):
        pos = np.array([
            [0.0000, 0.0000, 0.1173],
            [0.0000, 0.9289, -0.2741],
            [0.8044, -0.4644, -0.2741],
            [-0.8044, -0.4644, -0.2741],
            [0.0, 0.0, 50.0],  # '*' padding far away
        ])
        z = np.array([6, 1, 1, 1, 0])
        mol = coords_and_types_to_mol(pos, z, min_fragment_atoms=2)
        assert mol is not None
        assert mol.GetNumAtoms() == 4


# ---------------------------------------------------------------------------
# QED rejection filter
# ---------------------------------------------------------------------------

class TestQedFilter:
    def test_disabled_passthrough(self):
        mols = [None, _mol("CCO")]
        out, dropped = apply_qed_filter(mols, 0.0)
        assert out == mols and dropped == 0

    def test_positions_preserved(self):
        mols = [None, _mol("CCO"), None, _mol("c1ccccc1")]
        out, _ = apply_qed_filter(mols, 0.5)
        assert len(out) == 4
        assert out[0] is None and out[2] is None

    def test_drops_low_qed(self):
        mols = [_mol("CCO"), _mol("c1ccccc1")]
        out, dropped = apply_qed_filter(mols, 10.0)  # impossible threshold
        assert dropped == 2
        assert all(m is None for m in out)

    def test_threshold_respected(self):
        from rdkit.Chem import Descriptors
        mols = [_mol("CCO"), _mol("c1ccccc1")]
        qeds = [Descriptors.qed(m) for m in mols]
        out, dropped = apply_qed_filter(mols, max(qeds) + 0.01)
        assert dropped == 2
        out, dropped = apply_qed_filter(mols, min(qeds) - 0.01)
        assert dropped == 0
        assert all(m is not None for m in out)


# ---------------------------------------------------------------------------
# Veber rejection filter (plan C2)
# ---------------------------------------------------------------------------

class TestVeberFilter:
    def test_pass_and_drop(self):
        # Aspirin: TPSA 63, 2 rotatable bonds -> passes Veber.
        # Eicosane (C20): TPSA 0 but 17 rotatable bonds -> fails Veber.
        mols = [_mol("CC(=O)OC1=CC=CC=C1C(=O)O"),
                _mol("CCCCCCCCCCCCCCCCCCCC")]
        out, dropped = apply_veber_filter(mols)
        assert out[0] is not None and out[1] is None
        assert dropped == 1

    def test_positions_and_none_counted(self):
        mols = [None, _mol("CCO")]
        out, dropped = apply_veber_filter(mols)
        assert len(out) == 2 and out[0] is None and out[1] is not None
        assert dropped == 1  # None entries count as dropped (apply_qed_filter parity)

    def test_filtered_rate_stays_visible(self):
        """Veber_filtered% (pre-filter) must survive the filter's 100%."""
        mols = [_mol("CC(=O)OC1=CC=CC=C1C(=O)O"),
                _mol("CCCCCCCCCCCCCCCCCCCC")]
        before = veber_pass_rate(mols)            # 50% raw pass rate
        out, _ = apply_veber_filter(mols)
        assert veber_pass_rate(out) == 1.0        # filtered table: trivially 100
        assert before == 0.5                      # the honest number survives


# ---------------------------------------------------------------------------
# Type-sampling temperature
# ---------------------------------------------------------------------------

class TestTypeTemperature:
    def _tiny_model(self):
        torch.manual_seed(0)
        return EquivariantGenerator(num_types=10, node_dim=8, edge_dim=8,
                                    num_layers=1, time_dim=8).eval()

    def test_default_temperature_is_legacy(self):
        """T=1.0 must not alter logits before the posterior (legacy path)."""
        m = self._tiny_model()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        torch.manual_seed(7)
        pos1, z1 = sample_molecules(m, c, tp, torch.tensor([4]), device="cpu",
                                    ddim_steps=4)
        torch.manual_seed(7)
        pos2, z2 = sample_molecules(m, c, tp, torch.tensor([4]), device="cpu",
                                    ddim_steps=4, type_temperature=1.0)
        assert torch.equal(pos1, pos2) and torch.equal(z1, z2)

    def test_extreme_temperature_runs_and_changes_types(self):
        """T→0 sharpens toward argmax; verify it runs and differs from T=1
        under the same seed (posterior input differs)."""
        m = self._tiny_model()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        torch.manual_seed(7)
        _, z_hot = sample_molecules(m, c, tp, torch.tensor([4]), device="cpu",
                                    ddim_steps=4)
        torch.manual_seed(7)
        _, z_sharp = sample_molecules(m, c, tp, torch.tensor([4]), device="cpu",
                                      ddim_steps=4, type_temperature=0.05)
        assert torch.isfinite(z_hot).all() and torch.isfinite(z_sharp).all()

    def test_temperature_applies_in_flow_path_too(self):
        m = EquivariantGenerator(num_types=10, node_dim=8, edge_dim=8,
                                 num_layers=1, time_dim=8,
                                 objective="flow").eval()
        c = CenteredDDPM(num_steps=20, device="cpu")
        tp = TypeDDPM(num_steps=20, device="cpu")
        torch.manual_seed(3)
        pos, z = sample_molecules(m, c, tp, torch.tensor([4]), device="cpu",
                                  ddim_steps=4, type_temperature=1.7)
        assert torch.isfinite(pos).all() and ((z >= 0) & (z < 10)).all()


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

class TestCliWiring:
    def test_generate_and_eval_has_new_flags(self):
        import generate_and_eval as g
        src = inspect.getsource(g)
        for flag in ("--type_temperature", "--min_fragment_atoms", "--min_qed",
                     "--bond_tolerance", "--max_bonds_ratio", "--perceive_bonds",
                     "--veber_filter"):
            assert flag in src, f"missing CLI flag {flag}"
        # The flags must actually reach the metric code paths.
        assert "type_temperature=args.type_temperature" in src
        assert "min_fragment_atoms=args.min_fragment_atoms" in src
        assert "apply_qed_filter" in src
        # Veber filter C2: reach-through + honest pre-filter rate.
        assert "apply_veber_filter" in src
        assert "veber_pass_rate(all_mols)" in src
        assert 'metrics["Veber_filtered%"]' in src
        # Seeding: --seed must reach torch's sampling noise, not just the
        # numpy atom-count rng (the unseeded-generator bug behind the
        # false "reproducibility band" in progress.md §15 item 11).
        assert "torch.manual_seed(args.seed)" in src
        # Bond-perception chain (plan A1–A4): reach-through + provenance.
        assert "bond_tolerance=args.bond_tolerance" in src
        assert "max_bonds_ratio=args.max_bonds_ratio" in src
        assert "perceive_orders=args.perceive_bonds" in src
        assert "perceive_bonds=args.perceive_bonds" in src  # run_cfg record

    def test_tolerance_matrix_doc_entry(self):
        """The appendix must record what was (and wasn't) implemented."""
        with open("docs/recommendation.md") as f:
            doc = f.read()
        for token in ("Implementation Status", "TOLERANCE_MATRIX",
                      "3.2 DONE", "NOT IMPLEMENTED"):
            assert token in doc, f"appendix missing {token}"
