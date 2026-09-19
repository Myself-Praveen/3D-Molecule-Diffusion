"""Phase 3 tests: BBB oracle classifier (range, known-positive, batch robustness).

The AUROC gate itself is verified by the training run
(scripts/train_bbb_classifier.py prints TEST AUROC); these tests cover the
inference API against the saved oracle, skipping gracefully when it has
not been trained yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from rdkit import Chem

from src.models.bbb_classifier import BBBClassifier, bbb_topology_edges

ORACLE = Path("models/bbb_oracle.pt")
CAFFEINE = "Cn1cnc2n(C)c(=O)n(C)c(=O)c12"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def _untrained():
    torch.manual_seed(0)
    return BBBClassifier()


def _trained():
    if not ORACLE.exists():
        pytest.skip("oracle not trained (run scripts/train_bbb_classifier.py)")
    clf = BBBClassifier()
    clf.load_state_dict(
        torch.load(ORACLE, map_location="cpu", weights_only=False)["model_state_dict"]
    )
    return clf


class TestTopologyEdges:
    def test_shape_and_self_loop_fallback(self):
        import numpy as np

        pos = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [9.0, 0.0, 0.0]])
        edge_index = bbb_topology_edges(pos, cutoff=2.5)
        assert edge_index.shape[0] == 2
        # atoms 0-1 bonded (both directions); isolated atom 2 has no edges
        # (GCNConv adds self-loops internally, so all nodes still pool).
        assert edge_index.shape[1] == 2
        pairs = {tuple(sorted(e)) for e in edge_index.t().tolist()}
        assert (0, 1) in pairs

    def test_empty(self):
        import numpy as np

        assert bbb_topology_edges(np.zeros((0, 3))).shape == (2, 0)


class TestOracleAPI:
    def test_forward_shape(self):
        clf = _untrained()
        z = torch.tensor([1, 1, 3, 0])
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        batch = torch.zeros(4, dtype=torch.long)
        assert clf(z, edge_index, batch).shape == (1,)

    def test_predict_range(self):
        clf = _untrained()
        for smi in (CAFFEINE, ASPIRIN, "CCO"):
            p = clf.predict_mol(Chem.MolFromSmiles(smi))
            assert 0.0 <= p <= 1.0, (smi, p)

    def test_predict_none(self):
        assert _untrained().predict_mol(None) == 0.0

    def test_predict_batch_with_none(self):
        clf = _untrained()
        out = clf.predict_batch([Chem.MolFromSmiles(CAFFEINE), None])
        assert len(out) == 2 and out[1] == 0.0
        assert 0.0 <= out[0] <= 1.0


class TestTrainedOracle:
    def test_known_bbb_positive(self):
        clf = _trained()
        assert clf.predict_mol(Chem.MolFromSmiles(CAFFEINE)) > 0.5

    def test_batch_matches_single(self):
        clf = _trained()
        mols = [Chem.MolFromSmiles(s) for s in (CAFFEINE, ASPIRIN)]
        singles = [clf.predict_mol(m) for m in mols]
        batched = clf.predict_batch(mols)
        for a, b in zip(singles, batched):
            assert abs(a - b) < 1e-5
