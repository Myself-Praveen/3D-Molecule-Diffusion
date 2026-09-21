"""Phase 4 tests: BBB-specific metrics (ranges, known molecules, oracle wiring)."""

from rdkit import Chem

from src.utils.evaluation import (
    bbb_permeability_rate,
    cns_mpo_score,
    lipinski_pass_rate,
    scaffold_coverage,
    scaffold_diversity,
    veber_pass_rate,
)

ASPIRIN = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
CAFFEINE = Chem.MolFromSmiles("Cn1cnc2n(C)c(=O)n(C)c(=O)c12")
BENZENE = Chem.MolFromSmiles("c1ccccc1")
TOLUENE = Chem.MolFromSmiles("Cc1ccccc1")


class TestExtendedMetrics:
    def test_scaffold_diversity_range(self):
        assert scaffold_diversity([]) == 0.0
        d = scaffold_diversity([ASPIRIN, CAFFEINE, BENZENE, None])
        assert 0.0 <= d <= 1.0

    def test_scaffold_diversity_counts(self):
        # Benzene + toluene share a scaffold; caffeine's fused rings differ.
        d = scaffold_diversity([BENZENE, TOLUENE, CAFFEINE])
        assert abs(d - 2 / 3) < 1e-9

    def test_scaffold_coverage(self):
        assert scaffold_coverage([], [ASPIRIN]) == 0.0
        assert scaffold_coverage([ASPIRIN], []) == 0.0
        assert scaffold_coverage([BENZENE, ASPIRIN], [BENZENE, CAFFEINE]) == 0.5

    def test_lipinski_known(self):
        # Aspirin is a marketed drug: passes Lipinski and Veber.
        assert lipinski_pass_rate([ASPIRIN]) == 1.0
        assert veber_pass_rate([ASPIRIN]) == 1.0
        assert lipinski_pass_rate([]) == 0.0
        assert veber_pass_rate([None]) == 0.0

    def test_cns_mpo_range(self):
        assert cns_mpo_score([]) == 0.0
        for m in (ASPIRIN, CAFFEINE):
            s = cns_mpo_score([m])
            assert 0.0 <= s <= 4.0, s

    def test_bbb_rate_bounds(self):
        assert bbb_permeability_rate([], None) == 0.0

    def test_bbb_rate_with_oracle(self):
        import torch

        from src.models.bbb_classifier import BBBClassifier

        torch.manual_seed(0)
        clf = BBBClassifier()
        r = bbb_permeability_rate([CAFFEINE, ASPIRIN, None], clf)
        assert 0.0 <= r <= 1.0
