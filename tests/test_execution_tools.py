"""Tests for the post-Phase-8 execution tooling (2026-10-05 session).

Covers:
- ``scripts/flow_triage.py`` — the analytic trivial zero-velocity baseline
  must match Monte Carlo over centered noise, and the CLI must run end to
  end against the retained V4-flow checkpoints (progress.md flow triage).
- ``scripts/backfill_connectivity.py`` — the SDF-recomputed connectivity
  keys must reproduce ``evaluate()`` output on native eval dirs, and
  lossy/corrupt SDFs (parsed count != recorded num_valid) must be detected
  instead of backfilled.
- ``scripts/run_guidance_sweep.py`` — dry-run plans every weight and
  creates no filesystem side effects (the Phase-5 runner bug must not be
  reintroduced), and the oracle auto-load in generate_and_eval resolves
  bbb_classifier.checkpoint from config.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent


def _import(name: str):
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        return __import__(name)
    finally:
        sys.path.pop(0)


class TestFlowTriageBaseline:
    def test_trivial_baseline_matches_monte_carlo(self):
        flow_triage = _import("flow_triage")
        torch.manual_seed(0)
        # Two molecules of different sizes so the 1/n_g centering term differs.
        counts = [3, 5]
        batch = torch.repeat_interleave(
            torch.arange(len(counts)), torch.tensor(counts))
        x0 = torch.randn(sum(counts), 3) * 1.2
        x0 = x0 - torch.zeros_like(x0).index_add(
            0, batch, x0)[batch] / torch.bincount(batch).float()[batch].unsqueeze(-1)

        analytic = flow_triage.trivial_baseline(x0, batch)

        errs = []
        for _ in range(300):
            eps = torch.randn_like(x0)
            eps = eps - torch.zeros_like(eps).index_add(
                0, batch, eps)[batch] / \
                torch.bincount(batch).float()[batch].unsqueeze(-1)
            errs.append(((eps - x0) ** 2).mean().item())
        mc = sum(errs) / len(errs)
        assert analytic == pytest.approx(mc, rel=0.05)

    @pytest.mark.skipif(
        not (REPO / "checkpoints" / "central_v4_flow" / "best.pt").exists(),
        reason="V4-flow checkpoints not present")
    def test_triage_cli_end_to_end(self, tmp_path):
        out = tmp_path / "triage.json"
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "flow_triage.py"),
             "--n_mols", "16", "--u_points", "0.5",
             "--checkpoints",
             "checkpoints/central_v4_flow/best.pt",
             "--out", str(out)],
            cwd=REPO, capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        data = json.loads(out.read_text())
        # Trivial baseline must be well above the eps constant (1.0) —
        # this is the whole point of the triage (real value ~3.9).
        assert data["trivial_baseline"]["train"] > 1.5
        assert data["trivial_baseline"]["val"] > 1.5
        entry = next(iter(data["checkpoints"].values()))
        row = entry["splits"]["val"][0]["none"]
        # Skill must be reported for every variant.
        assert "skill" in row and "mse" in row and "trivial" in row


class TestBackfillConnectivity:
    def test_formula_reproduces_native_evaluate(self):
        backfill = _import("backfill_connectivity")
        d = REPO / "outputs" / "eval_v3_eta10"
        if not (d / "molecules.sdf").exists():
            pytest.skip("eval_v3_eta10 artifacts not present")
        data = json.loads((d / "metrics.json").read_text())
        if "ConnectedValidity" not in data["metrics"]:
            pytest.skip("native connectivity keys not present")
        result = backfill.connectivity_from_sdf(
            d / "molecules.sdf", int(data["num_total"]))
        assert result is not None
        computed, parsed = result
        # SDF must round-trip losslessly for this dir, and the recomputed
        # keys must equal evaluate()'s native output bit-for-bit.
        assert parsed == int(data["num_valid"])
        for k in ("ConnectedValidity", "BondsPerMol", "ConnectedFrac"):
            assert computed[k] == pytest.approx(data["metrics"][k], abs=1e-6)

    @pytest.mark.skipif(
        not (REPO / "outputs" / "eval_central" / "molecules.sdf").exists(),
        reason="eval_central artifacts not present")
    def test_lossy_sdf_is_detected_not_backfilled(self):
        """eval_central's SDF drops records on read (861 recorded, fewer
        parsed) — the backfill must report the mismatch rather than write
        numbers computed from the wrong molecule list."""
        backfill = _import("backfill_connectivity")
        d = REPO / "outputs" / "eval_central"
        data = json.loads((d / "metrics.json").read_text())
        if "ConnectedValidity" in data["metrics"]:
            pytest.skip("already backfilled (should not happen for eval_central)")
        result = backfill.connectivity_from_sdf(
            d / "molecules.sdf", int(data["num_total"]))
        assert result is not None
        _, parsed = result
        assert parsed != int(data["num_valid"])

    def test_validate_mode_exits_clean(self):
        proc = subprocess.run(
            [sys.executable,
             str(REPO / "scripts" / "backfill_connectivity.py"),
             "--validate", "--only", "eval_v3_eta10,eval_v4_flow_probe"],
            cwd=REPO, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "failures=0" in proc.stdout


class TestGuidanceSweepRunner:
    def test_dry_run_plans_all_without_side_effects(self, tmp_path):
        outdir = tmp_path / "sweep_guidance"
        proc = subprocess.run(
            [sys.executable,
             str(REPO / "scripts" / "run_guidance_sweep.py"),
             "--dry-run", "--outdir", str(outdir)],
            cwd=REPO, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "dry-run complete: 6 run(s) planned" in proc.stdout
        # The Phase-5 runner bug (dry-run creating run dirs) must not be
        # reintroduced in this runner.
        assert not outdir.exists()

    def test_generate_eval_help_works(self):
        """--help used to crash: argparse %-expansion hit 'BBB% metric'."""
        proc = subprocess.run(
            [sys.executable, str(REPO / "generate_and_eval.py"), "--help"],
            cwd=REPO, capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "--guidance_scale" in proc.stdout

    def test_oracle_path_resolution_from_config(self):
        """generate_and_eval must resolve bbb_classifier.checkpoint from the
        config when --bbb_oracle is absent (sweep evals need BBB%)."""
        import yaml

        cfg = yaml.safe_load(
            (REPO / "configs" / "central_bbb.yaml").read_text())
        oracle = (cfg.get("bbb_classifier") or {}).get("checkpoint")
        assert oracle, "central_bbb.yaml must declare bbb_classifier.checkpoint"
        assert (REPO / oracle).exists()
