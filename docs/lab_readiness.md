# Lab-Readiness Stack — Roadmap & Status

**Status: stages 1–9 complete (21 Oct 2026).** Every current model arm
**FAILS** the gate — that is the honest measurement of today's outputs, not a
defect in the gate (see [Validation results](#validation-results)).

## Why this exists

The honest-ruler metrics (`src/utils/evaluation.py`) answer *is this a
connected, sanitizable molecule?* A synthesis lab asks a stricter question:
**can a chemist order this and make it?** This stack answers that in seven
stages, motivated by the measured negatives in `progress.md` §15 items 3, 4
and 13 (over-bonded soup at ConnV ≤ 1.5 across every objective arm).

Design rules, matching the repo's honesty conventions:

- Every entry point accepts `None` and returns a safe sentinel — a
  generated soup must never crash the eval measuring it.
- Each score is labelled for what it *is*: the complexity proxy is
  explicitly **not** the published Ertl SA score, the ADMET terms are
  structural alerts, not predictors. Saying so in the docstring is the point.
- Nothing changes the Validity/ConnectedValidity ruler. These are additive
  gates that run *after* a molecule has been accepted by the honest ruler.

## Stages

| Stage | Module | Question it answers | Status | Commit |
|---|---|---|---|---|
| 1 Perception | `src/utils/chem_qa.py` | Can bond orders and stereochemistry be read back from the 3D coordinates? (perception, never repair — the ruler is untouched) | Done | `75e80c9` |
| 2 Geometry | `src/utils/chem_qa.py` | How far do bond lengths sit from covalent norms (Cordero 2008)? What is the MMFF94/UFF strain energy? | Done | `80956ac` |
| 3 Synthesis | `src/utils/chem_qa.py` | 1–10 complexity proxy + PAINS (RDKit's real Baell–Holloway catalogue) | Done | `403dd7d` |
| 4 Domain | `src/utils/chem_qa.py` | Is the molecule inside the applicability domain the model learned (nearest-train Tanimoto, Morgan r2/2048)? | Done | `f1a63d1` |
| 5 ADMET | `src/utils/chem_qa.py` | ESOL solubility (exact Delaney coefficients) + hERG/CYP3A4/P-gp/polyaromatic structural alerts (auditable SMARTS) | Done | `daea294` |
| 6 Routes | `src/utils/retrosynthesis.py` | Can it be assembled from common building blocks within the bounded template horizon? | Done | `ff0eb7f` |
| 7 Gate | `src/utils/lab_gate.py`, `scripts/lab_readiness_gate.py` | Run all six stages over one cohort → one PASS/FAIL/INCOMPLETE verdict | Done | `b228ce6` |
| 8 Validation | — | End-to-end runs on the three committed BBB evals + one `generate_and_eval.py --lab_gate` run | Done | (artifacts below) |
| 9 Docs | this file, `progress.md` §15.15, `docs/project_report.tex` | Roadmap/status, progress record, report section | Done | (docs commit) |

Tests: `tests/test_chem_qa_stage1.py` … `tests/test_chem_qa_stage6.py` +
`tests/test_lab_gate.py`; full suite **354 passed**. Stage 6's tests pin
three implementation bugs found during development (the intermediate cap
discarded *discovered* routes; template-subset requests mislabelled
reactions; non-canonical building-block inputs KeyErrored in the provenance
walk), and stage 3's pin the defined-vs-potential stereocentre counting.

## The gate (stage 7)

Criteria are **data, not code** — `DEFAULT_CRITERIA` in
`src/utils/lab_gate.py` states metric, operator, threshold and a one-line
rationale per criterion, replaceable wholesale by callers:

| Criterion | Metric | Threshold | Needs |
|---|---|---|---|
| perception | `Perception%` | ≥ 95 | 3D coordinates |
| geometry | `MeanBondStrainPct` | ≤ 15 | 3D coordinates |
| pains | `PAINSFree%` | ≥ 95 | — |
| in_domain | `InDomain%` | ≥ 80 | training set |
| soluble | `ESOL_Soluble%` | ≥ 60 | — |
| routes | `RouteFeasibleRatio%` | ≥ 50 | training-set baseline |

Verdict semantics that keep it honest:

- **Sentinel −1.0 = "not measurable"** (same precedent as `BBB%` on QM9
  rows). A criterion reading it is reported as **SKIP**, never as pass or
  fail; **any SKIP without a FAIL → INCOMPLETE**, so missing evidence never
  counts as success.
- **Calibration is measured, not assumed.** The 2-step route horizon
  reaches only **0.74%** of the 1628 real BBBP training compounds (QM9-size
  molecules route near-100%), so the routes criterion is *relative* —
  half the routability real chemistry achieves under the identical bounded
  search. An absolute ≥50% threshold would fail purchasable drugs
  themselves.
- Thresholds are documented **first-pass floors, not literature standards**;
  tighten per assay, not globally.

## Usage

```bash
# Standalone over an eval dir (molecules.sdf preferred: stages 1–2 need 3D;
# without --config, stage 4 and the routes baseline SKIP → INCOMPLETE):
python scripts/lab_readiness_gate.py --eval-dir outputs/eval_bbb_full_final \
    --config configs/central_bbb.yaml
# exit codes: 0 = PASS, 1 = FAIL, 2 = INCOMPLETE
# writes <eval-dir>/lab_gate.json

# During generation: merges gate metrics into the metric table (raw AND
# relaxed LaTeX rows keep identical key sets) and persists the verdicts as
# lab_gate / lab_gate_relaxed inside metrics.json:
python generate_and_eval.py --checkpoint ... --lab_gate [--relax] --output_dir OUT
```

Programmatic: `run_gate(mols, train_mols)` →
`{"report", "metrics", "criteria", "verdicts", "overall"}` (JSON-ready).

## Validation results

The three committed BBB evals, gated standalone with their training configs:

| arm | n | perception % | strain % | PAINS % | in-domain % | soluble % | routes % | verdict |
|---|---|---|---|---|---|---|---|---|
| full 200k (best) | 554 | 0.72 | 12.19 | 100.0 | 0.00 | 78.5 | 0.00 | **FAIL** |
| geofix (best) | 418 | 0.96 | 14.98 | 100.0 | 0.00 | 82.8 | 0.00 | **FAIL** |
| λ₂-renorm (best) | 478 | 1.05 | 19.59 | 100.0 | 0.00 | 80.1 | 0.00 | **FAIL** |

Reading the table honestly:

- **The refusal is correct.** Perception succeeds on well under 1% of the
  outputs because the multi-fragment, over-bonded soup cannot be assigned
  bond orders at all (`DetermineBondOrders` fails with a charge mismatch),
  and the nearest training neighbour sits at Tanimoto ≈ 0.13 against the
  0.4 in-domain floor — both exactly where L1's soup diagnosis says they
  should be. Training-set baseline for routes: 0.74% (`TrainRouteFeasible%`).
- **The gate differentiates where the arms differ.** Geometry passes at
  12.19% (full) and 14.98% (geofix) strain but fails at 19.59% (renorm,
  mean MMFF 1774 vs ~987 kcal/mol), while PAINS and ESOL pass everywhere —
  it does not fail everything indiscriminately.
- **Relaxation flips only geometry** (16.3 → 2.35% strain in the
  `--lab_gate --relax` smoke run, every other criterion unchanged); the
  gate reports the relaxed verdict beside the raw one without letting it
  mask the raw verdict — consistent with report limitation L3.

## What this is NOT

- **Not a production retrosynthesis engine.** Stage 6 does bounded net
  transformations (connectivity only — no reagents, selectivity, yield or
  protecting groups, no commercial availability). Under a pool cap or
  beyond one fragment-growth coupling it may report "infeasible" for
  something a full engine could route; it must never report "feasible" for
  something its own templates cannot build.
- **Not calibrated predictors.** The complexity score is a transparent
  within-repo proxy; the ADMET terms are SMARTS heuristics with no IC50,
  EC50 or probability attached; ESOL is the one exactly-sourced model
  (published Delaney coefficients).
- **Not a replacement for the honest ruler.** Validity/ConnectedValidity
  remain the headline; the gate layers on top.

## Next steps

Conditional on the sampler work (report Future Work): re-run
`scripts/lab_readiness_gate.py` once connected molecules exist, and tighten
thresholds per assay rather than globally. A QM9-era eval can be gated too,
but its training baseline falls back to distance-built graphs (no SMILES
column) and is slow — expect minutes, not seconds.
