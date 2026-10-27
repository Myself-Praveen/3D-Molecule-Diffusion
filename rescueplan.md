# Rescue Plan — Fixing Zero Connectivity

**Date:** October 27, 2026
**Status:** Proposed — evidence gathered from the full codebase and every run output in `outputs/`
**Scope:** why generation produces 0–23% single-fragment molecules, and the ordered set of steps that fixes it

---

## 1. What the evidence says

Two new rulers were computed from the saved SDFs and `data/bbbp/processed_3d.pt`, both using the
eval's own bond window (`r_i + r_j + TOLERANCE_MATRIX[i,j]`):

- **orphan-atom rate** — % of atoms whose *nearest* neighbour falls outside the bond window
  (an orphan can never bond, so it is always a fragment tip).
- **local bond length** — median nearest-neighbour distance, split by element.

| arm | orphan rate | nn(H) | nn(heavy) | R_gyr / N^⅓ | ConnFrac |
|---|---|---|---|---|---|
| **real BBBP (training target)** | **0.1%** | **1.09 Å** | **1.10 Å** | **1.12** | 96.8% |
| V4-eps on QM9 (`eval_v4_eps_eta10_full`, 54% ConnV) | 0.3% | 1.08 | 1.08 | — | 87% |
| **V5** (`eval_pre_rl`, best BBBP arm) | **7.0%** | **1.30 (+19%)** | **1.21 (+10%)** | 1.11 ✓ | 22.6% |
| **V6** (`eval_v6_bbb_full`, radius_norm) | **57.4%** | **2.02 (+85%)** | **1.56 (+42%)** | **2.88 (2.6× too big)** | 0.4% |

### Finding A — V5 is not over-bonded; it is *stretched*

V5's global size is correct (1.11 vs 1.12 real) but every bond is 10–19% too long, so ~7% of atoms
end up with no in-window neighbour (≈2.7 orphans per molecule — the observed 4.85 fragments/mol).
Inter-fragment gaps sit **0.1–0.5 Å outside the window** (measured: widening the window by +0.15 Å
moves ConnFrac 23% → 39%).

The Step 1.6 gate's "bond geometry FAIL (37–38 bonds/mol vs ≈19)" is comparing **H-inclusive** bond
counts to a heavy-atom target. On the size-normalized ruler V5 is **0.95 bonds/atom vs 1.00 real** —
i.e. the bond *density* is right and the failing axis is fragmentation, not over-bonding.

### Finding B — V6 never had connectivity at any epoch

Four fresh probes (`outputs/probe_v6_ep{400,1000,2000,3400}`, n=200, ddim 50, same protocol) give
**0.0% ConnFrac at every epoch**. Its normalized output comes out 2.6× too spread and the error
grows with molecule size (small 1.88 → large 3.90) — the signature of the `R(N)` transform, not of
checkpoint selection. Widening the perception window does nothing for it (0% → 4% at +0.8 Å).

### Finding C — two latent unit bugs under `radius_norm`

In `src/sampling.py::_guide()`:

1. Å thresholds (`COVALENT_RADII`, bond windows) are applied to `x0`, which lives in **normalized**
   units (Å / R(N) ≈ ÷3.7) whenever `radius_norm != "none"` — every pair would look like a
   catastrophic clash.
2. `x0_clamp = 10.0` is 10 **normalized** units = 36 Å, i.e. a no-op.

Both were inert only because `steric_weight` and `valence_weight` were `0.0` in every V6 run.

### Finding D — checkpoint selection is decoupled from connectivity

V6 `best.pt` val loss **0.769** < V5 **0.887**, yet ConnFrac 0.4% vs 9%. `best.pt` is chosen by val
loss, and `_quick_validity()` hardcodes **18 atoms** (QM9 size; BBBP's mean is 46). Also
`min_snr_gamma: 5.0` down-weights the low-noise steps by γ/SNR — exactly the steps that set bond
lengths.

### Finding E — the perception levers are exhausted

- `--bond_tolerance` is **tighten-only** by construction (`min(base, cap)`); the CLI cannot widen
  the window at all.
- Measured on the saved SDFs by patching `TOLERANCE_MATRIX`:

| widen | V5 ConnFrac | V5 validity | V5 honest ConnV |
|---|---|---|---|
| +0.00 Å | 22.6% | 100% | 22.6% |
| +0.15 Å | 39.1% | 60% | 21.6% |
| +0.30 Å | 51.4% | 20% | 9.5% |

  Widening converts *disconnected-but-valid* into *connected-but-invalid* — **honest ConnV does not
  improve**. Shrinking coordinates ×0.85 gives 51% ConnFrac, but that is a scale lie, not a fix.

### Finding F — data scale is the structural gap

1640 train molecules, one ETKDG conformer each, 17% scaffold coverage, against ~134k for QM9 where
this exact architecture reaches 0.3% orphans. ConnFrac compounds with size: at a 7% orphan rate and
N = 40 you are mathematically capped near the observed 23%. Additionally, **66 molecules (3.2%) of
the training targets are multi-component salts** (29 with isolated atoms) — the model is taught that
disconnected output is a valid target.

### Finding G — no test-time mechanism exists that can *attract*

`src/guidance.py` contains only repulsive energies (`steric_guidance`, `valence_guidance`). Nothing
in the sampler can pull a fragment gap shut. Sampler knobs have been swept extensively already
(guidance 0–8, eta 0–1, ddim 20–1000, steric 1/5/10, coord_temp 0.5, relax, min_qed,
`bond_tolerance` tighten, `min_fragment_atoms`) and plateau at ~14% ConnV on V5.

---

## 2. The plan

### Tier 0 — fix the measurement and the bugs (hours, no training)

- **R1. Re-measure the geometry gate.** Replace "Bonds/mol ≈ 19" with **bonds/atom ≈ 1.0** and add
  **NN-outside-window** (target ≤1%) as the geometry axis, in `scripts/diagnose_bbb_clump.py`.
  Re-evaluate Step 1.6 against the corrected rulers before any RL decision.
- **R2. Fix the radius-norm unit bugs.** In `_guide()`: denormalize `x0` → apply guidance in Å →
  renormalize (or scale the thresholds by `R(N)`); make `x0_clamp` scale with `R(N)` (10 Å, not 10
  normalized units). Extend `tests/test_radius_norm.py` + `tests/test_steric_guidance.py`.
- **R3. Fix checkpoint selection.** Make `_quick_validity()` sample from the real atom-count
  histogram instead of 18 atoms, and pick `best.pt` by a generation metric (ConnFrac /
  NN-outside-window) rather than val loss.

### Tier 1 — sampler-side on V5, no retraining (~1 h per run)

- **R4. Attractive connectivity guidance (highest-leverage untried intervention).** New energy term
  in `src/guidance.py`: for each atom whose nearest neighbour lies within `window + 0.5 Å`, pull
  toward the covalent distance; ramp over the last 30% of the reverse trajectory (same contract as
  `steric_guidance`: `<= 0` is a no-op, per-atom displacement cap, CoM projection). This attacks the
  measured 0.1–0.5 Å gap directly.
- **R5. Run the knobs that have never been non-zero:** `valence_weight` has been `0.0` in every
  recorded eval, ancestral DDPM (`ddim_steps=None`) was never run, `keep_largest_fragment` never
  `True`. Quick checks only — history says the gain is small.
- **R6. Do not chase perception further.** Finding E is the measurement; widening is a lens change,
  not a generator fix.

### Tier 2 — training-side (the actual fix; one arm ≈ 33 h CPU)

- **R7. Lose the eps/min-SNR coupling on the coordinate head:** switch to an `x0` (or `v`) objective
  or drop `min_snr_gamma: 5.0`, so low-noise steps — which set bond lengths — carry full weight.
- **R8. Fix the data scale.** Pretrain on a large 3D corpus and fine-tune on BBBP; add conformer
  augmentation (currently 1 conformer/molecule); **filter the 3.2% multi-component salt targets**.
- **R9. V6 decision.** Either instrument it with an x0-recovery probe in Å on held-out data
  (normalize-only vs denormalize-only vs learned scale) to localize which half of the transform is
  wrong, or archive it and carry `radius_norm` as a V7 hypothesis to run *after* R7.

### Already unblocked (runs today, no training)

`outputs/reward_select_v5/elites_top{25,50}.sdf` (top-25 = 100% ConnFrac, 88% BBB%, bonds/atom
1.04) is exactly what `scripts/raft_finetune.py` consumes — RAFT can proceed while R7/R8 train.
DPO stays gated behind the corrected R1 gate.

---

## 3. Suggested order

1. **R1 + R2 + R3** (Tier 0) — small, testable, no training budget.
2. **R4** (attractive guidance) on `central_v5_bbb/best.pt`, Table-I protocol, vs `eval_pre_rl`.
3. **R7** as the next training arm; **R8** as the campaign after it.
4. **R9** decided once R7's arm lands.

**Success criterion:** NN-outside-window ≤ 1% (real = 0.1%) and ConnV ≥ 40% at n=1000 under the
unchanged Table-I protocol, without a validity sacrifice.

---

### Appendix — artifacts from this analysis

- `outputs/probe_v6_ep{400,1000,2000,3400}/` — four V6 checkpoint probes (all 0.0% ConnFrac).
- Tolerance-widen, coordinate-shrink and orphan-rate measurements — derived from existing
  `molecules.sdf` files, no regeneration.
- Knob-coverage audit across all 130 `outputs/**/metrics.json` run blocks (which flags were ever
  non-default).
