# fix.md — Metrics audit and repair plan

**Status:** analysis complete, no code changed yet. This document is the specification for
fixing the evaluation numbers in this repo — every defect below was verified by running the
current code on the current artifacts, not inferred from prose.

**Why this matters:** the last three sessions ended in contradictory conclusions
("over-bonded soup" vs "under-bonded", "R5-best is 59.5% validity — new best", "V7-nosnr is
disproved") that are **artifacts of the metric definitions**, not of the models. Several of
them are wrong in a way that would survive into the paper tomorrow.

---

## 0. TL;DR — the five findings that change conclusions

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| 1 | **`BondsPerMol` is compared against QM9's 19-bond reference, but the models generate BBBP-sized molecules where real is 47.3 bonds / 45.9 atoms (1.035 bonds/atom).** | Ground truth through the repo's own builder: 47.3 bonds, 45.9 atoms, 1.035 b/atom. V6-full: 20.7 bonds at 56.7 atoms = **0.366 b/atom**. V5-w5: 0.961 b/atom. | Every "over-bonded soup (32 bonds vs ~19 real)" statement in `progress.md`/`rescueplan.md` is **inverted**. These models are *under*-bonded, by 4–60 %. |
| 2 | **`keep_largest_fragment` improved "validity" by deleting atoms, and the table cannot see it.** | R5-best survivors average **27.7 atoms** vs 45.9 requested; V5 same checkpoint = 38.6. No `AtomsRetained%` metric exists. | "59.5 % vs 43.0 % validity" is **not comparable**. R5-best is a smaller-molecule generator scored on a truncated molecule. |
| 3 | **`Novelty`, `SNN`, `ScaffCov`, and the conditioning profile are computed against 20 of 1628 training molecules (1.2 %).** | `_data_to_rdkit_mol` survives **20/1628** — measured. `Novelty = 100.0` on every single arm is the giveaway. | Novelty/SNN/ScaffCov are meaningless for every BBB arm; the CFG "training-mean profile" is fit to 1.2 % of the data. |
| 4 | **`BBB%` divides by *valid* molecules, so it is not comparable across arms.** | pre-RL: 34.2 % of 398 valid = **136 per 1000 generated**. V6-full: 1.4 % of 503 = **7 per 1000**. V5-w5: 31.4 % of 86 = **135 per 200**. | The headline BBB column hides a 20× difference in actual yield. |
| 5 | **`best.pt` for V7-nosnr was selected on a 16-molecule, 20-step probe whose score ignores validity.** | Live probe at epoch 3400: validity 6.2 %, conn 100 %, nn-out 8.4 % → score **91.6** ⇒ saved. Same checkpoint, 200 samples @ ddim 50: validity **33.0 %**, conn **0.0 %**. | Model selection is noise. `best.pt` is whatever 16 molecules happened to look connected. |

---

## 1. How to reproduce this audit

```bash
# 1. Ground-truth control: real BBBP molecules through the eval's own bond builder
.venv/bin/python - <<'PY' 2>/dev/null
import numpy as np
from src.dataset_bbb import load_bbbp
from src.utils.evaluation import coords_and_types_to_mol, connectivity
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
ds, tr, te = load_bbbp(root='data/bbbp')
mols = [coords_and_types_to_mol(d.pos.numpy(), d.z.numpy().astype(int)) for d in ds]
v = sum(m is not None for m in mols)
print(f"validity {100*v/len(mols):.1f}%  {connectivity(mols)}")
print("mean atoms", np.mean([len(d.z) for d in ds]))
PY
# -> validity 97.0%  ConnV 94.2%  BondsPerMol 47.3  ConnectedFrac 97.2
# -> mean atoms 45.9

# 2. How many training molecules survive the reference converter
.venv/bin/python - <<'PY' 2>/dev/null
from src.dataset_bbb import load_bbbp
import generate_and_eval as G
ds, _, _ = load_bbbp(root='data/bbbp')
print(sum(G._data_to_rdkit_mol(d) is not None for d in ds), "/", len(ds))
PY
# -> 20 / 1628

# 3. Per-arm atom/bond geometry (valid molecules only, from the committed SDFs)
.venv/bin/python - <<'PY' 2>/dev/null
import glob, os, statistics
from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
for d in sorted(glob.glob('outputs/*/molecules.sdf')):
    try:
        n = [m.GetNumAtoms() for m in Chem.SDMolSupplier(d, sanitize=False) if m]
        b = [m.GetNumBonds() for m in Chem.SDMolSupplier(d, sanitize=False) if m]
    except OSError:  # corrupt SDFs exist (eval_central_v2, eval_lambda2)
        continue
    if n:
        ma, mb = statistics.mean(n), statistics.mean(b)
        print(os.path.basename(os.path.dirname(d)), len(n),
              round(ma, 1), round(mb, 1), round(mb / ma, 3))
PY
```

Reference values to hold every number against (BBBP, measured by this repo):

| quantity | real BBBP |
|---|---|
| Validity (through the eval ruler) | **97.0 %** |
| ConnectedValidity | **94.2 %** |
| ConnectedFrac | 97.2 % |
| MeanAtoms | 45.9 |
| BondsPerMol | 47.3 |
| **BondsPerAtom** | **1.035** |
| FragmentsPerMol | ~1.04 |
| NN distance p50 | 1.09 Å |

The ruler is **not** the bottleneck (97 % ceiling). The reporting is.

---

## 2. The table, as it should be read

`n` is the denominator that actually produced the number. `BBB of valid` is what the repo
prints; `BBB yield` is what a lab gets per generated sample. CIs are Wilson 95 %.

| arm | valid / N | Validity [CI] | ConnV [CI] | BBB of valid | **BBB yield (per 1000)** | atoms (kept) | bonds | **b/atom** | frags/mol |
|---|---|---|---|---|---|---|---|---|---|
| real BBBP | 1628/1628 | 97.0 | 94.2 | 76.5 (label rate) | 765 | 45.9 | 47.3 | **1.035** | 1.04 |
| pre-RL | 398/1000 | 39.8 [36.8, 42.9] | 9.0 [7.4, 10.9] | 34.2 % (136) | **136** [116, 159] | 39.2 | 37.3 | 0.952 | 4.85 |
| V5 bbb + steric w5 | 86/200 | 43.0 [36.3, 49.9] | 12.5 [8.6, 17.8] | 31.4 % (27) | **135** [94, 189] | 38.6 | 37.1 | 0.961 | 4.45 |
| V6 bbb radius | 86/200 | 43.0 [36.3, 49.9] | 0.0 [0, 1.8] | 4.7 % (4) | **20** [8, 50] | 60.7 | 24.8 | 0.409 | — |
| V6 bbb full | 503/1000 | 50.3 [47.2, 53.4] | 0.2 [0, 0.7] | 1.4 % (7) | **7** [3, 14] | 56.7 | 20.7 | 0.366 | 37.9 |
| **R5-best** | 119/200 | 59.5 [52.6, 66.1] | 59.5 | 36.1 % (43) | **215** [164, 277] | **27.7** | 29.1 | 1.051 | — |
| V7-nosnr best.pt (ddim 50, no relax) | 66/200 | 33.0 [26.9, 39.8] | 0.0 [0, 1.9] | 0.0 % (0) | **0** [0, 19] | 65.6 | 21.9 | 0.334 | — |

Read that table and the last three sessions resolve themselves:

* **R5-best is a real gain on yield** (215 vs 135 per 1000) — and a different product:
  28-atom molecules instead of 39. It must be reported as "fragment-pruned", never
  head-to-head against unpruned arms.
* **V6 is not "over-bonded soup", it is under-bonded**: half the bonds real molecules have, at
  *larger* size. That is a different bug with a different fix.
* **V7-nosnr is not "disproved on 16 samples"** — it is a 0/200 BBB-yield model. That is a much
  stronger result and it is measured.
* **pre-RL and V5-w5 are statistically tied** on yield (136 vs 135) and on validity (within CI).
  The "steric guidance w=5 gave 43 % vs 39.8 %" story is inside the noise band.

---

## 3. Defect catalogue

Severity: **P0** = changes a published conclusion; **P1** = uninterpretable / non-reproducible
number; **P2** = hygiene.

### F1 — P0 — `Novelty`, `SNN`, `ScaffCov` are measured against 1.2 % of the training set

**Where:** `generate_and_eval.py:724` `_data_to_rdkit_mol`, used at `:327`.

**What is wrong:** the reference converter bonds every pair closer than a hard-coded
`dist < 2.0 Å`, maps unknown elements to `"C"`, and leaves implicit hydrogens on. BBBP
molecules have 46 atoms at ~1.09 Å nearest-neighbour distance, so this over-bonds them into
non-sanitizable graphs.

**Evidence:** `20 / 1628` molecules survive. `train_smiles` therefore holds ≤ 20 SMILES and
`train_mols` ≤ 20 molecules — that is the reference set for Novelty, SNN and ScaffCov on
every BBB row. `Novelty = 100.0` on *every* arm in the repo is the symptom.

**Fix:** prefer the dataset's own SMILES (already used by
`generate_and_eval._gate_reference_mols`); only fall back to coordinate perception, and then
with the *same* builder as the eval.

```python
def _reference_mols_and_smiles(full_dataset):
    """Canonical reference set for Novelty / SNN / ScaffCov.

    Bond orders come from the dataset SMILES. Coordinate perception is the
    fallback, and it must use the eval's own covalent-radii builder — the old
    ``dist < 2.0`` rule destroyed 98.8 % of BBBP.
    """
    mols, smiles = [], set()
    for data in full_dataset:
        smi = getattr(data, "smiles", None)
        mol = Chem.MolFromSmiles(str(smi)) if smi else None
        if mol is None:
            mol = coords_and_types_to_mol(
                data.pos.numpy(), data.z.numpy().astype(int))
        if mol is None:
            continue
        mols.append(mol)
        smiles.add(Chem.MolToSmiles(mol))
    return mols, smiles
```

and in `_save_eval_outputs` / the metrics payload record
`"train_reference": {"n": len(mols), "requested": len(full_dataset),
"coverage": len(mols)/len(full_dataset)}`. **Hard-fail** the eval if coverage < 0.9 for a
dataset that carries SMILES.

**Test:** `tests/test_metrics_v2.py::test_reference_coverage` — build the reference for
`load_bbbp` and assert `coverage > 0.95` and `len(train_smiles) > 1500`.

**Effect:** `Novelty` drops from a fake 100 % to a real number; `SNN` becomes meaningful;
`ScaffCov` denominator stops being 12 scaffolds.

**Also affected:** `build_target_cond(..., properties_mean=...)` — the "training-mean profile"
fed to classifier-free guidance is currently the mean of those ≤ 20 molecules. Same fix, and
the eval must print the profile it used.

---

### F2 — P0 — `BondsPerMol` is compared to a QM9 number on BBBP-sized molecules

**Where:** `src/utils/evaluation.py:569` `connectivity()`; all "over-bonded soup" prose in
`progress.md` §13/§15, `rescueplan.md`, `scripts/run_rescue_generation.sh` summary header.

**What is wrong:** `BondsPerMol` is an absolute count. The table compares it to "~19 bonds
(real, 18-atom QM9)". BBBP molecules average 45.9 atoms and **47.3 bonds**. A 40-atom
molecule with 33 bonds is not over-bonded; it is missing ~14 bonds.

**Evidence:** see §2. V6-full is 0.366 bonds/atom against 1.035 real — under-bonded by 65 %.
The `scripts/diagnose_bbb_clump.py` verdict (`bonds/atom ~ 1.0` gate) already knows this; the
table never got the memo.

**Fix:** make the size-normalized quantity a first-class metric and keep the absolute count for
continuity.

```python
def connectivity(mols, reference_bonds_per_atom: float | None = None):
    valid = [m for m in mols if m is not None]
    total = len(mols)
    if not valid or total == 0:
        nan = float("nan")
        return {"ConnectedValidity": nan, "BondsPerMol": nan,
                "BondsPerAtom": nan, "MeanAtoms": nan,
                "ConnectedFrac": nan, "FragmentsPerMol": nan}
    bonds, atoms, connected, frags = [], [], 0, []
    for m in valid:
        try:
            n_at = m.GetNumAtoms()
            atoms.append(n_at)
            bonds.append(m.GetNumBonds())
            frags.append(len(Chem.GetMolFrags(m)))
            connected += int(frags[-1] == 1)
        except Exception:
            continue
    mean_atoms = float(np.mean(atoms)) if atoms else float("nan")
    mean_bonds = float(np.mean(bonds)) if bonds else float("nan")
    return {
        "ConnectedValidity": 100.0 * connected / total,
        "BondsPerMol": mean_bonds,
        "BondsPerAtom": mean_bonds / mean_atoms if mean_atoms else float("nan"),
        "MeanAtoms": mean_atoms,
        "ConnectedFrac": 100.0 * connected / len(valid),
        "FragmentsPerMol": float(np.mean(frags)) if frags else float("nan"),
    }
```

and `evaluate(..., reference=...)` adds a ratio so the direction of the error is explicit:

```python
"BondsPerAtom_rel": (bpa / reference["BondsPerAtom"]) if reference else None,  # 1.0 = real
```

**Reporting rule to adopt:** no bond statement without `bonds/atom`, and the acceptance band is
`0.75 × 1.035 ≤ b/a ≤ 1.25 × 1.035` (the gate `diagnose_bbb_clump.py` already uses
`1.25 ×` / `0.75 ×`). Absolute `BondsPerMol` stays in the payload but is never the basis of a
verdict.

**Test:** `test_bonds_per_atom_is_size_invariant` — two synthetic molecules, 18 atoms/18 bonds
and 46 atoms/46 bonds, must both report `BondsPerAtom ≈ 1.0`.

**Correction required in the record:** replace "over-bonded soup (32–33 bonds/mol vs ~19
real)" with "under-bonded (0.78–0.86 vs 1.035 bonds/atom)". The direction of the remaining
`ConnV ≈ 0` failure does not change, but the diagnosis (too few bonds, dispersed geometry) and
therefore the fix do.

---

### F3 — P0 — Post-processing that deletes atoms is invisible to the metrics

**Where:** `src/utils/evaluation.py:104` `coords_and_types_to_mol` (`min_fragment_atoms`,
`keep_largest_fragment`, `max_bonds_ratio`), `generate_and_eval.py:422`.

**What is wrong:** `keep_largest_fragment=True` prunes every component except the largest
*after* sampling. `coords_and_types_to_mol` returns only `Mol | None`, so the number of atoms
deleted — the thing that produced the validity gain — is never recorded. `num_total` counts
*molecules*, not atoms, so nothing in `metrics.json` reveals it.

**Evidence:** R5-best (`keep_largest_fragment` + valence 0.5) = 119/200 valid, survivors
average **27.7 atoms**; the same checkpoint without pruning (V5-w5) = 86/200, survivors
average **38.6 atoms**; requested mean is 45.9. `clump.json` for V5-w5 already reports
`largest_fragment_fraction: 0.80` — i.e. 20 % of atoms are satellites by construction.

**Fix:** account for it, in the builder and in the payload.

```python
def coords_and_types_to_mol(..., stats: dict | None = None) -> Chem.Mol | None:
    ...
    requested = len(atomic_numbers)
    # after the pruning block:
    if stats is not None:
        stats["requested_atoms"] = requested
        stats["kept_atoms"] = len(valid_atoms)
        stats["pruned_atoms"] = requested - len(valid_atoms)
        stats["pruned_fraction"] = (requested - len(valid_atoms)) / max(requested, 1)
    # ... on every early ``return None`` too, with kept_atoms=0
```

`generate_and_eval.py` accumulates a list of `stats` dicts and passes to `evaluate()`:

```python
"MeanAtoms_requested":  float(np.mean([s["requested_atoms"] for s in stats])),
"MeanAtoms_kept":       <MeanAtoms from connectivity()>,
"AtomsRetained%":       100.0 * sum(s["kept_atoms"] for s in stats)
                              / max(sum(s["requested_atoms"] for s in stats), 1),
```

**Reporting rule (non-negotiable):** a row with `AtomsRetained% < 99` is labelled
`post-hoc: fragment-pruned` and may not be placed in the same comparison column as unpruned
rows. If it is, the caption must state the atom-retention difference.

**Test:** `test_pruning_is_accounted` — a 10-atom input with one 3-atom satellite must report
`pruned_atoms == 3` and `AtomsRetained% == 70` with `keep_largest_fragment=True`, and
`pruned_atoms == 0` without.

**Effect on conclusions:** "R5-best = 59.5 % validity, new state of the art" becomes "R5-best =
59.5 % validity *at 60 % atom retention*, i.e. a 28-atom generator; on BBB+ yield per 1000
generated it is 215 vs 135 for V5-w5, and that comparison is legitimate." The yield claim
survives; the validity claim does not.

---

### F4 — P0 — `BBB%` denominator is *valid molecules*, and it is the headline column

**Where:** `src/utils/evaluation.py:666` `bbb_permeability_rate`.

**What is wrong:** `return sum(p > 0.5) / len(valid)`. Two arms with 40 % and 5 % validity are
compared on a denominator that differs by 8×, and a model with 2 valid molecules reports 50 %
BBB.

**Evidence:** §2 column "BBB yield". pre-RL 136/1000 vs V6-full 7/1000 — a 19× difference that
the printed 34.2 % vs 1.4 % understates as 24×, and neither number is the yield.

**Fix:** report both, with counts and CIs.

```python
def bbb_rates(mols, classifier) -> dict[str, float]:
    """BBB+ rate over valid molecules AND over all generated samples.

    ``BBB%`` keeps its historical meaning (of valid) for continuity; the
    comparable number across arms is ``BBB_yield%`` (of all samples).
    """
    valid = [m for m in mols if m is not None]
    if classifier is None:
        return {"BBB%": None, "BBB_yield%": None, "n_bbb": 0}
    if not valid:
        return {"BBB%": None, "BBB_yield%": 0.0, "n_bbb": 0}
    probs = classifier.predict_batch(valid)
    k = sum(1 for p in probs if p > 0.5)
    return {
        "BBB%": 100.0 * k / len(valid),
        "BBB%_ci95": list(wilson_ci(k, len(valid))),
        "BBB_yield%": 100.0 * k / len(mols),
        "BBB_yield%_ci95": list(wilson_ci(k, len(mols))),
        "n_bbb": k,
        "n_valid": len(valid),
        "n_total": len(mols),
    }
```

**Test:** `test_bbb_denominators` — 10 samples, 2 valid, 1 BBB+ ⇒ `BBB% == 50.0`,
`BBB_yield% == 10.0`.

---

### F5 — P0 — Means over 2 valid molecules are printed at 4 decimals as if they were measurements

**Where:** `src/utils/evaluation.py:785` `evaluate`, `print_metrics`, and the LaTeX row in
`generate_and_eval.py:518`.

**What is wrong:** `QED 0.448`, `LogP 0.367`, `ScaffDiv 0.500` for V7-nosnr come from
**2** molecules. Nothing in `metrics` carries the denominator, so downstream code
(`scripts/generate_paper_figures.py`, `ui/app.py`) cannot know.

**Fix:** put the counts *inside* the metric dict and make the printer honest.

```python
MIN_REPORTABLE_N = 50

def print_metrics(metrics: dict[str, float]) -> None:
    n = int(metrics.get("n_valid") or 0)
    for k, v in metrics.items():
        if isinstance(v, float) and n < MIN_REPORTABLE_N and k not in COUNTS:
            print(f"{k:<20} {'n/a (n=%d)' % n:>10}")
        else:
            ...
```

and `_save_eval_outputs` writes

```python
payload = {
  "metrics": metrics,           # carries n_valid / n_total / n_bbb
  "num_valid": ..., "num_total": ...,
  "reportable": n_valid >= MIN_REPORTABLE_N,
  ...
}
```

`Validity` and `ConnectedValidity` are exempt: they are proportions over `num_total` and are
legitimate at any n — but they must carry CIs (F6).

**Test:** `test_small_n_is_not_reported_as_a_mean` — 2-molecule eval prints `n/a (n=2)` for
QED and still prints `Validity`.

---

### F6 — P1 — No uncertainty anywhere; single seed

**Where:** whole suite. `grep -r "wilson\|confidence"` → 0 matches.

**What is wrong:** 43.0 % vs 39.8 % (V5-w5 vs pre-RL) is read as a win; with n=200/1000 the CIs
are [36.3, 49.9] and [36.8, 42.9] — overlapping. Conversely R5-best vs V5-w5 is [52.6, 66.1] vs
[36.3, 49.9] — separated, but only because of F3.

**Fix:** one helper, used by every percentage:

```python
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, in percent. Returns (low, high); (nan, nan) if n == 0."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (100.0 * max(0.0, centre - half), 100.0 * min(1.0, centre + half))
```

Add `--seeds 42 123 456` to `generate_and_eval.py`: loop the *whole* generation+eval per seed,
report `median [min, max]` per metric. Table I may only carry a single-seed row if it is
labelled `seed 42, no replication`.

**Test:** `test_wilson_matches_known_values` — `wilson_ci(0, 200) == (0.0, 1.8…)`,
`wilson_ci(200, 200) == (98.2…, 100.0)`, symmetry `wilson_ci(k) == 100 - reversed wilson_ci(n-k)`.

---

### F7 — P0 — The `relaxed` table is a structural no-op and has already corrupted one row

**Where:** `src/utils/evaluation.py:594` `relax_molecule`, `generate_and_eval.py:526`.

**What is wrong:** relaxation moves coordinates only. Bonds — and therefore
`Validity`, `ConnectedValidity`, `BondsPerMol`, `Uniqueness`, `QED`, `LogP`, `BBB%`,
`ScaffDiv`, `ScaffCov` — were fixed at construction and cannot change. Every `metrics_relaxed`
in the repo is byte-identical to `metrics` on 16/16 keys (verified for V5-w5, V6-full,
R5-best, V6-radius). It is not a second measurement; it is a duplicate.

Worse: `outputs/eval_pre_rl/metrics.json` has `metrics.BBB% = 34.17` but
`metrics_relaxed.BBB% = 0.0` — a `bbb_recompute` block patched one table and not the other.

**Fix:**
1. Keep `--relax`, but change what it reports: after relaxation recompute only the
   *geometry-dependent* quantities (`NN_Outside%`, `NN_p50`, `Rg`, `FragmentsPerMol` on the
   relaxed conformer) and set the invariant keys to `None` with `"invariant": true`.
2. Never write `metrics_relaxed` without also writing `"relaxed_metrics_are": "geometry-only"`.
3. Fix the pre-RL relaxed row (or delete it) — a row where the same molecules report 34.17 %
   and 0.0 % BBB cannot stay in the artifact set.
4. Paper text: state once that relaxation is a geometry control with no effect on graph
   metrics, instead of implying a second table.

**Test:** `test_relaxed_table_cannot_change_graph_metrics` — relax a molecule, assert the key
set of the relaxed payload excludes `Validity`/`BBB%`, or that they equal the raw ones.

---

### F8 — P1 — Stored `latex_row` goes stale and silently changes shape

**Where:** `generate_and_eval.py:518-520`, `scripts/generate_paper_figures.py:94`,
`src/utils/visualization.py:102`.

**What is wrong:** `latex_row` is frozen at eval time from `list(metrics.keys())`. Then
`bbb_recompute` edits `metrics` and not the row (`eval_pre_rl`: row says `0.0000` in the BBB
column, `metrics` says `34.17`). Rows gain keys depending on flags (`Veber_filtered%`,
gate keys, `Validity sd`, `n seeds` in the K rows) and
`plot_metric_comparison_table` unions keys across rows, so **column alignment depends on which
rows are present**.

**Fix:** stop storing the row; render it.

```python
# src/utils/evaluation.py
METRIC_ORDER = ("Validity", "ConnectedValidity", "BBB%", "BBB_yield%",
                "BondsPerAtom", "MeanAtoms", "AtomsRetained%", "FragmentsPerMol",
                "NN_Outside%", "QED", "LogP", "IntDiv_p", "ScaffDiv", "Uniqueness",
                "Novelty", "SNN", "Lipinski%", "Veber%", "CNS_MPO", "n_valid", "n_total")

def latex_row(metrics, order=METRIC_ORDER) -> str:
    cells = []
    for k in order:
        v = metrics.get(k)
        cells.append("---" if v is None else
                     f"{v:.4f}" if isinstance(v, float) else str(v))
    return "  &  ".join(cells) + r" \\"
```

`_save_eval_outputs` writes `"metrics_schema": 2` and no `latex_row`; the figure script
computes it. Anyone reading a 4-decimal cell can now map it to a column unambiguously.

**Test:** `test_latex_row_is_derived_not_stored` — mutate a metric dict, re-render, assert the
row changed; assert fixed column count for dicts with extra keys.

---

### F9 — P1 — The `-1.0` sentinel for "no oracle" leaks into arithmetic and is falsy-checked

**Where:** `src/utils/evaluation.py:791` (`else -1.0`), `ui/app.py:240`,
`ui/utils/mol_loader.py:102`.

**What is wrong:** `BBB% = -1.0` is a valid float; it sorts below real values, averages into
totals, and renders as `-1.0 %`. `mol_loader`/`app` guard with
`m.get('ConnV') or m.get('ConnectedValidity', 0.0)`, and `or` treats a genuine `0.0` as missing.

**Fix:** `None` in the payload (`null` in JSON), plus `"oracle": {"path":…, "val_auroc":…} or
None` in a `metrics_meta` block; consumers use explicit `is None` checks and render `—`.
That also kills the `-1.0` QM9 rows in `figures/table1_metrics.tex`.

---

### F10 — P1 — `connectivity()` reports "0 %" when there are no valid molecules

**Where:** `src/utils/evaluation.py:580` (`if not valid or total == 0: return {… 0.0 …}`).

**What is wrong:** `Validity 0.0` and `ConnectedValidity 0.0` are printed for
`eval_lambda2` and `eval_central_v2` (both `num_valid: 0`) as if connectivity had been
measured. `0.0` and "undefined" lead to different decisions.

**Fix:** covered by the F2 snippet — `nan` (JSON `null`) with `n_valid: 0` alongside. Downstream
renders `—` and the row is flagged `reportable: false`.

---

### F11 — P0 — `best.pt` is chosen by a 16-molecule probe whose score ignores validity

**Where:** `train.py:187` (`"score": conn_frac - nn_outside`), `train.py:893`,
`configs/central_v7_bbb_nosnr.yaml:89` (`num_samples: 16`), `_quick_validity` default
`ddim_steps=20`.

**What is wrong:** three independent defects stacked:
1. **Validity is not in the score.** A model that emits 16 valid, connected molecules out of
   200 scores the same as one that emits 16 valid out of 16. `conn_frac` is computed over
   *valid* molecules only (`train.py:181`).
2. **n = 16.** The 95 % CI on a 6–50 % rate at n=16 spans ~25 points.
3. **ddim 20** is a different sampler from the eval protocol (ddim 200 / eta 1.0 / quadratic).

**Evidence:** epoch 3400 saved with probe `validity 6.2 %, conn 100 %, nn_out 8.4 % →
score 91.6`. Same checkpoint at 200 samples / ddim 50: `validity 33.0 %, conn 0.0 %`. One of the
two numbers is wrong by 27 points and the checkpoint was chosen on it.

**Fix:**

```python
# train.py:181-190
n_valid = len(valid_mols)
conn_over_valid = 100.0 * connected / max(n_valid, 1)
# Expected yield, not conditional rate: validity gates connectivity.
score = (100.0 * n_valid / max(len(mols), 1)) / 100.0 * conn_over_valid - nn_outside
if n_valid < _PROBE_MIN_VALID:      # 4
    score = float("-inf")
```

Also: `configs/*.yaml generation.num_samples: 16 -> 64`, `_quick_validity(ddim_steps=50)`, and
record the probe settings in the checkpoint (`"probe": {...}`) so `best.pt` is auditable.

**Test:** `tests/test_checkpoint_selection.py::test_probe_prefers_the_valid_model` — fake
sampler A produces 16 parseable connected molecules out of 16; B produces 4 parseable out of
16. B must not win.

**Effect:** V7-nosnr would not have saved epoch 3400. This is the highest-value single change
in the document: it decides which weights every later eval measures.

---

### F12 — P1 — Two rulers, two files: the geometry axis is not in `metrics.json`

**Where:** `scripts/diagnose_bbb_clump.py` (a *separate* run) vs `evaluate()`.

**What is wrong:** `bonds_per_atom`, `fragments_per_mol`, `largest_fragment_fraction`,
`nn_outside_window_pct`, `nn_p50`, `radius_of_gyration` live only in `clump.json`, produced by
a second invocation that frequently was not run (`rescue_r5_best_n200`, `eval_v6_bbb_radius`
have no `clump.json` at all). `scripts/run_rescue_generation.sh` then prints a table mixing
`metrics.json` and `clump.json` with `nan` for half the rows.

**Fix:** fold the geometry summary into `evaluate()` (it is cheap: one `N²` distance matrix per
molecule, already computed for `nn_outside_window_pct`), store under the same `metrics` dict,
and keep `clump.json` for the extended diagnostics (pair counts, Pearson, per-atom maps).

**Test:** `test_evaluate_carries_the_geometry_axis` — key set contains
`{BondsPerAtom, FragmentsPerMol, NN_Outside%, NN_p50, Rg, MeanAtoms}`.

---

### F13 — P2 — `nn_outside_window_pct` under-counts orphans and lies about single atoms

**Where:** `src/utils/evaluation.py:601`.

**What is wrong:** atoms whose element is missing from `_COVALENT_RADII` hit `continue` — they
can never bond, yet are not counted as outside. A 1-atom molecule returns `0.0` (perfect
score).

**Fix:** count unknown-Z atoms as outside; return `nan` for `n < 2`; additionally report the
per-molecule mean (`NN_Outside%` is currently atom-weighted).

---

### F14 — P2 — `Novelty` mixes unique and non-unique counting

**Where:** `src/utils/evaluation.py:455`.

**What is wrong:** the numerator counts *unique* novel SMILES (`if s not in seen`), the
denominator counts *all* valid molecules. With 100 % uniqueness in every current arm this is
masked; the moment duplicate collapse appears the number is wrong.

**Fix:** `novel_unique / n_unique` (MOSES definition) with both counts recorded.

---

### F15 — P1 — Table I mixes protocols and n without saying so

**Where:** `scripts/generate_paper_figures.py:48` `TABLE_RUNS`, `:94` `load_table_rows`.

**What is wrong:** `load_table_rows` reads `metrics` only — it ignores `num_total`, `run`
(sampler, ddim, eta, relax, pruning flags) and `reportable`. Rows with n=200 and n=1000 sit in
one table; the V6-radius row (ddim 200 + relax) and the V7 probe row (ddim 50, no relax) differ
by 10 validity points *on the same checkpoint* (`43.0` vs `33.0`).

**Fix:** each row loads `{metrics, num_valid, num_total, run}` and the table prints
`n` and a protocol tag; `generate_paper_figures` exits non-zero if two rows in the same block
disagree on `ddim_steps/eta/step_schedule/relax/keep_largest_fragment` unless one carries
`"protocol_exception": "<reason>"`.

**Test:** `test_table_rejects_mixed_protocols` — two rows with different `ddim_steps` ⇒ exit 2.

---

### F16 — P2 — The generated size distribution is never checked, and validity is size-biased

**Where:** `generate_and_eval.py:347` (`_build_atom_count_histogram(full_dataset)`).

**What is wrong:** sizes are drawn from the histogram (mean 45.9), but the *surviving*
molecules are systematically off: V5-w5 38.6 atoms, V6-full **56.7** (larger than requested —
impossible for a representative pass set), V7-nosnr: the only 2 molecules that survived are
8 atoms each. So validity is a size-conditional rate and nothing records it.

**Fix:** report `MeanAtoms_requested` vs `MeanAtoms_kept` and a two-sample KS statistic
(`SizeMatch_KS`, scipy) — pass gate `KS < 0.1` and `0.8 ≤ kept/requested ≤ 1.2`. Also seed
the atom-count draws per (seed, index) rather than per batch, so n=200 is a prefix of n=1000
and runs are nested-comparable.

---

### F17 — P2 — The UI displays fabricated numbers next to real ones

**Where:** `ui/utils/mol_loader.py:44-53` (hard-coded "BBBP Ground Truth": `Validity 100`,
`ConnV 100`, `BBB% 76.5`, `BondsPerMol 24.2`), `ui/app.py:236-243`
(`delta=f"{conn_v - 8.75:.2f}% vs Base"`, `"-5.4 vs Clump"`, `"+12.4 (Clumped)"`).

**What is wrong:** those references and deltas are invented constants rendered in the same
`st.metric` widgets as measured values. `BondsPerMol 24.2` for real BBBP is also wrong (F2:
47.3).

**Fix:** compute the reference row with `evaluate()` over `load_bbbp` at startup (cache in
`st.cache_data`) and compute deltas against the selected run's baseline or omit them.

---

## 4. Canonical schema v2

Every `metrics.json` from now on:

```json
{
  "metrics_schema": 2,
  "metrics": {
    "Validity": 59.5, "Validity_ci95": [52.6, 66.1], "n_valid": 119,
    "ConnectedValidity": 59.5, "ConnectedValidity_ci95": [52.6, 66.1],
    "BBB%": 36.1, "BBB%_ci95": [28.1, 45.1],
    "BBB_yield%": 21.5, "BBB_yield%_ci95": [16.4, 27.7], "n_bbb": 43, "n_total": 200,
    "MeanAtoms": 27.7, "MeanAtoms_requested": 45.9, "AtomsRetained%": 60.3,
    "SizeMatch_KS": 0.41,
    "BondsPerMol": 29.1, "BondsPerAtom": 1.051, "BondsPerAtom_rel": 1.016,
    "FragmentsPerMol": 1.0, "ConnectedFrac": 100.0,
    "NN_Outside%": 0.0, "NN_p50": 1.09, "Rg": 2.9,
    "QED": 0.568, "LogP": 0.941, "IntDiv_p": 0.893, "ScaffDiv": 0.840,
    "Uniqueness": 100.0, "Novelty": 0.0, "SNN": 0.41, "ScaffCov": 0.31,
    "Lipinski%": 100.0, "Veber%": 80.7, "CNS_MPO": 3.34
  },
  "reportable": true,
  "num_valid": 119, "num_total": 200,
  "metrics_meta": {
    "oracle": {"path": "checkpoints/bbb_oracle.pt", "val_auroc": 0.91},
    "reference": {"dataset": "bbbp", "n": 1628, "coverage": 1.0,
                  "BondsPerAtom": 1.035, "MeanAtoms": 45.9, "FragmentsPerMol": 1.04},
    "relaxed_metrics_are": "geometry-only",
    "post_hoc": ["keep_largest_fragment"],
    "protocol": {"ddim_steps": 200, "eta": 1.0, "step_schedule": "quadratic",
                 "relax": true, "bond_tolerance": null, "min_fragment_atoms": 1,
                 "max_bonds_ratio": 0.0, "perceive_bonds": false},
    "versions": {"torch": "...", "rdkit": "...", "git": "..."}
  },
  "run": { "...existing args..." },
  "metrics_relaxed": { "...": "geometry-only keys; graph metrics omitted" }
}
```

Rules:

1. A rate is never reported without `n` and a CI.
2. A comparison is never made between rows whose `reportable`, `AtomsRetained%` or `protocol`
   differ, unless the difference is printed next to the number.
3. `BondsPerMol` never appears in prose without `BondsPerAtom`.
4. `None` means "not measured"; `0.0` means "measured zero". No `-1` sentinels.

---

## 5. Implementation order

| step | files | change | gate |
|---|---|---|---|
| **1 (P0)** | `train.py` | F11: validity-gated probe score, `n≥64`, ddim 50, `-inf` under 4 valid | `tests/test_checkpoint_selection.py` |
| **2 (P0)** | `src/utils/evaluation.py` | `wilson_ci`, `connectivity()` v2 (b/atom, atoms, frags, nan), `bbb_rates`, `evaluate` v2 with counts + geometry + `reference` | `tests/test_metrics_v2.py` (new) |
| **3 (P0)** | `generate_and_eval.py` | F1 reference set from SMILES; F3 stats plumbing; F5 reportability; `--seeds`; drop stored `latex_row` | new tests |
| **4 (P0)** | `generate_and_eval.py`, `src/utils/evaluation.py` | F7 relaxed payload = geometry-only; repair `outputs/eval_pre_rl/metrics.json` | test |
| **5 (P1)** | `scripts/generate_paper_figures.py`, `src/utils/visualization.py` | F15 canonical order, protocol check, `None → —` | test |
| **6 (P1)** | `ui/utils/mol_loader.py`, `ui/app.py` | F17 real reference row, delete invented deltas | manual |
| **7 (P2)** | `scripts/diagnose_bbb_clump.py`, `scripts/run_rescue_generation.sh` | read the new keys from `metrics.json`, keep `clump.json` optional; seeded reference sample | — |

Then regenerate the record:

```bash
# 1. Re-evaluate the four rows that carry conclusions, same protocol, 3 seeds
for ckpt in checkpoints/central_v5_bbb/best.pt \
            checkpoints/central_v6_bbb_radius/best.pt \
            checkpoints/central_v7_bbb_nosnr/best.pt; do
  .venv/bin/python generate_and_eval.py --checkpoint "$ckpt" \
    --num_samples 1000 --ddim_steps 200 --eta 1.0 --step_schedule quadratic \
    --relax --output_dir "outputs/v2_$(basename $(dirname $ckpt))" --seeds 42 123 456
done

# 2. R5-best, labelled as the post-hoc arm it is
.venv/bin/python generate_and_eval.py --checkpoint checkpoints/central_v5_bbb/best.pt \
  --num_samples 1000 --ddim_steps 200 --eta 1.0 --step_schedule quadratic \
  --keep_largest_fragment --valence_weight 0.5 \
  --output_dir outputs/v2_r5_best --seeds 42 123 456

# 3. Rebuild the table (now protocol-checked)
.venv/bin/python scripts/generate_paper_figures.py
```

`outputs/*` stays intact; v2 rows are written alongside so the old numbers remain auditable.

---

## 6. Acceptance checklist

- [ ] `wilson_ci` present for every percentage in `metrics.json`; `n` in the same dict.
- [ ] `BondsPerAtom` + `MeanAtoms` + `AtomsRetained%` in every row; no prose bond claim without them.
- [ ] Reference set coverage ≥ 95 % for BBBP; `Novelty` no longer 100.0 on every arm by construction.
- [ ] `BBB%` and `BBB_yield%` both present with counts; the table prints yield.
- [ ] `metrics_relaxed` contains only geometry keys; `eval_pre_rl` inconsistency resolved.
- [ ] No `-1` sentinels; `None` renders as `—`; `nan` for undefined connectivity.
- [ ] `best.pt` selection uses a validity-gated score with `n ≥ 64` (re-train V7-nosnr if the arm is kept).
- [ ] Table I refuses mixed protocols and mixed atom-retention.
- [ ] `roundtrip`: re-running one existing config reproduces its `Validity` to the float, proving the bond ruler is unchanged (all changes are additive keys).

---

## 7. Statements in the record that these fixes contradict

1. `progress.md` §13.14/§13.15/§15.13 — "over-bonded soup (32–33 bonds/mol vs ~19 real)".
   **Wrong reference.** It is 0.78–0.86 bonds/atom against 1.035 real: **under**-bonded.
2. "R5-best 59.5 % validity is the best arm" — true only with the label
   `fragment-pruned, 60 % atom retention, 27.7 mean atoms`.
3. "V6 bbb radius has the same validity as V5 (43.0 %) but worse connectivity" — the two 43.0s
   are different molecules (60.7 vs 38.6 atoms, 0.409 vs 0.961 b/atom). Not comparable.
4. "V7-nosnr hypothesis disproved on 16 samples" — it is disproved on **200 samples, 0 BBB+
   yield**, which is a stronger and properly measured statement.
5. "Novelty 100 %, SNN 0.09" on every BBB arm — both were computed against 20 molecules.
6. `outputs/eval_pre_rl/metrics.json` raw vs relaxed BBB% (34.17 vs 0.0) — one of the two is not
   a measurement; both must be regenerated.

---

## 8. Out of scope

* Any change to the sampler, guidance terms, or training recipe. This document only changes
  what is *measured* and how it is *reported*; F11 is included because model selection is a
  measurement decision, and it cannot be fixed downstream.
* Re-tuning `TOLERANCE_MATRIX`. The ground-truth control (§1) shows the ruler is at 97 %
  on real molecules; leave it alone and recalibrate only if the control drops.
* Deleting historical `outputs/` directories. Old rows stay as the audit trail; new rows are
  added with `metrics_schema: 2`.
