# Proposed Action Plan — Fixing Validity & Path to Publication

**Date:** September 20, 2026  
**Author:** Praveen  
**Status:** Proposed — Awaiting Team Discussion

---

## 1. Problem Summary

We have run **5 separate experiments** so far. Here are the honest results:

| # | Run | Validity | Verdict |
|---|-----|----------|---------|
| 1 | Centralized (old code) | 86.1% | ⚠️ **FAKE** — inflated by atom-index bug |
| 2 | Federated IID (old code) | 86.3% | ⚠️ **FAKE** — same bug |
| 3 | Federated Non-IID (old code) | 38.6% | ⚠️ **FAKE** — same bug |
| 4 | Centralized v2 (fixed index + λ₂=1.0) | 0.0% | ❌ **COLLAPSED** — λ₂ penalty broke training |
| 5 | Lambda2 experiment | 0.0% | ❌ **COLLAPSED** — same cause |

### What happened?

**Bug 1 — Atom Index Mapping (Runs 1–3):**  
The old `generate_and_eval.py` used a lookup table (`QM9_ATOMIC_NUMBERS`) that shifted every element. Model index `1` was mapped to Carbon instead of Hydrogen. This meant broken molecule fragments with wrong element labels trivially passed RDKit sanitization. The 86% validity was not real.  
**Fix:** Samyra correctly fixed this by using identity mapping (index = atomic number). ✅

**Bug 2 — λ₂ Valence Penalty on Noisy Data (Runs 4–5):**  
After fixing the index bug, Samyra added a soft valence penalty (`λ₂=1.0`) to prevent atoms from clumping together. However, this penalty was applied to `noisy_pos` — the intermediate positions at diffusion timestep *t*, which are deliberately corrupted with random Gaussian noise. Penalizing random noise positions forced the AI to distort its atom-type predictions to "compensate", which completely scrambled the learned chemistry. Result: 0% validity.  
**Fix:** Remove the λ₂ penalty entirely. ✅ (see Step 1 below)

---

## 2. The 4-Step Recovery Plan

### Step 1: Remove the Broken λ₂ Penalty ⏱️ 5 minutes

**Files to change:**
- `configs/central.yaml` → Set `valence_loss_weight: 0.0`

**What to keep:**
- ✅ Samyra's atom-index fix (correct)
- ✅ Sampler hardening — logit clamp ±15, bulletproof multinomial (good defensive code)
- ✅ The `full_pair_index` utility in `objectives.py` (useful later, just don't use it in training)

**Why remove instead of fix?**  
In diffusion models, the training loss operates on noise-corrupted data at timestep *t*. You mathematically cannot measure bond counts or valence on `x_t` because it is deliberately randomized. Chemical validity should be enforced during **generation** (post-hoc filtering or guided sampling), not during training.

---

### Step 2: Retrain Central QM9 with Better Hyperparameters ⏱️ 8–12 hours

**Recommended config changes:**

| Parameter | Current Value | Proposed Value | Reason |
|-----------|--------------|----------------|--------|
| `valence_loss_weight` | 1.0 | **0.0** | Remove broken penalty |
| `epochs` | 100 | **200** | More training with the now-working coordinate head |
| `patience` | 1000 | **30** | Real early stopping instead of effectively disabled |
| `lr` | 1e-3 | **5e-4** | Lower LR = more stable for diffusion models |
| `kNN` | 4 | **8** | More neighbors = richer message passing for geometry |
| `num_layers` | 4 | **6** | Deeper EGNN captures longer-range atomic interactions |
| `batch_size` | 32 | **64** | Better gradient estimates (if memory allows on CPU) |

**Command:**
```bash
python train.py --config configs/central.yaml --run_epochs 200
```

---

### Step 3: Fix Bond Reconstruction Cutoff ⏱️ 10 minutes

The over-connection problem (26 bonds/mol vs ~19 true) should be fixed at **evaluation time**, not training time.

**File:** `src/utils/evaluation.py`  
**Change:** `distance_cutoff` default from `2.5` Å → **`1.8`** Å (standard single-bond covalent radius sum)

This is the correct place to enforce chemical constraints — when assembling the final molecule from predicted coordinates, not during the noise-prediction training loop.

---

### Step 4: Full Experimental Campaign ⏱️ 2–3 days

Once centralized baseline achieves >60% **true** validity:

1. **Retrain Federated models** (IID + Non-IID) with the clean config
2. **Switch dataset** from QM9 → BBB using the already-built `dataset_bbb.py`
3. **Enable conditioning** (`cond_dim: 16`) for BBB-targeted generation
4. **Train BBB oracle** on the real BBB dataset
5. **Run the full experimental grid** for the paper:
   - Centralized vs Federated IID vs Federated Non-IID
   - QM9 baseline vs BBB-targeted
   - Conditioned vs Unconditioned generation
6. **Generate final molecules** and evaluate with all 7 MOSES metrics + BBB oracle scoring

---

## 3. Expected Outcomes After Fixes

| Scenario | Expected Validity | Notes |
|----------|------------------|-------|
| Central (QM9, clean) | **65–85%** | True validity with correct atom mapping + working coordinate head |
| Fed-IID (QM9) | **60–80%** | Slight drop from federated averaging is expected |
| Fed-Non-IID (QM9) | **30–50%** | Known tradeoff from the GraphGANFed paper |
| Central (BBB, conditioned) | **50–70%** | Smaller dataset, but conditioned generation helps |

> **Important:** A true 70% is actually a *better* result than the fake 86%. It means the model is generating molecules with correct element types, correct 3D geometry, AND valid chemical bonds — all from pure noise. That is genuinely impressive for a 3D diffusion model and is publication-worthy.

---

## 4. Timeline to Paper

| Week | Milestone |
|------|-----------|
| **Week 1 (Now)** | Fix config, retrain QM9 centralized baseline, verify true validity |
| **Week 2** | Retrain federated models (IID + Non-IID), evaluate full QM9 grid |
| **Week 3** | Switch to BBB dataset, train conditioned models, run BBB experiments |
| **Week 4** | Generate final molecules, run BBB oracle scoring, compile all tables |
| **Week 5** | Write paper draft, create figures, finalize results |

---

## 5. Team Assignments

- **Praveen:** Config fixes, retraining runs, paper coordination
- **Samyra:** Code review of fixes, federated experiment runs, evaluation analysis
- **Shreya:** Literature review updates, paper writing, figure generation

---

## 6. Key Takeaway

We are NOT starting over. The entire codebase infrastructure is complete and working:
- ✅ 3D Diffusion engine (DDPM + DDIM)
- ✅ Equivariant GNN denoiser (EGNN)
- ✅ Federated Learning system (FedAvg + FedProx)
- ✅ BBB dataset pipeline
- ✅ BBB property conditioning
- ✅ BBB oracle classifier (0.92 AUROC)
- ✅ Full MOSES evaluation suite

We just need to **remove one bad hyperparameter** (`λ₂=1.0 → 0.0`), **tune the config**, and **retrain**. The path to publication is clear.
