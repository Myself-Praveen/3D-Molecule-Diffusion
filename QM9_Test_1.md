# QM9 Baseline Test 1 - Results & Analysis

> [!CAUTION]
> **RETRACTED — do not cite.** The table below was measured with two compounding
> errors: (1) an atom-index shift in `generate_and_eval.py` that relabeled every
> element (H→C, C→S…), and (2) a distance-threshold bond builder under which
> disconnected fragment soup sanitizes as "valid" (real QM9 ground truth scored
> 0.7%). The 86% figures describe mislabeled fragments, not molecules.
> See `progress.md` §§7–10 and the honest replacement table in §2 below
> (fixed identity mapping + covalent-radii reconstruction + connectivity metrics).

**Date:** September 19, 2026
**Dataset:** QM9 (Standard 3D Molecular Benchmark)
**Goal:** Validate the 3D Equivariant Diffusion model and the Federated Learning architecture before migrating to the Blood-Brain Barrier (BBB) dataset.

---

## 1. Overview of the Run
This test evaluated the complete Phase 1.5 (3D Diffusion) and Phase 3 (Federated Learning) pipelines. The model was trained to generate 3D molecular coordinates and atomic types from scratch, starting from pure Gaussian noise. 

Three separate evaluation environments were tested:
1. **Centralized Baseline:** All data resides on a single machine.
2. **Federated IID:** Data is perfectly distributed among $K$ simulated clients (hospitals), maintaining an identical distribution across all clients.
3. **Federated Non-IID:** Data is skewed and unevenly distributed across clients, mimicking real-world data silos.

---

## 2. Metric Results (MOSES)
For each environment, the AI generated 1,000 novel 3D molecules. The output coordinates and atom types were converted into standard chemical representations and scored using the MOSES benchmark metrics.

| Metric | Centralized | Federated IID | Federated Non-IID |
| :--- | :---: | :---: | :---: |
| **Validity** | 86.1% | 86.3% | 38.6% |
| **Uniqueness** | 53.0% | 47.7% | 92.7% |
| **Novelty** | 53.0% | 47.7% | 92.7% |
| **IntDiv_p** | 0.376 | 0.324 | 0.647 |
| **QED** | 0.553 | 0.242 | 0.274 |
| **LogP** | 1.528 | 11.267 | 9.409 |

*(Retracted — see banner. Honest replacement below.)*

## 2b. Honest Table (Fixed Ruler, 1000 Samples, DDIM-50)

Measured after the identity-mapping fix, covalent-radii reconstruction, and
connectivity metrics (`ConnectedValidity` = % valid AND single-fragment;
QM9 ground truth: 94.7% validity, ~19 bonds/mol):

| Metric | Central (coordfix_v1) | Central (geo_esc) | Central (geo_esc, DDIM-200/eta0.5) |
| :--- | :---: | :---: | :---: |
| **Validity** | 20.8% | 29.9% | 47.1% |
| **ConnectedValidity** | 1.3% | 7.7% | 14.3% |
| **BondsPerMol** | 15.3 | 16.8 | 17.4 |
| **Uniqueness** | 100% | 100% | 100% |
| **QED** | 0.45 | 0.49 | 0.49 |

Federated rows are pending retrains (both pre-fix globals were coord-dead);
they will be evaluated with the adopted DDIM-200/eta0.5 protocol on completion.
The old "IID matches central" claim is **void** — it compared fragment soups.

---

## 3. Analysis & Conclusions

### Centralized vs. Federated IID
The Federated IID model achieved a **Validity of 86.3%**, perfectly matching the Centralized model's **86.1%**. 
* **Conclusion:** This proves the core Federated Learning architecture is functioning flawlessly. Clients can collaboratively train a robust 3D Diffusion model without centralizing data, maintaining strict data privacy while achieving state-of-the-art chemical generation.

### The Non-IID Tradeoff
Under Non-IID conditions (imbalanced data), **Validity dropped to 38.6%**, while **Novelty and Uniqueness spiked to 92.7%**. 
* **Conclusion:** This outcome exactly mirrors the theoretical tradeoff established in the original *GraphGANFed* paper. When clients possess highly skewed data, the global model struggles to unify the strict structural rules of chemistry (lowering validity). However, this confusion acts as a strong regularizer that forces the model to hallucinate wildly diverse structures (maximizing novelty/uniqueness). 

### Next Steps (Updated 20 Sep 2026 — Old Text Below Is Superseded)
Validated: oracle (0.92 AUROC), honest measurement (ground truth 94.7%),
monotonic geometry gains (connected validity 0.0 → 14.3%). Open: pushing
connected validity toward the 65–85% plan target (stop-grad λ₂ retrain,
capacity/schedule), federated Table I rows, then BBB phases.

### Next Steps (Original, Superseded)
The 3D Diffusion framework and Federated pipeline are fully validated. 
**Praveen is currently retraining the model with a higher patience level** to seek even better convergence. Once this final QM9 tuning is complete, the architecture is ready to transition to Phase 4: fine-tuning on the specialized Blood-Brain Barrier (BBB) datasets.
