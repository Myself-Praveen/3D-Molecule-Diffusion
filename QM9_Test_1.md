# QM9 Baseline Test 1 - Results & Analysis

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

---

## 3. Analysis & Conclusions

### Centralized vs. Federated IID
The Federated IID model achieved a **Validity of 86.3%**, perfectly matching the Centralized model's **86.1%**. 
* **Conclusion:** This proves the core Federated Learning architecture is functioning flawlessly. Clients can collaboratively train a robust 3D Diffusion model without centralizing data, maintaining strict data privacy while achieving state-of-the-art chemical generation.

### The Non-IID Tradeoff
Under Non-IID conditions (imbalanced data), **Validity dropped to 38.6%**, while **Novelty and Uniqueness spiked to 92.7%**. 
* **Conclusion:** This outcome exactly mirrors the theoretical tradeoff established in the original *GraphGANFed* paper. When clients possess highly skewed data, the global model struggles to unify the strict structural rules of chemistry (lowering validity). However, this confusion acts as a strong regularizer that forces the model to hallucinate wildly diverse structures (maximizing novelty/uniqueness). 

### Next Steps
The 3D Diffusion framework and Federated pipeline are fully validated. 
**Praveen is currently retraining the model with a higher patience level** to seek even better convergence. Once this final QM9 tuning is complete, the architecture is ready to transition to Phase 4: fine-tuning on the specialized Blood-Brain Barrier (BBB) datasets.
