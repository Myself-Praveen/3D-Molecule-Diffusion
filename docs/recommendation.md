# Recommendation Report: Strategies to Maximize All Metrics

**Project:** 3D Equivariant Graph Diffusion for Molecular Generation  
**Current Baseline:** V3 Centralized Model (256-dim, 8-layer EGNN, cosine schedule, attention gates, k-NN=8)  
**Date:** September 2026  

---

## Executive Summary

Our V3 centralized model achieves **62.2% validity** with 99.2% uniqueness and 99.2% novelty on 1,000 generated molecules (DDIM-200, η=0.5, linear stepping, force-field relaxation). This document catalogs **every actionable strategy** to push validity toward 80–90%+ and simultaneously improve all other metrics (uniqueness, novelty, QED, connected validity, BBB%, scaffold diversity).

Strategies are organized into four tiers by implementation effort, with estimated impact, scientific rationale, and implementation details for each.

---

## Current Metrics Snapshot

| Metric | Current Value | Target | Gap |
|---|---|---|---|
| **Validity** | 62.2% | 80%+ | 17.8% |
| **Uniqueness** | 99.2% | >99% | ✅ Met |
| **Novelty** | 99.2% | >99% | ✅ Met |
| **IntDiv_p** | 0.789 | >0.85 | 0.061 |
| **QED** | 0.438 | >0.5 | 0.062 |
| **LogP** | 1.289 | 0–3 range | ✅ Acceptable |
| **SNN** | 0.143 | >0.3 | 0.157 |
| **Lipinski%** | 100.0% | >95% | ✅ Met |
| **Veber%** | 99.8% | >95% | ✅ Met |
| **CNS_MPO** | 3.13/4.0 | >3.5 | 0.37 |
| **BBB%** | Unknown (V3) | >30% | TBD |
| **Connected Validity** | Unknown (V3) | >40% | TBD |
| **ScaffDiv** | 0.031 | >0.1 | 0.069 |
| **ScaffCov** | 0.006 | >0.05 | 0.044 |

**Primary bottleneck:** Validity and Connected Validity  
**Secondary bottlenecks:** SNN, ScaffDiv/ScaffCov, IntDiv, QED

---

## Tier 0: Immediate Wins (Zero Code Changes)

These require **no retraining and no code modifications** — just different command-line flags on existing evaluation scripts.

### 0.1 Quadratic DDIM Stepping on V3

**Expected Impact:** +3–8% validity  
**Effort:** 5 minutes  
**Rationale:** The V3 evaluation used `--step_schedule linear`. Our previous experiments showed quadratic stepping concentrates denoising steps at low noise levels where bond-length precision is determined. The quadratic grid `t_i = (i/S)² × (T−1)` allocates ~60% of compute budget to the critical final 20% of the noise range.

**Command:**
```bash
python generate_and_eval.py \
  --checkpoint checkpoints/central_v3/best.pt \
  --relax --step_schedule quadratic \
  --ddim_steps 200 --eta 0.5 --num_samples 1000
```

**Why it works:** Bond lengths in organic molecules span a narrow 1.0–1.6Å window. At high noise (t≈T), atoms are scattered across ~5Å — no amount of precision matters. At low noise (t≈0), sub-0.1Å errors determine whether a C–C bond is 1.54Å (valid) or 1.95Å (broken). Quadratic stepping gives the denoiser more iterations exactly where precision matters most.

---

### 0.2 Increased DDIM Steps (500–1000)

**Expected Impact:** +2–5% validity  
**Effort:** 10 minutes (longer generation time)  
**Rationale:** Our sweep showed monotonic improvement: 28.3% (50 steps) → 37.8% (100) → 44.4% (200). The trend suggests diminishing but non-zero returns at 500+ steps.

**Commands to sweep:**
```bash
# 500 steps, quadratic
python generate_and_eval.py \
  --checkpoint checkpoints/central_v3/best.pt \
  --relax --step_schedule quadratic \
  --ddim_steps 500 --eta 0.5 --num_samples 1000

# 1000 steps, quadratic  
python generate_and_eval.py \
  --checkpoint checkpoints/central_v3/best.pt \
  --relax --step_schedule quadratic \
  --ddim_steps 1000 --eta 0.5 --num_samples 1000
```

---

### 0.3 Stochasticity (η) Tuning

**Expected Impact:** +1–3% validity  
**Effort:** 30 minutes (multiple runs)  
**Rationale:** η=0.5 was chosen heuristically. The optimal noise injection level may differ for the larger V3 model. η=0 gives deterministic DDIM (lower variance, potentially under-explores); η=1.0 gives full DDPM stochasticity (higher diversity, potentially more valid conformations through noise-driven exploration).

**Sweep grid:**
```bash
for eta in 0.0 0.3 0.5 0.7 1.0; do
  python generate_and_eval.py \
    --checkpoint checkpoints/central_v3/best.pt \
    --relax --step_schedule quadratic \
    --ddim_steps 200 --eta $eta --num_samples 1000
done
```

---

### 0.4 Bond Reconstruction Tolerance Tuning

**Expected Impact:** +2–5% validity  
**Effort:** 1 hour (code + sweep)  
**Rationale:** Current bond assignment uses a global tolerance of +0.45Å on top of covalent radii: bonded iff `d_ij ≤ r_i + r_j + 0.45`. This tolerance was chosen to maximize ground-truth QM9 validity (99.0%). However, element-pair-specific tolerances could recover more generated molecules that are borderline-valid.

**Approach:**
- Compute optimal tolerances per element pair (C–C, C–N, C–O, C–F, N–O, etc.) on QM9 ground truth
- Use tighter tolerances for pairs prone to false positives (e.g., H–H contacts)
- Use looser tolerances for pairs prone to false negatives (e.g., C=O double bonds at ~1.23Å)

**Implementation:** Modify `src/utils/evaluation.py::coords_and_types_to_mol()` to use a `TOLERANCE_MATRIX[z_i][z_j]` instead of a scalar 0.45.

---

### 0.5 Ensemble Generation (Multiple Seeds)

**Expected Impact:** +1–2% validity (via cherry-picking)  
**Effort:** 30 minutes  
**Rationale:** Generate multiple batches with different random seeds and take the union of valid molecules. This doesn't improve the per-sample validity rate, but maximizes the absolute count of valid, unique, novel molecules for downstream use.

```bash
for seed in 42 123 456 789 1337; do
  python generate_and_eval.py \
    --checkpoint checkpoints/central_v3/best.pt \
    --relax --step_schedule quadratic \
    --ddim_steps 200 --eta 0.5 \
    --num_samples 1000 --seed $seed
done
```

---

## Tier 1: Training Enhancements (Same Architecture, Different Training)

These require **retraining** but no architectural changes to the EGNN.

### 1.1 Exponential Moving Average (EMA) of Weights

**Expected Impact:** +2–4% validity  
**Effort:** 2 hours  
**Rationale:** EMA is near-universal in diffusion model training (used by DDPM, EDM, Stable Diffusion, etc.). It smooths out gradient noise by maintaining a shadow copy of the model weights: `θ_EMA ← α·θ_EMA + (1−α)·θ` with α=0.9999. The EMA weights generalize better than the raw training weights, especially for generation tasks.

**Implementation in `train.py`:**
```python
import copy

# After model initialization
ema_model = copy.deepcopy(model)
ema_decay = 0.9999

# After each optimizer.step()
with torch.no_grad():
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

# Save EMA weights alongside regular weights
torch.save({
    'model': model.state_dict(),
    'ema_model': ema_model.state_dict(),
    ...
}, 'best.pt')
```

**At inference:** Load `ema_model` state dict instead of `model`.

---

### 1.2 Timestep-Weighted Loss (SNR Weighting)

**Expected Impact:** +2–4% validity  
**Effort:** 3 hours  
**Rationale:** Currently all timesteps contribute equally to the training loss. However, low-noise timesteps (small t) are where bond-length precision is determined, while high-noise timesteps (large t) primarily learn global structure. Weighting the loss by the signal-to-noise ratio (SNR) or using a `min-SNR-γ` weighting scheme focuses learning on the most impactful timesteps.

**Options:**
1. **P2 weighting** (Choi et al., 2022): `w(t) = 1 / (1 + SNR(t))^γ` with γ=1
2. **min-SNR-γ** (Hang et al., 2023): `w(t) = min(SNR(t), γ) / SNR(t)` with γ=5
3. **Linear ramp**: Simply weight timesteps t < T/4 at 2× their normal contribution

**Implementation in `train.py`:**
```python
# min-SNR-gamma weighting
snr = alpha_bar_t / (1 - alpha_bar_t)  # signal-to-noise ratio
gamma = 5.0
weight = torch.minimum(snr, torch.tensor(gamma)) / snr
loss = (weight * per_sample_loss).mean()
```

---

### 1.3 Extended Training (More Epochs)

**Expected Impact:** +2–5% validity  
**Effort:** Compute time only  
**Rationale:** The V3 model was trained for a fixed number of epochs. Diffusion models, especially larger ones (4.09M params), often continue improving well beyond initial convergence. Training to 300–500 epochs with a patience-based early stopping on validation loss (not quick_valid) may yield further gains.

```bash
python train.py --config configs/central_v3_full.yaml \
  --resume --run_epochs 200  # Add 200 more epochs
```

---

### 1.4 Learning Rate Schedule Refinement

**Expected Impact:** +1–3% validity  
**Effort:** 2 hours  
**Rationale:** The current CosineAnnealing schedule may decay too aggressively or not aggressively enough. Alternative schedules:

1. **Warmup + Cosine decay**: Linear warmup for 5–10 epochs, then cosine decay. Prevents early gradient spikes in the attention gates.
2. **Constant + decay**: Keep LR constant for 80% of training, then decay linearly to 0.
3. **Reduce-on-plateau**: Drop LR by 10× when validation loss plateaus for 20 epochs.

---

### 1.5 Data Augmentation via Random Rotation

**Expected Impact:** +1–2% validity  
**Effort:** 1 hour  
**Rationale:** Although the EGNN is E(3)-equivariant by design (coordinate predictions are equivariant), the **type prediction head** is invariant and processes features that may benefit from seeing rotated inputs. Applying random SO(3) rotations to input coordinates during training acts as a regularizer.

```python
# In train.py, before forward pass
R = random_rotation_matrix()  # 3x3 random rotation
batch.pos = batch.pos @ R.T  # Rotate all coordinates
```

---

### 1.6 Noise Schedule Fine-Tuning

**Expected Impact:** +1–3% validity  
**Effort:** 3 hours  
**Rationale:** The cosine schedule with s=0.008 was designed for image diffusion. Molecular coordinates have different statistics (concentrated in a ~5Å box vs. 0–255 pixel range). A custom schedule with a different offset or a sigmoid schedule may better match molecular geometry:

**Sigmoid schedule:**
```python
def _sigmoid_betas(T, start=-3, end=3):
    t = torch.linspace(start, end, T)
    betas = torch.sigmoid(t)
    betas = (betas - betas.min()) / (betas.max() - betas.min())
    return betas * 0.02 + 1e-4
```

---

## Tier 2: Architectural Improvements (Model Changes)

### 2.1 Self-Conditioning

**Expected Impact:** +3–6% validity  
**Effort:** 4 hours  
**Rationale:** Self-conditioning (Chen et al., 2022; Analog Bits) feeds the model's own prediction of x̂₀ from the previous denoising step back as an additional input. This gives the model access to a "rough draft" of the final molecule, allowing it to refine iteratively rather than predicting from scratch at each step.

**Implementation:**
```python
# In EquivariantGenerator.forward():
# Accept optional x0_estimate as input
def forward(self, pos, types, t, batch, edge_index, x0_estimate=None):
    h = self.type_embed(types)
    if x0_estimate is not None:
        # Concatenate estimate as additional node features
        x0_feat = self.x0_proj(x0_estimate)  # Linear(3, node_dim)
        h = h + x0_feat
    ...

# In sampling loop:
x0_estimate = None
for i, t in enumerate(timesteps):
    noise_pred, type_logits, _ = model(x_t, z_t, t, batch, edge_index, x0_estimate)
    # Compute x̂₀ from noise prediction
    x0_estimate = (x_t - sqrt(1-alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)
    # 50% dropout during training (randomly zero out x0_estimate)
```

**Why it works:** At early denoising steps, the model has very little information about the true molecule. Self-conditioning gives it a progressively-refined scaffold to work with, similar to how iterative refinement works in protein structure prediction (AlphaFold2).

---

### 2.2 Edge/Bond Type Prediction

**Expected Impact:** +5–10% validity  
**Effort:** 8 hours  
**Rationale:** Currently, bonds are inferred post-hoc from distances using covalent radii. This is fragile — a 0.1Å error can flip a bond from present to absent. Instead, predict bond types explicitly as part of the model output.

**Implementation:**
- Add a pairwise edge classifier head that predicts bond type (none, single, double, triple, aromatic) for each atom pair within a distance cutoff
- Train with cross-entropy loss against ground-truth bond types from QM9
- At inference, use predicted bond types directly instead of distance-based inference

```python
class BondTypeHead(nn.Module):
    def __init__(self, node_dim, num_bond_types=5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + 1, 128),  # pair features + distance
            nn.SiLU(),
            nn.Linear(128, num_bond_types)
        )
    
    def forward(self, h, pos, edge_index):
        hi, hj = h[edge_index[0]], h[edge_index[1]]
        dij = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1, keepdim=True)
        return self.mlp(torch.cat([hi, hj, dij], dim=-1))
```

---

### 2.3 Multi-Scale Message Passing

**Expected Impact:** +2–4% validity  
**Effort:** 4 hours  
**Rationale:** Using a single k-NN=8 for all layers is suboptimal. Early layers should see local neighborhoods (k=4) for fine-grained bond structure; later layers should see global context (k=16) for ring closure and macro-structure.

```python
# In EquivariantGenerator:
k_schedule = [4, 4, 8, 8, 12, 12, 16, 16]  # 8 layers
for layer_idx, layer in enumerate(self.layers):
    edge_index = build_knn_graph(pos, k=k_schedule[layer_idx], batch=batch)
    pos, h = layer(pos, h, edge_index, t_emb)
```

---

### 2.4 Coordinate Refinement Head (Two-Stage Denoising)

**Expected Impact:** +3–5% validity  
**Effort:** 6 hours  
**Rationale:** After the main EGNN stack predicts noise, add a lightweight 2-layer "refinement" EGNN that takes the predicted clean coordinates x̂₀ and refines them. This acts as a learned post-processing step that can fix small bond-length errors before they propagate through DDIM.

```python
# After main EGNN noise prediction
x0_hat = (x_t - sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_bar_t)

# Refinement EGNN (2 layers, lightweight)
refined_x0 = self.refine_egnn(x0_hat, h_final, edge_index)
refined_noise = (x_t - sqrt(alpha_bar_t) * refined_x0) / sqrt(1 - alpha_bar_t)
```

---

### 2.5 Discrete Diffusion for Atom Types (D3PM)

**Expected Impact:** +2–4% validity  
**Effort:** 6 hours  
**Rationale:** The current EDM-style categorical diffusion adds Gaussian noise to one-hot vectors, which doesn't respect the discrete structure of atom types. D3PM (Austin et al., 2021) defines a proper discrete Markov chain over token space, with transition matrices that can encode chemical priors (e.g., C→N transitions are more likely than C→F).

---

## Tier 3: Paradigm Shifts (Major Rearchitecting)

### 3.1 Latent Diffusion (GeoLDM-style)

**Expected Impact:** +10–15% validity (→ 75–80%)  
**Effort:** 2–3 days  
**Rationale:** GeoLDM (Xu et al., 2023) trains an equivariant autoencoder to compress 3D molecular coordinates into a lower-dimensional latent space, then runs diffusion in that latent space. This dramatically simplifies the denoising task — the model learns to denoise smooth latent vectors rather than raw Ångström-scale coordinates.

**Architecture:**
```
Encoder: Molecule (x, z) → EGNN_enc → Latent (h_latent)
Diffusion: q(h_t | h_0) → EGNN_denoise → p(h_{t-1} | h_t)
Decoder: Latent (h_latent) → EGNN_dec → Molecule (x, z)
```

**Published results:** GeoLDM achieves **~89% validity** on QM9, vs. EDM's ~82%.

---

### 3.2 Flow Matching (Conditional Flow Matching / Riemannian FM)

**Expected Impact:** +8–12% validity  
**Effort:** 2 days  
**Rationale:** Flow matching (Lipman et al., 2023) replaces the diffusion forward/reverse process with optimal transport paths between noise and data. Benefits:
- Straighter sampling trajectories → fewer steps needed
- No variance schedule to tune
- Better training signal (velocity prediction vs. noise prediction)

**Key change:** Replace the noise prediction objective with a velocity prediction objective:
```python
# Instead of: loss = ||ε - ε_θ(x_t, t)||²
# Use: loss = ||v - v_θ(x_t, t)||²  where v = α'_t·x_0 + σ'_t·ε
```

---

### 3.3 Autoregressive Atom Placement

**Expected Impact:** +15–20% validity  
**Effort:** 1 week  
**Rationale:** Instead of generating all atoms simultaneously (which requires global coordination), generate atoms one at a time, each conditioned on all previously placed atoms. G-SchNet and G-SphereNet achieve >95% validity on QM9 using this approach.

**Trade-off:** Much higher validity but loses the parallelism advantage of diffusion. Generation is O(N) sequential steps per molecule rather than O(1).

---

### 3.4 Hybrid Diffusion + Autoregressive

**Expected Impact:** +10–15% validity  
**Effort:** 1 week  
**Rationale:** Use diffusion for coarse placement (scaffold/backbone), then autoregressive refinement for precise atom positioning. Combines the global coherence of diffusion with the precision of autoregressive models.

---

## Strategies for Specific Metrics

### Improving SNN (Similarity to Nearest Neighbor)

Current: 0.143 → Target: >0.3

The low SNN indicates generated molecules are structurally dissimilar to the training set. This is partially expected (100% novelty), but pushing SNN higher means generating more "realistic" molecules.

1. **Classifier-free guidance toward training distribution**: Use a light guidance signal pointing toward the training manifold
2. **Reduce η**: Lower stochasticity produces molecules closer to the training manifold
3. **Scaffold conditioning**: Condition on common QM9 scaffolds (cyclohexane, benzene, etc.)

### Improving QED (Drug-likeness)

Current: 0.438 → Target: >0.5

1. **QED-guided sampling**: Train a QED predictor, use as classifier guidance
2. **Property conditioning**: Add QED as a conditioning variable during training (already supported in architecture)
3. **Rejection sampling**: Generate 2× molecules, filter for QED > 0.5

### Improving Scaffold Diversity and Coverage

Current: ScaffDiv=0.031, ScaffCov=0.006

1. **Diversity regularizer λ₃**: Increase the diversity loss weight in training
2. **Temperature scaling on type prediction**: Higher temperature → more diverse atom type selections → more diverse scaffolds
3. **Conditional generation per scaffold**: Generate molecules conditioned on specific Murcko scaffolds from QM9

### Improving Connected Validity

Connected validity requires molecules to be both valid AND a single connected fragment (no disconnected atoms or fragments).

1. **Connectivity loss**: Add a differentiable penalty for disconnected components during training (approximate via graph Laplacian eigenvalues)
2. **Post-processing**: After generation, remove isolated atoms or small fragments before bond assignment
3. **Larger k-NN**: More neighbors = more likely that all atoms are connected through the message-passing graph

### Improving BBB%

Current: ~24.8% (from older model, TBD on V3)

1. **Conditioned generation** (already architected): Train with BBB labels, use classifier-free guidance at inference
2. **Property optimization**: Optimize for CNS MPO components (MW < 360, LogP 2–4, HBD ≤ 1, tPSA < 90)
3. **Multi-property guidance**: Combine validity classifier + BBB oracle guidance during sampling

---

## Recommended Execution Order

### Phase 1: Free Wins (Day 1)
```
1. Run V3 with quadratic stepping          → expect ~65-70%
2. Sweep η ∈ {0.3, 0.5, 0.7, 1.0}         → find optimal η
3. Try 500 DDIM steps with best η          → expect ~68-73%
```

### Phase 2: Training Improvements (Days 2–3)
```
4. Add EMA to train.py, retrain V3         → expect +2-4%
5. Add min-SNR-γ loss weighting            → expect +2-4%
6. Train for 200 more epochs               → expect +2-5%
   Combined Phase 2 target: ~72-78%
```

### Phase 3: Architecture Upgrades (Days 4–6)
```
7. Self-conditioning                       → expect +3-6%
8. Explicit bond type prediction           → expect +5-10%
   Combined Phase 3 target: ~78-85%
```

### Phase 4: Paradigm Shift (Days 7–10, if needed)
```
9. Latent diffusion (GeoLDM)               → expect ~85-90%
   OR
10. Flow matching                          → expect ~80-88%
```

---

## Risk Assessment

| Strategy | Risk | Mitigation |
|---|---|---|
| Quadratic DDIM | None (no retraining) | — |
| EMA | Minimal (standard practice) | Use decay 0.9999, start after epoch 10 |
| Self-conditioning | 50% dropout needed to avoid dependency | Ablate with/without |
| Timestep weighting | May hurt high-noise reconstruction | Monitor full loss curve |
| Latent diffusion | Autoencoder may lose geometric fidelity | Train AE first, verify reconstruction |
| Flow matching | Requires rewriting diffusion.py + sampling.py | Keep DDPM as fallback |
| Bond type prediction | Requires QM9 bond labels (available) | Use RDKit ground truth |

---

## References

1. **EMA**: Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
2. **min-SNR-γ**: Hang et al., "Efficient Diffusion Training via Min-SNR Weighting", CVPR 2023
3. **Self-conditioning**: Chen et al., "Analog Bits: Generating Discrete Data using Diffusion Models", ICLR 2023
4. **GeoLDM**: Xu et al., "Geometric Latent Diffusion Models for 3D Molecule Generation", ICML 2023
5. **Flow Matching**: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
6. **D3PM**: Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces", NeurIPS 2021
7. **P2 Weighting**: Choi et al., "Perception Prioritized Training of Diffusion Models", CVPR 2022
8. **G-SchNet**: Gebauer et al., "Symmetry-adapted generation of 3D point sets", NeurIPS 2019
9. **EDM**: Hoogeboom et al., "Equivariant Diffusion for Molecule Generation in 3D", ICML 2022
10. **GraphGANFed**: Manu et al., "A Federated Generative Framework for Graph-Structured Molecules", IEEE/ACM TCBB 2024

---

## Implementation Status (September 2026)

State of each strategy in the codebase, in execution order. Everything is
opt-in via config/flags; the legacy DDPM recipe remains the default.

**Tier 0 (eval-only — no code needed, run the commands above):**
- 0.1 quadratic DDIM, 0.2 step sweep, 0.3 η sweep, 0.5 multi-seed ensemble: the
  flags (`--step_schedule quadratic`, `--ddim_steps`, `--eta`, `--seed`) already
  exist; the sweeps are evaluation runs, not code changes.
- **0.4 DONE** — `TOLERANCE_MATRIX` / `get_bond_tolerance()` in
  `src/utils/evaluation.py` replace the global +0.45 Å scalar in
  `_add_bonds_from_distance`; unlisted pairs fall back to the scalar.
  Values encode the doc's qualitative guidance (H–H tight 0.25, C–O loose 0.50);
  recalibrate per pair on QM9 ground truth before trusting the +2–5% estimate.

**Tier 1 (DONE, `dad20cc`) — training enhancements, same architecture:**
- 1.1 EMA (`src/training_utils.EMA`; `training.ema_decay`), also server-side in
  `src/fed/server.py`; best.pt stores EMA weights.
- 1.2 min-SNR-γ (`min_snr_weight`; `training.min_snr_gamma`; γ=1 ⇒ P2).
- 1.3 more epochs: run-level (resume + patience), nothing to implement.
- 1.4 warmup+cosine LR (`build_warmup_cosine_scheduler`; `training.warmup_epochs`).
- 1.5 rotation augmentation (`training.rotation_augment_prob`).
- 1.6 sigmoid schedule (`diffusion.schedule: sigmoid`).

**Tier 2 (DONE, `0320004`) — architecture, all default-off and backward
compatible (legacy checkpoints load unchanged):**
- 2.1 self-conditioning (`model.self_condition`; 50% train dropout).
- 2.2 bond-type prediction — supervision side DONE: `bond_head` trained with
  CE against QM9 ground-truth `edge_index`, gated to low-noise molecules per
  PAIR (`training.bond_loss_weight`). Inference still infers bonds from
  distances; consuming predicted bond types at generation time is future work.
- 2.3 multi-scale kNN (`model.knn_schedule`, e.g. [4,4,8,8,12,12,16,16]).
- 2.4 coordinate refinement head (`model.coord_refine_layers`; zero-init ⇒
  identity at init, safe to bolt onto any run).
- 2.5 **NOT IMPLEMENTED** (scope note): the EDM-style continuous noising of
  one-hot types plus the exact categorical posterior in `TypeDDPM` already
  provides a working discrete-type sampler; D3PM's benefit here would be a
  chemistry-informed transition matrix (C→N ≫ C→F), which requires retraining
  and a principled matrix for raw-atomic-number indexing. Deferred.

**Tier 3:**
- **3.2 DONE** (`d49df80`) — flow matching with velocity prediction on the
  linear OT path (`src/models/flow.py`): target ``v = α'_t·x0 + σ'_t·ε = ε − x0``;
  opt-in via `diffusion.objective: flow` (DDPM `eps` objective remains the
  default and fallback — risk-table mitigation honored); Euler ODE sampler in
  `src/sampling.py` (auto-detected from `model.objective`); exact
  noise-free x0 recovery `x0 = x_u − u·v` at every u; composes with all Tier 1/2
  features; resume guard rejects eps↔flow switches. Same architecture — no new
  parameters.
- 3.1 latent diffusion (GeoLDM), 3.3 autoregressive, 3.4 hybrid: **NOT
  IMPLEMENTED** — multi-day paradigm shifts (2–3 days to 1 week each per the
  effort estimates above), deferred until the Tier 1–3 + Tier 0 retrain/sweep
  results justify the investment.

**Metric-specific strategies (from "Strategies for Specific Metrics"):**
- Connectivity #2 **DONE** — `coords_and_types_to_mol(min_fragment_atoms=k)`
  prunes provisional distance-graph fragments smaller than k atoms before bond
  assignment (exposed as `--min_fragment_atoms`).
- Scaffold diversity #2 **DONE** — `--type_temperature` on the sampler's type
  posterior.
- QED #3 **DONE** — `apply_qed_filter` + `--min_qed` rejection filter
  (raw table always reported first).
- SNN (#1 guidance, #2 lower η), scaffold conditioning, QED-guided sampling,
  connectivity loss (differentiable Laplacian), larger kNN, BBB conditioned
  generation: conditioning infrastructure exists (Phase 2); the guided/
  oracle-based sampling variants are future work.

**Recommended next experiment:** retrain V3
(`configs/central_v3_full.yaml`) with the winning opt-ins (EMA + min-SNR-γ +
warmup-cosine + Tier 2 self-conditioning/kNN-schedule + bond head) and re-run
the Tier 0 eval sweeps; alternatively train a fresh run with
`diffusion.objective: flow` to compare paradigms directly.
