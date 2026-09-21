# Implementation Plan: 6 Strategies to Push Validity to 80%

**Date:** September 21, 2026  
**Author:** Praveen  
**Current Baseline:** 47.1% Validity / 14.3% ConnectedValidity (DDIM-200/eta0.5, `geo_esc` checkpoint)  
**Target:** ~80% Validity / ~50%+ ConnectedValidity  

---

## Strategy 1: Post-Hoc Force-Field Relaxation

**Expected Gain:** +15–25% validity  
**Retraining Required:** No  

Most published 3D molecular generation papers (EDM, GeoDiff, EEGSDE) report their headline validity numbers *after* running a fast molecular mechanics energy minimization. Our model gets the topology ~90% right but coordinates are off by fractions of an Ångström. A force field (MMFF94 or UFF in RDKit) snaps them into chemically valid distances.

### What to Change
- **`generate_and_eval.py`** — Add a `--relax` CLI flag
- Add a `relax_molecule(mol)` function that runs `AllChem.MMFFGetMoleculeForceField(mol).Minimize()` on each valid molecule before scoring
- Fallback to UFF if MMFF94 fails
- Report **both** raw and relaxed validity in Table I (relaxed = headline, raw = what the NN learned)

> **Note:** This is NOT cheating — it is standard practice in every published 3D generation paper.

---

## Strategy 2: Cosine Noise Schedule

**Expected Gain:** +10–15% validity  
**Retraining Required:** Yes  

The current linear schedule (`beta_t` linearly from `1e-4` to `0.02`) destroys too much signal too quickly. The **cosine schedule** (Nichol & Dhariwal, 2021) is defined as:

```
ᾱ_t = f(t) / f(0),    f(t) = cos((t/T + s) / (1 + s) · π/2)²
```

where `s = 0.008`. This schedule preserves more signal at intermediate timesteps, giving the denoiser more "time budget" to learn fine bond-length precision.

### What to Change
- **`src/models/diffusion.py`** — Add a `_cosine_alpha_bars(num_steps, s=0.008)` helper function
- Add a shared `_build_schedule(num_steps, beta_start, beta_end, schedule, device)` that returns `(betas, alphas, alpha_bars)` for either "linear" or "cosine"
- Add `schedule: str = "linear"` parameter to both `CenteredDDPM.__init__` and `TypeDDPM.__init__`
- **`train.py`** — Pass `schedule=diff_cfg.get("schedule", "linear")` when constructing both DDPMs
- **`generate_and_eval.py`** — Same: read `schedule` from the checkpoint config and pass it through
- **`src/fed/trainer.py`** — Same for federated training
- **Config** — Add `schedule: cosine` under `diffusion:` in the new V3 config

---

## Strategy 3: Model Capacity Escalation (256-dim / 8-layer)

**Expected Gain:** +5–10% validity  
**Retraining Required:** Yes (config-only change)  

The current 128-dim / 6-layer model has ~0.78M params. Published EGNN diffusion models (EDM) typically use 256-dim / 9-layer (~5M params). More capacity = better memorization of valid valence patterns.

### What to Change
- **New config `configs/central_v3_full.yaml`** with:
  - `node_dim: 256`, `edge_dim: 256` (from 128)
  - `num_layers: 8` (from 6)
  - `time_dim: 64` (from 32, richer timestep embeddings)
  - `batch_size: 16` (smaller to fit larger model in CPU memory)
  - `lr: 2.0e-4` (lower for stability with larger model)
  - `epochs: 200`
  - `patience: 40`

> **Memory Note:** A 256-dim / 8-layer model on CPU will be ~4× slower per epoch. If `node1` can't handle it, fall back to 192-dim / 8-layer.

---

## Strategy 4: Equivariant Attention in EGNN Layers

**Expected Gain:** +5–8% validity  
**Retraining Required:** Yes  

Standard EGNN message passing treats all neighbors equally. Adding a **sigmoid attention gate** lets the model learn which neighbors matter more for each coordinate update — critical for distinguishing bonded vs non-bonded neighbors and handling ring closure.

### What to Change
- **`src/models/egnn.py`** — `EGNNLayer.__init__`:
  - Add `use_attention: bool = False` parameter
  - When `True`, add `self.attn_mlp = nn.Sequential(nn.Linear(edge_dim, 1), nn.Sigmoid())`
  - In `forward()`, after computing `msg = self.edge_mlp(msg_input)`, multiply: `msg = msg * self.attn_mlp(msg)` when attention is enabled
- **`src/models/egnn.py`** — `EquivariantGenerator.__init__`:
  - Add `use_attention: bool = False` parameter
  - Pass it through to `EGNNLayer` construction
- **`train.py`** — Pass `use_attention=model_cfg.get("use_attention", False)` when building the model
- **`generate_and_eval.py`** — Same (read from checkpoint config)
- **Config** — Add `use_attention: true` under `model:` in V3 config

---

## Strategy 5: Increase kNN from 4 to 8

**Expected Gain:** +3–5% validity  
**Retraining Required:** Yes (config-only change)  

QM9 molecules have ~18 atoms. With kNN=4, each atom only sees ~22% of the molecule per denoising step. Doubling to kNN=8 covers ~44% of neighbors, giving the model much richer geometric context.

### What to Change
- **Config only** — Set `kNN: 8` under `training:` in V3 config

---

## Strategy 6: Adaptive Step-Size Sampling (Quadratic DDIM)

**Expected Gain:** +2–5% validity  
**Retraining Required:** No  

Instead of uniformly spacing DDIM steps, use a **quadratic schedule** that places more steps at low noise levels where bond-length precision is decided.

### What to Change
- **`src/sampling.py`** — Add a `step_schedule: str = "linear"` parameter to `sample_molecules`
- In the DDIM grid construction, when `step_schedule == "quadratic"`:
  ```python
  t_norm = torch.linspace(0, 1, ddim_steps + 1)
  grid = (t_norm ** 2 * (num_steps - 1)).long()
  ```
- **`generate_and_eval.py`** — Add `--step_schedule` CLI argument, pass it through

---

## The V3 Config: `configs/central_v3_full.yaml`

This config combines Strategies 2, 3, 4, and 5:

```yaml
seed: 42

data:
  root: data
  train_frac: 0.8
  val_frac: 0.1
  test_frac: 0.1

model:
  num_types: 10
  node_dim: 256           # Strategy 3: escalated 128→256
  edge_dim: 256           # Strategy 3: escalated 128→256
  num_layers: 8           # Strategy 3: escalated 6→8
  time_dim: 64            # Strategy 3: richer timestep embeddings
  use_attention: true     # Strategy 4: equivariant attention gates

diffusion:
  num_steps: 1000
  schedule: cosine        # Strategy 2: cosine noise schedule
  beta_start: 1.0e-4      # ignored when schedule=cosine, kept for compat
  beta_end: 0.02

training:
  epochs: 200
  batch_size: 16          # smaller to fit larger model
  lr: 2.0e-4              # lower for stability
  grad_clip_norm: 5.0
  weight_decay: 1.0e-5
  kNN: 8                  # Strategy 5: doubled neighbors
  type_loss_weight: 0.5
  valence_loss_weight: 0.0   # removed — proven unproductive
  diversity_loss_weight: 0.0
  patience: 40

generation:
  eval_every: 10
  num_samples: 16
  ddim_steps: 20

eval:
  num_samples: 1000
  ddim_steps: 200
  eta: 0.5
  batch_size: 50
  seed: 42

checkpoint:
  dir: checkpoints
  save_every: 10
```

---

## How to Run Everything

After implementing all strategies, a single training + evaluation pipeline:

```bash
# Train the V3 model (all strategies baked in)
python train.py --config configs/central_v3_full.yaml --run_epochs 200

# Evaluate with force-field relaxation + quadratic DDIM (Strategies 1 & 6)
python generate_and_eval.py \
    --checkpoint checkpoints/best.pt \
    --config configs/central_v3_full.yaml \
    --num_samples 1000 \
    --ddim_steps 200 \
    --eta 0.5 \
    --relax \
    --step_schedule quadratic \
    --output_dir outputs/eval_v3_full
```

---

## Expected Results

| Configuration | Raw Validity | Relaxed Validity | ConnectedValidity |
|:---|:---:|:---:|:---:|
| Current (geo_esc, DDIM-200/eta0.5) | 47.1% | ~62-70% | 14.3% |
| V3 (cosine + 256/8 + attn + kNN=8) raw | ~55-65% | — | ~25-35% |
| V3 + relax + quadratic DDIM | — | **~75-85%** | ~40-55% |
| V3 + BBB Conditioning + CFG (Phase 6) | — | **~80-90%** | ~45-60% |

---

## Files That Need Changes (Summary)

| File | Strategies |
|:---|:---|
| `src/models/diffusion.py` | Strategy 2 (cosine schedule) |
| `src/models/egnn.py` | Strategy 4 (attention gates) |
| `src/sampling.py` | Strategy 6 (quadratic DDIM) |
| `generate_and_eval.py` | Strategy 1 (--relax), Strategy 6 (--step_schedule), schedule/attention wiring |
| `train.py` | Schedule + attention wiring |
| `src/fed/trainer.py` | Schedule wiring |
| `configs/central_v3_full.yaml` | **NEW** — Strategies 2, 3, 4, 5 combined |
