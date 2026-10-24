# 3D Equivariant Graph Diffusion for Molecular Generation — Guide

This is a research codebase for training and evaluating a diffusion-based 3D molecular generator, with a federated-learning option and BBB-permeability evaluation on top. It lives at the intersection of equivariant graph networks, diffusion sampling, and chemistry-aware evaluation, and the project's honest status is documented in `progress.md`.

## What this is

A **central training loop** (`train.py`) for a joint position+type EGNN denoiser with optional property conditioning, a **federated path** (`fed_train.py`) for IID/non-IID formula-partitioned splits, and a **full evaluation pipeline** (`generate_and_eval.py` + `src/utils/evaluation.py`) that produces MOSES-style metrics plus BBB-specific, connectivity, and lab-readiness numbers.

The honest summary: the generator can produce float-valid 3D atom arrangements, but reliable **connected, non-over-bonded, chemically sensible molecules** is the hard part and is still the project's open problem. Everything below is documented there.

## Getting started

### Environment

```bash
bash setup_env.sh          # creates the conda env + installs requirements
# or, on Windows:
.\setup_env.ps1
```

Then activate the env and confirm the smoke checks pass:

```bash
conda activate bio_diffusion
python -c "import torch, rdkit, e3nn, torch_geometric; print('ok')"
.venv/bin/python -m pytest tests/test_setup.py -q
```

### Data

- **QM9** is expected at `data/` (130k molecules, processed). The loader in `src/dataset.py` builds PyG `Data(pos, z, ...)` objects.
- **BBB datasets** (BBBP, B3DB) are downloaded and converted to 3D in `src/dataset_bbb.py`, cached under `data/bbbp/` / `data/b3db/` as `processed_3d.pt`. Conversion uses ETKDGv3 + MMFF; failures are tracked and reported.

### A first training run

```bash
# central QM9, short smoke (resume-capable)
OMP_NUM_THREADS=8 .venv/bin/python train.py --config configs/central.yaml --run_epochs 20
```

Training is resumable: `--resume --run_epochs N` continues from `checkpoints/last.pt`. For runs that exceed a session, launch with `setsid` so a tool timeout doesn't kill the process:

```bash
setsid nohup bash -c 'OMP_NUM_THREADS=8 .venv/bin/python train.py --config configs/central_bbb_full.yaml' > logs/bbb_full.log 2>&1 &
```

### A first evaluation

```bash
.venv/bin/python generate_and_eval.py \
  --checkpoint checkpoints/best.pt \
  --num_samples 1000 --ddim_steps 50 \
  --output_dir outputs/eval_smoke
```

With the BBB oracle available:

```bash
.venv/bin/python generate_and_eval.py \
  --checkpoint checkpoints/bbb_full/best.pt \
  --config configs/central_bbb_full.yaml \
  --num_samples 1000 --ddim_steps 200 --eta 1.0 \
  --step_schedule quadratic --relax \
  --min_fragment_atoms 2 --bond_tolerance 0.25 --max_bonds_ratio 1.4 \
  --bbb_oracle models/bbb_oracle.pt \
  --output_dir outputs/eval_bbb_full
```

## What's implemented

- **Diffusion**: centered DDPM over coordinates + categorical TypeDDPM over atom types (`src/models/diffusion.py`), ancestral DDPM + DDIM samplers (`src/sampling.py`).
- **Denoiser**: timestep-conditioned EGNN with joint coordinate/type heads and a bond head (`src/models/egnn.py`).
- **Federated**: Flower-compatible clients, formula-partitioned IID/non-IID QM9 splits, weighted FedAvg/FedProx, personal heads, λ₂/λ₃ losses (`fed_train.py`, `src/fed/`, `src/objectives.py`).
- **BBB pipeline**: ETKDGv3+MMFF SMILES→3D, Murcko-scaffold splits, property caching (`src/dataset_bbb.py`).
- **BBB oracle**: 3-layer GCN + mean-pool + MLP, best val AUROC ~0.91, test ~0.92 (`src/models/bbb_classifier.py`).
- **Conditioning**: classifier-free guidance over class + [QED, LogP, tPSA, MW], label dropout (`src/sampling.py`, `src/fed/trainer.py`).
- **Multi-objective**: soft valence penalty (all-pairs, x0-gated), bond-head supervision, diversity, and a gated rotation-invariant "geofix" arm (Kabsch-aligned MSE + pairwise-distance loss).
- **Eval metrics**: MOSES suite (Validity, Uniqueness, Novelty, IntDiv, QED, LogP, SNN) + BBB-specific (BBB%, ScaffDiv, ScaffCov, Lipinski%, Veber%, CNS_MPO) + connectivity (ConnectedValidity, BondsPerMol, ConnectedFrac) + V5 (bond tolerance cap, over-bond rejection gate, bond-order perception).
- **Lab-readiness stack**: stages 1–7 (bond-order perception, geometry QA, synthetic complexity + PAINS, applicability domain, ADMET proxies, retrosynthesis, six-stage verdict) + an 8th stage that re-runs it end-to-end.
- **Experiment runners**: sweep, ablation, guidance-sweep, and chain scripts that wait on checkpoints and regenerate paper figures.

## Where things stand

The honest current status is in `progress.md`. The short version:

- **QM9 central** runs reproduce connected molecules at non-trivial rates (the V4-eps arm hits ~54% ConnV), and the eval tooling is now calibrated on QM9 ground truth.
- **BBB central** at full budget (200k steps) produces a genuine denoiser that beats the trivial predictor at every timestep, but sampled geometry is still over-bonded atom soup: ~2% connected, ~32 bonds/mol vs ~19 real. Conditioning steers BBB% (up to ~58% at high guidance) but never fixes connectivity.
- Objective-side ablations (λ₂ renormalization, geofix shape supervision) did **not** fix connectivity and are recorded as negatives.
- The current working hypothesis is geometric (sampler + checkpoint selection), and the V5 BBB retrain (V4-eps architecture on BBBP) is the next structure-level attempt.

## How to continue

1. Read `progress.md` §16 first — it records the latest in-session measurement and what was kept vs dropped against the plan.
2. If you want to **reproduce a number**, find the `outputs/eval_bbb_*` directory for the arm and read its `metrics.json` + `run` (the saved run config reproduces the exact command).
3. If you want to **change the bond chain**, look at `src/utils/evaluation.py` (`coords_and_types_to_mol`, `TOLERANCE_MATRIX`, `get_bond_tolerance`) and `generate_and_eval.py` (CLI flags + `run` recording). Calibrate against ground truth before deploying any tolerance change — `scripts/calibrate_bond_tolerance.py`.
4. If you want to **train something new**, copy a config and verify its `checkpoint.dir` before launching (there's an ops incident in `progress.md` §15 item 8 about mistyping it).
5. If you want to **run the lab gate**, `python scripts/lab_readiness_gate.py --eval-dir outputs/eval_bbb_*` and/or `generate_and_eval.py --lab_gate`.

## Conventions

- Reproducibility is now enforced: `generate_and_eval.py` seeds both numpy and torch from `--seed`, so identical flags + seed give identical molecules.
- Every eval output records its full run config in `metrics.json` (`run` key), so any row is regenerable.
- Do not claim relaxed-table gains for connectivity — `--relax` is a no-op for the headline connectivity numbers; the raw table is the honest one.

## References

- The published plan that motivated the latest work is stored in `plan.md`.
- The honest project record is `progress.md`. This guide is the orientation; `progress.md` is the truth.
