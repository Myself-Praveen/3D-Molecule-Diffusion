# Contributing to 3D-Molecule-Diffusion

Thank you for considering a contribution to this research project. This
document explains how to set up the environment, follow the project
conventions, and submit changes.

## Project context

This repository replaces GraphGANFed's MLP-GAN generator with an E(3)-equivariant
diffusion model over 3D molecular coordinates, keeping its federated protocol
(FedAvg → FedProx, IID vs non-IID by molecular formula). The research analysis,
recommendations, and phased plan live in `.idea/` — read
`.idea/03_implementation_plan.md` before proposing large changes so your work
fits the roadmap.

## Getting started

1. Fork or branch from `main`.
2. Set up the environment (see the README's Installation section, or run
   `./setup_env.sh` on Linux/macOS / `.\setup_env.ps1` on Windows).
3. Verify the installation:

   ```bash
   python test_setup.py      # dependency, GPU, RDKit, and QM9 checks
   python -m pytest tests/ -q   # full unit test suite must pass
   ```

The first QM9 load downloads several gigabytes into `data/`.

## How to contribute

### Reporting bugs

Open an issue that includes:

- What you did (exact command and config file)
- What you expected vs what happened (paste the full traceback if any)
- Environment details: OS, GPU/CPU, `python -c "import torch; print(torch.__version__)"`

For numerical issues in sampling/training (NaN losses, exploding coordinates),
include the config used and whether the model was trained from scratch.

### Proposing features

Open an issue first for anything larger than a small fix. Reference the
relevant phase/step in `.idea/03_implementation_plan.md` where applicable.
Features that change experimental methodology (losses, metrics, partitioning)
must state how results remain comparable with the paper's protocol.

### Submitting pull requests

- Create a feature branch (`feat/<topic>` or `fix/<topic>`).
- Keep PRs focused: one logical change per PR.
- Update tests for any behavior change; new modules need unit tests under
  `tests/`.
- Update the README when adding user-facing entry points or configs.
- Run before pushing:

  ```bash
  python -m pytest tests/ -q
  ```

## Code conventions

- Python 3.10+, type hints on public functions (`from __future__ import annotations`).
- No code comments unless they explain *why*; keep docstrings for public APIs.
- Follow existing module structure:
  - Model/diffusion math → `src/models/`
  - Samplers → `src/sampling.py`
  - Losses/objectives → `src/objectives.py`
  - Federated logic → `src/fed/`
  - Chemistry/metrics → `src/utils/evaluation.py`
- All runs are configuration-driven via YAML in `configs/`; do not hardcode
  hyperparameters inside training loops.
- Seed all randomness (`set_seed`) so experiments are reproducible.

## Research-code specifics

- **Correctness of physics/chemistry matters**: changes to diffusion schedules,
  centering, equivariance, valence logic, or metric definitions require a test
  demonstrating the invariant (e.g., noise mean ≈ 0 per molecule, rotation
  equivariance, posterior probabilities summing to 1).
- **Comparability**: don't silently change defaults that affect published
  numbers (`configs/*.yaml`, seeds, splits). Note any intentional changes in
  the PR description.
- **Performance claims**: report timing on both CPU and CUDA when touching
  hot paths (kNN graph construction, message passing, samplers).

## Commit messages

Use concise, imperative one-liners, e.g.:

```
feat: add FedProx proximal term to local trainer
fix: clamp predicted x0 to prevent DDIM coordinate blow-up
docs: document federated configuration knobs
```

## License

By contributing, you agree that your contributions will be licensed under the
project's license once one is added (see the README License section).
