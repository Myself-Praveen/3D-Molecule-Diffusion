# 3D-Molecule-Diffusion

## 3D Equivariant Graph Diffusion for Molecular Generation

This research repository extends GraphGANFed (Manu et al., 2024) from 2D graph-matrix generation to 3D molecular diffusion. It trains an E(3)-equivariant graph neural network (EGNN) as the denoiser of a DDPM over QM9 atomic coordinates and atom types, evaluates generated molecules with MOSES-style chemistry metrics, and reproduces GraphGANFed's federated protocol (weighted FedAvg → FedProx, IID vs non-IID by molecular formula) with a diffusion generator.

## Motivation

GraphGANFed represents each molecule with 2D adjacency and node-label matrices and uses an MLP-based GAN generator. That representation is useful for topology, but it does not directly model molecular geometry. A 3D diffusion formulation provides a path toward learning spatial conformations while avoiding the adversarial optimization and mode-collapse risks commonly associated with GAN training.

## Current Scope

Phases 1.5–3 of the implementation plan (`.idea/03_implementation_plan.md`) are implemented:

- QM9 loading through PyTorch Geometric, including atomic numbers (`z`) and 3D coordinates (`pos`).
- A timestep-conditioned EGNN denoiser with joint coordinate + atom-type diffusion (EDM-style categorical process) and a bond-type head for chemistry reconstruction.
- Centered DDPM forward noising, ancestral DDPM and DDIM reverse samplers with per-molecule re-centering and x0 clamping.
- MOSES-style metric suite: Validity, Uniqueness, Novelty, IntDiv_p, QED, LogP, SNN (`src/utils/evaluation.py`).
- Federated training over formula-partitioned QM9: IID and non-IID splits mirroring the GraphGANFed protocol, weighted FedAvg aggregation, Flower-compatible client (`src/fed/`).
- FedProx client-side proximal term (configurable μ), FedPer-style personal type/bond heads, and multi-objective losses λ₂ (soft valence penalty) / λ₃ (diversity regularizer) (`src/objectives.py`).

## GraphGANFed Comparison

| Capability | GraphGANFed baseline | This repository |
| --- | --- | --- |
| Representation | 2D adjacency and node-label matrices | 3D coordinates and atomic numbers |
| Generator | MLP within a WGAN-GP pipeline | EGNN denoiser within a DDPM pipeline |
| Geometry | Topological only | Coordinate-aware and translation/rotation equivariant |
| Optimization | Adversarial training | Noise-prediction MSE |
| Distribution setting | Federated, non-IID focus | Federated (IID + non-IID) with a centralized QM9 baseline |
| Personalization | Not studied | FedProx proximal term + FedPer-style personal type/bond heads |

## Project Layout

```text
3D-Molecule-Diffusion/
|-- checkpoints/           # Model checkpoints; generated files are ignored
|-- configs/
|   |-- central.yaml       # Centralized training configuration
|   |-- fed_iid.yaml       # Federated IID / FedAvg configuration
|   `-- fed_niid.yaml      # Federated non-IID / FedProx configuration
|-- data/                  # Downloaded QM9 + persisted partitions; ignored
|-- outputs/               # Samples, plots, and metrics; generated files ignored
|-- src/
|   |-- dataset.py         # QM9 loading and batched 3D data helpers
|   |-- sampling.py        # Ancestral DDPM + DDIM reverse samplers
|   |-- objectives.py      # λ₂ valence penalty, λ₃ diversity regularizer
|   |-- models/
|   |   |-- diffusion.py   # Centered DDPM + categorical TypeDDPM
|   |   `-- egnn.py        # Timestep-conditioned EGNN denoiser
|   |-- fed/
|   |   |-- partition.py   # Formula-based IID / non-IID partitioning
|   |   |-- trainer.py     # Local training loops (FedProx, personal heads)
|   |   |-- client.py      # Flower NumPyClient wrapper
|   |   `-- server.py      # Round-based FedAvg/FedProx simulation
|   `-- utils/
|       |-- evaluation.py  # MOSES-style metric suite + mol reconstruction
|       `-- graph.py       # Vectorized k-NN graph construction
|-- tests/                 # Unit tests for all phases
|-- generate_and_eval.py   # End-to-end generation + metric evaluation
|-- CODE_OF_CONDUCT.md     # Contributor Covenant code of conduct
|-- CONTRIBUTING.md        # Contribution guide
|-- README.md
|-- requirements.txt
|-- setup_env.ps1          # Windows PowerShell setup
|-- setup_env.sh           # Linux/macOS Bash setup
|-- test_setup.py          # Dependency, GPU, RDKit, and QM9 checks
|-- train.py               # Centralized training loop
`-- fed_train.py           # Federated training entry point
```

## Requirements

- Python 3.10
- Conda, Miniconda, or Anaconda
- Internet access for package installation and the first QM9 download
- NVIDIA GPU with a compatible CUDA driver for accelerated training; CPU execution is supported for smoke tests
- Approximately several gigabytes of available storage for QM9 and package caches

## Installation

### Windows PowerShell

```powershell
.\setup_env.ps1
conda activate bio_diffusion
```

The PowerShell script creates the environment when necessary, installs the CUDA 12.1 PyTorch wheel, installs the remaining dependencies, and runs the verification script. On a CPU-only machine, install the CPU PyTorch build instead.

### Linux or macOS

```bash
chmod +x setup_env.sh
./setup_env.sh
conda activate bio_diffusion
```

The Bash script selects the CUDA 12.1 PyTorch index when `nvidia-smi` is available and otherwise selects the CPU index.

### Manual installation

```bash
conda create --yes --name bio_diffusion python=3.10
conda activate bio_diffusion
python -m pip install --upgrade pip
python -m pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cu121
python -m pip install --requirement requirements.txt
```

For CPU-only execution, replace the CUDA index URL with `https://download.pytorch.org/whl/cpu`.

## Verification

Run the setup check after installation:

```bash
python test_setup.py
```

The check reports the Torch version, CUDA device details when visible, imports PyTorch Geometric and e3nn, parses the sample SMILES `CCO`, downloads or loads QM9, and prints the first molecule's `pos` and `z` tensors.

## Training

### Centralized

```bash
conda activate bio_diffusion
python train.py --config configs/central.yaml
```

For each batch the trainer samples one diffusion timestep per molecule, adds per-molecule centered Gaussian noise to coordinates and categorical noise to atom types, builds a coordinate k-NN graph, predicts the spatial noise with the timestep-conditioned EGNN, and optimizes the joint loss `L = L_pos + λ_type·L_type` with an 80:10:10 train/val/test split, early stopping, and checkpointing.

The k-NN construction uses a vectorized pure-PyTorch implementation, so no optional `pyg-lib` binary backend is required.

### Generation & evaluation

```bash
python generate_and_eval.py --checkpoint checkpoints/best.pt --num_samples 1000
```

Samples molecules via DDIM (or full DDPM), reconstructs RDKit molecules with valence-aware bond assignment, and reports Validity, Uniqueness, Novelty, IntDiv_p, QED, LogP, and SNN. Additional flags: `--ddim_steps`, `--eta`, `--batch_size`, `--seed`, `--device`.

## Federated training

Run the simulated GraphGANFed protocol over formula-partitioned QM9:

```bash
python fed_train.py --config configs/fed_iid.yaml    # IID, FedAvg
python fed_train.py --config configs/fed_niid.yaml   # non-IID, FedProx
```

Configuration knobs (see the YAML files):

- `fed.num_clients` / `fed.mode`: K clients with `iid` (class-stratified) or `niid` (Dirichlet-style unbalanced) partitions, persisted to `data/partitions/K{K}_{mode}.json`.
- `fed.proximal_mu`: > 0 enables the FedProx client-side proximal term μ/2·‖w − w_global‖².
- `fed.personal_heads`: keeps type/bond heads local (FedPer-style) while aggregating only the EGNN trunk.
- `training.valence_loss_weight` / `training.diversity_loss_weight`: multi-objective λ₂ and λ₃ from Step 3.3.
- `data.max_molecules`: subsample QM9 for fast smoke runs.

Per-round server loss, per-client losses, validation loss, and periodic DDIM sample validity/uniqueness are logged to `history.json` in the output directory.

## Testing

Run the full unit test suite (noise centering, rotation equivariance, categorical posteriors, partitioning invariants, weighted FedAvg arithmetic, FedProx/personalization behavior, multi-objective losses):

```bash
python -m pytest tests/ -q
```

## Community

- **Contributing** — see [CONTRIBUTING.md](CONTRIBUTING.md) for environment setup, code conventions, and pull-request guidelines.
- **Code of Conduct** — this project follows the [Contributor Covenant](CODE_OF_CONDUCT.md); be respectful and constructive in all project spaces.

## Research Roadmap

Completed:

1. Reverse DDPM + DDIM samplers (`src/sampling.py`).
2. Chemical reconstruction and MOSES-style evaluation (`src/utils/evaluation.py`, `generate_and_eval.py`).
3. Molecule-size and atom-type conditioning (joint categorical diffusion + size-histogram sampling).
4. Federated training with Flower-compatible clients and IID/non-IID partitions (`src/fed/`, `fed_train.py`).
5. FedProx, FedPer-style personal heads, and multi-objective λ₂/λ₃ losses (`src/objectives.py`).

Remaining:

6. Full-scale experiment sweeps: K ∈ {1, 2, 4, 7} × {IID, non-IID} × μ ∈ {0, 0.01, 0.1, 1.0} and the Validity–Uniqueness Pareto plot over (λ₂, λ₃).
7. Results write-up mirroring the paper's experimental axes, including the diffusion-vs-GAN mode-collapse analysis.
8. Optional stretch: classifier-free guidance conditioned on QED-bucket / uniqueness-proxy.

## References

1. Manu, D., Yao, J., Liu, W., and Sun, X. (2024). *GraphGANFed: A Federated Generative Framework for Graph-Structured Molecules Towards Efficient Drug Discovery*. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 21(2), 342-353.
2. Satorras, V. G., Hoogeboom, E., and Welling, M. (2021). *E(n) Equivariant Graph Neural Networks*. ICML 2021.
3. Ho, J., Jain, A., and Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
4. Wu, Z., et al. (2018). *MoleculeNet: A Benchmark for Molecular Machine Learning*. Chemical Science, 9(2), 513-530.

## License

No license file has been added yet. Add a project license before distributing this research code publicly or incorporating it into another project (see the note in [CONTRIBUTING.md](CONTRIBUTING.md)).
