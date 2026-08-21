# 3D-Molecule-Diffusion

## 3D Equivariant Graph Diffusion for Molecular Generation

This research repository explores an extension of GraphGANFed (Manu et al., 2024) from 2D graph-matrix generation to centralized 3D molecular diffusion. The current Phase 1 implementation trains an E(3)-equivariant graph neural network to predict centered Gaussian noise on QM9 atomic coordinates.

## Motivation

GraphGANFed represents each molecule with 2D adjacency and node-label matrices and uses an MLP-based GAN generator. That representation is useful for topology, but it does not directly model molecular geometry. A 3D diffusion formulation provides a path toward learning spatial conformations while avoiding the adversarial optimization and mode-collapse risks commonly associated with GAN training.

## Current Scope

Phase 1 includes:

- QM9 loading through PyTorch Geometric, including atomic numbers (`z`) and 3D coordinates (`pos`).
- A coordinate-updating EGNN implemented with PyTorch and PyTorch Geometric message-passing primitives.
- A centered DDPM forward-noising process with one timestep per molecule in a batch.
- A centralized MSE training loop for predicted versus sampled coordinate noise.
- RDKit, CUDA, PyTorch Geometric, e3nn, and QM9 environment verification.

The current model predicts coordinate noise and does not yet include a reverse-time sampler, bond reconstruction, molecular validity evaluation, or federated training. The `e3nn` dependency is installed for the planned equivariant-model expansion; the current EGNN layer uses explicit E(3)-equivariant coordinate updates.

## GraphGANFed Comparison

| Capability | GraphGANFed baseline | This repository |
| --- | --- | --- |
| Representation | 2D adjacency and node-label matrices | 3D coordinates and atomic numbers |
| Generator | MLP within a WGAN-GP pipeline | EGNN denoiser within a DDPM pipeline |
| Geometry | Topological only | Coordinate-aware and translation/rotation equivariant |
| Optimization | Adversarial training | Noise-prediction MSE |
| Distribution setting | Federated, non-IID focus | Centralized QM9 baseline in Phase 1 |

## Project Layout

```text
3D-Molecule-Diffusion/
|-- checkpoints/           # Model checkpoints; generated files are ignored
|-- data/                  # Downloaded and processed QM9 data; ignored
|-- outputs/               # Samples, plots, and metrics; generated files ignored
|-- src/
|   |-- __init__.py
|   |-- dataset.py         # QM9 loading and batched 3D data helpers
|   |-- models/
|   |   |-- __init__.py
|   |   |-- diffusion.py   # Centered DDPM noise schedule
|   |   `-- egnn.py        # Coordinate denoiser
|   `-- utils/
|       `-- __init__.py    # Reserved for evaluation utilities
|-- README.md
|-- requirements.txt
|-- setup_env.ps1          # Windows PowerShell setup
|-- setup_env.sh           # Linux/macOS Bash setup
|-- test_setup.py          # Dependency, GPU, RDKit, and QM9 checks
`-- train.py               # Centralized Phase 1 training loop
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

Start the centralized Phase 1 loop from the repository root:

```bash
conda activate bio_diffusion
python train.py
```

The default run uses 50 epochs and a batch size of 32. For each batch it samples one diffusion timestep per molecule, adds per-molecule centered Gaussian noise, builds a coordinate k-NN graph, predicts the spatial noise with the EGNN, and optimizes mean squared error.

The training script includes a pure-PyTorch k-NN fallback, so it does not require the optional `pyg-lib` binary backend. This is useful for Windows and CPU environments where a matching PyG extension wheel is unavailable.

## Research Roadmap

1. Add a reverse DDPM sampler for generating noisy-to-clean coordinate trajectories.
2. Add chemical reconstruction and evaluation: validity, uniqueness, novelty, stability, and geometry metrics.
3. Add molecule-size and atom-type conditioning for generation beyond fixed graph inputs.
4. Integrate Flower for simulated federated clients and compare IID and non-IID partitions.
5. Study personalized federated objectives such as FedProx and multi-objective validity/uniqueness/novelty weighting.

## References

1. Manu, D., Yao, J., Liu, W., and Sun, X. (2024). *GraphGANFed: A Federated Generative Framework for Graph-Structured Molecules Towards Efficient Drug Discovery*. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 21(2), 342-353.
2. Satorras, V. G., Hoogeboom, E., and Welling, M. (2021). *E(n) Equivariant Graph Neural Networks*. ICML 2021.
3. Ho, J., Jain, A., and Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
4. Wu, Z., et al. (2018). *MoleculeNet: A Benchmark for Molecular Machine Learning*. Chemical Science, 9(2), 513-530.

## License

No license file has been added yet. Add a project license before distributing this research code publicly or incorporating it into another project.
