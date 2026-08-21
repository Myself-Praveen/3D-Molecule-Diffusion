# 3D Molecule Diffusion

Research scaffold for extending GraphGANFed (Manu et al., 2024) toward a 3D equivariant graph diffusion model on QM9 using PyTorch, PyTorch Geometric, e3nn, and RDKit.

## Quickstart

From this directory, run:

```bash
bash setup_env.sh
conda activate bio_diffusion
python test_setup.py
```

The setup script creates the `bio_diffusion` Conda environment with Python 3.10. It installs the CUDA 12.1 PyTorch wheels when `nvidia-smi` is available and otherwise installs the CPU wheels. The first verification run downloads and processes QM9, so it needs internet access and several gigabytes of disk space.

## Layout

- `data/`: downloaded QM9 data and processed graphs
- `checkpoints/`: model weights
- `outputs/`: generated molecules, metrics, and plots
- `src/dataset.py`: QM9 loading and preprocessing helpers
- `src/models/`: equivariant model implementations
- `src/utils/`: evaluation utilities
