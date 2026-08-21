"""Verify the core dependencies and a small QM9 data sample."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            print(f"GPU: {properties.name}")
            print(f"VRAM: {properties.total_memory / (1024**3):.2f} GiB")
            print(f"Compute capability: {properties.major}.{properties.minor}")

        import torch_geometric
        import e3nn
        from rdkit import Chem

        print(f"PyTorch Geometric: {torch_geometric.__version__}")
        print(f"e3nn: {e3nn.__version__}")

        molecule = Chem.MolFromSmiles("CCO")
        if molecule is None or molecule.GetNumAtoms() != 3:
            raise RuntimeError("RDKit could not parse the sample SMILES 'CCO'.")
        print("RDKit: parsed 'CCO' successfully")

        from torch_geometric.datasets import QM9

        dataset_root = Path(__file__).resolve().parent / "data"
        print(f"Loading QM9 into: {dataset_root}")
        dataset = QM9(root=str(dataset_root))
        if len(dataset) < 5:
            raise RuntimeError(f"QM9 contains only {len(dataset)} molecules; expected at least 5.")

        subset = [dataset[index] for index in range(5)]
        first = subset[0]
        if not hasattr(first, "pos") or not hasattr(first, "z"):
            raise RuntimeError("QM9 sample does not contain 'pos' and 'z' fields.")

        print(f"QM9 molecules checked: {len(subset)}")
        print(f"First molecule coordinates (pos):\n{first.pos}")
        print(f"First molecule atomic numbers (z):\n{first.z}")
        print("\033[1mSUCCESS: PyTorch, PyG, e3nn, RDKit, and QM9 are ready.\033[0m")
        return 0
    except Exception as error:  # Provide one actionable failure point for setup checks.
        print(f"\nSETUP CHECK FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
