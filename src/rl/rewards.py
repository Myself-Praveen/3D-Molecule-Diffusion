"""Composite reward for RL fine-tuning of BBB molecule generation (Step 2.1).

Three interface corrections from the corrected plan are load-bearing here and
are pinned by tests:

1. **The oracle is a plain ``nn.Module``.** ``BBBClassifier`` is not a
   Lightning module: there is no ``load_from_checkpoint()`` and no
   ``predict_proba()``. Loading is ``torch.load(...)`` + ``load_state_dict(...)``
   (``ckpt["model_state_dict"]`` or a raw state dict), and scoring goes through
   the project's own inference helpers, which keep the training-time graph
   construction (explicit Hs + distance-gated edges).
2. **Invalid molecules are kept, not dropped.** They score exactly
   ``-1.0`` so they become the *rejected* side of a preference pair — the best
   negative examples available, since the failure mode we want the model to
   stop producing is precisely an invalid clump.
3. **The bond target is measurable.** The plan's ``target_bonds=19`` is a
   heavy-atom count, while this project's ruler counts every bond including
   hydrogens (:data:`REFERENCE_BONDS_PER_MOL` is the measured value for real
   BBBP under exactly that ruler). An absolute target in the wrong unit would
   silently punish every correct molecule, so both are exposed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from rdkit import Chem
from rdkit.Chem import QED
from torch_geometric.data import Data

from src.guidance import COVALENT_RADII
from src.models.bbb_classifier import BBBClassifier

# Measured with scripts/diagnose_bbb_clump.py against 200 real BBBP training
# molecules using the covalent-radii ruler (H-inclusive): 26.88 bonds/mol,
# 0.999 bonds/atom. Pass this as ``target_bonds`` when scoring molecules that
# carry explicit hydrogens; the plan's 19 is the heavy-atom figure.
REFERENCE_BONDS_PER_MOL: float = 26.88
REFERENCE_BONDS_PER_ATOM: float = 1.0

# An unusable sample is not "absent" — it is a strong negative example.
INVALID_REWARD: float = -1.0

DEFAULT_WEIGHTS: dict[str, float] = {
    "bbb": 0.4,
    "validity": 0.2,
    "qed": 0.15,
    "connectivity": 0.15,
    "bonds": 0.1,
    "clash": 0.1,
    "fragments": 0.05,
}

_oracle_cache: dict[str, torch.nn.Module] = {}


def load_bbb_oracle(
    checkpoint_path: str = "models/bbb_oracle.pt",
    device: str | torch.device = "cpu",
) -> torch.nn.Module | None:
    """Load the frozen BBB oracle as a plain ``nn.Module``.

    The corrected plan's call sequence, verbatim in behaviour: manual
    ``torch.load`` with ``weights_only=False``, ``state_dict`` extraction from
    either a wrapped or a bare checkpoint, ``load_state_dict``, ``eval()``.
    Returns ``None`` when the file is missing, so a run without an oracle
    degrades to ``bbb_prob = 0.0`` instead of crashing.
    """
    device = torch.device(device)
    key = f"{checkpoint_path}@{device}"
    if key in _oracle_cache:
        return _oracle_cache[key]
    if not Path(checkpoint_path).exists():
        return None
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = int(ckpt.get("hidden_dim", 128)) if isinstance(ckpt, dict) else 128
    oracle = BBBClassifier(hidden_dim=hidden_dim)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    oracle.load_state_dict(state)
    oracle.eval().to(device)
    _oracle_cache[key] = oracle
    return oracle


def _mol_geometry(mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(pos, z)`` tensors from the molecule's own conformer, or ``None``."""
    if mol is None or mol.GetNumConformers() == 0 or mol.GetNumAtoms() == 0:
        return None
    pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long)
    return pos, z


def bbb_probability(mol: Chem.Mol | None, oracle: torch.nn.Module | None) -> float:
    """BBB+ probability in [0, 1], or 0.0 when unavailable.

    Generated molecules already carry a conformer, so they are scored from
    those coordinates through ``data_to_input`` — the same zero-shift path the
    eval's ``BBB%`` uses — instead of re-running ETKDG for every reward call.
    ``None``/geometry-less inputs fall back to ``predict_mol`` (which embeds a
    conformer itself) and finally to 0.0.
    """
    if mol is None or oracle is None:
        return 0.0
    try:
        geom = _mol_geometry(mol)
        if geom is None:
            return float(oracle.predict_mol(mol))
        pos, z = geom
        prepared = BBBClassifier.data_to_input(Data(pos=pos, z=z))
        if prepared is None or prepared.z.numel() == 0:
            return 0.0
        device = next(oracle.parameters()).device
        with torch.no_grad():
            logit = oracle(
                prepared.z.to(device), prepared.edge_index.to(device),
                torch.zeros(prepared.z.numel(), dtype=torch.long, device=device),
            )
        return float(torch.sigmoid(logit).item())
    except Exception:
        return 0.0


def compute_steric_clash_score(mol: Chem.Mol) -> float:
    """Fraction of atom pairs closer than ``r_i + r_j - 0.30`` A.

    Severe overlaps only — a real covalent bond sits at ``r_i + r_j``, so it is
    never charged. Scored on the molecule's own conformer (no re-embedding),
    over unordered pairs, so the denominator is ``n(n-1)/2``.
    """
    geom = _mol_geometry(mol)
    if geom is None:
        return 0.0
    pos, z = geom
    pos = pos.numpy()
    radii = [COVALENT_RADII.get(int(v), 0.76) for v in z.tolist()]
    n = len(radii)
    clashes = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            limit = radii[i] + radii[j] - 0.30
            dist = float(((pos[i] - pos[j]) ** 2).sum() ** 0.5)
            total += 1
            if dist < limit:
                clashes += 1
    return clashes / max(total, 1)


def composite_reward(
    mol: Chem.Mol | None,
    bbb_classifier: torch.nn.Module | None = None,
    target_bonds: float = 19.0,
    weights: dict[str, float] | None = None,
) -> float:
    """Composite scalar reward (plan weights: bbb .4, validity .2, qed .15,
    connectivity .15, −bonds .1, −clash .1, −fragments .05).

    Returns ``INVALID_REWARD`` (−1.0) for an invalid molecule — the plan's
    correction 5: those are the *rejected* side of a pair, not rubbish to drop.
    ``target_bonds`` defaults to the plan's 19 (heavy-atom convention); pass
    :data:`REFERENCE_BONDS_PER_MOL` for molecules carrying explicit hydrogens.
    """
    if mol is None:
        return INVALID_REWARD
    try:
        Chem.SanitizeMol(Chem.Mol(mol))
    except Exception:
        return INVALID_REWARD

    if weights is None:
        weights = DEFAULT_WEIGHTS

    try:
        qed = float(QED.qed(mol))
    except Exception:
        qed = 0.0
    frags = max(len(Chem.GetMolFrags(mol)), 1)
    connectivity = 1.0 if frags == 1 else 0.0
    bonds = float(mol.GetNumBonds())
    bond_penalty = abs(bonds - target_bonds) / max(target_bonds, 1.0)
    clash = compute_steric_clash_score(mol)
    frag_penalty = (frags - 1) / max(frags, 1)

    reward = (
        weights["bbb"] * bbb_probability(mol, bbb_classifier)
        + weights["validity"] * 1.0
        + weights["qed"] * qed
        + weights["connectivity"] * connectivity
        - weights["bonds"] * bond_penalty
        - weights["clash"] * clash
        - weights["fragments"] * frag_penalty
    )
    return float(reward)


def reward_batch(
    mols: Sequence[Chem.Mol | None],
    bbb_classifier: torch.nn.Module | None = None,
    target_bonds: float = 19.0,
    weights: dict[str, float] | None = None,
) -> list[float]:
    """Reward for a sequence of molecules, positionally aligned with the input."""
    return [
        composite_reward(m, bbb_classifier, target_bonds, weights) for m in mols
    ]
