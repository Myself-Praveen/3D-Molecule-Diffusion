"""BBB permeability datasets: BBBP (MoleculeNet) + B3DB, SMILES -> 3D PyG Data.

Phase 1 of implementation.md. Each item is a PyG ``Data`` object with::

    data.pos:    (N, 3) float — 3D coordinates from RDKit ETKDGv3 + MMFF
    data.z:      (N,)   long  — atomic numbers (hydrogens explicit)
    data.y:      (1,)   long  — BBB label (0 = BBB-, 1 = BBB+)
    data.smiles: str          — original SMILES (for evaluation)
    data.qed:    (1,)   float — RDKit QED
    data.logp:   (1,)   float — Crippen LogP
    data.tpsa:   (1,)   float — topological polar surface area
    data.mw:     (1,)   float — molecular weight

Raw tables are cached under ``data/bbbp/`` / ``data/b3db/``; converted 3D
tensors are cached to ``processed_3d.pt`` (saved incrementally every 500
molecules, so a killed run resumes instead of restarting).

Notes:
- ``deepchem`` is listed in requirements for spec compliance, but loading
  here uses a direct CSV download + own Murcko scaffold split: importing
  deepchem pulls in tensorflow, which is unnecessary bulk for this step.
- ~5-10% of SMILES fail 3D embedding (expected; failures are logged and
  reported, not fatal).
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data
from tqdm import tqdm

logger = logging.getLogger(__name__)

BBBP_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
B3DB_URL = (
    "https://raw.githubusercontent.com/theochem/B3DB/main/"
    "B3DB/B3DB_classification.tsv"
)

# Fallback normalization constants (overwritten by train-set stats at load).
PROPERTY_STATS = {
    "qed": {"mean": 0.5, "std": 0.2},
    "logp": {"mean": 2.0, "std": 1.5},
    "tpsa": {"mean": 60.0, "std": 30.0},
    "mw": {"mean": 300.0, "std": 100.0},
}
PROPERTY_KEYS = ("qed", "logp", "tpsa", "mw")

CACHE_EVERY = 500  # incremental cache writes (kill-safe conversion)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("Reusing cached %s", dest)
        return dest
    logger.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    return dest


def _murcko_scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return "__invalid__"
        try:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            # Salts / multi-fragment records: scaffold of the largest fragment.
            frags = Chem.GetMolFrags(mol, asMols=True)
            if not frags:
                return "__invalid__"
            biggest = max(frags, key=lambda m: m.GetNumAtoms())
            try:
                return MurckoScaffold.MurckoScaffoldSmiles(mol=biggest)
            except Exception:
                return "frag:" + Chem.MolToSmiles(biggest)
    except Exception:
        return "__invalid__"


def scaffold_split(
    smiles_list: list[str],
    labels: list[int],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Murcko-scaffold grouped 80/10/10 split (indices into the input lists).

    Scaffold groups (largest first) are greedily assigned to the currently
    smallest split — same strategy as deepchem's ScaffoldSplitter — so no
    scaffold appears in two splits.
    """
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for idx, smi in enumerate(smiles_list):
        groups.setdefault(_murcko_scaffold(smi), []).append(idx)
    # Largest groups first; ties shuffled for determinism via seeded rng.
    order = sorted(groups, key=lambda s: -len(groups[s]))
    splits: list[list[int]] = [[], [], []]
    targets = (train_frac, val_frac, 1.0 - train_frac - val_frac)
    totals = [0, 0, 0]
    n = len(smiles_list)
    for scaf in order:
        size = len(groups[scaf])
        # Assign to the split furthest below its target fraction.
        fracs = [totals[k] / max(n, 1) for k in range(3)]
        deficits = [targets[k] - fracs[k] for k in range(3)]
        # Break exact ties randomly (seeded) to avoid scaffold-size bias.
        best = max(range(3), key=lambda k: (deficits[k], rng.random()))
        splits[best].extend(groups[scaf])
        totals[best] += size
    return splits[0], splits[1], splits[2]


# ---------------------------------------------------------------------------
# SMILES -> 3D conversion
# ---------------------------------------------------------------------------

def smiles_to_3d_data(
    smiles: str,
    bbb_label: int,
    seed: int = 42,
    mmff_iters: int = 200,
) -> Data | None:
    """Convert one SMILES string to a 3D PyG ``Data`` object.

    Returns ``None`` when RDKit embedding fails (~5-10% of drug-like SMILES
    lack a valid 3D conformer); callers log and skip these.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = True
    try:
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol, maxIters=mmff_iters)
        conf = mol.GetConformer()
        pos = np.array(
            [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
             for i in range(mol.GetNumAtoms())],
            dtype=np.float32,
        )
        z = np.array(
            [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64,
        )
        qed = float(Descriptors.qed(mol))
        logp = float(Descriptors.MolLogP(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
        mw = float(Descriptors.MolWt(mol))
    except Exception:
        return None
    if not np.all(np.isfinite(pos)):
        return None
    return Data(
        pos=torch.from_numpy(pos),
        z=torch.from_numpy(z),
        y=torch.tensor([int(bbb_label)], dtype=torch.long),
        smiles=smiles,
        qed=torch.tensor([qed], dtype=torch.float32),
        logp=torch.tensor([logp], dtype=torch.float32),
        tpsa=torch.tensor([tpsa], dtype=torch.float32),
        mw=torch.tensor([mw], dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def compute_property_stats(data_list: list[Data]) -> dict[str, dict[str, float]]:
    """Mean/std of QED/LogP/TPSA/MW over a (train) split for z-scoring."""
    stats: dict[str, dict[str, float]] = {}
    for key in PROPERTY_KEYS:
        vals = torch.stack([d[key] for d in data_list]).float().view(-1)
        stats[key] = {
            "mean": float(vals.mean()),
            "std": max(float(vals.std()), 1e-6),
        }
    return stats


def normalize_properties(
    data: Data,
    stats: dict[str, dict[str, float]] | None = None,
) -> torch.Tensor:
    """Return a (4,) tensor of z-scored [QED, LogP, tPSA, MW]."""
    stats = stats or PROPERTY_STATS
    out = []
    for key in PROPERTY_KEYS:
        s = stats[key]
        out.append((float(data[key].view(-1)[0]) - s["mean"]) / s["std"])
    return torch.tensor(out, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Generic conversion driver (with incremental cache)
# ---------------------------------------------------------------------------

def _convert_with_cache(
    smiles_list: list[str],
    labels: list[int],
    cache_path: Path,
    seed: int = 42,
) -> tuple[list[Data], int]:
    """Convert all SMILES, resuming from / periodically writing cache.

    Returns (items_by_index, n_failed) where values are ``Data`` or ``None``.
    Cache stores per-index ``Data`` objects plus a fingerprint of the inputs;
    a fingerprint mismatch discards it.
    """
    fingerprint = hashlib.md5(
        ("\n".join(f"{s}\t{labels[i]}" for i, s in enumerate(smiles_list))).encode(),
    ).hexdigest()
    cached: dict[int, Data | None] = {}
    if cache_path.exists():
        try:
            blob = torch.load(cache_path, map_location="cpu", weights_only=False)
            if blob.get("fingerprint") == fingerprint:
                cached = blob["items"]
                logger.info("Resuming conversion from %s (%d done)", cache_path, len(cached))
            else:
                logger.info("Cache fingerprint mismatch; reconverting")
        except Exception as e:
            logger.warning("Ignoring unreadable cache %s (%s)", cache_path, e)

    pending = [i for i in range(len(smiles_list)) if i not in cached]
    for count, idx in enumerate(
        tqdm(pending, desc=f"SMILES->3D [{cache_path.parent.name}]"), start=1
    ):
        # None marks a failed conversion so resumes skip it.
        cached[idx] = smiles_to_3d_data(smiles_list[idx], labels[idx], seed=seed + idx)
        if (len(cached) % CACHE_EVERY == 0) or (count == len(pending)):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            torch.save({"fingerprint": fingerprint, "items": cached}, tmp)
            tmp.replace(cache_path)

    n_failed = sum(1 for v in cached.values() if v is None)
    return cached, n_failed


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_table(url: str, dest: Path, sep: str = ",") -> pd.DataFrame:
    _download(url, dest)
    return pd.read_csv(dest, sep=sep)


def _detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    cols = {c.lower(): c for c in df.columns}
    smiles_col = next(
        (cols[c] for c in ("smiles", "canonical_smiles", "isomeric_smiles",
                           "molecule", "compound") if c in cols),
        None,
    )
    label_col = next(
        (cols[c] for c in ("p_np", "bbb", "bbb_class", "label", "class",
                           "bbb+/-", "bbb+/bbb-", "bbb_label", "permeability")
         if c in cols),
        None,
    )
    if smiles_col is None or label_col is None:
        raise ValueError(f"Cannot detect SMILES/label columns in {list(df.columns)}")
    return smiles_col, label_col


def _coerce_label(value) -> int | None:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "bbb+", "+", "positive", "permeable"):
            return 1
        if v in ("0", "false", "no", "bbb-", "-", "negative", "non-permeable",
                 "nonpermeable", "impermeable"):
            return 0
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _prepare_split_data(
    root: str | Path,
    name: str,
    df: pd.DataFrame,
    seed: int = 42,
) -> tuple[list[Data], list[Data], list[Data], dict]:
    """Scaffold-split, convert, cache; returns (train, val, test, info)."""
    root = Path(root)
    smiles_col, label_col = _detect_columns(df)
    smiles_list: list[str] = []
    labels: list[int] = []
    for _, row in df.iterrows():
        smi = row[smiles_col]
        lab = _coerce_label(row[label_col])
        if not isinstance(smi, str) or not smi.strip() or lab not in (0, 1):
            continue
        smiles_list.append(smi.strip())
        labels.append(lab)

    train_idx, val_idx, test_idx = scaffold_split(smiles_list, labels, seed=seed)
    split_of = {}
    for i in train_idx:
        split_of[i] = 0
    for i in val_idx:
        split_of[i] = 1
    for i in test_idx:
        split_of[i] = 2

    cache_path = root / "processed_3d.pt"
    items, n_failed = _convert_with_cache(smiles_list, labels, cache_path, seed=seed)

    # Exact index-based split mapping (failed conversions drop out).
    splits: tuple[list[Data], list[Data], list[Data]] = ([], [], [])
    for idx, split_id in split_of.items():
        data = items.get(idx)
        if data is not None:
            splits[split_id].append(data)
    train_data, val_data, test_data = splits

    stats = compute_property_stats(train_data) if train_data else dict(PROPERTY_STATS)
    info = {
        "n_raw": len(df),
        "n_usable": len(smiles_list),
        "n_failed_3d": n_failed,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_test": len(test_data),
        "frac_bbb_pos": float(np.mean(labels)) if labels else 0.0,
        "property_stats": stats,
    }
    logger.info(
        "%s: raw=%d usable=%d failed3d=%d train/val/test=%d/%d/%d BBB+ %.1f%%",
        name, info["n_raw"], info["n_usable"], n_failed,
        len(train_data), len(val_data), len(test_data),
        100 * info["frac_bbb_pos"],
    )
    return train_data, val_data, test_data, info


def load_bbbp(
    root: str | Path = "data/bbbp",
    seed: int = 42,
) -> tuple[list[Data], list[Data], list[Data]]:
    """Load BBBP (MoleculeNet, ~2k compounds) with scaffold split.

    Returns (train, val, test) lists of 3D PyG ``Data``. Property stats are
    available via :func:`get_bbbp_stats`.
    """
    root = Path(root)
    df = _load_table(BBBP_URL, root / "BBBP.csv", sep=",")
    train, val, test, info = _prepare_split_data(root, "bbbp", df, seed=seed)
    _LAST_INFO["bbbp"] = info
    return train, val, test


def load_b3db(
    root: str | Path = "data/b3db",
    seed: int = 42,
) -> tuple[list[Data], list[Data], list[Data]]:
    """Load B3DB classification set (~7.8k compounds) with scaffold split."""
    root = Path(root)
    df = _load_table(B3DB_URL, root / "B3DB_classification.tsv", sep="\t")
    train, val, test, info = _prepare_split_data(root, "b3db", df, seed=seed)
    _LAST_INFO["b3db"] = info
    return train, val, test


_LAST_INFO: dict[str, dict] = {}


def get_dataset_info(name: str) -> dict:
    """Info dict from the most recent :func:`load_bbbp` / :func:`load_b3db` call."""
    return _LAST_INFO.get(name, {})
