"""Reward-as-selector: best-of-N ranking with the composite reward (no training).

Uses ``src/rl/rewards.py`` as a *rejection sampler*, not as a DPO gradient:
roll out N candidates, score every one with the composite reward, keep the top
K, and report what the selected pool looks like under the standard Table-I
metric suite and the clump ruler. This is the cheapest way to answer "is the
reward even correlated with the property we care about?" before spending a
training run on preference optimization, and the selected set is exactly the
elite subset ``scripts/raft_finetune.py`` would consume.

Two facts about the pool this operates on, both of which the output records:

* an eval's ``molecules.sdf`` holds the *valid* molecules of the rollout, so
  the invalid ones (which the reward scores ``INVALID_REWARD = -1.0`` and would
  never be selected) are already excluded — selection from the saved pool is
  equivalent to best-of-N over the full rollout, and ``--rollout_size`` keeps
  the denominator honest so ``ConnectedValidity@K`` is expressed against N.
* a subset's ``Validity`` is 100 by construction; the honest headline is the
  within-pool ``ConnectedFrac`` plus the ``@K`` rates.

The ``--random`` baseline is the whole pool (the expectation of a uniform
draw), so every gain printed is a genuine ranking gain, not a sampling one.

Usage:
    .venv/bin/python scripts/reward_select.py \
        --candidates outputs/eval_pre_rl/molecules.sdf \
        --config configs/central_v5_bbb.yaml \
        --oracle models/bbb_oracle.pt --rollout_size 1000 \
        --topk 25,50,100,200,all --out_dir outputs/reward_select_v5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.diagnose_bbb_clump import (  # noqa: E402
    aggregate,
    load_reference,
    molecule_statistics,
)
from src.rl.rewards import (  # noqa: E402
    REFERENCE_BONDS_PER_MOL,
    load_bbb_oracle,
    reward_batch,
)
from src.utils.evaluation import evaluate  # noqa: E402


def load_candidates(paths: list[str]) -> list[Chem.Mol]:
    """Read valid molecules from one or more eval SDFs, in file order."""
    mols: list[Chem.Mol] = []
    for path in paths:
        supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
        kept = 0
        for mol in supplier:
            if mol is None or mol.GetNumConformers() == 0:
                continue
            mols.append(mol)
            kept += 1
        print(f"  {path}: {kept} usable molecules")
    return mols


def clump_stats(mols: list[Chem.Mol]) -> dict:
    """Clump diagnostic on a molecule subset (same ruler as the triage script)."""
    stats = []
    for mol in mols:
        conf = mol.GetConformer()
        pos = np.array([
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
             conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ], dtype=float)
        z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
        if len(z) < 2:
            continue
        stats.append(molecule_statistics(pos, z))
    return aggregate(stats)


def pool_summary(mols: list[Chem.Mol], oracle) -> dict:
    """Selection-relevant property distribution of a pool (not the reward)."""
    if not mols:
        return {"n": 0}
    connected, bonds, qed_vals, bbb_vals, clashes = [], [], [], [], []
    from rdkit.Chem import QED
    from src.rl.rewards import compute_steric_clash_score

    for mol in mols:
        connected.append(len(Chem.GetMolFrags(mol)) == 1)
        bonds.append(mol.GetNumBonds())
        try:
            qed_vals.append(float(QED.qed(mol)))
        except Exception:
            qed_vals.append(0.0)
        try:
            from src.rl.rewards import bbb_probability
            bbb_vals.append(bbb_probability(mol, oracle))
        except Exception:
            bbb_vals.append(0.0)
        clashes.append(compute_steric_clash_score(mol))
    return {
        "n": len(mols),
        "connected_pct": 100.0 * float(np.mean(connected)),
        "bonds_per_mol": float(np.mean(bonds)),
        "qed": float(np.mean(qed_vals)),
        "bbb_prob": float(np.mean(bbb_vals)),
        "clash_frac": float(np.mean(clashes)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", nargs="+", required=True,
                        help="One or more eval molecules.sdf paths to pool")
    parser.add_argument("--config", default="configs/central_v5_bbb.yaml",
                        help="Dataset config (for novelty/SNN/scaffold baselines)")
    parser.add_argument("--oracle", default="models/bbb_oracle.pt")
    parser.add_argument("--target_bonds", type=float,
                        default=REFERENCE_BONDS_PER_MOL,
                        help="Bond target in the H-inclusive ruler (the SDF "
                             "molecules carry explicit H)")
    parser.add_argument("--topk", default="25,50,100,200,all",
                        help="Comma-separated K ladder, or 'all'")
    parser.add_argument("--rollout_size", type=int, default=None,
                        help="N the candidates were drawn from (for @K rates); "
                             "defaults to the pool size")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    mols = load_candidates(args.candidates)
    if not mols:
        raise SystemExit("no usable candidates")
    rollout = args.rollout_size or len(mols)

    oracle = load_bbb_oracle(args.oracle)
    if oracle is None:
        print(f"WARNING: no oracle at {args.oracle} — BBB term contributes 0.0")
    target_bonds = args.target_bonds
    print(f"\nScoring {len(mols)} candidates "
          f"(target_bonds={target_bonds:.2f}, oracle={'yes' if oracle else 'no'})")
    rewards = reward_batch(mols, oracle, target_bonds=target_bonds)

    order = np.argsort(-np.asarray(rewards), kind="stable")
    ranked = [mols[i] for i in order]
    ranked_rewards = [float(rewards[i]) for i in order]

    # Dataset baselines for novelty / SNN / scaffold coverage.
    import yaml as _yaml
    from generate_and_eval import _data_to_rdkit_mol
    from src.dataset_bbb import load_bbbp

    cfg = _yaml.safe_load(open(args.config))
    data_name = str(cfg["data"].get("dataset", "bbbp")).lower()
    if data_name == "bbbp":
        full_dataset, _, _ = load_bbbp(root=cfg["data"]["root"])
    else:
        raise SystemExit(f"unsupported dataset {data_name!r} (this script is "
                         f"for the BBB arms)")
    train_smiles: set[str] = set()
    train_mols: list[Chem.Mol] = []
    for data in full_dataset:
        mol = _data_to_rdkit_mol(data)
        if mol is not None:
            train_smiles.add(Chem.MolToSmiles(mol))
            train_mols.append(mol)

    # 'all' anywhere in the ladder means the whole pool.
    ks: list[int] = []
    for token in args.topk.split(","):
        token = token.strip().lower()
        if not token:
            continue
        ks.append(len(ranked) if token == "all" else min(int(token), len(ranked)))
    ks = sorted(set(ks))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'pool':<28}{'n':>6}{'ConnFrac':>10}{'ConnV@K':>9}"
          f"{'bonds':>8}{'BBB%':>8}{'QED':>7}{'IntDiv':>8}{'CNS_MPO':>9}"
          f"{'frags':>7}{'Rgyr':>7}")
    rows: list[dict] = []

    def _report(tag: str, subset: list[Chem.Mol], k: int | None) -> dict:
        metrics = evaluate(subset, train_smiles, train_mols, bbb_classifier=oracle)
        clump = clump_stats(subset)
        connected_in_subset = 100.0 * float(
            np.mean([len(Chem.GetMolFrags(m)) == 1 for m in subset])
        ) if subset else 0.0
        connv_at_k = (100.0 * sum(1 for m in subset
                                  if len(Chem.GetMolFrags(m)) == 1) / rollout
                      if subset else 0.0)
        summary = {
            "pool": tag,
            "k": k if k is not None else len(subset),
            "connected_pct": connected_in_subset,
            "connected_validity_at_k": connv_at_k,
            "metrics": metrics,
            "clump": clump,
            "properties": pool_summary(subset, oracle),
        }
        print(f"{tag:<28}{len(subset):>6}{connected_in_subset:>10.1f}"
              f"{connv_at_k:>9.2f}"
              f"{metrics['BondsPerMol']:>8.2f}{metrics['BBB%']:>8.2f}"
              f"{metrics['QED']:>7.3f}{metrics['IntDiv_p']:>8.3f}"
              f"{metrics['CNS_MPO']:>9.3f}"
              f"{clump.get('fragments_per_mol', float('nan')):>7.2f}"
              f"{clump.get('radius_of_gyration', float('nan')):>7.2f}")
        rows.append(summary)
        return summary

    _report("random (whole pool)", ranked, None)

    for k in ks:
        subset = ranked[:k]
        _report(f"top-{k} by reward", subset, k)
        # The elite set RAFT would consume: top-K, written as an SDF.
        writer = Chem.SDWriter(str(out_dir / f"elites_top{k}.sdf"))
        for mol in subset:
            writer.write(mol)
        writer.close()

    # Per-candidate audit trail so the ranking can be inspected, not trusted.
    with open(out_dir / "rewards.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "smiles", "reward", "n_atoms", "n_bonds",
                         "n_fragments"])
        for rank, (mol, reward) in enumerate(zip(ranked, ranked_rewards), 1):
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:
                smi = ""
            writer.writerow([rank, smi, f"{reward:.6f}", mol.GetNumAtoms(),
                             mol.GetNumBonds(),
                             len(Chem.GetMolFrags(mol))])
    payload = {
        "candidates": args.candidates,
        "config": args.config,
        "oracle": args.oracle,
        "target_bonds": target_bonds,
        "n_candidates": len(mols),
        "rollout_size": rollout,
        "reward_stats": {
            "mean": float(np.mean(rewards)),
            "median": float(np.median(rewards)),
            "max": float(np.max(rewards)),
            "min": float(np.min(rewards)),
        },
        "pools": rows,
    }
    with open(out_dir / "selection.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved -> {out_dir}/selection.json, rewards.csv, elites_top*.sdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
