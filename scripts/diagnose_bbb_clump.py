"""Clump diagnostics for BBBP generation (the triage plan's Step 4).

Answers "is the failure atom soup, and of which kind?" from artifacts that
already exist, so it costs no generation budget:

1. **Pairwise-distance distribution** — nearest-neighbour percentiles. Real
   BBBP molecules sit at ~1.1 A (bonded), so a generated p50 well above that is
   the collapsed/over-bonded signature.
2. **Contact density** — fraction of atom pairs inside the eval's bond window
   (``r_i + r_j + tolerance``). A clump has many more near-contact pairs per
   atom than a real molecule.
3. **Bonds-per-molecule vs atom count** — the plan's diagnostic (3): if the
   bond count is high and *uncorrelated* with size, the generator is producing
   a fixed-size dense blob rather than size-appropriate chains.
4. **Per-element-pair contact counts** — whether the excess sits in C-C (a
   clump) or is spread over heteroatom pairs (a perception problem).

Everything is measured against a reference distribution (real BBBP molecules by
default) because an absolute number has no meaning without one: 19 bonds/mol is
right for BBBP and wrong for QM9.

Usage:
    python scripts/diagnose_bbb_clump.py --eval_dir outputs/eval_bbb_v5
    python scripts/diagnose_bbb_clump.py --eval_dir outputs/eval_bbb_v5 \
        --reference bbbp --reference_limit 300 --out outputs/bbb_v5_clump.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.guidance import COVALENT_RADII  # noqa: E402
from src.utils.evaluation import get_bond_tolerance  # noqa: E402

# Element pairs reported individually (the common organic ones); everything
# else is aggregated so the table stays readable.
# Keys are canonical (min Z, max Z) — the pair loop always queries that form.
_NAMED_PAIRS = {
    (1, 1): "H-H", (1, 6): "C-H", (1, 7): "N-H", (1, 8): "O-H",
    (1, 16): "S-H", (6, 6): "C-C", (6, 7): "C-N", (6, 8): "C-O",
    (6, 9): "C-F", (6, 16): "C-S", (6, 17): "C-Cl", (6, 35): "C-Br",
    (7, 7): "N-N", (7, 8): "N-O", (8, 8): "O-O", (16, 16): "S-S",
}


def bond_window(z_i: int, z_j: int, bond_tolerance: float | None = None) -> float:
    """Distance below which the eval ruler calls a pair bonded (A).

    Mirrors ``_add_bonds_from_distance`` exactly (covalent radii + per-pair
    tolerance with the optional tighten-cap), so a "contact" here is a bond
    there — the diagnostic and the headline metric cannot disagree.
    """
    ri = COVALENT_RADII.get(int(z_i))
    rj = COVALENT_RADII.get(int(z_j))
    if ri is None or rj is None:
        return float("nan")
    return ri + rj + get_bond_tolerance(int(z_i), int(z_j), cap=bond_tolerance)


def molecule_statistics(
    pos: np.ndarray,
    atomic_numbers: np.ndarray,
    bond_tolerance: float | None = None,
) -> dict:
    """Clump statistics for one molecule (all distances in A)."""
    pos = np.asarray(pos, dtype=float)
    z = np.asarray(atomic_numbers, dtype=int)
    n = len(z)
    if n < 2:
        return {
            "n_atoms": n, "n_pairs": 0, "contacts": 0,
            "contact_fraction": float("nan"), "nn_mean": float("nan"),
            "nn_p50": float("nan"), "radius_of_gyration": 0.0,
            "pair_counts": {},
        }

    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    zi, zj = z[iu[0]], z[iu[1]]

    windows = np.array([bond_window(a, b, bond_tolerance) for a, b in zip(zi, zj)])
    valid = np.isfinite(windows)
    contacts = int(((d <= windows) & valid).sum())

    nn = dist + np.eye(n) * 1e9
    nearest = nn.min(axis=1)
    centered = pos - pos.mean(axis=0, keepdims=True)
    rog = float(np.sqrt((centered ** 2).sum(-1).mean()))

    pair_counts: dict[str, int] = defaultdict(int)
    bonded = (d <= windows) & valid
    for a, b, dd, w in zip(zi, zj, d, windows):
        if not np.isfinite(w) or dd > w:
            continue
        key = _NAMED_PAIRS.get(
            (min(int(a), int(b)), max(int(a), int(b))), "other"
        )
        pair_counts[key] += 1

    # Connected components of the same contact graph the eval ruler builds.
    # This is the number that separates the two soup signatures: a single
    # over-dense blob (few components, tiny Rgyr) from fragmented clusters
    # (many components, large Rgyr, local bond counts near the reference).
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(iu[0][bonded], iu[1][bonded]):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    sizes: dict[int, int] = defaultdict(int)
    for a in range(n):
        sizes[find(a)] += 1

    return {
        "n_atoms": n,
        "n_pairs": int(len(d)),
        "contacts": contacts,
        "contact_fraction": contacts / max(len(d), 1),
        "nn_mean": float(nearest.mean()),
        "nn_p50": float(np.median(nearest)),
        "radius_of_gyration": rog,
        "fragments": len(sizes),
        "largest_fragment_fraction": max(sizes.values()) / n,
        "pair_counts": dict(pair_counts),
    }


def aggregate(stats: list[dict]) -> dict:
    """Mean-of-means summary plus the bonds-vs-size correlation the plan asks for."""
    usable = [s for s in stats if s["n_atoms"] >= 2]
    if not usable:
        return {"n_molecules": 0}
    sizes = np.array([s["n_atoms"] for s in usable], dtype=float)
    bonds = np.array([s["contacts"] for s in usable], dtype=float)
    per_atom = bonds / np.maximum(sizes, 1.0)

    pair_totals: dict[str, float] = defaultdict(float)
    for s in usable:
        for k, v in s["pair_counts"].items():
            pair_totals[k] += v
    pair_per_mol = {
        k: v / len(usable) for k, v in sorted(pair_totals.items())
    }
    corr = float(np.corrcoef(sizes, bonds)[0, 1]) if len(usable) > 2 and \
        sizes.std() > 0 and bonds.std() > 0 else float("nan")

    return {
        "n_molecules": len(usable),
        "n_atoms_mean": float(sizes.mean()),
        "bonds_per_mol": float(bonds.mean()),
        "bonds_per_atom": float(bonds.mean() / max(sizes.mean(), 1.0)),
        "contact_fraction": float(np.mean([s["contact_fraction"] for s in usable])),
        "nn_mean": float(np.mean([s["nn_mean"] for s in usable])),
        "nn_p50": float(np.median([s["nn_p50"] for s in usable])),
        "radius_of_gyration": float(np.mean([s["radius_of_gyration"] for s in usable])),
        "bonds_vs_size_pearson": corr,
        "fragments_per_mol": float(np.mean([s["fragments"] for s in usable])),
        "largest_fragment_fraction": float(
            np.mean([s["largest_fragment_fraction"] for s in usable])
        ),
        "single_fragment_pct": 100.0 * float(
            np.mean([s["fragments"] == 1 for s in usable])
        ),
        # Per-atom bond count is the scale-free comparison (real BBBP ~1.0).
        "pair_counts_per_mol": pair_per_mol,
    }


def load_sdf(path: str | Path, limit: int | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Read 3D coordinates and atomic numbers from an eval's molecules.sdf."""
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    positions, numbers = [], []
    for mol in supplier:
        if mol is None:
            continue
        if mol.GetNumConformers() == 0:
            continue
        conf = mol.GetConformer()
        pos = np.array([
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
             conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ], dtype=float)
        z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
        if len(z) < 2:
            continue
        positions.append(pos)
        numbers.append(z)
        if limit is not None and len(positions) >= limit:
            break
    return positions, numbers


def load_reference(name: str, limit: int = 200) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Real molecules to compare against (``bbbp``/``b3db``/``qm9``)."""
    from src.dataset import load_qm9

    name = name.lower()
    if name == "qm9":
        dataset = load_qm9(root="data")
    elif name == "bbbp":
        from src.dataset_bbb import load_bbbp

        dataset, _, _ = load_bbbp(root="data/bbbp")
    elif name == "b3db":
        from src.dataset_bbb import load_b3db

        dataset, _, _ = load_b3db(root="data/b3db")
    else:
        raise ValueError(f"Unknown reference {name!r} (expected qm9/bbbp/b3db)")

    positions, numbers = [], []
    for data in dataset:
        positions.append(np.asarray(data.pos, dtype=float))
        numbers.append(np.asarray(data.z, dtype=int))
        if len(positions) >= limit:
            break
    return positions, numbers


def _fmt(value: float, width: int = 8) -> str:
    if value != value:  # NaN
        return "-".rjust(width)
    return f"{value:.3f}".rjust(width)


def print_comparison(generated: dict, reference: dict | None) -> None:
    rows = [
        ("molecules", "n_molecules", 0),
        ("atoms/mol", "n_atoms_mean", 1),
        ("bonds/mol", "bonds_per_mol", 2),
        ("bonds/atom", "bonds_per_atom", 3),
        ("contact frac", "contact_fraction", 3),
        ("NN mean (A)", "nn_mean", 3),
        ("NN p50 (A)", "nn_p50", 3),
        ("Rgyr (A)", "radius_of_gyration", 2),
        ("corr(bonds,N)", "bonds_vs_size_pearson", 3),
        ("fragments/mol", "fragments_per_mol", 2),
        ("largest frac", "largest_fragment_fraction", 3),
        ("1-frag %", "single_fragment_pct", 1),
    ]
    print(f"\n{'metric':<14}{'generated':>12}{'reference':>12}")
    print("-" * 38)
    for label, key, nd in rows:
        gen = generated.get(key, float("nan"))
        line = f"{label:<14}{(_fmt(gen) if nd else str(gen)).rjust(12)}"
        if reference is not None:
            ref = reference.get(key, float("nan"))
            line += f"{(_fmt(ref) if nd else str(ref)).rjust(12)}"
        print(line)

    print("\nper-element-pair contacts per molecule (bonds)")
    print(f"{'pair':<10}{'generated':>12}{'reference':>12}")
    print("-" * 34)
    gen_pairs = generated.get("pair_counts_per_mol", {})
    ref_pairs = (reference or {}).get("pair_counts_per_mol", {})
    for key in sorted(set(gen_pairs) | set(ref_pairs)):
        line = f"{key:<10}{_fmt(gen_pairs.get(key, 0.0)):>12}"
        if reference is not None:
            line += f"{_fmt(ref_pairs.get(key, 0.0)):>12}"
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval_dir", required=True,
                        help="Eval output directory containing molecules.sdf")
    parser.add_argument("--sdf", default=None,
                        help="Explicit SDF path (default: <eval_dir>/molecules.sdf)")
    parser.add_argument("--reference", default="bbbp",
                        help="Real dataset to compare against (qm9/bbbp/b3db; "
                             "'' disables the reference)")
    parser.add_argument("--reference_limit", type=int, default=200,
                        help="Number of reference molecules to measure")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap generated molecules measured (default: all)")
    parser.add_argument("--bond_tolerance", type=float, default=None,
                        help="Tighten-cap on the per-pair tolerance, exactly as "
                             "generate_and_eval's --bond_tolerance")
    parser.add_argument("--out", default=None,
                        help="Write the diagnostic table as JSON here")
    args = parser.parse_args()

    sdf_path = Path(args.sdf) if args.sdf else Path(args.eval_dir) / "molecules.sdf"
    if not sdf_path.exists():
        raise SystemExit(f"missing {sdf_path} — run generate_and_eval with --output_dir first")

    positions, numbers = load_sdf(sdf_path, args.limit)
    if not positions:
        raise SystemExit(f"no usable molecules in {sdf_path}")

    gen_stats = [
        molecule_statistics(p, z, args.bond_tolerance)
        for p, z in zip(positions, numbers)
    ]
    generated = aggregate(gen_stats)

    reference = None
    if args.reference:
        ref_pos, ref_z = load_reference(args.reference, args.reference_limit)
        reference = aggregate([
            molecule_statistics(p, z, args.bond_tolerance)
            for p, z in zip(ref_pos, ref_z)
        ])

    print(f"\nGenerated: {sdf_path} (n={len(positions)})")
    if reference is not None:
        print(f"Reference: {args.reference} (n={reference['n_molecules']})")
    print_comparison(generated, reference)

    verdict = _verdict(generated, reference)
    print(f"\nVerdict: {verdict}")

    if args.out:
        payload = {
            "sdf": str(sdf_path),
            "reference": args.reference or None,
            "generated": generated,
            "reference_stats": reference,
            "verdict": verdict,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved -> {out}")
    return 0


def _verdict(generated: dict, reference: dict | None) -> str:
    """Plain-language read of the soup signatures.

    Three cases are distinguishable with these measurements, and they call for
    different fixes:

    - **over-dense blob**: bonds/atom above the reference with a *small* radius
      of gyration and few fragments — the atoms really did collapse.
    - **fragmented clumps**: locally normal bonds/atom but many fragments and a
      radius of gyration well above the reference — dense specks scattered in
      space (the sampler never forms one molecule).
    - **dispersed**: bonds/atom far below the reference — atoms sit too far
      apart to bond at all.
    """
    if not generated.get("n_molecules"):
        return "no measurable molecules"
    bpa = generated["bonds_per_atom"]
    rgyr = generated["radius_of_gyration"]
    frags = generated["fragments_per_mol"]
    nn = generated["nn_p50"]
    ref = reference or {}
    ref_bpa = ref.get("bonds_per_atom")
    ref_rgyr = ref.get("radius_of_gyration")
    ref_frags = ref.get("fragments_per_mol")
    ref_nn = ref.get("nn_p50")

    if ref_bpa and bpa > 1.25 * ref_bpa:
        kind = (f"OVER-DENSE (fusion) — {bpa:.2f} bonds/atom vs {ref_bpa:.2f} real")
    elif ref_bpa and bpa < 0.75 * ref_bpa:
        kind = (f"UNDER-BONDED / DISPERSED — {bpa:.2f} bonds/atom vs "
                f"{ref_bpa:.2f} real")
    elif ref_frags and frags > 3 * ref_frags:
        kind = (f"FRAGMENTED CLUMPS — {frags:.1f} fragments/mol vs "
                f"{ref_frags:.1f} real at reference-like local bonding "
                f"({bpa:.2f} vs {ref_bpa:.2f} bonds/atom)")
    else:
        kind = f"local bonding near reference ({bpa:.2f} vs {ref_bpa} bonds/atom)"

    if ref_rgyr and rgyr > 1.3 * ref_rgyr:
        kind += f"; radius of gyration {rgyr:.2f} A vs {ref_rgyr:.2f} real (too spread)"
    elif ref_rgyr:
        kind += f"; radius of gyration {rgyr:.2f} A vs {ref_rgyr:.2f} real"
    if ref_nn is not None and nn > ref_nn + 0.3:
        kind += f"; NN {nn:.2f} A vs {ref_nn:.2f} real (too loose)"
    return kind


if __name__ == "__main__":
    raise SystemExit(main())
