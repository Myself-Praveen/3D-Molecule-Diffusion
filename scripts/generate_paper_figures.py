"""Generate all paper figures (implementation.md §7.2 / file map).

Usage:
    .venv/bin/python scripts/generate_paper_figures.py            # auto data
    .venv/bin/python scripts/generate_paper_figures.py --synthetic
    .venv/bin/python scripts/generate_paper_figures.py --outdir figs/

Real inputs, when present:
    outputs/fed_*/history.json          -> Fig. 1 (convergence), Fig. 3
    outputs/eval_*/metrics.json         -> Table I
    outputs/sweep_lambda_pareto/**/results.json -> Fig. 2 (else synthetic)
    outputs/ablation_*/seed_*/results.json       -> Fig. 4 (else synthetic)
    outputs/eval_*/molecules.sdf        -> Fig. 6 (else embedded SMILES)
    outputs/sweep_guidance/**/results.json       -> Fig. 7 (else synthetic)

Missing inputs fall back to seeded synthetic data so the pipeline always
produces all seven figures + Table I (verification §7.2). Every figure is
written as both .pdf and .png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from src.utils.visualization import (  # noqa: E402
    plot_ablation_bars, plot_bbb_property_distribution,
    plot_convergence_curves, plot_guidance_scale_sweep,
    plot_metric_comparison_table, plot_mode_collapse_analysis,
    plot_pareto_front, render_molecules_3d,
)

REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"

# Canonical Table I row order (headline runs first).
TABLE_RUNS = [
    ("V3 (QM9)", "eval_v3"),
    ("V4-eps eta=1.0", "eval_v4_eps_eta10_full"),
    ("V4-eps eta=0.5", "eval_v4_eps"),
    ("Fed IID (QM9)", "eval_fed_iid"),
    ("Fed non-IID (QM9)", "eval_fed_niid"),
]


def find_histories() -> dict[str, str]:
    return {
        p.parent.name.replace("fed_", "Fed "): str(p)
        for p in sorted(OUTPUTS.glob("fed_*/history.json"))
    }


def load_table_rows() -> dict[str, dict]:
    rows = {}
    for label, dirname in TABLE_RUNS:
        path = OUTPUTS / dirname / "metrics.json"
        if path.exists():
            try:
                rows[label] = json.loads(path.read_text())["metrics"]
            except (json.JSONDecodeError, KeyError, OSError):
                continue
    return rows


def load_results_dirs(pattern: str) -> list[dict]:
    out = []
    for path in sorted(OUTPUTS.glob(pattern)):
        try:
            out.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def pareto_rows() -> list[dict]:
    rows = load_results_dirs("sweep_lambda_pareto/**/seed_*/results.json")
    cooked = []
    for r in rows:
        metrics = r.get("metrics") or r.get("requested") or {}
        overrides = r.get("combo") or r.get("overrides") or {}
        label = ",".join(f"{k.split('.')[-1]}={v}"
                         for k, v in sorted(overrides.items()))
        cooked.append({
            "Validity": metrics.get("Validity", 0.0),
            "Uniqueness": metrics.get("Uniqueness", 100.0),
            "label": label,
        })
    return cooked


def ablation_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for r in load_results_dirs("ablation_*/seed_*/results.json"):
        arm = r.get("ablation", "unknown")
        metrics = r.get("metrics") or {}
        if arm not in rows and metrics:
            rows[arm] = {
                k: metrics.get(k, 0.0)
                for k in ("Validity", "ConnectedValidity", "BBB%")
                if k in metrics
            }
    return rows


def guidance_rows() -> dict:
    rows = load_results_dirs("sweep_guidance/**/results.json")
    out = {}
    for r in rows:
        w = (r.get("combo") or {}).get("guidance_scale")
        if w is None:
            continue
        out[float(w)] = (r.get("metrics") or {}).get("BBB%", 0.0)
    return out


def load_generated_sdf() -> list:
    from rdkit import Chem

    for path in sorted(OUTPUTS.glob("eval_*/molecules.sdf")):
        try:
            mols = [m for m in Chem.SDMolSupplier(str(path), removeHs=False)
                    if m is not None]
        except Exception:
            continue
        if mols:
            return mols
    return []


# ---------------------------------------------------------------------------
# Synthetic fallbacks (seeded, for pipeline verification)
# ---------------------------------------------------------------------------

def synthetic_pareto(rng) -> list[dict]:
    rows = []
    for l2 in (0.0, 0.1, 0.5, 1.0, 2.0):
        for l3 in (0.0, 0.01, 0.05):
            rows.append({
                "Validity": float(45 + 12 * np.tanh(l2) + rng.normal(0, 2)),
                "Uniqueness": float(96 + 8 * np.tanh(l3) + rng.normal(0, 1)),
                "label": f"l2={l2},l3={l3}",
            })
    return rows


def synthetic_ablation(rng) -> dict[str, dict]:
    base = {"Validity": 58.0, "ConnectedValidity": 51.0, "BBB%": 62.0}
    return {
        arm: {k: max(0.0, v + rng.normal(0, 4)) for k, v in base.items()}
        for arm in ("full_model", "no_conditioning", "no_valence_penalty",
                    "no_diversity_loss", "fewer_layers", "no_3d")
    }


def synthetic_guidance() -> dict:
    return {0.0: 24.0, 0.5: 33.0, 1.0: 44.0, 2.0: 57.0, 4.0: 65.0,
            8.0: 69.0}


def synthetic_property_groups(rng):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    pos = ("CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
           "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "c1ccc2c(c1)ccc1ccccc12",
           "CN(C)CCc1c[nH]c2ccccc12", "CC(=O)Nc1ccc(O)cc1")
    neg = ("OCC1OC(O)C(O)C(O)C1O", "OC[C@H]1OC(O)C(O)[C@@H](O)[C@@H]1O",
           "CCCCCCCCCCCC(=O)O", "NCC(=O)O", "OCC(O)CO",
           "C(C(=O)O)N")
    groups = {}
    for label, smis in (("BBB+", pos), ("BBB-", neg)):
        mols = []
        for smi in smis:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            if AllChem.EmbedMolecule(mol, randomSeed=int(rng.integers(1 << 30))) != 0:
                continue
            mols.append(mol)
        groups[label] = mols
    return groups


def synthetic_molecules(n: int = 6):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smis = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
            "c1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "CN(C)CCc1c[nH]c2ccccc12", "CC(=O)Nc1ccc(O)cc1"]
    mols = []
    for smi in smis[:n]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            continue
        mols.append(mol)
    return mols


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--outdir", default="outputs/figures")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic data for every figure")
    args = parser.parse_args()
    outdir = REPO / args.outdir
    rng = np.random.default_rng(42)
    written: list[Path] = []
    synthetic_notes: list[str] = []

    def track(result):
        if isinstance(result, list):
            written.extend(result)

    # Fig. 1 + Fig. 3 — histories (real when present)
    histories = find_histories()
    if not histories:
        histories = {}
        synthetic_notes.append("Fig.1/3 (histories)")
    track(plot_convergence_curves(histories, outdir / "fig1_convergence"))
    track(plot_mode_collapse_analysis(histories, outdir / "fig3_mode_collapse"))

    # Table I
    rows = {} if args.synthetic else load_table_rows()
    if not rows:
        synthetic_notes.append("Table I")
        rng_rows = synthetic_ablation(rng)
        rows = {"synthetic": m for m in rng_rows.values()}
    latex = plot_metric_comparison_table(rows, outdir / "table1_metrics")
    if isinstance(latex, str):  # placeholder returns a path list instead
        written.extend([outdir / "table1_metrics.pdf",
                        outdir / "table1_metrics.png"])

    # Fig. 2 — Pareto
    pareto = synthetic_pareto(rng) if args.synthetic else pareto_rows()
    if not pareto:
        pareto = synthetic_pareto(rng)
        synthetic_notes.append("Fig.2 (pareto)")
    track(plot_pareto_front(pareto, outdir / "fig2_pareto"))

    # Fig. 4 — ablation
    ablation = {} if args.synthetic else ablation_rows()
    if not ablation:
        ablation = synthetic_ablation(rng)
        synthetic_notes.append("Fig.4 (ablation)")
    track(plot_ablation_bars(ablation, outdir / "fig4_ablation"))

    # Fig. 5 — property distributions (generated sets are unlabeled -> synthetic)
    groups = synthetic_property_groups(rng)
    track(plot_bbb_property_distribution(groups,
                                         outdir / "fig5_properties"))

    # Fig. 6 — molecules (real SDF when present)
    mols = [] if args.synthetic else load_generated_sdf()
    if not mols:
        mols = synthetic_molecules()
        synthetic_notes.append("Fig.6 (molecules)")
    track(render_molecules_3d(mols, outdir / "fig6_molecules"))

    # Fig. 7 — guidance sweep
    guidance = {} if args.synthetic else guidance_rows()
    if not guidance:
        guidance = synthetic_guidance()
        synthetic_notes.append("Fig.7 (guidance)")
    track(plot_guidance_scale_sweep(guidance, outdir / "fig7_guidance"))

    print(f"wrote {len(written)} files to {outdir}")
    for p in written:
        try:
            print(f"  {p.relative_to(REPO)}")
        except ValueError:  # outdir outside the repo (e.g. /tmp)
            print(f"  {p}")
    if synthetic_notes:
        print("synthetic fallbacks used for: " + ", ".join(synthetic_notes))
    missing = [p for p in written if not p.exists()]
    if missing:
        print(f"ERROR: {len(missing)} declared files missing")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
