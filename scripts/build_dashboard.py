#!/usr/bin/env python
"""Build a self-contained HTML dashboard (2D-only) from eval outputs.

Usage:
    .venv/bin/python scripts/build_dashboard.py [--out outputs/dashboard.html]
                                                [--molcap 48] [--runs name1,name2]

Scans outputs/eval_*/ directories (each with metrics.json, smiles.txt and
optionally molecules.sdf) and writes one self-contained HTML file that opens
in any browser — no server, no network, no JavaScript dependencies:

* metric cards per run + a cross-run comparison table
* RDKit-rendered 2D SVG gallery of every valid molecule with per-molecule
  stats (formula, MW, QED, LogP, rings, fragments)

3D viewing is deliberately removed for now (WebGL was unavailable on the
target machine); see git history for the 3Dmol.js / canvas-fallback versions
if it needs to come back.

Runs predating the covalent-radii eval fix (commit 17fd4f7) are auto-badged
"inflated lens" and collapsed, so stale numbers can't mislead.
"""

from __future__ import annotations

import argparse
import html
import sys
from datetime import datetime
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "dashboard.html"

# Eval dirs known to predate the covalent-radii eval fix (commit 17fd4f7):
# their numbers were measured through the broken bond-reconstruction lens.
INFLATED_RUNS = {
    "eval_pre_coordfix",
    "eval_central",
    "eval_fed_iid",
    "eval_fed_niid",
}

# ---------------------------------------------------------------- per-mol stats


def _safe(fn, default=0.0):
    def inner(mol):
        try:
            return fn(mol)
        except Exception:
            return default
    return inner


def mol_stats(mol) -> dict:
    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        formula = "?"
    return {
        "Formula": formula,
        "MW": round(_safe(Descriptors.MolWt)(mol), 1),
        "QED": round(_safe(Descriptors.qed)(mol), 2),
        "LogP": round(_safe(Descriptors.MolLogP)(mol), 2),
        "Heavy": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1),
        "Rings": mol.GetRingInfo().NumRings(),
        "Frags": len(Chem.GetMolFrags(mol)),
    }


def mol_svg(mol, w: int = 220, h: int = 170) -> str:
    try:
        AllChem.Compute2DCoords(mol)
        d = rdMolDraw2D.MolDraw2DSVG(w, h)
        opts = d.drawOptions()
        opts.bondLineWidth = 1.5
        opts.padding = 0.08
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        return d.GetDrawingText()
    except Exception:
        return f'<svg width="{w}" height="{h}"><text x="10" y="80" fill="#888">render failed</text></svg>'


# ---------------------------------------------------------------- io helpers

def read_json(p: Path):
    try:
        import json
        return json.loads(p.read_text())
    except Exception:
        return None


def read_smiles(p: Path) -> list[str]:
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def summarize_run(d: Path, molcap: int) -> dict:
    """Collect metrics + rendered molecules for one eval_* directory."""
    m = read_json(d / "metrics.json") or {}
    smiles = read_smiles(d / "smiles.txt")
    mols: list = []
    n_unparsable = 0
    sdf_path = d / "molecules.sdf"
    if sdf_path.exists() and sdf_path.stat().st_size > 0:
        try:
            supp = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
            for x in supp:
                if x is None:
                    n_unparsable += 1
                else:
                    mols.append(x)
        except (OSError, RuntimeError):
            n_unparsable = 0  # unreadable SDF (e.g. zero-valid run); treat as no molecules
    mols = mols[:molcap]
    svgs = [mol_svg(mol) for mol in mols]
    stats = [mol_stats(mol) for mol in mols]
    return {
        "dir": d.name,
        "metrics": m.get("metrics", {}),
        "run": m.get("run", {}),
        "num_valid": m.get("num_valid", len(smiles)),
        "num_total": m.get("num_total", "?"),
        "smiles": smiles,
        "svgs": svgs,
        "stats": stats,
        "n_unparsable": n_unparsable,
    }


def protocol(r: dict) -> str:
    run = r.get("run", {})
    bits = [f"DDIM-{run.get('ddim_steps', '?')}"]
    eta = run.get("eta", 0.0)
    if eta:
        bits.append(f"η={eta}")
    for flag in ("relax", "quadratic", "step_schedule"):
        if run.get(flag):
            bits.append(str(flag))
    return " · ".join(bits)


# ---------------------------------------------------------------- html pieces

BADGE = {
    True: '<span class="badge warn">⚠ inflated lens</span>',
    False: '<span class="badge ok">✓ honest eval</span>',
}

METRIC_CARDS = [
    "Validity", "Uniqueness", "Novelty", "IntDiv_p", "QED", "LogP", "SNN",
    "ConnectedValidity", "BondsPerMol", "ConnectedFrac", "ScaffDiv", "Lipinski%",
]
CARD_FMT = {
    "Validity": "{:.1f}%", "Uniqueness": "{:.1f}%", "Novelty": "{:.1f}%",
    "IntDiv_p": "{:.3f}", "QED": "{:.3f}", "LogP": "{:.2f}", "SNN": "{:.3f}",
    "ConnectedValidity": "{:.1f}%", "BondsPerMol": "{:.1f}",
    "ConnectedFrac": "{:.1f}%", "ScaffDiv": "{:.3f}", "Lipinski%": "{:.1f}%",
}


def esc(x) -> str:
    return html.escape(str(x))


def gallery(r: dict, total_valid: int) -> str:
    cards = []
    for i, st in enumerate(r["stats"]):
        stat_line = " · ".join(f"{k} {esc(v)}" for k, v in st.items() if k != "Formula")
        frag_note = (
            '<span class="frag">multi-fragment</span>'
            if st["Frags"] > 1
            else '<span class="frag single">single</span>'
        )
        cards.append(
            f'<div class="molcard">'
            f'<div class="molimg">{r["svgs"][i]}</div>'
            f'<div class="molmeta"><b>{esc(st["Formula"])}</b>{frag_note}<br>'
            f'<span>{stat_line}</span></div>'
            f"</div>"
        )
    more = ""
    if total_valid > len(r["stats"]):
        more = f'<p class="more">showing {len(r["stats"])} of {total_valid} valid molecules (raise --molcap for more)</p>'
    if not r["stats"]:
        more = '<p class="more">No valid molecules were produced — nothing to display.</p>'
    return f'<div class="grid">{"".join(cards)}</div>{more}'


def run_section(r: dict, inflated: bool) -> str:
    m = r["metrics"]
    cards = []
    for name in METRIC_CARDS:
        v = m.get(name)
        if isinstance(v, (int, float)):
            cards.append(
                f'<div class="metric"><div class="metric-val">{esc(CARD_FMT[name].format(v))}</div>'
                f'<div class="metric-label">{esc(name)}</div></div>'
            )
    detail = (
        f'<p class="proto">{esc(protocol(r))} · {r["num_valid"]}/{r["num_total"]} valid '
        f'· checkpoint: {esc(r["run"].get("checkpoint", "?"))}</p>'
    )
    total = r["num_valid"] if isinstance(r["num_valid"], int) else len(r["smiles"])
    g = "" if inflated else gallery(r, total)
    return (
        f'<section id="{esc(r["dir"])}" class="run{" inflated" if inflated else ""}">'
        f"<h2>{esc(r['dir'])} {BADGE[inflated]}</h2>"
        f'{detail}<div class="cards">{"".join(cards)}</div>'
        f'<details{" open" if not inflated else ""}><summary>Molecule gallery ({len(r["stats"])} shown)</summary>{g}</details>'
        f"</section>"
    )


def comparison_table(runs: list[dict]) -> str:
    cols = ["Validity", "Uniqueness", "Novelty", "IntDiv_p", "QED", "LogP", "SNN",
            "ConnectedValidity", "BondsPerMol"]
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    rows = []
    for r in runs:
        inflated = r["dir"] in INFLATED_RUNS
        tds = ""
        for c in cols:
            v = r["metrics"].get(c)
            tds += f"<td>{v:.3f}</td>" if isinstance(v, (int, float)) else "<td>—</td>"
        rows.append(
            f'<tr class="{"row-inflated" if inflated else ""}">'
            f'<td><a href="#{esc(r["dir"])}">{esc(r["dir"])}</a></td>{tds}'
            f"<td>{BADGE[inflated]}</td></tr>"
        )
    return (
        '<table class="cmp"><thead><tr><th>Run</th>' + head + "<th>Lens</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


CSS = """
:root { --bg:#0f1420; --panel:#171e2e; --line:#26304a; --text:#dde4f0; --dim:#8fa0bd;
        --accent:#4da3ff; --ok:#3ecf8e; --warn:#f5b04d; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:14px/1.45 -apple-system,'Segoe UI',Roboto,sans-serif; }
header { padding:26px 32px 10px; }
h1 { margin:0 0 4px; font-size:22px; }
.sub { color:var(--dim); font-size:13px; }
main { padding:0 32px 60px; }
h2 { font-size:17px; margin:34px 0 6px; }
.badge { font-size:11px; padding:2px 8px; border-radius:10px; vertical-align:middle; }
.badge.ok { background:rgba(62,207,142,.15); color:var(--ok); }
.badge.warn { background:rgba(245,176,77,.15); color:var(--warn); }
.proto { color:var(--dim); font-size:12px; margin:4px 0 12px; }
.cards { display:flex; flex-wrap:wrap; gap:10px; margin:8px 0 14px; }
.metric { background:var(--panel); border:1px solid var(--line); border-radius:10px;
          padding:10px 16px; min-width:104px; }
.metric-val { font-size:19px; font-weight:600; }
.metric-label { color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
table.cmp { border-collapse:collapse; width:100%; margin:14px 0 6px; font-size:13px; }
.cmp th, .cmp td { border:1px solid var(--line); padding:7px 10px; text-align:right; }
.cmp th:first-child, .cmp td:first-child { text-align:left; }
.cmp thead th { background:var(--panel); color:var(--dim); font-weight:600; }
.cmp a { color:var(--accent); text-decoration:none; }
tr.row-inflated td { opacity:.55; }
details { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px 16px; }
details summary { cursor:pointer; color:var(--dim); }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:14px; margin-top:14px; }
.molcard { background:var(--bg); border:1px solid var(--line); border-radius:12px; overflow:hidden;
           display:flex; flex-direction:column; }
.molimg { background:#fff; }
.molimg svg { display:block; width:100%; height:auto; }
.molmeta { padding:8px 12px 10px; font-size:12px; }
.molmeta span { color:var(--dim); }
.frag { float:right; font-size:10px; padding:1px 7px; border-radius:9px;
        background:rgba(245,176,77,.15); color:var(--warn); }
.frag.single { background:rgba(62,207,142,.15); color:var(--ok); }
.more { color:var(--dim); font-size:12px; }
.run.inflated { opacity:.8; }
"""


def build(out: Path, molcap: int, runs_filter: list[str] | None) -> None:
    eval_dirs = sorted(ROOT.glob("outputs/eval_*"))
    if runs_filter:
        eval_dirs = [d for d in eval_dirs if d.name in runs_filter]
    if not eval_dirs:
        sys.exit("No outputs/eval_*/ directories found.")
    runs = [summarize_run(d, molcap) for d in eval_dirs]

    honest = [r for r in runs if r["dir"] not in INFLATED_RUNS]
    inflated = [r for r in runs if r["dir"] in INFLATED_RUNS]
    honest.sort(key=lambda r: r["metrics"].get("Validity", 0), reverse=True)

    sections = [run_section(r, inflated=False) for r in honest]
    sections += [run_section(r, inflated=True) for r in inflated]

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>QM9 Diffusion — Eval Dashboard</title>
<style>{CSS}</style></head>
<body>
<header>
  <h1>Molecule Diffusion — Eval Dashboard</h1>
  <div class="sub">Generated {datetime.now():%Y-%m-%d %H:%M} · {len(honest)} honest runs · {len(inflated)} pre-fix runs (collapsed below) · 2D structures only</div>
</header>
<main>
  <h2>Run comparison</h2>
  {comparison_table(runs)}
  {''.join(sections)}
</main>
</body></html>"""

    out.write_text(page)
    print(f"Wrote {out} ({out.stat().st_size/1e6:.1f} MB, {len(runs)} runs, molcap={molcap})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--molcap", type=int, default=48,
                    help="max molecules embedded per run (2D gallery)")
    ap.add_argument("--runs", type=str, default=None,
                    help="comma-separated eval dir names to include (default: all)")
    args = ap.parse_args()
    build(args.out, args.molcap, args.runs.split(",") if args.runs else None)


if __name__ == "__main__":
    main()
