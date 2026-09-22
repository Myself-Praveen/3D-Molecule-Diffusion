#!/usr/bin/env python
"""Build a self-contained HTML dashboard from eval outputs.

Usage:
    .venv/bin/python scripts/build_dashboard.py [--out outputs/dashboard.html]
                                                [--molcap 48] [--runs name1,name2]

Scans outputs/eval_*/ directories (each with metrics.json, smiles.txt and
optionally molecules.sdf), renders 2D SVG structures with RDKit, embeds
3Dmol.js (downloaded once, cached in /tmp), and writes one self-contained
HTML file that opens in any browser — no server, no network needed.

Notes
-----
* smiles.txt holds the valid molecules only (that is what the eval counted),
  so the gallery shows exactly what the Validity metric measured.
* Runs produced before the covalent-radii eval fix are auto-badged as
  "inflated lens" and collapsed by default, so stale numbers can't mislead.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "dashboard.html"
GL3DMOL = Path("/tmp/3Dmol-min.js")

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


# ---------------------------------------------------------------- 3Dmol.js

def get_3dmol_js() -> str:
    """Inline 3Dmol.js (cached in /tmp); CDN fallback tag if unavailable."""
    if GL3DMOL.exists() and GL3DMOL.stat().st_size > 100_000:
        return "<script>" + GL3DMOL.read_text() + "</script>"
    try:
        req = urllib.request.Request(
            "https://3Dmol.org/build/3Dmol-min.js",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        data = urllib.request.urlopen(req, timeout=15).read()
        GL3DMOL.write_bytes(data)
        return "<script>" + data.decode() + "</script>"
    except Exception:
        return '<script src="https://3Dmol.org/build/3Dmol-min.js"></script>'


# ---------------------------------------------------------------- io helpers

def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def read_smiles(p: Path) -> list[str]:
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def summarize_run(d: Path, molcap: int) -> dict:
    """Collect metrics + parsed molecules for one eval_* directory."""
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
    # Snapshot the real 3D conformers NOW, before the 2D gallery rendering
    # mutates them in place (Compute2DCoords/PrepareAndDrawMolecule flatten z).
    molblocks = [
        Chem.MolToMolBlock(mol)
        for mol in mols[:molcap]
        if mol.GetNumConformers() > 0
    ]
    return {
        "dir": d.name,
        "metrics": m.get("metrics", {}),
        "run": m.get("run", {}),
        "num_valid": m.get("num_valid", len(smiles)),
        "num_total": m.get("num_total", m.get("metrics", {}).get("num_total", "?")),
        "smiles": smiles,
        "mols": mols,
        "molblocks": molblocks,
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
    ("Validity", "{:.1f}%", "good_high"),
    ("Uniqueness", "{:.1f}%", "good_high"),
    ("Novelty", "{:.1f}%", "good_high"),
    ("IntDiv_p", "{:.3f}", "good_high"),
    ("QED", "{:.3f}", "good_high"),
    ("LogP", "{:.2f}", "neutral"),
    ("SNN", "{:.3f}", "neutral"),
    ("ConnectedValidity", "{:.1f}%", "good_high"),
    ("BondsPerMol", "{:.1f}", "neutral"),
    ("ConnectedFrac", "{:.1f}%", "good_high"),
    ("ScaffDiv", "{:.3f}", "neutral"),
    ("Lipinski%", "{:.1f}%", "good_high"),
]


def fmt_card(name: str, val) -> tuple[str, str] | None:
    """Return (label, formatted value) for a metric card, or None to skip."""
    if val is None or not isinstance(val, (int, float)):
        return None
    for label, spec, _ in METRIC_CARDS:
        if label == name:
            return label, spec.format(val)
    return None


def esc(x) -> str:
    return html.escape(str(x))


def gallery(r: dict, molcap: int) -> str:
    n_show = min(len(r["mols"]), molcap)
    cards = []
    for i in range(n_show):
        mol, st = r["mols"][i], r["stats"][i]
        stat_line = " · ".join(
            f"{k} {esc(v)}" for k, v in st.items() if k != "Formula"
        )
        frag_note = (
            '<span class="frag">multi-fragment</span>'
            if st["Frags"] > 1
            else '<span class="frag single">single</span>'
        )
        cards.append(
            f'<div class="molcard">'
            f'<div class="molimg">{mol_svg(mol)}</div>'
            f'<div class="molmeta"><b>{esc(st["Formula"])}</b>{frag_note}<br>'
            f'<span>{stat_line}</span></div>'
            f'<button class="btn3d" onclick="show3D(\'{r["dir"]}\',{i})">3D view</button>'
            f"</div>"
        )
    more = ""
    if len(r["mols"]) > molcap:
        more = f'<p class="more">showing {molcap} of {len(r["mols"])} valid molecules (raise --molcap for more)</p>'
    if n_show == 0:
        more = '<p class="more">No valid molecules were produced — nothing to display.</p>'
    return f'<div class="grid">{"".join(cards)}</div>{more}'


def run_section(r: dict, molcap: int, inflated: bool, idx: int) -> str:
    m = r["metrics"]
    cards = []
    for name, (_, fmt) in zip([c[0] for c in METRIC_CARDS], [(c[0], c[1]) for c in METRIC_CARDS]):
        if name in m and isinstance(m[name], (int, float)):
            label, val = fmt_card(name, m[name]) or (name, str(m[name]))
            cards.append(
                f'<div class="metric"><div class="metric-val">{esc(val)}</div>'
                f'<div class="metric-label">{esc(label)}</div></div>'
            )
    detail = (
        f'<p class="proto">{esc(protocol(r))} · {r["num_valid"]}/{r["num_total"]} valid '
        f'· checkpoint: {esc(r["run"].get("checkpoint", "?"))}</p>'
    )
    g = "" if inflated else gallery(r, molcap)
    hidden = " hidden" if inflated else ""
    return (
        f'<section id="{esc(r["dir"])}" class="run{" inflated" if inflated else ""}">'
        f"<h2>{esc(r['dir'])} {BADGE[inflated]}</h2>"
        f'{detail}<div class="cards">{"".join(cards)}</div>'
        f'<details{" open" if not inflated else ""}{hidden}><summary>Molecule gallery ({len(r["mols"])} valid)</summary>{g}</details>'
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


def embed_3d_data(runs: list[dict]) -> str:
    """Per-run SDF text blocks for the 3D viewer, as a JS object.

    Uses the molblocks snapshotted at load time — re-serializing the RDKit
    objects here would embed whatever the 2D gallery renderer left behind
    (flattened z=0 coordinates, which produce a blank 3D view).
    """
    data = {r["dir"]: r["molblocks"] for r in runs if r["molblocks"]}
    return "<script>const MOLDATA=" + json.dumps(data) + ";</script>"


VIEWER_JS = """
function show3D(run, idx) {
  const modal = document.getElementById('modal');
  modal.classList.add('open');
  const holder = document.getElementById('viewer');
  holder.innerHTML = '';
  const label = document.getElementById('modal-label');
  label.textContent = run + ' — molecule #' + (idx + 1);
  const sdf = MOLDATA[run] && MOLDATA[run][idx];
  if (!sdf) { label.textContent += ' (no 3D conformer)'; return; }
  const v = $3Dmol.createViewer(holder, {backgroundColor: 'white'});
  v.addModel(sdf, 'sdf');
  v.setStyle({}, {stick: {radius: 0.18}, sphere: {scale: 0.22}});
  v.zoomTo();
  v.render();
  v.zoom(1.3, 800);
}
function hide3D() { document.getElementById('modal').classList.remove('open'); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') hide3D(); });
"""

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
.molmeta { padding:8px 12px 4px; font-size:12px; }
.molmeta span { color:var(--dim); }
.frag { float:right; font-size:10px; padding:1px 7px; border-radius:9px;
        background:rgba(245,176,77,.15); color:var(--warn); }
.frag.single { background:rgba(62,207,142,.15); color:var(--ok); }
.btn3d { margin:8px 12px 12px; align-self:flex-start; background:var(--accent); color:#00254d;
         border:0; border-radius:8px; padding:6px 14px; font-weight:600; cursor:pointer; }
.more { color:var(--dim); font-size:12px; }
.run.inflated { opacity:.8; }
#modal { position:fixed; inset:0; background:rgba(0,0,0,.72); display:none;
         align-items:center; justify-content:center; z-index:50; }
#modal.open { display:flex; }
.modal-box { background:var(--panel); border:1px solid var(--line); border-radius:14px;
             padding:16px; width:min(640px,92vw); }
#modal-label { margin:0 0 10px; font-weight:600; }
#viewer { width:100%; height:440px; border-radius:10px; background:#fff; position:relative; }
.close { float:right; cursor:pointer; color:var(--dim); border:0; background:none; font-size:18px; }
"""


def build(out: Path, molcap: int, runs_filter: list[str] | None) -> None:
    eval_dirs = sorted(ROOT.glob("outputs/eval_*"))
    if runs_filter:
        eval_dirs = [d for d in eval_dirs if d.name in runs_filter]
    if not eval_dirs:
        sys.exit("No outputs/eval_*/ directories found.")
    runs = [summarize_run(d, molcap) for d in eval_dirs]
    for r in runs:
        r["stats"] = [mol_stats(m) for m in r["mols"]]

    honest = [r for r in runs if r["dir"] not in INFLATED_RUNS]
    inflated = [r for r in runs if r["dir"] in INFLATED_RUNS]
    honest.sort(key=lambda r: r["metrics"].get("Validity", 0), reverse=True)

    sections = [run_section(r, molcap, inflated=False, idx=i) for i, r in enumerate(honest)]
    sections += [run_section(r, molcap, inflated=True, idx=i) for i, r in enumerate(inflated)]

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>QM9 Diffusion — Eval Dashboard</title>
<style>{CSS}</style></head>
<body>
<header>
  <h1>Molecule Diffusion — Eval Dashboard</h1>
  <div class="sub">Generated {datetime.now():%Y-%m-%d %H:%M} · {len(honest)} honest runs · {len(inflated)} runs measured through the pre-fix eval lens (collapsed below) · click any molecule for 3D</div>
</header>
<main>
  <h2>Run comparison</h2>
  {comparison_table(runs)}
  {''.join(sections)}
</main>
<div id="modal" onclick="if(event.target===this)hide3D()">
  <div class="modal-box">
    <button class="close" onclick="hide3D()">✕</button>
    <p id="modal-label"></p>
    <div id="viewer"></div>
  </div>
</div>
{embed_3d_data(runs)}
{get_3dmol_js()}
<script>{VIEWER_JS}</script>
</body></html>"""

    out.write_text(page)
    print(f"Wrote {out} ({out.stat().st_size/1e6:.1f} MB, {len(runs)} runs, molcap={molcap})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--molcap", type=int, default=48,
                    help="max molecules embedded per run (2D gallery + 3D data)")
    ap.add_argument("--runs", type=str, default=None,
                    help="comma-separated eval dir names to include (default: all)")
    args = ap.parse_args()
    build(args.out, args.molcap, args.runs.split(",") if args.runs else None)


if __name__ == "__main__":
    main()
