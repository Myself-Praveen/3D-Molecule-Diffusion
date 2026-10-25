"""Visualization renderers for 2D depictions, 3Dmol.js viewers, and Plotly diagnostics."""

from __future__ import annotations

import base64
import html
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, rdmolops


def render_2d_svg(
    smiles_or_mol: str | Chem.Mol,
    width: int = 240,
    height: int = 240,
    dark_mode: bool = True
) -> str:
    """Generate clean SVG string for 2D molecular structure with high-contrast bonds."""
    if isinstance(smiles_or_mol, str):
        try:
            mol = Chem.MolFromSmiles(smiles_or_mol)
        except Exception:
            mol = None
    else:
        mol = smiles_or_mol

    if mol is None:
        return f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#090d16" rx="6"/>
            <text x="50%" y="50%" fill="#f43f5e" font-size="12" font-family="Inter, sans-serif" text-anchor="middle" dominant-baseline="middle">Invalid Structure</text>
        </svg>"""

    try:
        mol_copy = Chem.Mol(mol)
        rdDepictor.Compute2DCoords(mol_copy)
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        opts.bondLineWidth = 2.0
        if dark_mode:
            opts.setBackgroundColour((0.06, 0.08, 0.12, 1.0))
            opts.updateAtomPalette({
                0: (0.75, 0.80, 0.88),   # default / carbon bond lines
                1: (0.80, 0.85, 0.90),   # H
                6: (0.82, 0.86, 0.92),   # C
                7: (0.38, 0.70, 1.00),   # N (sky blue)
                8: (0.95, 0.38, 0.38),   # O (coral red)
                9: (0.35, 0.85, 0.75),   # F (cyan/teal)
                15: (1.00, 0.65, 0.20),  # P (orange)
                16: (0.95, 0.85, 0.25),  # S (yellow)
                17: (0.35, 0.88, 0.45),  # Cl (emerald)
                35: (0.90, 0.45, 0.20),  # Br (amber)
                53: (0.60, 0.35, 0.85),  # I (violet)
            })
        drawer.DrawMolecule(mol_copy)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception as exc:
        return f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#090d16" rx="6"/>
            <text x="50%" y="50%" fill="#64748b" font-size="11" font-family="Inter, sans-serif" text-anchor="middle" dominant-baseline="middle">Rendering Error: {html.escape(str(exc)[:25])}</text>
        </svg>"""


def render_3dmol_viewer(
    sdf_or_pdb_data: str,
    format: str = "sdf",
    style: str = "ball-and-stick",
    color_scheme: str = "element",
    height: int = 450,
    width: str = "100%",
    element_valences: Optional[Dict[int, Dict[str, Any]]] = None,
    fragment_mapping: Optional[Dict[int, int]] = None,
    clash_pairs: Optional[List[Tuple[int, int]]] = None,
    viewer_id: str = "mol_viewer",
) -> str:
    """Generate self-contained HTML embedding 3Dmol.js with custom representations and coloring."""
    clean_data = sdf_or_pdb_data.replace("\r\n", "\n")
    data_b64 = base64.b64encode(clean_data.encode("utf-8")).decode("utf-8")

    # Map styles to 3Dmol representation JSON
    style_configs = {
        "stick": "{stick: {radius: 0.15}}",
        "ball-and-stick": "{stick: {radius: 0.14}, sphere: {scale: 0.28}}",
        "sphere": "{sphere: {scale: 0.85}}",
        "line": "{line: {linewidth: 2.0}}",
        "surface": "{stick: {radius: 0.12}}",
    }
    base_style = style_configs.get(style, style_configs["ball-and-stick"])

    # Fragment or valence coloring hooks
    coloring_script = ""
    if color_scheme == "element":
        coloring_script = "// Standard CPK default"
    elif color_scheme == "fragment" and fragment_mapping:
        colors = ["#38bdf8", "#34d399", "#fbbf24", "#f43f5e", "#a78bfa", "#60a5fa", "#f59e0b", "#e879f9"]
        lines = []
        for atom_idx, frag_id in fragment_mapping.items():
            col = colors[frag_id % len(colors)]
            lines.append(f'viewer.setStyle({{index: {atom_idx}}}, {{stick: {{color: "{col}", radius: 0.14}}, sphere: {{color: "{col}", scale: 0.30}}}});')
        coloring_script = "\n".join(lines)
    elif color_scheme == "valence" and element_valences:
        lines = []
        for atom_idx, val_info in element_valences.items():
            if val_info.get("val", 0) > val_info.get("v_max", 4):
                col = "#f43f5e"  # Rose for hypervalent
                scale = 0.45
            else:
                col = "#34d399"  # Emerald for satisfied
                scale = 0.25
            lines.append(f'viewer.setStyle({{index: {atom_idx}}}, {{stick: {{radius: 0.14}}, sphere: {{color: "{col}", scale: {scale}}}}});')
        coloring_script = "\n".join(lines)
    elif color_scheme == "bbb":
        coloring_script = 'viewer.setStyle({}, {stick: {color: "#34d399", radius: 0.14}, sphere: {color: "#34d399", scale: 0.28}});'

    surface_script = ""
    if style == "surface":
        surface_script = 'viewer.addSurface($3Dmol.SurfaceType.VDW, {opacity: 0.70, color: "white"});'

    clash_script = ""
    if clash_pairs:
        for a1, a2 in clash_pairs[:30]:
            clash_script += f'viewer.addCylinder({{start: {{atom: {a1}}}, end: {{atom: {a2}}}, radius: 0.05, color: "#f43f5e", dashed: true}});\n'

    html_code = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #090d16; font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif; }}
        #container_{viewer_id} {{ width: 100%; height: {height}px; position: relative; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.08); }}
        .badge-overlay {{ position: absolute; top: 12px; left: 14px; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(8px); padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 500; color: #94a3b8; border: 1px solid rgba(255,255,255,0.08); z-index: 10; letter-spacing: 0.02em; }}
        .reset-btn {{ position: absolute; bottom: 12px; right: 14px; background: #0f172a; color: #cbd5e1; border: 1px solid rgba(255, 255, 255, 0.12); padding: 5px 10px; border-radius: 6px; font-size: 11px; font-weight: 500; cursor: pointer; z-index: 10; transition: background 0.15s ease, border-color 0.15s ease; }}
        .reset-btn:hover {{ background: #1e293b; border-color: rgba(56, 189, 248, 0.35); color: #f8fafc; }}
    </style>
</head>
<body>
    <div id="container_{viewer_id}">
        <div class="badge-overlay">Conformation · {style} · {color_scheme}</div>
        <button class="reset-btn" onclick="viewer.zoomTo(); viewer.render();">Reset View</button>
    </div>
    <script>
        const rawData = atob("{data_b64}");
        const element = document.getElementById("container_{viewer_id}");
        const viewer = $3Dmol.createViewer(element, {{ backgroundColor: "#090d16" }});
        viewer.addModel(rawData, "{format}");
        viewer.setStyle({{}}, {base_style});
        {coloring_script}
        {surface_script}
        {clash_script}
        viewer.zoomTo();
        viewer.render();
    </script>
</body>
</html>"""
    return html_code


def render_distance_heatmap(
    positions: np.ndarray,
    atom_labels: Optional[List[str]] = None,
    title: str = "Pairwise Atomic Distance Matrix (Å)"
) -> go.Figure:
    """Plot pairwise distance heatmap with clash highlights (<1.1 Å)."""
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diffs ** 2, axis=-1))

    if atom_labels is None:
        atom_labels = [f"Atom {i}" for i in range(len(positions))]

    fig = px.imshow(
        dist_matrix,
        x=atom_labels,
        y=atom_labels,
        color_continuous_scale="Viridis_r",
        zmin=0.0,
        zmax=6.0,
        labels=dict(x="Atom", y="Atom", color="Dist (Å)"),
        title=title,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, sans-serif", size=11, color="#cbd5e1"),
        height=420,
    )
    return fig


def render_nn_histogram(
    positions: np.ndarray,
    reference_peak: float = 1.54
) -> go.Figure:
    """Plot nearest neighbor distance distribution highlighting clump mass (<1.0 Å)."""
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dist_matrix, np.inf)
    nn_distances = np.min(dist_matrix, axis=-1)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=nn_distances,
        nbinsx=20,
        name="Generated Nearest Neighbors",
        marker_color="#38bdf8",
        opacity=0.85,
    ))
    fig.add_vline(
        x=reference_peak, line_width=2, line_dash="dash", line_color="#34d399",
        annotation_text="Real BBBP C-C (1.54 Å)", annotation_position="top right"
    )
    fig.add_vrect(
        x0=0.0, x1=1.05, fillcolor="#f43f5e", opacity=0.18, line_width=0,
        annotation_text="Clash Regime (<1.05 Å)", annotation_position="top left"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        title="Nearest-Neighbor Distance Distribution",
        xaxis_title="Distance to Nearest Neighbor (Å)",
        yaxis_title="Count",
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, sans-serif", size=11, color="#cbd5e1"),
        height=320,
    )
    return fig


def render_rg_gauge(rg_val: float, ref_range: Tuple[float, float] = (3.09, 3.43)) -> go.Figure:
    """Plot Radius of Gyration gauge comparing sample against real BBBP range."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rg_val,
        title={'text': "Radius of Gyration (Å)", 'font': {'size': 13, 'color': '#cbd5e1', 'family': 'Inter, sans-serif'}},
        gauge={
            'axis': {'range': [0, 8.0], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': "#38bdf8"},
            'bgcolor': "#0f172a",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.08)",
            'steps': [
                {'range': [0, 2.5], 'color': 'rgba(244, 63, 94, 0.20)'},
                {'range': [ref_range[0], ref_range[1]], 'color': 'rgba(52, 211, 153, 0.35)'},
                {'range': [5.0, 8.0], 'color': 'rgba(244, 63, 94, 0.20)'},
            ],
            'threshold': {
                'line': {'color': "#34d399", 'width': 3},
                'thickness': 0.8,
                'value': (ref_range[0] + ref_range[1]) / 2.0
            }
        }
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        font=dict(family="Inter, sans-serif", size=12, color="#cbd5e1"),
        height=260,
        margin=dict(l=30, r=30, t=30, b=20),
    )
    return fig


def render_lab_gate_funnel(stage_counts: Dict[str, int], total: int = 1000) -> go.Figure:
    """Plot 7-stage biochemical lab-readiness survival funnel."""
    stages = list(stage_counts.keys())
    values = list(stage_counts.values())

    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textinfo="value+percent initial",
        marker=dict(
            color=["#38bdf8", "#60a5fa", "#818cf8", "#a78bfa", "#fbbf24", "#34d399", "#10b981"]
        ),
        connector={"line": {"color": "#334155", "dash": "dot", "width": 1.5}},
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#090d16",
        plot_bgcolor="#090d16",
        title="7-Stage Biochemical Lab-Readiness Gate Funnel",
        font=dict(family="Inter, sans-serif", size=11, color="#cbd5e1"),
        height=380,
        margin=dict(l=160, r=40, t=40, b=30),
    )
    return fig
