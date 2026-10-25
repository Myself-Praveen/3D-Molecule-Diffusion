"""Screen 2: Interactive 3D Molecule Viewer with CPK, Fragment, and Valence Coloring."""

import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdmolops

from utils.mol_loader import discover_evaluated_runs, load_molecules_for_run, DEFAULT_VALENCE_MAX
from utils.renderers import render_3dmol_viewer

st.set_page_config(page_title="3D Molecule Viewer", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .stApp {
        background-color: #090d16;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    
    .viewer-header {
        margin-bottom: 20px;
    }
    .viewer-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0 0 4px 0;
    }
    .viewer-sub {
        font-size: 0.9rem;
        color: #94a3b8;
        margin: 0;
    }
    
    .prop-box {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    
    .status-pill {
        display: block;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.80rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .pill-ok {
        background: rgba(52, 211, 153, 0.10);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.25);
    }
    .pill-warn {
        background: rgba(251, 191, 36, 0.10);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.25);
    }
    .pill-err {
        background: rgba(244, 63, 94, 0.10);
        color: #f43f5e;
        border: 1px solid rgba(244, 63, 94, 0.25);
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 12px;
    }
    .metric-item {
        background: #090d16;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 6px;
        padding: 10px;
    }
    .metric-item-label {
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 2px;
    }
    .metric-item-val {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f1f5f9;
        font-variant-numeric: tabular-nums;
    }
    
    div.stButton > button, div[data-testid="stDownloadButton"] > button {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em !important;
        padding: 7px 14px !important;
        transition: all 0.15s ease !important;
    }
    div.stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover {
        background-color: #334155 !important;
        border-color: rgba(56, 189, 248, 0.5) !important;
        color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)

runs = discover_evaluated_runs()
run_keys = list(runs.keys())
saved_run = st.session_state.get("selected_run", run_keys[0] if run_keys else "None")

with st.sidebar:
    st.markdown("### Run Selection")
    active_run = st.selectbox(
        "Evaluation Run:",
        options=run_keys,
        index=run_keys.index(saved_run) if saved_run in run_keys else 0,
        format_func=lambda r: f"{r} [Conformers]" if runs[r].get("has_molecules") else f"{r} [Metrics Only]"
    )
    st.session_state["selected_run"] = active_run
    st.markdown("---")

# Header
st.markdown(f"""
<div class="viewer-header">
    <div class="viewer-title">3D Molecule Conformer Viewer: {active_run}</div>
    <div class="viewer-sub">Inspect Cartesian geometry, identify hypervalent atom coordination, and visualize isolated fragments.</div>
</div>
""", unsafe_allow_html=True)

info = runs[active_run]
mols = load_molecules_for_run(info, max_mols=400)

if not mols:
    st.info(f"Evaluation run `{active_run}` contains summary metrics, but no 3D conformers were saved to this folder. Select an evaluation run that contains conformers:")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("View BBBP Ground Truth Reference", key="sw3d_ref", use_container_width=True):
            st.session_state["selected_run"] = "[Reference] BBBP Ground Truth"
            st.rerun()
    with c2:
        if st.button("View eval_geo_relaxed_quad (498 mols)", key="sw3d_geo", use_container_width=True):
            st.session_state["selected_run"] = "eval_geo_relaxed_quad"
            st.rerun()
    with c3:
        if st.button("View eval_v3 (622 mols)", key="sw3d_v3", use_container_width=True):
            st.session_state["selected_run"] = "eval_v3"
            st.rerun()
    st.stop()

# Selected molecule index
mol_options = list(range(len(mols)))
saved_idx = st.session_state.get("inspect_mol_idx", 0)
default_idx = saved_idx if 0 <= saved_idx < len(mols) else 0

current_pos = st.sidebar.selectbox(
    "Select Molecule Index:",
    options=mol_options,
    index=default_idx,
    format_func=lambda i: f"Molecule #{i+1} ({mols[i]['num_atoms']} atoms, {mols[i]['num_frags']} frags)"
)
st.session_state["inspect_mol_idx"] = current_pos

m = mols[current_pos]
rdkit_mol = m["mol"]

# Layout: Left metadata (3 cols) | Center 3D viewer (6 cols) | Right controls (3 cols)
col_meta, col_viewer, col_ctrl = st.columns([3, 6, 3])

with col_ctrl:
    st.markdown("### Viewer Controls")
    rep_style = st.selectbox(
        "Representation:",
        ["ball-and-stick", "stick", "sphere", "surface", "line"],
        index=0
    )
    
    color_scheme = st.selectbox(
        "Atom Color Scheme:",
        ["element", "fragment", "valence", "bbb"],
        index=0,
        help="'fragment': colors each disconnected component differently. 'valence': red indicates hypervalency."
    )
    
    show_clashes = st.checkbox("Highlight Steric Clashes (< 1.1 Å)", value=False)
    
    st.markdown("---")
    st.markdown("**Bond Orders**")
    perceive_orders = st.checkbox("Perceive Double/Triple Bonds", value=False)
    
    st.download_button(
        "Download Conformer (SDF)",
        data=m["sdf_block"],
        file_name=f"{active_run}_mol_{current_pos}.sdf",
        mime="chemical/x-mdl-sdfile",
        use_container_width=True
    )

with col_meta:
    st.markdown("### Structure Diagnostics")
    st.markdown("**Canonical SMILES**")
    st.code(m["smiles"], language="text")
    
    # Status pills
    if m["is_connected"]:
        st.markdown('<div class="status-pill pill-ok">Single Connected Component</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-pill pill-warn">Multi-Fragment ({m["num_frags"]} fragments)</div>', unsafe_allow_html=True)
        
    if m["valence_violations"] > 0:
        st.markdown(f'<div class="status-pill pill-err">{m["valence_violations"]} Hypervalent Atoms Detected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill pill-ok">Valency Criteria Satisfied</div>', unsafe_allow_html=True)
        
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-item">
            <div class="metric-item-label">Atoms</div>
            <div class="metric-item-val">{m['num_atoms']}</div>
        </div>
        <div class="metric-item">
            <div class="metric-item-label">Bonds</div>
            <div class="metric-item-val">{m['num_bonds']}</div>
        </div>
        <div class="metric-item">
            <div class="metric-item-label">Radius Gyration</div>
            <div class="metric-item-val">{m['radius_of_gyration']} Å</div>
        </div>
        <div class="metric-item">
            <div class="metric-item-label">Drug-Likeness (QED)</div>
            <div class="metric-item-val">{m['qed']:.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Compute atom valence and fragment mapping
fragment_mapping = {}
if rdkit_mol:
    frags = rdmolops.GetMolFrags(rdkit_mol, asMols=False)
    for frag_id, atom_indices in enumerate(frags):
        for a_idx in atom_indices:
            fragment_mapping[a_idx] = frag_id

element_valences = {}
if rdkit_mol:
    for atom in rdkit_mol.GetAtoms():
        z = atom.GetAtomicNum()
        v_max = DEFAULT_VALENCE_MAX.get(z, 4)
        val = atom.GetExplicitValence() + atom.GetImplicitValence()
        element_valences[atom.GetIdx()] = {"z": z, "val": val, "v_max": v_max}

# Detect clashes (< 1.1 A)
clash_pairs = []
if show_clashes and m.get("positions") is not None:
    pos = m["positions"]
    diffs = pos[:, None, :] - pos[None, :, :]
    dists = (diffs ** 2).sum(-1) ** 0.5
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if dists[i, j] < 1.10:
                clash_pairs.append((i, j))

with col_viewer:
    st.markdown(f"### Conformer Geometry (#{current_pos + 1})")
    viewer_html = render_3dmol_viewer(
        m["sdf_block"],
        format="sdf",
        style=rep_style,
        color_scheme=color_scheme,
        height=520,
        element_valences=element_valences,
        fragment_mapping=fragment_mapping,
        clash_pairs=clash_pairs,
        viewer_id=f"view_{m['index']}"
    )
    st.components.v1.html(viewer_html, height=540)
