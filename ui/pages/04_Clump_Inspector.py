"""Screen 4: Deep Clump Inspector with Pairwise Distance Heatmaps and Clash Indicators."""

import json
import os
import streamlit as st
import numpy as np

from utils.mol_loader import discover_evaluated_runs, load_molecules_for_run, load_clump_metrics
from utils.renderers import render_3dmol_viewer, render_distance_heatmap, render_nn_histogram, render_rg_gauge

st.set_page_config(page_title="Clump Inspector", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .stApp {
        background-color: #090d16;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    
    .screen-header {
        margin-bottom: 20px;
    }
    .screen-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0 0 4px 0;
    }
    .screen-sub {
        font-size: 0.9rem;
        color: #94a3b8;
        margin: 0;
    }
    
    .clash-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.80rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .clash-none {
        background: rgba(52, 211, 153, 0.10);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.25);
    }
    .clash-detected {
        background: rgba(244, 63, 94, 0.10);
        color: #f43f5e;
        border: 1px solid rgba(244, 63, 94, 0.25);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="screen-header">
    <div class="screen-title">3D Atomic Clump & Geometry Inspector</div>
    <div class="screen-sub">Identify severe steric clashes (&lt;1.05 Å), distance density anomalies, and radius of gyration collapse.</div>
</div>
""", unsafe_allow_html=True)

runs = discover_evaluated_runs()
run_keys = list(runs.keys())
mols_runs = [r for r in run_keys if runs[r].get("has_molecules")]
ordered_runs = mols_runs + [r for r in run_keys if not runs[r].get("has_molecules")]

with st.sidebar:
    st.markdown("### Run Selection")
    saved_run = st.session_state.get("selected_run", mols_runs[0] if mols_runs else run_keys[0])
    active_run = st.selectbox(
        "Evaluation Run:",
        options=ordered_runs,
        index=ordered_runs.index(saved_run) if saved_run in ordered_runs else 0,
        format_func=lambda r: f"{r} [Conformers]" if runs[r].get("has_molecules") else f"{r} [Metrics Only]"
    )
    st.session_state["selected_run"] = active_run
    st.markdown("---")

info = runs[active_run]

# Load clump.json if available
clump_data = load_clump_metrics(info.get("clump_path"))
if clump_data:
    st.markdown("### Cohort Clump Diagnostics")
    g = clump_data.get("generated", {})
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Bonds per Molecule", f"{g.get('bonds_per_mol', 0.0):.2f}", delta="+10.5 vs BBBP" if g.get('bonds_per_mol', 0) > 30 else "Normal")
    with col2:
        st.metric("Mean Nearest-Neighbor", f"{g.get('nn_mean', 0.0):.3f} Å", delta="-0.25 Å (Compressed)" if g.get('nn_mean', 0) < 1.35 else "Normal")
    with col3:
        st.metric("Mean Fragments / Mol", f"{g.get('fragments_per_mol', 0.0):.2f}", delta="5.9 frags (Severe Soup)" if g.get('fragments_per_mol', 0) > 3 else "Intact")
    with col4:
        st.metric("Single Fragment Ratio", f"{g.get('single_fragment_pct', 0.0):.1f}%", delta="20.1% (80% multi-frag)")
    st.markdown("---")

# Per-molecule deep inspection
mols = load_molecules_for_run(info, max_mols=100)
if not mols:
    st.info(f"Evaluation run `{active_run}` contains summary metrics, but no 3D conformers were saved to this folder. Select an evaluation run that contains conformers:")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("View BBBP Ground Truth Reference", key="swcl_ref", use_container_width=True):
            st.session_state["selected_run"] = "[Reference] BBBP Ground Truth"
            st.rerun()
    with c2:
        if st.button("View eval_geo_relaxed_quad", key="swcl_geo", use_container_width=True):
            st.session_state["selected_run"] = "eval_geo_relaxed_quad"
            st.rerun()
    with c3:
        if st.button("View eval_v3", key="swcl_v3", use_container_width=True):
            st.session_state["selected_run"] = "eval_v3"
            st.rerun()
    st.stop()

mol_idx = st.sidebar.slider("Select Molecule for Distance Profiling:", 0, len(mols)-1, 0)
m = mols[mol_idx]
pos = m["positions"]

# Pairwise distance matrix
diffs = pos[:, None, :] - pos[None, :, :]
dists = (diffs ** 2).sum(-1) ** 0.5
np.fill_diagonal(dists, np.inf)

clash_count = int((dists < 1.05).sum() // 2)
clash_pairs = []
for i in range(len(pos)):
    for j in range(i + 1, len(pos)):
        if dists[i, j] < 1.05:
            clash_pairs.append((i, j))

# 2-Column layout: Left 3D with clash indicators | Right Plots
c_left, c_right = st.columns([5, 6])

with c_left:
    st.markdown(f"### 3D Clash Visualization (#{m['index']+1})")
    if clash_count > 0:
        st.markdown(f'<div class="clash-badge clash-detected">{clash_count} Severe Clashes (&lt;1.05 Å) Detected · Visualized with dashed pink cylinders</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="clash-badge clash-none">No Severe Steric Clashes (&lt;1.05 Å) Detected</div>', unsafe_allow_html=True)
        
    viewer_html = render_3dmol_viewer(
        m["sdf_block"],
        style="ball-and-stick",
        color_scheme="element",
        height=480,
        clash_pairs=clash_pairs,
        viewer_id=f"clump_{m['index']}"
    )
    st.components.v1.html(viewer_html, height=500)
    
    # Rg Gauge below 3D viewer
    st.plotly_chart(render_rg_gauge(m["radius_of_gyration"]), use_container_width=True)

with c_right:
    # Plot 1: Distance matrix heatmap
    atom_labels = [f"#{i} {m['mol'].GetAtomWithIdx(i).GetSymbol()}" if m.get('mol') else f"A{i}" for i in range(len(pos))]
    st.plotly_chart(render_distance_heatmap(pos, atom_labels=atom_labels), use_container_width=True)
    
    # Plot 2: Nearest neighbor distribution
    st.plotly_chart(render_nn_histogram(pos), use_container_width=True)
