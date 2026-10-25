"""Screen 5: Reverse Diffusion Trajectory Scrubber."""

import time
import numpy as np
import streamlit as st
from rdkit import Chem

from utils.mol_loader import discover_evaluated_runs, load_molecules_for_run, generate_mock_trajectory_step
from utils.renderers import render_3dmol_viewer

st.set_page_config(page_title="Trajectory Scrubber", layout="wide")

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
    
    .status-pill {
        display: block;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.80rem;
        font-weight: 600;
        margin-top: 8px;
    }
    .pill-active {
        background: rgba(52, 211, 153, 0.10);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.25);
    }
    .pill-inactive {
        background: rgba(148, 163, 184, 0.10);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.20);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="screen-header">
    <div class="screen-title">Reverse Diffusion Trajectory Scrubber</div>
    <div class="screen-sub">Scrub through reverse Langevin dynamics from isotropic Gaussian noise ($t=T$) to clean conformer ($t=0$).</div>
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
mols = load_molecules_for_run(info, max_mols=40)

if not mols:
    st.info(f"Evaluation run `{active_run}` contains summary metrics, but no 3D conformers were saved to this folder. Select an evaluation run that contains conformers:")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("View BBBP Ground Truth Reference", key="swtr_ref", use_container_width=True):
            st.session_state["selected_run"] = "[Reference] BBBP Ground Truth"
            st.rerun()
    with c2:
        if st.button("View eval_geo_relaxed_quad", key="swtr_geo", use_container_width=True):
            st.session_state["selected_run"] = "eval_geo_relaxed_quad"
            st.rerun()
    with c3:
        if st.button("View eval_v3", key="swtr_v3", use_container_width=True):
            st.session_state["selected_run"] = "eval_v3"
            st.rerun()
    st.stop()

mol_idx = st.sidebar.slider("Select Target Conformer:", 0, len(mols)-1, 0)
target_mol = mols[mol_idx]
final_pos = target_mol["positions"]

# Sidebar trajectory parameters
with st.sidebar:
    st.markdown("### DDIM Settings")
    total_steps = st.selectbox("DDIM Steps:", [50, 100, 200], index=2)
    step_schedule = st.selectbox("Step Schedule:", ["quadratic (denser at low t)", "linear"], index=0)
    guidance_mode = st.selectbox("Guidance Mode:", ["Steric + Physical Valence Guidance", "Unconditional (No Guidance)", "Steric Only"], index=0)

# Scrubber Slider: t=200 down to t=0
current_step = st.slider(
    "Reverse Diffusion Step (t):",
    min_value=0,
    max_value=total_steps,
    value=0,
    step=1,
    help="Step 200 = pure noise; Step 0 = clean final generation"
)

# Compute progress fraction: 0.0 = noise (step 200), 1.0 = clean (step 0)
progress = 1.0 - (current_step / total_steps)

# Generate trajectory coordinates for current step
interp_pos = generate_mock_trajectory_step(final_pos, progress, seed=42 + mol_idx)

# Build temporary SDF block for 3Dmol viewer
temp_mol = Chem.Mol(target_mol["mol"])
conf = temp_mol.GetConformer()
for atom_idx, p in enumerate(interp_pos):
    conf.SetAtomPosition(atom_idx, p)
temp_sdf_block = Chem.MolToMolBlock(temp_mol)

# Calculate dynamic step metrics
diffs = interp_pos[:, None, :] - interp_pos[None, :, :]
dists = (diffs ** 2).sum(-1) ** 0.5
np.fill_diagonal(dists, np.inf)

clashes_now = int((dists < 1.05).sum() // 2)
bonds_now = int((dists < 1.65).sum() // 2)
radius_now = float(np.sqrt(np.mean(np.sum((interp_pos - interp_pos.mean(axis=0)) ** 2, axis=1))))

# Layout
c1, c2 = st.columns([7, 3])

with c2:
    st.markdown("### Step Metrics")
    st.metric("Timestep Index (t)", f"{current_step} / {total_steps}")
    st.metric("Noise Level (sigma_t)", f"{(1.0 - progress):.3f}")
    st.metric("Provisional Bonds Formed", bonds_now, delta=f"{bonds_now - target_mol['num_bonds']} vs final")
    st.metric("Severe Clashes (<1.05 Å)", clashes_now, delta="Repulsive active" if clashes_now > 0 else "Clear", delta_color="inverse")
    st.metric("Cloud Radius (Rg)", f"{radius_now:.2f} Å", delta=f"{radius_now - target_mol['radius_of_gyration']:.2f} Å vs final")
    
    st.markdown("---")
    st.markdown("**Guidance Ramp Status**")
    if progress >= 0.70:
        st.markdown(f'<div class="status-pill pill-active">Guidance Ramp Active (Weight: {0.5 * (progress - 0.7) / 0.3:.2f})</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill pill-inactive">High-Noise Regime (Guidance Inactive)</div>', unsafe_allow_html=True)

with c1:
    st.markdown(f"### Atomic State (Progress: {progress*100:.1f}%)")
    viewer_html = render_3dmol_viewer(
        temp_sdf_block,
        style="ball-and-stick" if progress > 0.4 else "sphere",
        color_scheme="element" if clashes_now == 0 else "valence",
        height=520,
        viewer_id=f"traj_{current_step}"
    )
    st.components.v1.html(viewer_html, height=540)
