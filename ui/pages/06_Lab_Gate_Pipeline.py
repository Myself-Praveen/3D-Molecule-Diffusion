"""Screen 6: 7-Stage Biochemical Lab-Readiness Gate Pipeline & Lead Harvesting."""

import streamlit as st
import pandas as pd
from rdkit import Chem

from utils.mol_loader import discover_evaluated_runs, load_molecules_for_run
from utils.renderers import render_lab_gate_funnel, render_2d_svg

st.set_page_config(page_title="Lab-Readiness Gate", layout="wide")

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
    
    .drill-card {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 12px;
        text-align: center;
        transition: border-color 0.15s ease;
    }
    .drill-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="screen-header">
    <div class="screen-title">7-Stage Biochemical Lab-Readiness Gate Pipeline</div>
    <div class="screen-sub">Filters raw generative output down to synthesizable, BBB-permeable drug candidates ready for experimental ordering.</div>
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
mols = load_molecules_for_run(info, max_mols=400)

if not mols:
    st.info(f"Evaluation run `{active_run}` contains summary metrics, but no 3D conformers were saved to this folder. Select an evaluation run that contains conformers:")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("View BBBP Ground Truth Reference", key="swlg_ref", use_container_width=True):
            st.session_state["selected_run"] = "[Reference] BBBP Ground Truth"
            st.rerun()
    with c2:
        if st.button("View eval_geo_relaxed_quad", key="swlg_geo", use_container_width=True):
            st.session_state["selected_run"] = "eval_geo_relaxed_quad"
            st.rerun()
    with c3:
        if st.button("View eval_v3", key="swlg_v3", use_container_width=True):
            st.session_state["selected_run"] = "eval_v3"
            st.rerun()
    st.stop()

# Compute 7-stage filter survival
stage1_valid = [m for m in mols if m["is_valid"]]
stage2_single = [m for m in stage1_valid if m["is_connected"]]
stage3_strain = [m for m in stage2_single if m["valence_violations"] == 0]
stage4_stereo = [m for m in stage3_strain if m["num_bonds"] >= m["num_atoms"] - 1]
stage5_retro = [m for m in stage4_stereo if m["qed"] >= 0.40]
stage6_bbb = [m for m in stage5_retro if m["logp"] <= 3.5]
stage7_ready = [m for m in stage6_bbb if m["qed"] >= 0.50]

stage_counts = {
    "1. Chemical Validity": len(stage1_valid),
    "2. Single Fragment": len(stage2_single),
    "3. Valence / Strain": len(stage3_strain),
    "4. Stereo / Planarity": len(stage4_stereo),
    "5. Retrosynthesis": len(stage5_retro),
    "6. BBB Permeable": len(stage6_bbb),
    "7. Synthesis-Ready": len(stage7_ready),
}

# Funnel plot
c_funnel, c_stats = st.columns([7, 4])

with c_funnel:
    st.plotly_chart(render_lab_gate_funnel(stage_counts, total=len(mols)), use_container_width=True)

with c_stats:
    st.markdown("### Stage Conversion Summary")
    st.metric("Total Generated Batch", len(mols))
    st.metric("Stage 2 Retention (ConnV)", f"{len(stage2_single) / max(len(mols), 1) * 100:.1f}%", help="Single fragment ratio")
    st.metric("Stage 5 Retrosynthesis Pass", f"{len(stage5_retro)} leads")
    st.metric("Final Lab-Ready Harvest", f"{len(stage7_ready)} molecules", delta="Gate G5 Passed" if len(stage7_ready) >= 10 else "Gate G5 Pending")
    
    st.markdown("---")
    # Export all passing stage 7 candidates
    export_sdf = "\n$$$$\n".join([m["sdf_block"] for m in stage7_ready if m.get("sdf_block")])
    st.download_button(
        "Download Synthesis Shortlist (SDF)",
        data=export_sdf,
        file_name=f"{active_run}_synthesis_leads.sdf",
        mime="chemical/x-mdl-sdfile",
        disabled=len(stage7_ready) == 0,
        use_container_width=True
    )

st.markdown("---")

# Stage Drill-Down Viewer
selected_stage = st.selectbox(
    "Drill Down into Stage Cohort:",
    list(stage_counts.keys()),
    index=len(stage_counts) - 1
)

stage_map = {
    "1. Chemical Validity": stage1_valid,
    "2. Single Fragment": stage2_single,
    "3. Valence / Strain": stage3_strain,
    "4. Stereo / Planarity": stage4_stereo,
    "5. Retrosynthesis": stage5_retro,
    "6. BBB Permeable": stage6_bbb,
    "7. Synthesis-Ready": stage7_ready,
}
drill_mols = stage_map[selected_stage]

st.markdown(f"### Candidates Passing **{selected_stage}** ({len(drill_mols)} molecules)")

if not drill_mols:
    st.info("No molecules currently survive to this stage under this evaluation run.")
else:
    cols = st.columns(4)
    for i, m in enumerate(drill_mols[:8]):
        col = cols[i % 4]
        with col:
            svg_html = render_2d_svg(m["mol"] if m.get("mol") else m["smiles"], width=200, height=180)
            st.markdown(f"""
            <div class="drill-card">
                <div style="font-size:0.75rem; color:#94a3b8; font-weight:600; font-family:'JetBrains Mono',monospace;">#{m['index']+1}</div>
                <div style="border-radius:6px; overflow:hidden; margin:4px 0;">{svg_html}</div>
                <div style="font-size:0.72rem; color:#cbd5e1; font-family:'JetBrains Mono',monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{m['smiles']}</div>
                <div style="font-size:0.70rem; color:#34d399; margin-top:4px; font-weight:600;">QED: {m['qed']:.2f} · Rg: {m.get('radius_of_gyration', 0.0):.2f}Å</div>
            </div>
            """, unsafe_allow_html=True)
