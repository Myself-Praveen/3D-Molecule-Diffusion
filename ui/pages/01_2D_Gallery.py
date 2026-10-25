"""Screen 1: 2D Molecule Gallery with Chemical Depictions and Property Filtering."""

import streamlit as st
from utils.mol_loader import discover_evaluated_runs, load_molecules_for_run
from utils.renderers import render_2d_svg

st.set_page_config(page_title="2D Molecule Gallery", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .stApp {
        background-color: #090d16;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    
    .gallery-header {
        margin-bottom: 20px;
    }
    .gallery-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0 0 4px 0;
    }
    .gallery-sub {
        font-size: 0.9rem;
        color: #94a3b8;
        margin: 0;
    }
    
    .mol-card {
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 14px;
        display: flex;
        flex-direction: column;
        align-items: center;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .mol-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.4);
    }
    
    .status-badge {
        font-size: 0.70rem;
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-left: 4px;
    }
    .badge-emerald { background: rgba(52, 211, 153, 0.12); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.25); }
    .badge-amber { background: rgba(251, 191, 36, 0.12); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.25); }
    .badge-rose { background: rgba(244, 63, 94, 0.12); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.25); }
    .badge-cyan { background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.25); }
    
    .smi-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #94a3b8;
        background: #090d16;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 4px 8px;
        border-radius: 6px;
        max-width: 220px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        margin-top: 8px;
        letter-spacing: -0.01em;
    }
    
    .mol-stats-row {
        width: 100%;
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
        font-size: 0.74rem;
        color: #64748b;
        font-variant-numeric: tabular-nums;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        padding-top: 6px;
    }
    .mol-stats-row b {
        color: #cbd5e1;
        font-weight: 600;
    }
    
    div.stButton > button, div[data-testid="stDownloadButton"] > button {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 6px !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em !important;
        padding: 6px 10px !important;
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

# Sidebar filters & run selection
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
    
    st.markdown("### Filters")
    show_connected_only = st.checkbox("Single Fragment Only (Connected)", value=False)
    show_valid_only = st.checkbox("Chemically Valid Only", value=True)
    min_qed = st.slider("Minimum QED (Drug-likeness)", 0.0, 1.0, 0.0, 0.05)
    page_size = st.selectbox("Molecules per page:", [12, 24, 48], index=1)

# Main Header
st.markdown(f"""
<div class="gallery-header">
    <div class="gallery-title">2D Molecule Gallery: {active_run}</div>
    <div class="gallery-sub">Browse 2D chemical structures, inspect connectivity status, and filter by pharmacological properties.</div>
</div>
""", unsafe_allow_html=True)

info = runs[active_run]
molecules = load_molecules_for_run(info, max_mols=400)

if not molecules:
    st.info(f"Evaluation run `{active_run}` contains summary metrics, but no 3D conformers were saved to this folder. Select an evaluation run that contains conformers:")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("View BBBP Ground Truth Reference", key="sw_ref", use_container_width=True):
            st.session_state["selected_run"] = "[Reference] BBBP Ground Truth"
            st.rerun()
    with c2:
        if st.button("View eval_geo_relaxed_quad (498 mols)", key="sw_geo", use_container_width=True):
            st.session_state["selected_run"] = "eval_geo_relaxed_quad"
            st.rerun()
    with c3:
        if st.button("View eval_v3 (622 mols)", key="sw_v3", use_container_width=True):
            st.session_state["selected_run"] = "eval_v3"
            st.rerun()
    st.stop()

# Apply filters
filtered_mols = []
for m in molecules:
    if show_valid_only and not m["is_valid"]:
        continue
    if show_connected_only and not m["is_connected"]:
        continue
    if m["qed"] < min_qed:
        continue
    filtered_mols.append(m)

st.caption(f"Showing {len(filtered_mols)} of {len(molecules)} loaded molecules")

# Pagination
total_pages = max(1, (len(filtered_mols) + page_size - 1) // page_size)
cur_page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1, step=1)
start_idx = (cur_page - 1) * page_size
end_idx = min(start_idx + page_size, len(filtered_mols))
page_mols = filtered_mols[start_idx:end_idx]

# Render Grid
cols_per_row = 4
rows = [page_mols[i:i + cols_per_row] for i in range(0, len(page_mols), cols_per_row)]

for row in rows:
    cols = st.columns(cols_per_row)
    for col, m in zip(cols, row):
        with col:
            svg_html = render_2d_svg(m["mol"] if m.get("mol") else m["smiles"], width=220, height=200)
            
            conn_badge = '<span class="status-badge badge-emerald">Connected</span>' if m['is_connected'] else f'<span class="status-badge badge-amber">{m["num_frags"]} Frags</span>'
            valid_badge = '<span class="status-badge badge-emerald">Valid</span>' if m['is_valid'] else '<span class="status-badge badge-rose">Invalid</span>'
            
            st.markdown(f"""
            <div class="mol-card">
                <div style="width:100%; display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-size:0.75rem; font-weight:600; color:#94a3b8; font-family:'JetBrains Mono',monospace;">#{m['index'] + 1}</span>
                    <div>{conn_badge}{valid_badge}</div>
                </div>
                <div style="display:flex; justify-content:center; width:100%; border-radius:6px; overflow:hidden;">{svg_html}</div>
                <div class="smi-text" title="{m['smiles']}">{m['smiles']}</div>
                <div class="mol-stats-row">
                    <span>Bonds: <b>{m['num_bonds']}</b></span>
                    <span>QED: <b>{m['qed']:.2f}</b></span>
                    <span>Rg: <b>{m.get('radius_of_gyration', 0.0):.2f}Å</b></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button("Inspect 3D", key=f"insp_{m['index']}", use_container_width=True):
                    st.session_state["inspect_mol_idx"] = m["index"]
                    st.switch_page("pages/02_3D_Viewer.py")
            with b_col2:
                st.download_button(
                    "Download SDF",
                    data=m["sdf_block"] or "",
                    file_name=f"mol_{m['index']}.sdf",
                    key=f"dl_{m['index']}",
                    disabled=not m.get("sdf_block"),
                    use_container_width=True
                )
