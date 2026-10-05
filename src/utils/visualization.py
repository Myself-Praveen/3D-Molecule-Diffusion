"""Phase 7: all paper figures and tables (implementation.md §7.1).

Every ``plot_*`` / ``render_*`` function:
- is defensive against missing/empty inputs (draws an annotated placeholder
  instead of raising — sweep/ablation data may not exist yet),
- saves BOTH ``<output_path>.pdf`` (paper) and ``<output_path>.png``
  (slides) via :func:`save_figure`,
- uses seaborn's colorblind palette on a whitegrid theme.

``scripts/generate_paper_figures.py`` wires real experiment outputs (and
synthetic fallbacks) into these functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind")

PALETTE = sns.color_palette("colorblind")


def save_figure(fig, output_path: str | Path) -> list[Path]:
    """Save ``fig`` as both PDF and PNG next to ``output_path``.

    A suffix on ``output_path`` is stripped, so ``figs/fig1.pdf`` and
    ``figs/fig1`` both produce ``figs/fig1.pdf`` + ``figs/fig1.png``.
    Returns the written paths.
    """
    out = Path(output_path)
    if out.suffix:
        out = out.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [out.with_suffix(".pdf"), out.with_suffix(".png")]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return paths


def _placeholder(output_path: str | Path, message: str):
    """Empty-data figure so a pipeline run never crashes on missing data."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11,
            style="italic", color="0.4")
    return save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Fig. 1 — federated convergence
# ---------------------------------------------------------------------------

def plot_convergence_curves(history_paths: dict[str, str], output_path: str):
    """Server loss vs round for multiple configs (Fig. 1).

    Args:
        history_paths: {"IID K=4": "outputs/fed_iid/history.json", ...}
    """
    import json

    plotted = 0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, path in history_paths.items():
        try:
            history = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rounds, losses = [], []
        for entry in history:
            loss = entry.get("server_loss")
            if loss is None:
                continue
            rounds.append(entry.get("round", len(rounds) + 1))
            losses.append(float(loss))
        if not rounds:
            continue
        ax.plot(rounds, losses, marker="o", markersize=3, label=label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return _placeholder(output_path, "No history.json data found")
    ax.set_xlabel("Round")
    ax.set_ylabel("Server loss")
    ax.set_title("Federated convergence")
    ax.legend(fontsize=8)
    return save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Table I — metric comparison
# ---------------------------------------------------------------------------

def plot_metric_comparison_table(results: dict[str, dict], output_path: str):
    """LaTeX-ready comparison table (Table I).

    ``results`` maps run name -> metric dict. Draws the table as a figure
    and returns the LaTeX tabular source (also written to ``<stem>.tex``).
    """
    if not results:
        return _placeholder(output_path, "No results to tabulate")
    cols: list[str] = []
    for metrics in results.values():
        for key in metrics:
            if key not in cols:
                cols.append(key)
    cols = [c for c in cols if c not in ("LogP",)] or cols

    n_rows = len(results)
    fig, ax = plt.subplots(figsize=(0.9 * len(cols) + 2, 0.5 * n_rows + 1.5))
    ax.axis("off")
    cell_text = []
    for name, metrics in results.items():
        row = []
        for col in cols:
            val = metrics.get(col, "")
            row.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        cell_text.append(row)
    table = ax.table(
        cellText=cell_text,
        colLabels=cols,
        rowLabels=list(results.keys()),
        cellLoc="right",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    ax.set_title("Table I — generated-molecule metrics", fontsize=10)
    paths = save_figure(fig, output_path)

    header = " & ".join(cols) + " \\\\"
    latex_lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}", header,
                   "\\hline"]
    for name, row in zip(results.keys(), cell_text):
        latex_lines.append(name + " & " + " & ".join(row) + " \\\\")
    latex_lines.append("\\end{tabular}")
    latex = "\n".join(latex_lines)
    Path(paths[0]).with_suffix(".tex").write_text(latex + "\n")
    return latex


# ---------------------------------------------------------------------------
# Fig. 2 — lambda Pareto front
# ---------------------------------------------------------------------------

def plot_pareto_front(sweep_results: list[dict], output_path: str):
    """Validity vs Uniqueness Pareto front over the λ₂×λ₃ grid (Fig. 2).

    Each dict needs ``Validity`` and ``Uniqueness``; optional ``label``
    annotates the point.
    """
    pts = [r for r in sweep_results
           if isinstance(r, dict) and "Validity" in r and "Uniqueness" in r]
    if not pts:
        return _placeholder(output_path, "No sweep results (run sweep_lambda)")
    fig, ax = plt.subplots(figsize=(6, 5))
    xs = [float(r["Validity"]) for r in pts]
    ys = [float(r["Uniqueness"]) for r in pts]
    ax.scatter(xs, ys, s=60, zorder=3)
    for r, x, y in zip(pts, xs, ys):
        if r.get("label"):
            ax.annotate(str(r["label"]), (x, y), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Validity (%)")
    ax.set_ylabel("Uniqueness (%)")
    ax.set_title("λ₂ × λ₃ Pareto front")
    return save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Fig. 3 — mode-collapse analysis
# ---------------------------------------------------------------------------

def plot_mode_collapse_analysis(history_paths: dict, output_path: str):
    """IntDiv and scaffold diversity vs round (Fig. 3).

    Reads ``gen`` snapshots from fed history files; missing series are
    skipped, and if nothing is available an annotated placeholder is saved.
    """
    import json

    series: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for label, path in history_paths.items():
        try:
            history = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        found = {"IntDiv_p": [], "ScaffDiv": []}
        for entry in history:
            gen = entry.get("gen") or {}
            rnd = entry.get("round")
            for key in found:
                if key in gen and gen[key] is not None:
                    found[key].append((rnd, float(gen[key])))
        if any(found.values()):
            series[label] = found
    if not series:
        return _placeholder(
            output_path,
            "No IntDiv/ScaffDiv snapshots in history (needs sample_eval_every)",
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, found in series.items():
        for ax, key, title in ((axes[0], "IntDiv_p", "Internal diversity"),
                               (axes[1], "ScaffDiv", "Scaffold diversity")):
            if found[key]:
                xs, ys = zip(*found[key])
                ax.plot(xs, ys, marker="o", markersize=3, label=label)
    for ax, title in zip(axes, ("Internal diversity", "Scaffold diversity")):
        ax.set_xlabel("Round")
        ax.set_title(title)
        ax.legend(fontsize=8)
    return save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Fig. 4 — ablation bars
# ---------------------------------------------------------------------------

def plot_ablation_bars(ablation_results: dict, output_path: str):
    """Grouped bar chart of ablation results (Fig. 4).

    ``ablation_results``: {arm_name: {metric: value}}. Bars are grouped by
    metric across arms.
    """
    if not ablation_results:
        return _placeholder(output_path, "No ablation results (run run_ablation)")
    metric_names: list[str] = []
    for metrics in ablation_results.values():
        for key in metrics:
            if key not in metric_names:
                metric_names.append(key)
    metric_names = metric_names[:4]  # keep the chart readable
    arms = list(ablation_results.keys())

    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(arms)), 4.8))
    width = 0.8 / max(len(metric_names), 1)
    xs = np.arange(len(arms))
    for m_idx, metric in enumerate(metric_names):
        vals = [float(ablation_results[a].get(metric, 0.0)) for a in arms]
        ax.bar(xs - 0.4 + (m_idx + 0.5) * width, vals, width, label=metric)
    ax.set_xticks(xs)
    ax.set_xticklabels(arms, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Ablation study")
    ax.legend(fontsize=8)
    return save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Fig. 5 — BBB property distributions
# ---------------------------------------------------------------------------

def _mol_properties(mol) -> dict[str, float] | None:
    from rdkit.Chem import Descriptors, rdMolDescriptors

    try:
        return {
            "QED": Descriptors.qed(mol),
            "LogP": Descriptors.MolLogP(mol),
            "tPSA": rdMolDescriptors.CalcTPSA(mol),
            "MW": Descriptors.MolWt(mol),
        }
    except Exception:
        return None


def plot_bbb_property_distribution(mols, output_path: str):
    """Violin plots of QED / LogP / tPSA / MW per group (Fig. 5).

    ``mols`` is either ``{"BBB+": [mols...], "BBB-": [mols...]}`` or a list
    of ``(mol, label)`` pairs.
    """
    if isinstance(mols, dict):
        groups = {str(k): list(v) for k, v in mols.items()}
    else:
        groups = {}
        for item in mols:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                return _placeholder(
                    output_path,
                    "plot_bbb_property_distribution expects dict[str, list] "
                    "or list[(mol, label)]",
                )
            mol, label = item
            groups.setdefault(str(label), []).append(mol)

    rows = []
    for label, group in groups.items():
        for mol in group:
            props = _mol_properties(mol)
            if props:
                rows.append({"group": label, **props})
    if len(rows) < 4:
        return _placeholder(output_path, "Not enough molecules with properties")

    import pandas as pd

    df = pd.DataFrame(rows)
    long = df.melt(id_vars="group", var_name="Property", value_name="Value")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=long, x="Property", y="Value", hue="group",
                   inner="quart", ax=ax, cut=0)
    ax.set_title("Generated-molecule property distributions")
    ax.legend(fontsize=8)
    return save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Fig. 6 — 3D molecule renderings
# ---------------------------------------------------------------------------

_CPK_COLORS = {
    1: "#d9d9d9", 6: "#404040", 7: "#3050f8", 8: "#ff0d0d",
    9: "#90e050", 16: "#ffff30", 17: "#1ff01f", 53: "#940094",
    15: "#ff8000", 35: "#a62929",
}
_CPK_SIZES = {1: 40, 6: 90, 7: 85, 8: 80, 9: 70, 16: 100, 17: 100,
              53: 110, 15: 100, 35: 105}


def render_molecules_3d(mols: Sequence, output_path: str, max_mols: int = 6):
    """Ball-and-stick renderings of representative molecules (Fig. 6).

    Positions come from the first conformer; bonds from the distance graph
    already assigned by the caller's RDKit Mol (GetBonds).
    """
    usable = []
    for m in list(mols)[:max_mols]:
        if m is None:
            continue
        try:
            if m.GetNumConformers() == 0:
                continue
            usable.append(m)
        except Exception:
            continue
    if not usable:
        return _placeholder(output_path, "No renderable molecules")

    n = len(usable)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    for i, mol in enumerate(usable):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        conf = mol.GetConformer()
        pos = np.array([[conf.GetAtomPosition(k).x,
                         conf.GetAtomPosition(k).y,
                         conf.GetAtomPosition(k).z]
                        for k in range(mol.GetNumAtoms())])
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ax.plot(*zip(pos[a], pos[b]), color="0.35", linewidth=1.4,
                    zorder=1)
        for idx, atom in enumerate(mol.GetAtoms()):
            z = atom.GetAtomicNum()
            ax.scatter(*pos[idx], s=_CPK_SIZES.get(z, 80),
                       c=_CPK_COLORS.get(z, "#f0c8a0"),
                       edgecolors="black", linewidths=0.5, depthshade=False,
                       zorder=2)
        try:
            from rdkit.Chem import rdMolDescriptors
            title = rdMolDescriptors.CalcMolFormula(mol)
        except Exception:
            title = f"mol {i + 1}"
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
        _set_axes_equal(ax, pos)
    return save_figure(fig, output_path)


def _set_axes_equal(ax, pts: np.ndarray) -> None:
    center = pts.mean(axis=0)
    radius = float(np.abs(pts - center).max()) + 1e-3
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


# ---------------------------------------------------------------------------
# Fig. 7 — guidance-scale controllability
# ---------------------------------------------------------------------------

def plot_guidance_scale_sweep(results: dict, output_path: str):
    """BBB% vs guidance scale w (Fig. 7 — controllability).

    ``results``: {w: bbb_percent} or {w: {"BBB%": value}}.
    """
    clean: dict[float, float] = {}
    for w, v in (results or {}).items():
        try:
            if isinstance(v, dict):
                v = v.get("BBB%", v.get("BBB", None))
            if v is None:
                continue
            clean[float(w)] = float(v)
        except (TypeError, ValueError):
            continue
    if not clean:
        return _placeholder(output_path, "No guidance-sweep results")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = sorted(clean)
    ys = [clean[x] for x in xs]
    ax.plot(xs, ys, marker="o", linewidth=2, color=PALETTE[0])
    ax.axhline(50.0, linestyle="--", color="0.5", label="random baseline")
    ax.set_xlabel("Guidance scale w")
    ax.set_ylabel("BBB% (oracle)")
    ax.set_title("Classifier-free guidance controllability")
    ax.legend(fontsize=8)
    return save_figure(fig, output_path)
