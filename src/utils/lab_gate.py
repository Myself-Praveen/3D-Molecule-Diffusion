"""Lab-readiness gate: run all six stages and render one verdict.

The six stages in ``src/utils/chem_qa.py`` and ``src/utils/retrosynthesis.py``
each answer one question a synthesis lab asks before ordering a compound.
This module runs them all over one cohort and turns the numbers into a
single PASS / FAIL / INCOMPLETE verdict a script (or CI) can act on:

    stage 1  perception     can bond orders + stereo be read back?
    stage 2  geometry       are bond lengths anywhere near covalent norms?
    stage 3  synthesis      complexity proxy clean, PAINS-free?
    stage 4  domain         are candidates inside the learned domain?
    stage 5  ADMET          soluble enough? liability alerts firing?
    stage 6  routes         assemblable from common building blocks?

Conventions, matching the rest of the repo:

* **Sentinel -1.0** for "could not measure" (same precedent as ``BBB%`` in
  ``evaluate()``): ``Perception%`` and ``MeanBondStrainPct`` need 3D
  coordinates, ``InDomain%`` needs a training set. A criterion reading a
  negative value is reported as SKIP, never as a pass or a fail — "we could
  not measure this" must not masquerade as "this is fine".
* **Denominators are valid molecules** (MOSES convention), except the
  perception/geometry stages which additionally require 3D coordinates and
  say so via ``n_3d``.
* **Every threshold is data, not code**: ``DEFAULT_CRITERIA`` states
  metric, operator, value and a one-line rationale for each, and callers
  can replace the dict wholesale. The defaults are documented heuristics
  chosen as first-pass floors, not literature standards — they are meant
  to be tightened per assay, not quoted.
* Verdict: any FAIL -> FAIL; otherwise any SKIP -> INCOMPLETE (some
  criterion could not be evaluated); otherwise PASS. Incomplete evidence
  never counts as success.
"""

from __future__ import annotations

from typing import Sequence

from rdkit import Chem

from src.utils import chem_qa
from src.utils.retrosynthesis import feasibility_rate

# Each criterion: metric key (must exist in the flattened report),
# comparison operator, threshold, and why the floor exists.
DEFAULT_CRITERIA: dict[str, dict] = {
    "perception": {
        "metric": "Perception%", "op": ">=", "threshold": 95.0,
        "why": "bond-order perception must succeed on essentially every "
               "3D sample; a failure means the structure cannot be drawn "
               "unambiguously",
    },
    "geometry": {
        "metric": "MeanBondStrainPct", "op": "<=", "threshold": 15.0,
        "why": "mean |measured - ideal| bond length within 15% of the "
               "covalent norm (loose first-pass floor for distance-built "
               "geometry)",
    },
    "pains": {
        "metric": "PAINSFree%", "op": ">=", "threshold": 95.0,
        "why": "PAINS alerts are assay artifacts; a model that rediscovers "
               "them optimises against interference, not activity",
    },
    "in_domain": {
        "metric": "InDomain%", "op": ">=", "threshold": 80.0,
        "why": "most candidates should sit inside the applicability domain; "
               "beyond it, learned oracles (BBB%, QED, ESOL) extrapolate",
    },
    "soluble": {
        "metric": "ESOL_Soluble%", "op": ">=", "threshold": 60.0,
        "why": "majority should clear the conventional logS >= -4 mol/L "
               "aqueous-solubility floor",
    },
    "routes": {
        "metric": "RouteFeasibleRatio%", "op": ">=", "threshold": 50.0,
        "why": "at least half the routability that REAL training compounds "
               "achieve under the identical bounded search — absolute 2-step "
               "rates are dataset-specific (BBBP-like space: ~1% even for "
               "purchasable drugs; QM9-size: near 100%), so an absolute "
               "threshold would fail real chemistry",
    },
}

# Sentinel for "not measurable with the inputs given" (BBB% precedent).
NA = -1.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def lab_readiness_report(
    mols: Sequence[Chem.Mol | None],
    train_mols: Sequence[Chem.Mol] | None = None,
) -> dict:
    """Run all six stages over ``mols``; returns a nested stage report.

    ``train_mols`` feeds stage 4 (applicability domain) and the stage 6
    routability baseline; without it both report the -1.0 sentinel and
    their criteria will SKIP.
    Safe on empty input and on ``None`` entries (module convention).
    """
    valid_all = list(mols)
    valid = [m for m in valid_all if m is not None]
    has_3d = [m for m in valid if m.GetNumConformers() > 0]
    train_list = list(train_mols) if train_mols else []

    # --- stage 1: perception ------------------------------------------------
    bos = [chem_qa.bond_order_summary(m) for m in valid]
    sts = [chem_qa.stereo_summary(m) for m in valid]
    s1: dict[str, float] = {
        **{k: _mean([b[k] for b in bos])
           for k in ("NonSingleFrac", "AromaticFrac", "MeanBondOrder")},
        **{k: _mean([s[k] for s in sts])
           for k in ("StereocentresPerMol", "StereoDefinedFrac")},
    }
    if has_3d:
        ok = sum(
            1 for m in has_3d
            if chem_qa.perceive_bond_orders(m)[1] in ("ok", "already_ordered")
        )
        stereo_ok = sum(
            1 for m in has_3d
            if chem_qa.assign_stereochemistry(m)[1] == "ok"
        )
        s1["Perception%"] = 100.0 * ok / len(has_3d)
        s1["StereoReadback%"] = 100.0 * stereo_ok / len(has_3d)
    else:
        s1["Perception%"] = NA
        s1["StereoReadback%"] = NA

    # --- stage 2: geometry --------------------------------------------------
    s2: dict[str, float] = {"n_3d": float(len(has_3d))}
    if has_3d:
        devs = [chem_qa.bond_length_deviation(m) for m in has_3d]
        measured = [d for d in devs if d["n_bonds_measured"] > 0]
        s2["MeanBondStrainPct"] = (
            _mean([d["MeanBondStrainPct"] for d in measured])
            if measured else NA
        )
        s2["MaxBondStrainPct"] = (
            max(d["MaxBondStrainPct"] for d in measured)
            if measured else NA
        )
        energies = [e for e in (chem_qa.strain_energy(m) for m in has_3d)
                    if e is not None]
        s2["MeanStrainKcal"] = _mean(energies) if energies else NA
    else:
        s2["MeanBondStrainPct"] = NA
        s2["MaxBondStrainPct"] = NA
        s2["MeanStrainKcal"] = NA

    # --- stage 3: synthetic accessibility + PAINS ---------------------------
    sas = [s for s in (chem_qa.synthetic_complexity_score(m) for m in valid)
           if s is not None]
    s3 = {
        "SAComplexity": _mean(sas) if sas else NA,
        "PAINSFree%": chem_qa.pains_free_rate(valid),
    }

    # --- stage 4: applicability domain --------------------------------------
    if valid and train_list:
        s4 = dict(chem_qa.applicability_domain(valid, train_list))
    else:
        # No reference set -> not measurable, not zero. -1.0 makes the
        # criterion SKIP instead of failing on a fabricated 0.0.
        s4 = {"InDomain%": NA, "MeanMaxTanimoto": NA,
              "PctBeyond40": NA, "PctBeyond60": NA}

    # --- stage 5: ADMET -----------------------------------------------------
    s5 = dict(chem_qa.admet_summary(valid))

    # --- stage 6: retrosynthesis --------------------------------------------
    s6 = dict(feasibility_rate(valid))
    # Routability RELATIVE to real chemistry under the identical search.
    # Calibration measured, not assumed: on the BBBP training set the
    # default 2-step bounds route only ~1% of *purchasable drugs*, while
    # QM9-size molecules route near-100% — no absolute threshold serves
    # both, so the criterion is "half as routable as the training set".
    # Large training sets are subsampled deterministically (fixed-seed draw)
    # so the baseline stays seconds, not minutes.
    if valid and train_list:
        route_ref = train_list
        if len(route_ref) > 2000:
            import random as _random

            route_ref = _random.Random(0).sample(route_ref, 500)
        train_rate = feasibility_rate(route_ref)["RouteFeasible%"]
        s6["TrainRouteFeasible%"] = train_rate
        s6["RouteFeasibleRatio%"] = (
            100.0 * s6["RouteFeasible%"] / train_rate
            if train_rate > 0 else NA
        )
    else:
        s6["TrainRouteFeasible%"] = NA
        s6["RouteFeasibleRatio%"] = NA

    return {
        "n_total": len(valid_all),
        "n_valid": len(valid),
        "n_3d": len(has_3d),
        "stage1_perception": s1,
        "stage2_geometry": s2,
        "stage3_synthesis": s3,
        "stage4_domain": s4,
        "stage5_admet": s5,
        "stage6_routes": s6,
    }


def flatten_report(report: dict) -> dict[str, float]:
    """One dict of every stage metric + bookkeeping counts, for metrics.json.

    Keys are unique across stages and disjoint from ``evaluate()``'s keys,
    so an eval can merge them directly (``metrics.update(...)``).
    """
    flat: dict[str, float] = {
        "n_total": report["n_total"],
        "n_valid": report["n_valid"],
        "n_3d": report["n_3d"],
    }
    for key, stage in report.items():
        if key.startswith("stage"):
            flat.update(stage)
    return flat


def run_gate(
    mols: Sequence[Chem.Mol | None],
    train_mols: Sequence[Chem.Mol] | None = None,
    criteria: dict[str, dict] | None = None,
) -> dict:
    """Compute the report, apply criteria, return a JSON-ready verdict.

    Returns ``{"report", "metrics", "criteria", "verdicts", "overall"}``
    where ``metrics`` is the flattened report (merge into an eval's metric
    dict) and ``overall`` is PASS / FAIL / INCOMPLETE.
    """
    report = lab_readiness_report(mols, train_mols)
    metrics = flatten_report(report)
    crit = dict(criteria) if criteria is not None else dict(DEFAULT_CRITERIA)

    verdicts: dict[str, dict] = {}
    for name, c in crit.items():
        value = metrics.get(c["metric"])
        if report["n_valid"] == 0:
            status, detail = "SKIP", "no valid molecules to evaluate"
        elif value is None or value < 0:
            status, detail = "SKIP", "not measurable with these inputs " \
                                     "(missing 3D coords or training set)"
        else:
            ops = {">=": value >= c["threshold"], "<=": value <= c["threshold"]}
            if c["op"] not in ops:
                raise ValueError(f"unsupported op {c['op']!r} in criterion {name!r}")
            status = "PASS" if ops[c["op"]] else "FAIL"
            detail = ""
        verdicts[name] = {
            "metric": c["metric"], "value": value,
            "op": c["op"], "threshold": c["threshold"],
            "status": status, "detail": detail, "why": c["why"],
        }

    statuses = [v["status"] for v in verdicts.values()]
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "SKIP" in statuses:
        overall = "INCOMPLETE"
    else:
        overall = "PASS"
    return {
        "report": report, "metrics": metrics, "criteria": crit,
        "verdicts": verdicts, "overall": overall,
    }


def print_gate(gate: dict) -> None:
    """Print the verdict table: one row per criterion + the overall verdict."""
    report = gate["report"]
    print("\n" + "=" * 74)
    print(f"LAB-READINESS GATE  ({report['n_total']} total, "
          f"{report['n_valid']} valid, {report['n_3d']} with 3D coords)")
    print("=" * 74)
    print(f"{'criterion':<12} {'metric':<20} {'value':>9} "
          f"{'op':>3} {'thresh':>7}  status")
    print("-" * 74)
    for name, v in gate["verdicts"].items():
        val = f"{v['value']:.2f}" if v["value"] is not None else "n/a"
        print(f"{name:<12} {v['metric']:<20} {val:>9} "
              f"{v['op']:>3} {v['threshold']:>7.1f}  {v['status']}")
        if v["status"] == "SKIP" and v["detail"]:
            print(f"{'':<12}   -> {v['detail']}")
    print("-" * 74)
    fails = [n for n, v in gate["verdicts"].items() if v["status"] == "FAIL"]
    if fails:
        for n in fails:
            v = gate["verdicts"][n]
            print(f"  FAIL {n}: {v['metric']}={v['value']:.2f} misses "
                  f"{v['op']}{v['threshold']:.1f} — {v['why']}")
    print(f"OVERALL: {gate['overall']}")
    print("=" * 74)
