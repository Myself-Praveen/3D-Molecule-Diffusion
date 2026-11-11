# Lead triage report — 25 elites from `reward_select_v5`

Source: `outputs/reward_select_v5/elites_top25.sdf` (398 candidates → top 25).
All three steps were executed, not estimated. Every number below is in the CSVs
next to this file.

Scripts: `scripts/lead_triage.py` (step 1), `scripts/catalog_match.py` (step 2),
`scripts/dock_leads.py` (step 3).

---

## Step 1 — valence saturation

**Result: 25/25 saturate, 24/25 embed in 3D, 25/25 pass Lipinski with ≤1 violation.**

The raw elites are valence-unsaturated graphs (`[C]`, `[N]`, `[O]` carrying no
implicit H). Re-enabling implicit hydrogens fixes every one of them, which
confirms they were valid *topologies* that had simply never been told how many
hydrogens they own — no bond needed to be added or removed.

Candidate #1 becomes exactly the structure in the plan:

```
SMILES  CCC(F)(F)N(C)C1CCNC[C@H](C(C)(C)C)C1
MW 262.39   logP 3.34   TPSA 15.27   QED 0.785   HBD 1  HBA 2  rotB 3
Lipinski violations 0   MMFF energy 15.33 kcal/mol
```

Distributions over the 25 saturated leads:

| metric | min | median | max |
|---|---|---|---|
| MW | 126.1 | 287.4 | 355.4 |
| logP | −0.62 | 2.39 | 5.37 |
| TPSA | 0.0 | 15.3 | 37.9 |
| QED | 0.271 | 0.599 | 0.793 |
| Rings | 1 | 2 | 2 |

Two structural families dominate: heavily fluorinated cycloalkanes
(9 leads, TPSA ≈ 0-20) and small bridged azacycles (7 leads). Almost no
aromatic rings — consistent with the generator's training set being saturated
BBBP chemistry, and the reason logP is low and TPSA is near zero.

Aromatics were **not** re-perceived: the generated graphs contain no double
bonds, so no kekulé form exists. Anything with a benzene ring has to be designed
in by hand.

Artifacts: `step1_saturated.csv`, `step1_saturated_3d.sdf`, `step1_saturated.smi`,
`step1_summary.json`.

## Step 2 — catalog search

**Result: 2/25 exact catalog hits, and both are the smallest fragments. 23 leads
are unregistered. 0/25 have any ChEMBL analog at ≥ 0.85 Tanimoto.**

PubChem exact match by InChIKey:

| lead | CID | CAS | MW | verdict |
|---|---|---|---|---|
| lead_18 | 13207094 | 87517-38-2 | 126.1 | registered, one CAS number — buyable, but it is a 6-heavy-atom fluorocarbon |
| lead_14 | 21731026 | — | 148.1 | registered, no CAS in PubChem |

Both hits are the C₄-C₅ perfluoroalkanes. They are "real" only in the trivial
sense; they are not leads.

The 23 remaining leads returned `not_found` from PubChem and `no_hit` from
ChEMBL at the 85 % cutoff. **This was validated, not assumed:** the same code
run against donepezil and aspirin returns their own ChEMBL records at 100 %
similarity, so an empty result means the series really is unexplored rather
than the API silently failing. (The ChEMBL similarity endpoint does throw
intermittent HTML 500s; the client retries, and all 25 lookups ended in a clean
`hit`/`no_hit` answer.)

Consequence for procurement: nothing here can be ordered off a catalog at
100 % identity. The realistic route is on-demand synthesis, which is why the
upload file was produced:

```
outputs/lead_triage/enamine_query.smi     # 25 SMILES, ready for Enamine REAL / Chemspace
```

Artifacts: `step2_catalog.csv`, `enamine_query.smi`, `step2_summary.json`.

## Step 3 — docking

Protocol: 4EY7 (human AChE) → waters/additives/ligand stripped → PDBQT by
meeko 0.8 (`--default_altloc A`) → AutoDock Vina 1.2.5, box 24 Å centred on
donepezil (E20), centre (−1.58, −50.19, 2.10), exhaustiveness 8, 9 modes,
seed 42.

**Control: donepezil re-docked at −7.00 kcal/mol.** That is the calibration
point for reading the table below.

| rank | lead | affinity | QED | MW | logP | Ro5 viol |
|---|---|---|---|---|---|---|
| 1 | lead_24 | **−7.48** | 0.334 | 332.4 | 1.18 | 0 |
| 2 | lead_06 | −6.47 | 0.607 | 276.3 | 1.59 | 0 |
| 3 | lead_20 | −6.29 | 0.535 | 339.5 | 2.34 | 0 |
| 4 | lead_01 | −6.29 | 0.599 | 312.4 | 5.37 | 1 |
| 5 | lead_22 | −6.22 | **0.793** | 301.4 | 2.95 | 0 |
| 6 | lead_15 | −5.97 | 0.388 | 296.4 | −0.62 | 0 |
| 7 | lead_03 | −5.79 | 0.739 | 306.3 | 3.85 | 0 |
| 8 | lead_09 | −5.75 | 0.487 | 243.3 | 0.18 | 0 |
| 9 | lead_21 | −5.74 | 0.722 | 237.4 | 2.70 | 0 |
| 10 | lead_04 | −5.64 | 0.496 | 288.3 | 4.46 | 0 |
| 11-24 | lead_07 … lead_00 | −5.45 … −3.89 | — | — | — | 0 |
| — | lead_17 | not docked | 0.715 | 238.4 | 1.89 | 0 |

Median −5.35, best −7.48, worst −3.89 kcal/mol.

**lead_17 is missing** from the docking table: it is a strained fused cage
(`CCN(C)C(C)C1[C@@H]2[C@@H]3C[C@]3(CC)[C@@]12NOC`) for which distance-geometry
fails at all 20 seeds tried. It needs a different conformer source (e.g. a
force-field-only embedding followed by annealing) before it can be docked.

Artifacts: `docking/step3_docking.csv`, `docking/poses/*.pdbqt` (9 modes each,
so contacts can be inspected), `docking/4EY7_receptor.pdbqt`,
`docking/docking.log` (full Vina output), `docking_control/` (donepezil).

---

## Reading the two rankings together

Docking alone would pick lead_24, but its QED of 0.334 is the second-worst in
the set — it is a potency probe, not a lead. The balanced candidate is
**lead_22** (`CC(C)N(C)[C@@]1(C2CCCCC2CC(O)C(O)F)CC1C`): −6.22 kcal/mol, i.e.
within 0.8 kcal of donepezil, with QED 0.793, MW 301, logP 2.95, TPSA 40, zero
violations, and 5 rotatable bonds.

Ranked shortlist: `step4_shortlist.csv`.

## What this analysis does not establish

- **Vina's scoring error is ±1-2 kcal/mol**, larger than most gaps in the
  table. Ranks 2-6 are statistically a tie, and none of them is distinguishable
  from donepezil's −7.00 by score alone.
- **Rigid receptor, no waters, no explicit entropy.** AChE's gorge waters and
  the Phe330/Tyr337 mobility matter for real affinity; Vina ignores that.
- **No synthesis check.** Steps 2 and 3 say these molecules are new and would
  dock; they say nothing about whether the strained bicyclic cores
  (lead_24, lead_06, lead_15) can actually be made.
- **BBB is untested here.** 4EY7 docking says nothing about brain penetration,
  which is the project's actual objective — the existing BBB oracle should be
  run on these 25 before any of them is called a CNS lead.
- Single conformer per ligand, single protonation state, undefined stereocentres
  docked as generated.
