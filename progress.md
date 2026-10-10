# Project Progress & Rationale

**Project:** Novel Molecular Generation using Graph GANs/Diffusion for BBB Permeability
**Current Phase:** Phases 1–4 implemented; recommendation.md Tiers 0–3 implemented (V3 baseline: 62.2% validity, 99.2% uniq/novel)
**Last Updated:** 24 Sep 2026 (CPU-only machine, `node1`)

---

## 1. Where We Started (The 75% Completed Code)

The codebase arrived with the heavy machinery built and tested:
1. **3D Diffusion Mathematics** — centered DDPM over coordinates + categorical TypeDDPM over atom types (`src/models/diffusion.py`), ancestral DDPM + DDIM samplers (`src/sampling.py`).
2. **Equivariant GNN denoiser** — timestep-conditioned EGNN with joint coordinate/type heads and a bond head (`src/models/egnn.py`).
3. **Federated Learning System** — Flower-compatible clients, formula-partitioned IID/non-IID QM9 splits, weighted FedAvg/FedProx, personal heads, λ₂/λ₃ multi-objective losses (`fed_train.py`, `src/fed/`, `src/objectives.py`).
4. **MOSES-style evaluation** — Validity, Uniqueness, Novelty, IntDiv, QED, LogP, SNN (`src/utils/evaluation.py`, `generate_and_eval.py`).

The missing 25% — BBB datasets, targeted (conditioned) generation, and the paper experiments — is mapped in `implementation.md` (Phases 1–8). The end goal is unchanged: a **"Guided AI Designer"** you can instruct (*"generate 100 BBB-permeable molecules"*), trained with privacy-preserving federated learning. Nobody has combined 3D federated diffusion with BBB permeability before — that is the publication.

## 2. Environment & QM9 Baseline (Done)

- Built a local `.venv` (Python 3.12): CPU PyTorch 2.14, PyG, e3nn, RDKit, plus Phase-1 additions `deepchem`/`ogb`/`seaborn`, `pytest`.
- Downloaded QM9 into `data/` (130,831 processed molecules). `test_setup.py` reports SUCCESS.
- **Path verification:** all training paths (`data.root`, `checkpoint.dir`, `output_dir`, `partition_cache`) confirmed correct and relative to repo root.
- **Bugs fixed in `train.py`:** (a) `random_split` lengths summed to `n + n_val` (always crashed) → `[n_train, n_val, n_test]`; (b) `torch DataLoader` can't batch PyG graphs → `torch_geometric.loader.DataLoader`; (c) added `max_molecules` subsampling (mirrors `fed_train.py`).

## 3. Smoke Runs & Resumable Training (Done)

- Central smoke (500 mols, 2 epochs) and federated smoke (IID + non-IID, 2 rounds) all converge cleanly.
- Because full runs exceed 1-hour sessions, training is now **resumable in fixed chunks**: `train.py --resume --run_epochs N` (`checkpoints/last.pt` every epoch: model+optimizer+scheduler+best/patience/RNG, atomic writes) and `fed_train.py --resume --run_rounds N` (`last_global.pt` + appended `history.json`). Chunking verified by kill-and-resume tests. Use `setsid` + instant return for background launches (a tool-timeout once killed a run; resume lost nothing).
- Helper: `scripts/check_status.py` prints central + federated progress in one command.

## 4. Full QM9 Training, Round 1 (Done — Superseded)

- **Central:** 100 epochs (~1.5h). Early stopping fired at epoch 36 (patience 10 after a 12-epoch plateau), so patience was raised to 1000 to force the full run. Best val **1.3161** (epoch 49).
- **Federated:** IID 50 rounds (server 1.37→1.35, val 1.46) and non-IID 50 rounds with FedProx μ=0.1 (server 1.44→1.37, val 1.48). Mechanics verified; federated learns glacially vs central (fresh Adam per round, constant LR) — an open tuning item, not a bug.
- **Eval table v1 (RETRACTED — see §7):** central 85.7%, IID 86.0%, non-IID 36.1% validity. These numbers were measured with the wrong atom mapping and are invalid; the honest comparison is being redone.

## 5. Eval Persistence & Status Tooling (Done)

- `generate_and_eval.py` accepts federated checkpoints (`global_state` in addition to `model_state_dict`) and `--output_dir` saving `metrics.json` (numbers + LaTeX row + run config), `smiles.txt`, and `molecules.sdf` with 3D conformers (`outputs/eval_*/`).

## 6. Phase 1 — BBB Dataset Pipeline (Done, Committed `2bbf472`)

- New `src/dataset_bbb.py`: ETKDGv3+MMFF SMILES→3D conversion to PyG `Data(pos, z, y, qed, logp, tpsa, mw, smiles)`, Murcko-scaffold 80/10/10 splits, kill-safe incremental `processed_3d.pt` cache, train-set property stats + z-scoring. `tests/test_bbb_pipeline.py` (16 tests).
- **Results:** BBBP 1628/205/205 (76.4% BBB+, 0.6% 3D failures); B3DB 6144/770/779 (63.5% BBB+, 1.5% failures); geometry sane (mean NN ~1.1Å); scaffolds disjoint across splits.
- **Bugs fixed:** this RDKit's `MurckoScaffoldSmiles` takes `mol=` as keyword (positional call collapsed every split to all-train); B3DB label header `BBB+/BBB-` added to detection; cache path de-nested.
- **Deviation:** `deepchem` is installed per spec but never imported at runtime (it drags in tensorflow); loading uses direct CSV download + own scaffold split.

## 7. Phase 2 — Conditioning (Done, Committed `5b17b0e`) + Two Major Discoveries

- **Built:** conditioned EGNN (`cond_dim`, class + [QED, LogP, tPSA, MW] → timestep embedding, `cond=None` unconditional path), classifier-free guidance in `sampling.py` (`guidance_scale`, verified bit-identical at 0.0), trainer label dropout + cond extraction (`src/fed/trainer.py`). `tests/test_phase2_conditioning.py` (11 tests). All 69 tests green.
- **Discovery A — the coordinate head never learned.** `forward` aliased `initial_pos = pos` (no copy) while `encode` rebinds locally, so `noise_pred` was exactly **0** with no gradient path (proven: all-zero output + autograd error). Every prior run trained only atom types (`pos_loss≈1.0` was the constant baseline). Fixed via `clone()` + `encode` returning `(h, updated_pos)`; zero-init identity behavior preserved.
- **Discovery B — model indices ARE atomic numbers.** PyG QM9 `z` = raw Z {1:H, 6:C, 7:N, 8:O, 9:F} (QM9 contains only these; classes 0,2–5 are dead). The eval-time `QM9_ATOMIC_NUMBERS` remap shifted every element (H→C, C→S…) — this fabricated the §4 validity table (old model: 0.0 bonds/mol fragments that sanitize trivially) and zeroed the retrained model. Fixed to identity mapping; trainer `TYPE_TO_Z` corrected; oracle keeps its frozen self-consistent dense map (still 0.92 AUROC, untouched).

## 8. Phase 3 — BBB Oracle (Done, Committed `ac8a0ec`)

- New `src/models/bbb_classifier.py` (3-layer GCN + mean-pool + MLP) and `scripts/train_bbb_classifier.py` (BCE, pos-weighted, early stop on val AUROC → `models/bbb_oracle.pt`, gitignored). `tests/test_bbb_oracle.py` (8 tests).
- **Result:** best val AUROC **0.9116**, TEST **0.9218** (gate ≥0.88 ✓, literature ~0.92 ✓); caffeine (known BBB+) scores > 0.5.
- **Bug fixed:** first inference used heavy-atom bond topology while training saw H-inclusive distance graphs (caffeine scored 0.13). Inference now mirrors training exactly (AddHs → ETKDG → distance edges) plus `data_to_input()` for scoring generated molecules with zero shift. No retrain needed — the model was fine.

## 9. Geometry Retraining — Current Status (In Progress)

Retraining central QM9 with the fixed coordinate head (old checkpoints archived to `checkpoints/pre_coordfix/`, `checkpoints/coordfix_v1/`):
1. **v1 (no λ₂):** best val 0.666 (vs 1.316 dead-head), type head 99.8% accurate — but generation 0% valid: **26.1 bonds/mol vs ~19 true** (over-connected clumps; H with valence 4).
2. **λ₂=0.01 attempt:** no effect (25.0 bonds/mol) — diagnosed the surrogate as blind: kNN-capped counting scored real 0.99 vs clump 1.06. Fixed `soft_valence_penalty` to all-pairs counting (real **0.046** vs clump **0.081**), recalibrated λ₂ **0.01 → 1.0**, added live `quick_valid` probes (DDIM-20) every 10 epochs in `train.py` + a clump-vs-chain separation test.
3. **Sampler hardening** (NaN crash seen with confident models): sanitized `posterior_probs`, logit clamp ±15, bulletproof multinomial fallback.
4. **λ₂=1.0 run (v2) — DONE, FAILED:** full 100 epochs finished (best val 0.9034 @ epoch 58; test 0.9213) but **quick_valid = 0.0% at every probe** (epochs 10–100) — the all-pairs surrogate at λ₂=1.0 did not de-clump generation. Loss curves flat after ~epoch 30. Checkpoints archived to `checkpoints/lambda2_v2/`.
5. **Escalation run (§11.3 capacity/schedule valve) — RUNNING:** `configs/central_geo_esc.yaml` — node_dim/edge_dim 64→128, num_layers 4→6 (0.78M params, ~4× v2), epochs 100→150; λ₂=1.0, kNN, batch, probe cadence and seed unchanged for comparability. pid in `logs/central_geo_esc.pid`, log `logs/central_geo_esc.log`; resumable via `--resume --run_epochs N`. ~2 min/epoch → ~5 h total.
6. **Escalation stability hardening (`train.py`, pending commit):** the 128/6 model NaN-diverged at epoch 1 twice (lr 1e-3 and 3e-4), while a 6-step offline probe with identical code paths showed both 64/4 and 128/6 training cleanly at lr 3e-4 — so the culprit is a rare pathological batch inside the full ~3,270-step epoch, not capacity or lr per se. Fixes: (a) NaN/Inf grad guard — batches with non-finite grads are skipped instead of poisoning weights (2 such batches caught in escalated epoch 1); (b) `training.grad_clip_norm` (default 0.0 = off, legacy recipes unchanged; esc run uses 5.0) clips global grad norm so finite-but-huge batches take bounded steps. lr cut 1e-3→3e-4 kept for the bigger model. Evidence it works: escalated val improved 1.048→1.013 over epochs 1–2 (v2-trajectory), while epoch-1 *train averages* still show the few finite-but-huge batches (cosmetic; val unaffected).
7. **Discovery C — the eval bond builder failed GROUND TRUTH (fixed):** `coords_and_types_to_mol` could not rebuild even real QM9 molecules: **0.7% valid** at its 2.5 Å cutoff, **20.7%** at a proposed 1.8 Å (methyl H···H ≈ 1.78 Å and 1-3 ring contacts ≈ 2.42 Å were spuriously bonded; worse, the old threshold ladder assigned *triple* bonds to every C–H at 1.09 Å — the "C, 9" valence errors in every log). Replaced with element-aware covalent radii (bond iff dist ≤ r_i + r_j + 0.45 Å; single-bond topology since distance alone can't order bonds). Ground truth now: **99.0% valid, 10.83 bonds/mol vs 10.96 true**. Consequences: (a) v2's honest validity is **~9.4%** (32 samples, DDIM-20), not 0%; (b) v2's failure mode is **dispersed/under-bonded geometry (1.3 bonds/mol vs ~16 expected for 18 atoms) — coordinate precision, NOT clumping**: val_pos RMSE ≈ 0.54 Å vs ≤0.45 Å bonding windows. The earlier "26.1 bonds/mol clumps" diagnosis and both 0.0% eval tables were artifacts of the broken lens. The λ₂-clump rationale is unproven; λ₂ stays in the escalation for comparability, a λ₂=0 arm is optional. Escalated run restarted with `--resume` (epoch 22, best val 0.9078) so its probes use the fixed eval — epochs 30+ probes are the first honest ones. NOTE: `tests/test_phase1_5.py::test_loss_beats_baseline` is a pre-existing order-dependent flake (fails identically with the fix stashed; passes in isolation).
8. **Escalated run FINISHED (150/150):** best epoch **146** (val 0.871), final probe **quick_valid=25.0%** — first honest nonzero validity in project history. Full eval of best (`outputs/eval_geo_esc/`, DDIM-50): Validity **28.3**, Uniqueness/Novelty 100, IntDiv 0.90, QED 0.48, LogP ≈ 0, SNN 0.15, 283 molecules saved.
9. **Sampler operating-point sweep (no retraining):** more DDIM steps + mild stochasticity lift validity monotonically — DDIM-50/eta0 28.3/7.7 → DDIM-100 37.8/10.2 → DDIM-200 44.4/13.2 → DDIM-100/eta0.5 40.5/12.1 → **DDIM-200/eta0.5 47.1/14.3 (adopted as Table I protocol**, recorded in `central_geo_esc.yaml`). Geometry recipe unchanged; the gain is pure sampling.
10. **Connectivity metrics (anti-fraud honesty):** plain Validity passes disconnected single atoms, so pre_coordfix scored 99.4% on fragment soup. Added `ConnectedValidity` (% valid AND single-fragment), `BondsPerMol`, `ConnectedFrac` to `evaluation.py` (+ `tests/test_eval_connectivity.py`) and persisted in every `metrics.json`. True Table I (fixed ruler, 1000 samples DDIM-50): pre_coordfix 99.4/**0.0** (0.1 bonds), coordfix_v1 20.8/**1.3** (15.3), lambda2_v1 24.3/**1.6** (15.1), geo_esc 29.9/**7.7** (16.8) → truth 94.7% (~19). Monotonic real progress on the honest number.

## 10. Phase 4 — Extended Metrics (Done)

- New BBB-specific metrics in `src/utils/evaluation.py`, wired through `evaluate(..., bbb_classifier=...)` (backward compatible; `BBB%` is -1 without an oracle): **BBB%** (oracle P>0.5 rate), **ScaffDiv** (unique Murcko scaffolds / valid), **ScaffCov** (train scaffolds reproduced), **Lipinski%**, **Veber%**, **CNS_MPO** (Wager ramps for MW/LogP/HBD/TPSA; pKa+CLogD need proprietary predictors and are omitted, so 0–4 scale, documented). `tests/test_extended_metrics.py` (7 tests). `generate_and_eval.py --bbb_oracle` loads the trained oracle into eval.
- **Baseline on unconditioned geo_esc molecules (471 saved SMILES, no generation cost): BBB% 24.8**, ScaffDiv 0.53, Lipinski 100%, CNS_MPO 3.29/4 — the number Phase 6 guidance must beat.

## 11. Federated Retrains With Winning Recipe (Running)

- Both fed globals were still coord-dead, so `configs/fed_iid.yaml` + `fed_niid.yaml` moved to the geo_esc recipe (128-dim/6-layer, lr 3e-4, λ₂=1.0 all-pairs) and both full retrains launched backgrounded (`logs/fed_iid_retrain.log`, `logs/fed_niid_retrain.log`; old outputs archived to `outputs/fed_*_v1/`). `LocalTrainer` gained the same grad-clip + NaN-skip guard as central (its fresh-per-round Adam faces the same 128-dim instability).
- Smoke-verified (`fed_iid_smoke` 2 rounds clean) before launch; suite green modulo the known flake.
- geo_esc completed 150/150 epochs (best val 0.8710 @ epoch 146, test_loss 0.8818; probe band 6–31%, full-eval 47.1/14.3 under the adopted DDIM-200/eta0.5 protocol). Fed retrains in flight at the time of the validity-80 commit (their `outputs/` churn deliberately excluded from it).

## 12. Validity-80 Plan Implemented (Done)

All 6 strategies from validity_80_plan.md, wired through every entry point (`train.py`, `generate_and_eval.py`, `fed_train.py`, `src/fed/trainer.py`):
- **S1 relax** — `relax_molecule` (MMFF94, UFF fallback) in `src/utils/evaluation.py` + `--relax` flag; **S6 quadratic DDIM** — grid extracted to testable `ddim_timestep_grid` (dense low-noise spacing). Both already part of the adopted Table I operating point (DDIM-200/eta0.5/quad).
- **S2 cosine schedule** — `schedule=cosine` (Nichol & Dhariwal s=0.008) on `CenteredDDPM`/`TypeDDPM`; ᾱ computed straight from the f64 reference formula, not a betas round-trip. Resume guards reject schedule mismatches (checkpoint vs config).
- **S4 attention gates** — per-edge sigmoid `attn_mlp` on `EGNNLayer`/`EquivariantGenerator` via `use_attention` (default off; state-dict superset keeps legacy checkpoints loadable).
- **S3+S5 V3 config** — `configs/central_v3_full.yaml`: 256-dim/8-layer, kNN 8, λ₂=0, patience 40, lr 2e-4, batch 16, own checkpoint dir `checkpoints/central_v3` (geo_esc owns `checkpoints/`).
- **Bug fix en route**: `train.py`'s test-eval call of `sample_noisy_types` was missing `batch=batch_data.batch` (per-atom timesteps silently mismatched vs the conditioned t); all 3 call sites now consistent with the fed trainer and sampler.
- Tests: `tests/test_validity80.py` (17) + relax cases in `test_eval_connectivity.py` (7) — 24 new, all green; full suite 108/109 modulo the known pre-existing order-dependent flake.
- **V3 run launched** (pid in `logs/central_v3.pid`, log `logs/central_v3.log`): 4.09M params (5× geo_esc), sanity-verified fwd/bwd finite before launch. Measured ~13 min/epoch on CPU → **~43 h for 200 epochs**, resumable (`--resume --run_epochs N`); first `quick_valid` probe lands ~2.2 h in (epoch 10), value accrues early if stopped at any checkpoint.

## 13. Eval Dashboard (Done)

`scripts/build_dashboard.py` generates a self-contained `outputs/dashboard.html` from all `outputs/eval_*/` dirs — no server, no new deps, opens in any browser:
- **Metric cards** per run + a sortable comparison table; runs predating the covalent-radii eval fix are auto-badged "⚠ inflated lens" and collapsed so stale numbers can't mislead (honest runs sort by Validity).
- **2D gallery**: RDKit-rendered SVG of every valid molecule (capped at 48/run via `--molcap`) with formula, MW, QED, LogP, rings, fragment count.
- **2D-only by decision**: interactive 3D (3Dmol.js + a WebGL-free canvas fallback renderer) was built and verified but removed — WebGL proved unavailable on the target machine. The 3D-capable versions live in git history (`f088ba0`..3d-fallback) if needed later.
- Verified: 14 runs embedded, 385 molecule cards, zero JS errors in headless Chrome, no 3D remnants. Rebuild after any eval: `.venv/bin/python scripts/build_dashboard.py`.

## 13.5 Recommendation-Report Implementation (Done — Tiers 1–3 + remaining; `docs/recommendation.md`)

All of `docs/recommendation.md`'s actionable strategies implemented as opt-ins (defaults unchanged; legacy recipes and checkpoints valid throughout). Full per-item status lives in the doc's new "Implementation Status" appendix.

- **Tier 1 (`dad20cc`)** — new `src/training_utils.py`: EMA (decay ramp; server-side variant in `src/fed/server.py` via a `_StateView` adapter; best.pt stores EMA weights, last.pt raw), min-SNR-γ timestep weighting (mean-1 normalized, per-molecule broadcast; γ=1 ⇒ P2), SO(3) rotation augmentation, warmup+cosine LR; `schedule="sigmoid"` in `CenteredDDPM`/`TypeDDPM`; everything gated by new `training.*` config keys, all default-off. `tests/test_tier1_training.py` (22 tests). Test-design lessons: min-SNR **caps** low-noise dominance (w = min(SNR,γ)/SNR); rotation must preserve per-molecule pairwise distances.
- **Tier 2 (`0320004`)** — `EquivariantGenerator` gains `self_condition` (x0_proj, 50% train dropout), `coord_refine_layers` (zero-init refine EGNN ⇒ exact identity at init), `knn_schedule` (per-layer multi-scale graphs); bond-head supervision vs QM9 ground-truth `edge_index` (per-PAIR low-noise gate, `node_h.detach()` aux head, `training.bond_loss_weight`); resume guard extended to the Tier 2 flags; wired through train.py, fed trainer/server, sampler (SC auto-detect) and `generate_and_eval.py`. `tests/test_tier2_architecture.py` (17 tests). Verified: 3-epoch smoke train with ALL features on + resume guard + sampling. Coordinate-head lessons: everything coord-side is zero-init, so architecture tests must compare **type logits** and refine-head training tests need MSE vs a **nonzero target**.
- **Tier 3.2 flow matching (`d49df80`)** — new `src/models/flow.py`: linear OT path `x_u = (1−u)·x0 + u·ε` with target velocity `v = ε − x0` (exactly the doc's `v = α'_t·x0 + σ'_t·ε` for α=1−u, σ=u), per-molecule re-centered noise, exact noise-free x0 recovery `x0 = x_u − u·v̂` at every u. Opt-in `diffusion.objective: flow` (DDPM `eps` stays default/fallback — the doc's risk-table mitigation); same architecture, no new parameters, legacy checkpoints load either way; u tied to the same t that drives the time embedding/type chain; Euler ODE sampler in `src/sampling.py` (auto-detected from `model.objective`); `x0_valence_penalty` made objective-agnostic; resume guard rejects eps↔flow switches; composes with all Tier 1/2 features. `tests/test_tier3_flow.py` (26 tests). Smoke train + flow-checkpoint generation verified.
- **Tier 0.4 tolerance matrix** — `TOLERANCE_MATRIX`/`get_bond_tolerance()` in `src/utils/evaluation.py` replace the global +0.45 Å scalar in distance-based bonding: per-element-pair tolerances (H–H tight 0.25 vs methyl H···H false positives; C–O loose 0.50 vs carbonyl false negatives), unlisted pairs fall back to the scalar. Values encode the doc's qualitative guidance — recalibrate per pair on QM9 ground truth before trusting the +2–5% estimate.
- **Connectivity post-processing** — `coords_and_types_to_mol(min_fragment_atoms=k)` prunes provisional distance-graph fragments < k atoms (union-find over the covalent-radii graph) before bond assignment; `--min_fragment_atoms` flag (1 = legacy).
- **Type temperature** — `type_temperature` on `sample_molecules`' type posterior (flattens predicted type distributions → more scaffold diversity); `--type_temperature` flag.
- **QED rejection filter** — `apply_qed_filter` + `--min_qed` (raw table reported first, then the filtered table).
- **2.5 D3PM + 3.1/3.3/3.4** — scope notes in the doc's appendix: D3PM deferred (the EDM-style continuous noising + exact categorical posterior already provides a working discrete-type sampler; the chemistry-prior transition matrix requires retraining), Tier 3.1/3.3/3.4 are multi-day paradigm shifts deliberately not implemented.
- Tests: full suite 173 passed + the known pre-existing order-dependent flake (`test_loss_beats_baseline`).

**Next experiment:** retrain V3 with the winning opt-ins (EMA + min-SNR-γ + warmup-cosine + Tier 2 features + bond head) and re-run the Tier 0 eval sweeps (quadratic DDIM, η sweep, 500–1000 steps, tolerance matrix + min_fragment_atoms eval); or train fresh with `diffusion.objective: flow` for a direct paradigm comparison.

## 13.6 V4 Retrain — Both Arms (In Progress)

The "next experiment" above, launched (`a6fef55`). Two configs identical except the prediction objective (user decision: both, sequential; 400 epochs, patience 60):
- `configs/central_v4_eps.yaml` — V3 recipe (256/8, attention, cosine, kNN 8, lr 2e-4, grad-clip 5) + EMA 0.9999, min-SNR γ=5, warmup 10, rotation 0.5, self-conditioning, bond head 0.1, kNN schedule [4,4,8,8,12,12,16,16], refine 2. Checkpoints `checkpoints/central_v4_eps/`, log `logs/central_v4_eps.log`.
- `configs/central_v4_flow.yaml` — same everything, `objective: flow`, min-SNR 0 (inapplicable). Runs chained after eps via one `setsid bash -c` wrapper (pid `logs/central_v4.chain.pid`).
- **Smoke-gate before launch:** 2-epoch full-scale runs of both exact configs (60 molecules) + 20-step sampling of both checkpoints. This caught a real sampler landmine: a barely-trained eps model returns huge noise predictions when the high-noise self-conditioning input (x̂₀ structurally clamped at ±10) is OOD, and the trajectory then NaN-poisons two steps later → silent 0% validity. Fixed by a predict-zero substitution on non-finite predictions in both sampler paths (`2827c3a`, same philosophy as the type-posterior guard). Also: `train.py` must run with `python -u` when backgrounded — the first launch died silently with an unflushed log.
- **eps arm DONE:** early stop at epoch 177/400 (best val 0.8252 ≈ epoch 117; test 0.8506). NOTE: val is NOT comparable to V3's 0.6252 — min-SNR reweights the uniform-t val MSE by design, and self-conditioning trains with richer inputs than val passes. `quick_valid` probes band 0–56% (16-sample DDIM-20, noisy; peak 56.2% at epoch 120, adjacent to best-val 117). Full Table I eval in `outputs/eval_v4_eps/` (+ `outputs/eval_v4_eps_prune2/` with `--min_fragment_atoms 2`).
- **eps arm RESULTS (1000 samples, DDIM-200/η0.5/quadratic/relax — identical V3 protocol):** Validity **55.0% raw → 56.0% with fragment pruning** (vs V3 62.2% — a 6–7 pt REGRESSION on the headline metric), but every secondary metric jumped: IntDiv_p 0.789→**0.891** (target >0.85 MET), QED 0.438→**0.494** (≈0.5 target), ScaffDiv 0.031→**0.409** (13×, target 0.1 blown past), SNN 0.143→0.191, CNS_MPO 3.13→3.27, Uniq/Novel 100%. Reading: the Tier 1+2 stack (rotation augment + diversity-side pressure + multi-scale kNN + SC) bought a much better diversity/quality profile at the cost of geometric precision — the validity bottleneck the doc warns is hardest. The +1 pt from pruning shows ~all remaining invalids are genuine geometry/chemistry misses, not fragment soup. Candidate causes to test: min-SNR changing the effective objective, SC train/eval input mismatch, checkpoint selection under the reweighted val. **η=0.5 suboptimal — confirmed, see below.** Flow-arm A/B result in the flow bullet below.
- **eps arm η sweep (doc §0.3, 200 samples, DDIM-200/quadratic, seed 42, `outputs/eval_v4_eps_eta{00,03,07,10}/`):** validity rises monotonically with η — 0.0→44%, 0.3→55%, 0.7→57.5%, 1.0→60% (base η=0.5→55% at n=1000). The deterministic path (η=0) craters; stochastic noise rescues geometric validity. IntDiv_p stays ≥0.875 across the sweep (still >0.85 target); Uniq/Novel 100% everywhere. So a large share of the "6–7 pt regression" vs V3's 62.2% is a sampler-noise artifact of η=0.5 on the new model, not training damage — at η=1.0 the gap is within noise (n=200 ±3.5 pt).
- **eps arm η sweep FULL n=1000 confirmations (`outputs/eval_v4_eps_eta{07,10}_full/`):** η=0.7→**58.9%**, η=1.0→**62.2% — exact parity with V3** while every secondary metric dominates (IntDiv_p 0.875 vs 0.789, QED 0.494 vs 0.438, SNN 0.207 vs 0.143, ScaffDiv 0.412 vs 0.031, CNS_MPO 3.26 vs 3.13; Uniq/Novel 99.8/100%). `--min_fragment_atoms 2` at η=1.0 gives 61.7% (`outputs/eval_v4_eps_eta10_full_prune2/`) — pruning neutral here. **Conclusion: V4-eps at η=1.0 matches V3 validity with a strictly better diversity/quality profile; adopt DDIM-200/η=1.0/quadratic as the V4 eval protocol.** The remaining open questions (min-SNR vs SC as the residual geometry cost, checkpoint selection) are moot for the headline.
- **flow arm DONE — catastrophic 0.4% validity:** early stop at epoch 95 (patience 60; best val 3.5123 at epoch 35, val_pos never below ~3.17). Epoch-1 turbulence: 9 non-finite-grad batch skips (`x0_proj`/refine layers) before train_pos collapsed 7.9M→2.6 — grad-clip + skip guard recovered cleanly. quick_valid probes (DDIM-20 auto-Ode path) never exceeded 18.8% (n=16, mostly 0–6.2%). Full Table I eval (`outputs/eval_v4_flow/metrics.json`, watcher auto-launched on exit): **Validity 0.4% (4/1000)** — no non-finite/NaN hits in `logs/eval_v4_flow.log`, so not the known landmine; consistent with the training probes, so not an eval-harness artifact. Secondary metrics on the 4 valid mols (IntDiv 0.862, QED 0.492) are noise-level n. **A/B verdict: eps objective wins decisively (62.2% vs 0.4%) — but flow's `train_pos` plateau at ~2.11 looks close to the trivial zero-velocity predictor (≈1+Var(x0)), so undertrained-velocity vs ODE-path-bug is an OPEN triage question before declaring the paradigm result final.** Checkpoints `checkpoints/central_v4_flow/{best,last}.pt retained for triage (last.pt epoch 95 vs best.pt epoch 35, step-count probes, u↔t convention audit).
- **Eval protocol reminder (flow arm):** `--ddim_steps 200` is the Euler grid size for the ODE path; `--eta` and `--step_schedule` are DDIM-only concepts and are ignored under flow.

## 13.7 Flow Triage — the "Trivial Predictor" Hypothesis Is Refuted (Done, `1f4738f`)

§13.6 left the flow arm's 0.4% validity with an open question: `train_pos` plateaued at ~2.11, which looked close to the trivial zero-velocity predictor. `scripts/flow_triage.py` answers it with the **exact analytic baseline** instead of the wrong constant:

- For flow the predict-zero loss is `E‖eps_c − x0‖² = mean(1 − 1/n_g + ‖x0‖²/3)` — **3.9024 train / 3.9648 val**, NOT 2.11 and NOT 1.0. `train.py` had been logging the eps constant `1.0` for *both* objectives, which is what made 2.11 look trivial. Formula verified against Monte Carlo (analytic 2.0912 vs mc 2.0931, n=18×64).
- Fixed in `a4df4e1`: per-objective `total_baseline` accumulated per batch (flow uses the exact formula, eps keeps 1.0). Verified with 1-epoch smoke runs (eps `baseline=1.0000`, flow `baseline=3.8829`).
- **Verdict: the flow model is NOT a trivial predictor.** Per-u skill (`sc=none` / `sc=draft`) for `best.pt`: u=0.05 **+0.858/+0.901**, u=0.25 +0.71/+0.86, u=0.5 +0.17/+0.46, **u=0.75 +0.02 (cos 0.12) — collapse**, u=0.95 +0.18/+0.44. Real velocity learned at low/mid u with a high-u weakness; not an ODE-path or u↔t convention bug.
- **Checkpoint/step probes (n=200, DDIM-200 unless noted):** `best.pt` (epoch 35) 1.0%@50 steps, 0.5%@200, 0.0%@1000 Euler steps ⇒ step count is not the issue. `epoch_50` 6.5%, `epoch_70` 11.5%, `last.pt` (epoch 95) **18.0%** (ConnV 15.5), `last` n=1000 **20.2%** (ConnV 16.2). Validity climbs monotonically with training ⇒ **val-loss best-checkpoint selection picked a geometrically worse checkpoint** (epoch 35) than `last` (epoch 95).
- Artifact: `outputs/flow_triage.json`.

## 13.8 Connected Validity Rewrites the V3-vs-V4 Verdict (Done, `2b194b7`)

V3's 62.2% "validity" was never checked for connectivity. `scripts/backfill_connectivity.py` recomputes `ConnectedValidity/BondsPerMol/ConnectedFrac` from the saved SDF + recorded `num_total` (invalid mols are never written, so the SDF is exactly the valid list). Validated against natively-keyed evals to 1e-6 (12/12, `failures=0`); skips lossy/corrupt SDFs where the parsed count ≠ `round(Validity/100 × num_total)`.

| run | Validity | ConnectedValidity | BondsPerMol |
|---|---|---|---|
| `eval_v3` | 62.2 | **0.0** | 11.7 |
| `eval_v4_eps_eta10_full` | 62.2 | **54.4** | 17.8 |
| `eval_v4_eps_eta10` | 52.5 | — | — |
| `eval_v4_eps_eta07_full` | 58.9 | 46.0 | — |
| `eval_v4_eps_eta03` | — | 30.5 | — |
| `eval_v4_eps_eta00` | — | 25.5 | — |
| `eval_v4_eps_prune2` | — | 41.9 | — |

**The A/B flips: V4-eps@η=1.0 is the real winner for connected molecules (54.4% vs 0.0%, ConnectedFrac 87.5%).** 24 tables backfilled. Three dirs could not be backfilled (lossy SDF, parsed ≠ recorded): `eval_central` (720 vs 861), `eval_fed_iid` (396 vs 863), `eval_fed_niid` (234 vs 386).
Also fixed a real Phase-7 bug: `Chem.SDMolsupplier` does not exist (correct: `Chem.SDMolSupplier`) — it was in `scripts/generate_paper_figures.py` (Fig. 6 silently fell into its `except Exception`) and in the backfill script.

## 13.9 Table I Rows Are Now Sampler-Matched at η=1.0 (Done, `6f666aa`, `b6dd41d`)

Re-evaluated the headline runs under the adopted protocol (DDIM-200 / η=1.0 / quadratic / relax) so Table I compares like with like:

- `outputs/eval_v3_eta10/` — Validity **72.9** (729/1000), **ConnV 0.0**, BBB% 71.74, IntDiv 0.762, ScaffDiv 0.023.
- `outputs/eval_fed_iid_eta10/` — Validity 18.1, ConnV 0.6, BBB% 31.49, IntDiv 0.877, ScaffDiv 0.414.
- `outputs/eval_fed_niid_eta10/` — Validity 31.0, ConnV 0.0, BBB% 0.0, IntDiv 0.755, ScaffDiv 0.145.

Note η=1.0 moves V3 a lot (62.2→72.9) but leaves V4-eps flat (62.2) — V3's number was sampler-sensitive.

## 13.10 BBB Generation Is Atom Soup — Root-Caused to Undertraining (Diagnosis done; retrain running)

Real BBB central training completed: early stop at epoch 81 (patience 20), val 0.8598, quick_valid 37.5% at epoch 80, test pos_loss 0.4838 / type 0.8216.

**The sample quality is catastrophic despite decent-looking validity:** ConnectedValidity **0.0**, BondsPerMol ≈9, SMILES with 26–111 fragments, generated NN distance p50 **1.78 Å vs 1.09 Å** for real data.

Ruled out, each by a direct A/B:
- **Not guidance** — w=0 and w=2 are identical (V 93.5, ConnV 0 both).
- **Not dataset/model centering** — BBBP train CoM ‖·‖ mean 0.98 (QM9 0.96), coord rms 2.27.
- **Not the eval protocol** — DDIM-20, η=0.5/linear, η=0.0/quadratic all give ConnV 0.
- **Not `cond`** — `cond=None` also soups (45/48 valid, 0 connected); label+zeros 45/0; label+raw-means 43/0.

**Root cause (one-step x0 recovery probe, now `scripts/x0_recovery_probe.py`):** compare the closed-form x0 estimate against the trivial eps=0 predictor. A trained model must win at every t.

| t | QM9 V4-eps (control) | BBBP central |
|---|---|---|
| 20 | 0.0008 vs 0.0017 ✅ | 0.0076 vs 0.0063 ❌ **worse** |
| 50 | 0.0023 vs 0.0078 ✅ | 0.0329 vs 0.0309 ❌ **worse** |
| 100 | 0.0064 vs 0.0277 ✅ | 0.1212 vs 0.1170 ❌ **worse** |
| 200 | 0.0360 vs 0.1086 ✅ | 0.4628 vs 0.5231 ✅ |
| 900 | 1.9122 vs 39.0155 ✅ | 78.7153 vs 3695.6 ✅ |

The QM9 control wins throughout; the BBB model **adds noise instead of removing it at low t** — the signature of undertraining. It saw only ~21 batches × 81 epochs ≈ **1.7k optimizer steps** vs ~260k for QM9 V4. Working hypothesis: more steps fix the geometry.

**Also fixed (real bug, not the soup cause): OOD conditioning.** `extract_cond` (`src/fed/trainer.py:72`) feeds **raw** `qed/logp/tpsa/mw` at training time, but `build_target_cond` passed **zeros** (raw means: qed 0.6206, logp 2.2637, tpsa 71.34, mw 340.38). Now threaded from the training set (`c3e0e37`); verified end-to-end (`qed=0.621 logp=2.264 tpsa=71.343 mw=340.385`). A/B had already shown this is not the soup cause, but guided generation is now on-distribution.

## 13.11 Phase 5–8 Execution: Sweeps, Ablations, Guidance, Real-Data Figures (Done, `357e6d1`)

**61 of 64 runs succeeded** (K=21/24 — the 3 failures are all K=7 iid, i.e. the partitioner bug in §13.12 and being re-run; μ=12/12, λ=15/15, ablation=7/7, guidance=6/6). All eval dirs are under `outputs/sweep_*` / `outputs/ablation_*`. (The sweep logs' own failure counter says "6 failures" because it counts a failed train *and* the resulting skipped eval separately per run.)

**Every arm shows ConnectedValidity ≈ 0** — the single most important result of the sweep campaign:
- **Guidance sweep is flat** — w=0→8 gives V 92.8→92.4, ConnV 0.0 and BBB% **0.0** throughout. The classifier cannot steer a broken geometry.
- **λ sweep only buys parseability** — Validity climbs monotonically 42%→99.4% with valence weight, ConnV stays 0.0.
- **μ sweep** — same; seed variance is huge (V 15–99) with no connected arm.
- **Ablation** — `no_multi_obj` (42.6) and `no_valence_penalty` (49.2) have the *lowest* validity; all arms ConnV 0.

Figs 2/4/7 were regenerated from this real data (`scripts/generate_paper_figures.py`), replacing the synthetic fallbacks — verified as 15 pareto points / 7 ablation arms / 6 guidance points. `TABLE_RUNS` now points at the η=1.0 rows and `find_histories()` is restricted to `fed_iid`/`fed_niid`.

## 13.12 Federated IID Partitioner Bug (Fixed `2137c68`; K-sweep IID arms re-running)

`partition_iid` restarted its round-robin at client 0 for **every** formula class, so client 0 absorbed every singleton class and any client index ≥ the largest class size was **never fed**. On real BBBP (1371 formula classes, 1178 singletons, **max class size 5**):

| | before (buggy) | after (fixed) |
|---|---|---|
| K=2 iid | `[1419, 209]` | `[814, 814]` |
| K=4 iid | `[1374, 193, 45, 16]` | `[407, 407, 407, 407]` |
| K=7 iid | `[1371, 193, 45, 16, 3, 0, 0]` | `[233, 233, 233, 233, 232, 232, 232]` |

The old outputs reproduce the cached partition files bit-for-bit, so the diagnosis is certain. **Consequence: the K-sweep's IID arms were never IID** — K=4 iid was as skewed as K=4 niid — and K=7 crashed with two empty clients (`num_samples=0` in the DataLoader). Now: deal each class evenly across all clients and rotate the remainders; the IID cache key carries an algorithm version (`_v2`) so stale partitions cannot be reused. QM9's IID partition was already balanced (~1.0x) because its formula classes are large, so QM9 federated results are unaffected.

Regression test added (`tests/test_fed_phase23.py`); confirmed it **fails against the old implementation at every K**, so it is a real guard rather than a vacuous one. Suite status: **228 passed, 1 failed** — the failure is the pre-existing order-dependent flake `tests/test_phase1_5.py::TestTrainingConvergence::test_loss_beats_baseline` (passes in isolation, does not touch partitioning).

## 13.13 Task 4 Audit — Table I Rows Are Clean, but `--relax` Is a No-Op (Done)

Task 4 asked for "matched `Validity` + `ConnectedValidity` + relaxed numbers on every Table I row". Audit of the five `TABLE_RUNS` rows in `scripts/generate_paper_figures.py` shows **this is already satisfied** — no re-evaluation needed:

| Table I row | eval dir | n | η | Validity | ConnV | relaxed block |
|---|---|---|---|---|---|---|
| V3 (QM9) η=1.0 | `eval_v3_eta10` | 1000 | 1.0 | 72.9 | 0.0 | present |
| V4-eps (QM9) η=1.0 | `eval_v4_eps_eta10_full` | 1000 | 1.0 | 62.2 | 54.4 | present |
| V4-eps (QM9) η=0.5 | `eval_v4_eps` | 1000 | 0.5 | 55.0 | 37.6 | present |
| Fed IID (QM9) η=1.0 | `eval_fed_iid_eta10` | 1000 | 1.0 | 18.1 | 0.6 | present |
| Fed non-IID (QM9) η=1.0 | `eval_fed_niid_eta10` | 1000 | 1.0 | 31.0 | 0.0 | present |

All five ran DDIM-200, `step_schedule: quadratic`, `relax: true`, and all five carry `metrics_relaxed`. The three dirs §13.8 flagged as un-backfillable (`eval_central`, `eval_fed_iid`, `eval_fed_niid`) are **no longer referenced** — `TABLE_RUNS` points at their η=1.0 replacements. (`BBB% = -1.0` on the QM9 rows is the oracle-disabled sentinel, correct for non-BBBP molecules.)

**Finding: the `--relax` column is not a result.** `metrics_relaxed` is byte-identical to `metrics` in every row (0 of 16 keys differ in 4/5; `eval_fed_niid_eta10` differs by one key, SNN 0.137666→0.137655). Two structural reasons, both verified:

1. **Relaxation cannot rescue an invalid molecule.** `all_mols` holds `None` for every molecule that failed `coords_and_types_to_mol`, and `relax_molecule(None)` returns `None` (`src/utils/evaluation.py:423`). The invalid set is fixed *before* relaxation runs, so `Validity`/`ConnectedValidity` are invariant by construction. Confirmed on disk: raw and relaxed SDFs contain the same valid-only molecules (e.g. `eval_v3_eta10` 729 = 729, `eval_v4_eps_eta10_full` 622 = 622).
2. **The bond graph is already fixed when relaxation runs.** `coords_and_types_to_mol` perceives bonds via the covalent-radii builder from the *generated* coordinates; MMFF/UFF minimization then moves atoms without re-deriving bonds, so `ConnectedValidity`/`BondsPerMol`/`ConnectedFrac` cannot change. SDF-to-SDF SMILES comparison confirms only 3D-derived descriptors move: 0/729 changed in `eval_v3_eta10`, 24/622 in `eval_v4_eps_eta10_full`, 18/181 in `eval_fed_iid_eta10` — none of them the headline metrics.

**Consequence for the paper:** Table I must not present relaxed-vs-raw as a gain; the two columns are the same experiment. The `validity_80_plan.md` S1 hope that relaxation would "snap near-miss coordinates into chemically valid distances" is unreachable through this path — a real coordinate-level cleanup would have to re-perceive bonds after relaxation, and MMFF cannot be built for a structure that has no valid graph yet. This closes off post-hoc repair as a route to connectivity and pushes the soup back onto the objective/training (Tasks 1 and 3b).

## 13.14 λ₂ Valence Penalty Is Mis-Scaled While the Type Head Is Diffuse (Proposal — not applied)

Task 3b's "revisit the valence objective" half is analysis-only, so it runs without competing for CPU. New artifact: `scripts/valence_scale_probe.py`.

`soft_valence_penalty` is **one-sided** (`relu(1.2·bond_count − expected_valence)²`), so it can only push atoms apart. Its demand side uses the model's own predicted types, `Σ_k p_k · VALENCE[type_to_z[k]]`. Raw-Z indexing is correct for both datasets (`TYPE_TO_Z = {i: i}`; QM9's `z` is raw Z too — the `QM9_ATOMIC_NUMBERS` table was removed in `9b62eb3`), but that makes `max_valence = [0,1,0,0,0,0,4,3,2,1]`: classes **2–5 (Z=2,3,4,5) are dead yet still carry softmax mass**, diluting the carbon/nitrogen/oxygen demand toward ~1.1 bonds/atom early in training.

Measured on **real BBBP geometry** (48 train molecules — the output we actually want), λ₂ weight 1.0, versus the BBB `train_pos` MSE of 0.46:

| type head | current penalty | renorm-live variant |
|---|---|---|
| one-hot true (floor) | 0.0474 | 0.0474 |
| CE=0.82 (observed BBB head) | 0.0584 | 0.0095 |
| CE=1.00 | 0.0843 | 0.0091 |
| CE=1.50 | 0.2234 | 0.0096 |
| CE=1.82 | 0.3318 | 0.0120 |
| uniform | 0.4730 | 0.0217 |

**Reading, deliberately conservative:** at the BBB run's *observed* type-head confidence (CE ≈ 0.82) the penalty on correct geometry is only 0.058 ≈ 13% of the coordinate loss — so this is **not** the soup cause and does not displace the undertraining verdict in §13.10. But during **early** training (CE ≥1.5) it reaches 0.22–0.33, i.e. 50–72% of `train_pos`, and it is purely repulsive — a gradient of that size pointing away from correct bonded geometry is a plausible contributor to the generated NN distance p50 of 1.78 Å, which sits suspiciously right at `_BOND_CUTOFF = 1.8`.

**Proposed minimal fix (5 lines, ablation arm only):** renormalize the demand side over classes that can be a real atom — `p_live = p * (max_valence > 0); p_live /= p_live.sum(-1, keepdim=True)`. This keeps the model's own predicted distribution (so it still cannot dodge by predicting carbon everywhere) while removing the dead-class dilution: the penalty becomes ~0.01 at every confidence, i.e. never worse than the one-hot floor.

**Not applied yet**, for two reasons: (a) Task 3b only triggers if Task 1's trend says the soup is *not* pure undertraining; (b) `train.py` imports `src/objectives.py` at process start, so editing it mid-run would silently change the objective of the next sweep arm — the same hazard that applied to `partition.py`.

## 14. Commit History

- `c1a1ef5` resumable chunked training + central split/loader fixes
- `9bfb020` federated checkpoints + persistent eval outputs
- `2bbf472` Phase 1 BBB dataset pipeline
- `5b17b0e` Phase 2 conditioning + dead-coordinate-head fix
- `ac8a0ec` Phase 3 BBB oracle
- `9b62eb3` atom-index convention fix + sampler hardening
- `6ebc70e`→`8a1d892` λ₂ central training (after rebase onto remote `8fb2830` QM9 analysis doc); remote and local in sync
- `3648a5f` all-pairs λ₂ surrogate + live validity probes + recorded evals
- `0030a73` sampler sweep: DDIM-200/eta0.5 Table I protocol (47.1/14.3)
- `7f5deed` Phase 4 BBB metrics with oracle wiring and tests
- `ce1ac05` federated configs to geo_esc recipe with trainer grad-clip guard
- `dad20cc` Tier 1 training enhancements: EMA, min-SNR-gamma weighting, rotation augment, warmup-cosine LR, sigmoid schedule
- `0320004` Tier 2 architecture: self-conditioning, bond-head supervision, multi-scale kNN, coordinate refinement head
- `d49df80` Tier 3: flow matching objective and ODE sampler with DDPM fallback
- `2827c3a` Harden samplers: substitute predict-zero on non-finite predictions mid-trajectory
- `dfd6296` Per-pair bond tolerance matrix, fragment pruning, type temperature, QED filter, implementation-status appendix
- `a6fef55` Add V4 retrain configs: V3 recipe + Tier 1+2 opt-ins, eps and flow arms
- `4799ee6` Record V4-eps eval: 55–56% validity, diversity metrics up across the board
- `113eced` Record V4-eps eta sweep: validity rises with eta (n=200)
- `16cae2e` Record V4-eps full eta sweep: 62.2% validity at eta=1.0, parity with V3
- `3ad182b` Record V4-flow result: early stop epoch 95, 0.4% validity, objective A/B pending triage
- `b9747af` / `36a67b6` Phase 4: wire connectivity metrics into `evaluate()` and the LaTeX eval row
- `1dbe965` Phase 5: grid-sweep and ablation runners with dry-run support
- `1af7c39` Phase 6: dataset dispatch, conditioning/guidance wiring, label-skew partitioning
- `541c831` Phase 7: visualization pipeline + paper figure generator (PDF+PNG)
- `9a491a8` Phase 8: spec test suite (conditioning, oracle, metrics, runners, partitioning, figures)
- `b65523f` Fix `generate_and_eval --help`; auto-load BBB oracle from config; guidance-scale sweep runner
- `1f4738f` Flow triage diagnostics: exact zero-velocity baseline + per-u velocity skill
- `2b194b7` Backfill connectivity metrics from saved SDFs; fix `SDMolSupplier` API misuse
- `a4df4e1` Log the true per-objective predict-zero baseline (flow trivial loss ~3.9, not 1.0)
- `6f666aa` Record η=1.0 Table I evals + first flow checkpoint probes
- `b6dd41d` Point Table I at the matched η=1.0 rows; restrict convergence histories
- `4fbe41e` Test post-Phase-8 execution tooling; handle empty-SDF evals; side-effect-free guidance dry-run
- `c3e0e37` Condition guided generation on the training-set raw property means instead of OOD zeros
- `357e6d1` Record sweep/ablation/guidance results and regenerate Figs 2/4/7 from real data
- `2137c68` Fix `partition_iid` handing every singleton class to client 0 (+ versioned cache key, regression test)
- `c998540` Add the one-step x0 recovery probe as a reusable diagnostic
- `102d9d4` Probe-scale BBBP training budget + checkpoint override on the x0 probe
- `93d506a` Record the session's findings: flow triage, connectivity flip, BBB soup, partition bug
- `5f983ab` Split IID sweep configs + committed x0 trend monitor (the re-run was previously launched from throwaway temp files)
- `b09cfec` Record the Task 4 audit: Table I rows clean, `--relax` is a no-op

## 15. What's Next

0. **Host reboot, not a code failure (2026-10-06 13:55 → 10-10 15:11 idle):** every background job was killed by an OS reboot, ~4 days before the work was noticed. Both chains were relaunched on 10-10 15:11 (`setsid`, verified alive): training resumed from `last.pt` and confirmed via the checkpoint itself at `epoch=125` (not a silent restart from 0). Sweep stdout is block-buffered when redirected, so `logs/sweep_K_iid_*.log` looks empty for long stretches — read progress from `outputs/sweep_K_mode/*/*/results.json` instead. Two traps found in the interrupted state: `K=1 s=123` and `K=4 s=123` were killed mid-train, so their `eval/metrics.json` still hold **stale pre-fix** numbers (90.2 / 92.5); and `K=2`×3 / `K=4 s=456` are entirely pre-fix. Because the fix also changed within-client index order, K=1 s=42 moved 87.8→92.7, so all 12 arms are re-run as one uniform batch.
1. **Running now (probe-scale BBB retrain):** `configs/central_bbb_longprobe.yaml` — ~1400 epochs × 26 batches ≈ **36k steps** (~3h), early stopping disabled, checkpoints every 50 epochs in `checkpoints/bbb_longprobe/`. A monitor probes the low-t x0 error at every 100th checkpoint into `logs/bbb_longprobe_probe_trend.log`. **Decision point:** if the low-t x0 error drops below the trivial baseline, scale to the full (~200k-step) budget; if it plateaus above it, the soup is not pure undertraining and the architecture/objective needs attention.
2. **Running now (K-sweep IID re-run):** the 12 BBBP IID arms (K=1,2,4,7 × 3 seeds) retrained against the fixed partitioner, writing in place to `outputs/sweep_K_mode/numclients*_modeiid/`. Commit the results, then regenerate Table I + Figs 2/4/7 so the corrected partitions are reflected.
3. **Then:** the headline BBB story still needs resolution — every centralized, federated, ablated and guided BBB arm produces unconnected fragment soup (ConnV ≈ 0). Options: (a) finish the undertraining test above; (b) revisit the x0-valence all-pairs surrogate; (c) report soup as the honest finding and scope the paper's BBB claims accordingly. **Task 3b's analysis half is done (§13.14):** the λ₂ demand side is diluted by dead classes while the type head is diffuse; the fix is a 5-line renormalization, measured, **not yet applied** (see §13.14 for why). Treat it as the single Task 3b ablation arm once Task 1 reports.
4. **Task 4 closed (§13.13):** all five Table I rows already carry matched `Validity` + `ConnectedValidity` + `metrics_relaxed`; the three un-backfillable dirs are no longer referenced by `TABLE_RUNS`. **New:** the relaxed column is a no-op — do not claim a relaxed gain in Table I, and treat post-hoc relaxation as a dead end for connectivity.
5. **Doc/paper:** `progress.md` §13.7–§13.13 records this session; the paper's limitations section needs the undertraining, partition-bug and `--relax`-no-op caveats.
