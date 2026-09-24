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

## 15. What's Next

1. **Now:** monitor both fed retrains (`tail logs/fed_*_retrain.log`); on 50/50 completion, eval each `best_global.pt` with the adopted protocol (DDIM-200/eta0.5) + oracle for the federated Table I rows (validity, connected validity, BBB%).
2. **Then:** Phase 5 BBB configs + sweeps (conditioned training configs now that metrics/oracle/sampler are all in place), Phase 6 entry-point wiring (conditioned/BBB training + guided generation targeting BBB% ≫ 24.8%).
3. **After:** Phase 7 figures, Phase 8 tests + paper write-up; optional λ₂=0 ablation now that the clump theory is unproven.
