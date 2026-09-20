# Project Progress & Rationale

**Project:** Novel Molecular Generation using Graph GANs/Diffusion for BBB Permeability
**Current Phase:** Phases 1–3 implemented; QM9 geometry retraining in progress
**Last Updated:** 19 Sep 2026 (CPU-only machine, `node1`)

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

## 10. Commit History

- `c1a1ef5` resumable chunked training + central split/loader fixes
- `9bfb020` federated checkpoints + persistent eval outputs
- `2bbf472` Phase 1 BBB dataset pipeline
- `5b17b0e` Phase 2 conditioning + dead-coordinate-head fix
- `ac8a0ec` Phase 3 BBB oracle
- `9b62eb3` atom-index convention fix + sampler hardening
- `6ebc70e`→`8a1d892` λ₂ central training (after rebase onto remote `8fb2830` QM9 analysis doc); remote and local in sync
- Pending: λ₂=1.0 code (surrogate fix, validity probe, config) + v2 eval records + `central_geo_esc` escalation config/run

## 11. What's Next

1. **Immediate:** monitor the escalated run — `tail logs/central_geo_esc.log` (quick_valid probes every 10 epochs; chunked resume via `--resume --run_epochs N`). First real signal: the epoch-10 probe; v2 was 0.0% there, so anything > 0% is progress.
2. **If validity recovers:** full eval to a fresh `outputs/` dir for the honest before/after Table I, then fed retrains with the winning recipe (both fed globals are still coord-dead), then Phase 4 extended metrics (BBB% via the oracle, scaffold diversity, Lipinski, CNS-MPO).
3. **After the escalated run:** the sharpened diagnosis (Discovery C) is *under-bonded geometry from coordinate error* (RMSE ≈ 0.54 Å vs ≤0.45 Å bonding windows), so the valves are: capacity/training length (running), eps-prediction form, and higher-fidelity eval sampling (more DDIM steps). A λ₂=0 ablation arm is optional now that the clump theory is unproven.
4. **Then:** Phase 5 BBB configs + sweeps, Phase 6 entry-point wiring (conditioned/BBB training + guided generation), Phase 7 figures, Phase 8 tests + paper write-up.
