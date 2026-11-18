# Plan: raise the V5 model metrics to a good level

Updated 2026-11-19. Supersedes the metric targets in `docs/gravitas_enhancement_plan.md`
(§2) and the remaining open items of `proposed_plan.md` / `validity_80_plan.md`. The RL
stage those plans gate on has now actually run (`progress.md` §18); this document starts
from what that measurement produced and spends effort only where the record shows nothing
has moved.

## 0. Where we are

Best arm on record is `checkpoints/central_v5_bbb_rl_beta5/last.pt` (V5-eps + DPO β5.0),
same protocol as every row below: DDIM-200, η=1.0, seed 42, `--relax`, steric+valence
guidance, **no** largest-fragment collapse.

| metric | V5 pre-RL | DPO β5.0 (best) | best-known elsewhere | good level (target) |
|---|---|---|---|---|
| Validity % | 44.25 | **56.00** | 71.0 (`bbb_full_connfix`, but ConnV 2.2) | **≥ 60** |
| ConnectedValidity % | 10.50 | 13.50 | 14.0 (steric w=1, n=200 — did not survive n=1000) | **≥ 25** |
| Single-fragment % | 23.73 | 24.11 | 25 (steric w=1, n=1000) | **≥ 45** |
| Fragments / mol | 5.85 | 5.92 | 4.98 (steric w=1, n=1000) | **≤ 1.5** |
| Bonds / atom | 0.933 | 0.933 | 0.96 (steric w=5) | **≥ 0.99** |
| BBB_yield % (BBB ∩ valid) | 18.00 | 19.50 | — (never reported before §18) | **≥ 22** |
| QED | 0.561 | 0.552 | 0.611 (`bbb_full_connfix`) | **≥ 0.58** |
| Uniqueness % | 100 | 100 | 100 | **≥ 95** |

Two targets above are **corrected measurements, not inherited numbers**:

- **Bonds/mol ≈ 27 (inherited from the plan) is wrong for this dataset.** BBBP with
  explicit hydrogens is 46.38 atoms and **48.27 bonds per molecule (1.041 bonds/atom)**
  over all 2039 rows of `data/bbbp/BBBP.csv`. The model draws 46.97 atoms, so the like-for-like
  target is ~48.8 bonds, and the honest invariant is **bonds/atom**. The old 27 conflates a
  heavy-atom count with a hydrogen-inclusive one.
- BBB_yield, not raw BBB%, is the property metric. Raw BBB% falls when validity rises
  (40.68 → 34.82 for β5.0) purely because the denominator is a fixed 400-molecule cohort
  containing fewer invalid molecules; BBB among *valid* molecules rose 18.00 → 19.50.

## 1. Measure before optimizing (½ day, no training)

1.1 **Adopt the n=1000 ruler.** At n=400 the 95% CI on ConnectedValidity is ±3.4 pp, wider
than most of the effects we are chasing, and the steric w=1 result inverted between n=200
(14.0%) and n=1000 (9.8%). Every gate below is evaluated at n=1000; n=400 is for triage only.

1.2 **Resolve the size defect.** The sampler draws 46.97 atoms per molecule (matching BBBP's
46.38) but the evaluated set averages 41.3–43.1 atoms while `AtomsRetained%` reports 100.0
and `SizeMatch_KS` reports 0.0. One of those three numbers is not measuring what its name
says. Instrument the draw: per-molecule N requested, N in `molecules.sdf`, N after `--relax`.
Fragment statistics are dominated by molecule size, so this must be settled before any
fragment-count comparison across arms.

1.3 **Verified-not-broken list** — do not spend effort here again:
- `knn_schedule` *is* honored at sampling time (`src/sampling.py:277, 381, 452`); the
  "sampler rebuilds at k=4 only" concern in `progress.md` §17 is stale.
- DPO β ≈ 5 is correct for a difference-of-MSE reward; β = 0.05 is a log-prob-ratio value
  and provably inert here (loss pinned at log 2 for 30 epochs).
- DPO lr 1e-6; 1e-5 wrecks the geometry heads in a single epoch.
- `--coord_temperature` is a recorded negative (Validity 52.5 / ConnV 0.0 / 20.2 frags).

## 2. Workstream A — bond supervision (highest expected value)

**Diagnosis.** Fragment count cannot fall because nothing in training knows what a bond is.
`EquivariantGenerator` has a `bond_head` and `predict_bond_logits` (`src/models/egnn.py:189,
297`), and `train.py` implements the supervised pair loss (`_bond_pair_labels`, gated at
t < valence_tau), but `bond_loss_weight` is absent from the V5 config because
`smiles_to_3d_data` returns no `edge_index` (`src/dataset_bbb.py:175`) — the QM9 arms use
`data.edge_index` and BBBP never provided one. The sampler's graph is a kNN distance graph
(k=4…16), which is not a bond ruler, and the DPO reward sees validity/QED/BBB only. So every
lever tried so far has been blind to connectivity; steric guidance moves it only incidentally
(+5 pp at n=200, gone at n=1000).

**Change (small, ~30 lines).** In `smiles_to_3d_data`, after `AddHs`, emit
`edge_index` from `mol.GetBonds()` (both directions, molecule-local indices). The mol object
is already built there, so no new dependency and no new data. Then set
`bond_loss_weight: 0.1` in a V5-eps child config and train from scratch (or resume the
pre-RL `best.pt` — report which; the record should not mix the two).

- **Gate A1 (1 epoch):** `bond_loss` finite and falling, `val_pos` within 5% of its pre-RL
  value, and a 64-molecule probe still produces parseable SMILES.
- **Gate A2 (pilot ≤ 400 epochs, ≈ 12 h overnight at the measured ~2 min/epoch):** at n=400,
  bonds/atom ≥ 0.98 and single-fragment ≥ 32%. This is a *trend* gate, not the final number.
- **Stop rule:** if 400 epochs leave both bonds/atom and single-fragment inside the baseline
  CI, the bond head is not reaching the sampler — proceed to A3 rather than extending the run.

**A3 (fallback, no retraining).** Feed the *predicted* bonds into the sampler graph: replace
or augment the kNN `edge_index` with edges whose `predict_bond_logits` probability exceeds a
threshold, rebuilt each step from `x0_estimate`. Cheap to implement, one eval per threshold
(0.3 / 0.5 / 0.7).

## 3. Workstream B — build the sampler graph from x0, not from noise

The sampler rebuilds `build_knn_graph(pos, ...)` on the *noisy* coordinates every reverse step
(`src/sampling.py:379, 449`). At high t that graph is a near-random graph over scattered atoms,
so the message-passing never sees the molecule it is trying to denoise. Two variants, each one
eval at n=400 (≈ 40 min, no training):

- **B1:** always build from `x0_estimate`.
- **B2:** noisy graph for t > 50%, x0 graph below.

**Gate B:** single-fragment +5 pp with validity no worse than −3 pp. This is the cheapest
experiment in the plan and the most likely to explain fragmentation, so it runs first.

## 4. Workstream C — re-test connectivity guidance on the Tier-2 model

`connectivity_*` (attractive term pulling near-miss fragment gaps to covalent distance) is
implemented and wired (`src/sampling.py:340`, CLI in `generate_and_eval.py:621`), with exactly
one recorded attempt: `bbb_full` (Tier-1) gave Validity 71.0 / ConnV 2.2 / single-frag 3.1 —
a negative. It has never been run on the Tier-2 architecture, where the failure mode is
clumps rather than soup.

Run w ∈ {0.25, 0.5, 1.0} on the β5.0 checkpoint, ramp `start_frac` 0.7, n=400, everything
else identical. **Gate C:** single-fragment +10 pp with validity no worse than −3 pp.
**Stop rule:** flat at all three weights → retire the term and say so in `progress.md`.

## 5. Workstream D — give the RL reward a topology term

Only after A or B moves single-fragment. The measured evidence for the current blindness is
sharp: with a validity/QED/BBB reward, DPO raised validity 11.75 pp (p=0.0009) and left
connected validity at 10.5 → 13.5 (p=0.19) and single-fragment flat — it optimized precisely
what the reward could see.

Add to the reward: `+w1 · 1[single fragment]`, `−w2 · |bonds/atom − 1.0|`, keep the BBB and
validity terms, drop the `high_w` guidance-scale pair source (measured *worse*: 72.9% invalid
at w=4.0 vs 62.8% at w=0.0). Weights set so no single term exceeds 40% of the reward range;
record the reward spread over 760 rollouts before training, as §18 did.

Protocol, unchanged from the one that worked: lr 1e-6, β 5.0, one timestep per pair, 30
epochs, `ema_decay 0.0`, patience 30. Gate G3: pair margin ≥ 0.60 and ≥ 50% of pairs with a
positive gap after training. Gate G4 at n=1000.

## 6. Workstream E — checkpoint selection

Training val loss draws one random timestep per graph, so it swings 0.68 ↔ 3.3 between
consecutive epochs at constant training loss; the β5.0 arm's `best.pt` is therefore an
arbitrary epoch. Select on the generation probe (`num_samples 64`, `ddim_steps 50`) instead,
and report best-vs-last explicitly so a lucky epoch cannot masquerade as progress.

## 7. Ordering, cost, and what gets reported

| step | cost | gate |
|---|---|---|
| 1.1–1.3 ruler, size audit, targets | ~½ day, no GPU/CPU training | numbers agreed |
| B1 / B2 graph from x0 | 2 evals ≈ 1.5 h | single-frag +5 pp |
| C guidance w sweep | 3 evals ≈ 2 h | single-frag +10 pp |
| A1 bond-loss smoke | 1 epoch ≈ 5 min | loss finite, probe parses |
| A2 bond pilot ≤ 400 epochs | ≈ 12 h | bonds/atom ≥ 0.98, single-frag ≥ 32% |
| E probe-based selection | runs alongside A | best ≈ last |
| D topology reward | 30 epochs ≈ 75 min + 2 evals | G3 margin ≥ 0.60 |
| final table, n=1000, all arms | 4 evals ≈ 3 h | §0 targets |

B and C are evals of the existing checkpoint and can run concurrently with A's training.

**Reporting contract** (non-negotiable, because the record already contains the counterexamples):
n=1000; no `--keep_largest_fragment`; BBB_yield alongside raw BBB%; 95% CI for every rate;
seed stated; every arm logged in `progress.md` including negatives; no arm reported from its
own favourable protocol.

## 8. Falsification and the honest ceiling

If A2 and A3 both fail to drop fragments below ~3/mol, the conclusion is that an 8-layer
256-dim atom-level EGNN with 2 refine layers cannot place explicit hydrogens densely enough at
46 atoms, and no post-hoc guidance will fix it — the deliverable then becomes the best honest
number plus a documented ceiling, not another knob. The specific falsifiers, so they are not
rationalized after the fact: bonds/atom still < 0.95 after 400 supervised epochs; single-fragment
still < 30% with predicted-bond graphs at every threshold; connectivity guidance inert at all
three weights on the Tier-2 model. Any one of those three, on its own, is a negative result and
gets recorded as one.
