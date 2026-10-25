Follow-up 3/3: fragment-level numbers across runs (run 2026-10-25).

Source files read:
- outputs/eval_v5_bbb_base/clump.json
- outputs/eval_pre_rl/clump.json
- outputs/eval_v5_bbb_ctemp0p5/clump.json
- outputs/eval_v5_bbb_steric_w1/clump.json
- outputs/eval_v5_bbb_steric_w5/clump.json
- outputs/eval_v5_bbb_steric_w10/clump.json
- outputs/eval_v5_bbb_steric_w1_n1000/clump.json
- outputs/eval_v4_flow_last_n1000/metrics.json
- outputs/eval_v4_flow_probe_last1000/metrics.json
- outputs/sweep_guidance_full/*/metrics.json (6 scales)
- outputs/sweep_guidance/*/metrics.json (6 scales)

Clump-cluster (bonds_per_mol high, ~37, but fragmented):
- V5 BBB base n=174: FRAGMENTED CLUMPS — 5.9 frag/mol vs 1.0 real, 0.93 bonds/atom vs 1.00, Rg 3.75 vs 3.09 A, single-fragment 20%.
- Pre-RL ruler n=398: 4.8 frag/mol, 0.95 bonds/atom, Rg 3.68 vs 3.09, single-fragment 22.6%.
- V5 BBB ctemp 0.5 n=105: UNDER-BONDED / DISPERSED, 0.56 bonds/atom, 20.2 frag/mol, Rg 7.07 A, single-fragment 0%.
- W5 steric n=86: 4.5 frag/mol, 0.96 bonds/atom, nn_mean 1.28 vs 1.15, contact_fraction 6.6% vs 9.9%.
- W1 steric n=1000: 4.98 frag/mol, 0.95 bonds/atom, nn_mean 1.29, single-fragment 25%.
- W1 steric n=86: same family as W5.
- W10 steric: same family.

V4 flow metrics (no clump.json, read metrics.json):
- V4 flow last n=1000: Validity 20.2%, ConnectedValidity 16.2%, BBB% = -1.0 (masked), ScaffDiv 0.663, ScaffCov 0.011.
- V4 flow probe last n=200: Validity 17.5%, ConnectedValidity 15.0%, BBB% = -1.0.

Sweep, guidance_full over bbb_full/best.pt (n=1000, ddim=200, eta=1.0, quadratic, relax=False):
- gs0.0: Val=71.3 ConnV=0.9 BBB=0.0 Bonds/mol=32.8 ConnFrac=1.3 ScaffDiv=0.339 ScaffCov=0.167
- gs0.5: Val=74.8 ConnV=1.1 BBB=17.5 Bonds/mol=32.5 ConnFrac=1.5 ScaffDiv=0.305 ScaffCov=0.167
- gs1.0: Val=70.4 ConnV=0.9 BBB=14.6 Bonds/mol=33.0 ConnFrac=1.3 ScaffDiv=0.344 ScaffCov=0.167
- gs2.0: Val=53.1 ConnV=0.9 BBB=19.8 Bonds/mol=32.6 ConnFrac=1.7 ScaffDiv=0.420 ScaffCov=0.167
- gs4.0: Val=12.5 ConnV=1.3 BBB=58.4 Bonds/mol=24.8 ConnFrac=10.4 ScaffDiv=0.544 ScaffCov=0.167
- gs8.0: Val=0.7 ConnV=0.6 BBB=57.1 Bonds/mol=7.1 ConnFrac=85.7 ScaffDiv=0.571 ScaffCov=0.083

Sweep, base bbb/best.pt (n=1000, ddim=200, eta=1.0, quadratic, relax=False):
- gs0.0: Val=92.8 ConnV=0.0 BBB=0.0 Bonds/mol=8.7 ConnFrac=0.0 ScaffDiv=0.061 ScaffCov=0.083
- gs0.5: Val=93.5 ConnV=0.0 BBB=0.0 Bonds/mol=8.8 ConnFrac=0.0 ScaffDiv=0.055 ScaffCov=0.083
- gs1.0: Val=94.1 ConnV=0.0 BBB=0.0 Bonds/mol=8.5 ConnFrac=0.0 ScaffDiv=0.056 ScaffCov=0.083
- gs2.0: Val=92.7 ConnV=0.0 BBB=0.0 Bonds/mol=8.7 ConnFrac=0.0 ScaffDiv=0.058 ScaffCov=0.083
- gs4.0: Val=93.5 ConnV=0.0 BBB=0.0 Bonds/mol=9.2 ConnFrac=0.0 ScaffDiv=0.060 ScaffCov=0.083
- gs8.0: Val=92.4 ConnV=0.0 BBB=0.0 Bonds/mol=10.1 ConnFrac=0.0 ScaffDiv=0.079 ScaffCov=0.083

Takeaway: the sweep contradicts the earlier read. bbb_full can be pushed toward connectivity by strong guidance (ConnFrac 1.3% -> 85.7% from w=0 to w=8), but only by destroying validity (71% -> 0.7%) and collapsing Bonds/mol from ~33 to ~7. The base bbb checkpoint is flat at ConnV=0.0 / Bonds/mol ~8-10 across every guidance scale, so it has no connectable geometry to begin with. The two failure modes seen in the clump.json cluster (fragmented clumps vs under-bonded dispersed) both appear here across guidance scale.
