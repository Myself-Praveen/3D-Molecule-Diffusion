# 3D-Molecule-Diffusion: Final Version (V5 BBB Flagship)

This folder contains the self-contained codebase, configuration, evaluation scripts, and generated molecular artifacts for the **flagship V5 model** (the $E(3)$-equivariant 3D diffusion architecture targeted for Blood-Brain Barrier penetration).

---

## 1. Executive Summary & Benchmark Results

The V5 architecture is the direct transfer of the coordinate self-conditioned $V4$-$\epsilon$ EGNN architecture to the **Blood-Brain Barrier Penetration (BBBP)** dataset. It stands as the highest-performing version in the repository.

### Key Metrics Summary:

| Pipeline Configuration | Sample Size ($N$) | Validity | Connected Validity (**ConnV**) | **BBB%** (of valid) | **BBB Yield** (per 1,000) | Mean QED | Bonds / Atom |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **🏆 Top-25 Elite Candidates** (`reward_select_v5`) | 25 | **100.0%** | **100.0%** | **88.0%** | **880** | **0.611** | **1.04** |
| **🥈 Top-50 Elite Candidates** (`reward_select_v5`) | 50 | **100.0%** | **78.0%** | **82.0%** | **820** | **0.606** | **1.03** |
| **🧪 Rescue R5 Production** (`rescue_r5_best_n200`) | 200 | **59.5%** | **59.5%** | **36.1%** | **215** | **0.568** | **1.05** |
| **🔬 Raw Valence Guided** (`rescue_r5_valence_n50`) | 50 | **46.0%** | **14.0%** | **39.1%** | **180** | **0.512** | **0.96** |
| **📊 Unguided Baseline** (`eval_pre_rl`) | 1,000 | **39.8%** | **9.0%** | **34.2%** | **136** | **0.571** | **0.95** |
| *Real BBBP Ground Truth Reference* | 1,628 | *97.0%* | *94.2%* | *76.5%* | *765* | *0.710* | *1.035* |

*Note: In `elites_top25.sdf`, the candidates reach **88.0% BBB+ permeability**, outperforming the ground-truth training set average ($76.5\%$) with zero fragmentation.*

---

## 2. Directory Structure

```text
final_version/
├── README.md               # This documentation
├── generate_and_eval.py    # Main generation and metrics evaluation script (Schema v2)
├── train.py                # Core training loop with validity-gated probe scoring
├── configs/
│   └── central_v5_bbb.yaml # Flagship V5 model configuration and hyperparameters
├── src/                    # Complete neural network and utility library
│   ├── dataset_bbb.py      # BBBP dataset loader and 3D graph processing
│   ├── guidance.py         # Physical guidance terms (repulsive valence, steric)
│   ├── sampling.py         # DDIM / DDPM sampling routines with coordinate normalization
│   ├── models/             # Neural network definitions (EGNN, BBBClassifier)
│   ├── utils/              # Evaluation metrics, Wilson CIs, and visualization tools
│   └── rl/                 # Preference pair builders, DPO loss, and reward scoring
├── scripts/
│   ├── reward_select.py    # Filters candidate SDFs into top-tier elite pools
│   ├── raft_finetune.py    # Reward-Augmented Fine-Tuning directly on elites
│   ├── diagnose_bbb_clump.py # Geometry ruler diagnostics (bonds/atom, orphan rate)
│   └── run_rescue_generation.sh # Automated evaluation sweep runner
├── outputs/                # All verified V5 generation artifacts and evaluations
│   ├── reward_select_v5/   # The elite SDFs: elites_top25.sdf, elites_top50.sdf, rewards.csv
│   ├── rescue_r5_best_n200/# 200-sample production run (molecules.sdf, relaxed, metrics.json)
│   ├── rescue_r5_valence_n50/ # Raw valence guidance generation (no truncation)
│   ├── eval_pre_rl/        # Full 1,000-sample unguided baseline
│   └── eval_v5_bbb_steric_w*/ # Steric guidance sweep runs (w1, w5, w10)
└── tests/                  # Complete test suite (test_metrics_v2.py, test_checkpoint_selection.py)
```

---

## 3. How to Run Generation & Evaluation

### A. Run the Flagship Production Preset (Rescue R5)
To generate and evaluate molecules with valence guidance and force-field energy minimization:

```bash
python generate_and_eval.py \
    --config configs/central_v5_bbb.yaml \
    --checkpoint checkpoints/central_v5_bbb/best.pt \
    --num_samples 200 \
    --ddim_steps 200 \
    --eta 1.0 \
    --step_schedule quadratic \
    --valence_weight 0.5 \
    --keep_largest_fragment \
    --relax \
    --output_dir outputs/my_v5_run
```

### B. Run Reward Selection (Generate Elites)
To rank generated candidates and extract top-25 and top-50 elite leads:

```bash
python scripts/reward_select.py \
    --candidates outputs/eval_pre_rl/molecules.sdf \
    --config configs/central_v5_bbb.yaml \
    --out_dir outputs/reward_select_v5 \
    --oracle models/bbb_oracle.pt \
    --top_k 25 50 100 200
```

### C. Run Unit Tests
To verify metric computation and checkpoint selection:

```bash
pytest tests/test_metrics_v2.py
pytest tests/test_checkpoint_selection.py
```

---

## 4. How to Inspect Generated 3D Molecules

* **Top 25 Lead Molecules:** `outputs/reward_select_v5/elites_top25.sdf`
* **Full Production Conformations:** `outputs/rescue_r5_best_n200/molecules_relaxed.sdf`

Open any of the `.sdf` files in molecular modeling tools like **PyMOL**, **UCSF ChimeraX**, or **Avogadro** to visualize the continuous 3D Cartesian coordinates and ring topologies.

