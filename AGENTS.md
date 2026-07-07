# Agent Handoff — Yield Curve Dynamics ML

Use this file to onboard a new Cursor/agent chat quickly.

## Project

**Title:** Geometric No-Arbitrage Yield Curve Dynamics with Student-t CVAEs, Neural SDEs, and Jacobian Projection

**Repo:** `danield116/Yield-Curve-Dynamics-ML`  
**Main code:** `yield-curve-geometric-sde/`

## Research goal

Two-stage pipeline for yield-curve forecasting:

1. **Stage A** — learn a low-dimensional yield-curve manifold (VAE/CVAE/Student-t CVAE)
2. **Stage B** — learn latent dynamics with Neural SDE + constraint ablations

Stage B ablations (not separate models — same pipeline, different penalties):

| Ablation | Flags |
|----------|--------|
| `sde_only` | no constraints |
| `sde_pde` | no-arbitrage PDE + diagnostics |
| `sde_jacobian` | decoder-Jacobian manifold projection |
| `sde_both` | PDE + Jacobian |

**Important framing:** Neural SDE is the dynamics model. Jacobian projection is a geometric constraint on the decoder manifold, not a competitor to the SDE.

## What is implemented

- [x] Repo scaffold + config
- [x] FRED data download + preprocess (start `2001-07-01`, drop incomplete rows before imputation)
- [x] PyTorch datasets/dataloaders (pointwise + latent windows)
- [x] Models: VAE, CVAE, Student-t CVAE, Neural SDE (scaffolds/skeletons)
- [x] Constraints: `bond_math`, `no_arbitrage_pde`, `jacobian_projection`
- [x] `training/train_stage_a.py` — full training loop
- [x] `training/train_stage_b.py` — full training + ablations
- [x] `notebooks/colab_train.ipynb` — Colab pipeline (data → Stage A → all ablations)

## What is still TODO

- [ ] Full PDE Hessian term tuning (optional flag `include_hessian`, off by default)
- [ ] Optional `sde_activation` ablation flag (ReLU vs Tanh in SDE MLPs)

## Recently completed

- [x] Jacobian fix for LevelScript CVAE (`training/manifold_ops.py`)
- [x] Exact Student-t NLL in `models/student_t_vae.py`
- [x] `baselines/pca_var.py`, `baselines/nelson_siegel.py` (NSS / Nelson-Siegel-Svensson)
- [x] `evaluation/evaluate_run.py` — test-split RMSE, multi-horizon, arbitrage diagnostics
- [x] `experiments/run_full_comparison.py` — orchestrate eval + comparison plots
- [x] `evaluation/metrics.py` scorecard + `visualization/plot_curves.py`

## Key file map

```
yield-curve-geometric-sde/
├── data/
│   ├── download_fred_yields.py      # FRED CSV download
│   ├── preprocess_curves.py         # clean, split, scale, LevelScript
│   ├── datasets.py                  # tensors, LatentWindowDataset
│   └── dataloaders.py               # pointwise + latent window loaders
├── models/
│   ├── vae.py, cvae.py, student_t_vae.py
│   └── neural_sde.py                # mu_P, sigma, lambda, mu_Q, Euler-Maruyama
├── constraints/
│   ├── bond_math.py                 # P = exp(-y*tau), forwards, short rate
│   ├── no_arbitrage_pde.py          # PDE residual + total_constraint_loss
│   └── jacobian_projection.py       # tangent + re-encode projection
├── training/
│   ├── train_stage_a.py             # manifold training
│   └── train_stage_b.py             # SDE + ablation training
├── config/default.yaml
└── notebooks/colab_train.ipynb
```

## Where the math lives (not in training files)

| Math | File |
|------|------|
| `dz = mu_P dt + sigma dW`, `mu_Q = mu_P - sigma*lambda` | `models/neural_sde.py` |
| Euler-Maruyama step | `sde/integrators.py`, `neural_sde.euler_maruyama` |
| No-arbitrage PDE residual | `constraints/no_arbitrage_pde.py` |
| Jacobian projection | `constraints/jacobian_projection.py` |

Training scripts **call** these modules; they do not define the equations.

## Data conventions

- **Source:** FRED constant-maturity UST yields (11 tenors)
- **Default start:** `2001-07-01` (full 11-tenor overlap)
- **Preprocess:** drop rows with any missing tenor, then business-day reindex + short-gap imputation
- **Split:** chronological 70/15/15, train-only robust scaling (median/IQR)
- **LevelScript:** `level = 1Y`, `shape = curve - level` (index 3 in default tenor order)
- **Raw/processed data are gitignored** — must download in Colab or locally

## How to run

```bash
cd yield-curve-geometric-sde

# Data
python data/download_fred_yields.py
python data/preprocess_curves.py --levelscript

# Stage A
python training/train_stage_a.py

# Stage B (pick ablation)
python training/train_stage_b.py --ablation sde_pde
```

# Evaluate (test split, multi-horizon + baselines)
python evaluation/evaluate_run.py --split test --output-dir reports/comparison

# Full comparison (eval + plots; add --train-stage-a/b to retrain)
python experiments/run_full_comparison.py
```

**Colab:** open `notebooks/colab_train.ipynb`, set runtime to **GPU** (not TPU for Stage B constraints). Run through cell 9 for evaluation scorecard.

## Training outputs

- Stage A: `reports/checkpoints/stage_a/`, `reports/latents/stage_a/`
- Stage B: `reports/checkpoints/stage_b/`, `reports/forecasts/stage_b/`
- Evaluation: `reports/comparison/scorecard.csv`, `summary.json`, `figures/`

## Code style preferences

- Readable, notebook-like Python (user referenced `GeoGuesser_Primary_Model.ipynb`)
- Prefer simple functions over heavy abstraction
- Minimize scope — focused diffs only
- Do not commit unless user asks
- `__pycache__`, `data/raw/`, `data/processed/` are gitignored

## Design notes for next agent

- **Stage A default:** `student_t_cvae` with `use_levelscript: true`
- **Neural SDE uses Tanh** (stability); VAE/CVAE use ReLU — not empirically ablated yet
- **Stage B fit loss:** `latent_fit_weight` * latent MSE + `curve_loss_weight` * curve MSE + optional constraints
- **Stage B training horizons:** random sample from `train_horizons` (default `[1,5,21]`); val uses `val_horizon` (default `1`)
- **Stage B checkpoint:** saved on `checkpoint_metric` (default `curve_rmse`)
- **SDE input:** last `latent_history_steps` latent vectors (default 10), not only z_t
- **LevelScript decode at forecast:** uses last known level from `y_hist`, not future `y_fut` (no lookahead)
- **Persistence-residual forecast:** `y_pred = y_last + decode(z_pred) - decode(z_last)` (matches persistence at zero drift)
- **Constraint horizon gating:** PDE/diag apply on `pde_train_horizons` (default `[1]`); Jacobian on `jacobian_train_horizons` (default `[1,5,21]`)
- **Constraint weights (tuned):** `pde_penalty_weight=0.001`, `jacobian_projection_weight=0.1`
- **Jacobian warmup:** `jacobian_warmup_epochs=40` to separate from `sde_only` after fit stabilizes
- **Eval geometry tables:** `reports/comparison/constraint_ablation_h1.csv`, `h5.csv`, `h21.csv` + arb/manifold bar plots
- **Stage A KL:** `kl_weight` default `0.1` (lower KL → better reconstruction for Stage B)
- **Constraints are soft penalties** during training; diagnostics also used at eval
- **AlphaVantage not recommended** — less tenor coverage than FRED, API limits
- **Local training not required** — Colab GPU preferred, especially for PDE/Jacobian ablations

## Suggested next tasks (priority)

1. Re-run Stage B on Colab with new curve loss + latent history (old checkpoints incompatible)
2. Confirm test `curve_rmse` beats `persistence` baseline, then chase `nss` / `pca_var`

## References

- Paper PDF in repo root: `Yield_Curve_Dynamics_Pred_Paper.pdf`
- README: `yield-curve-geometric-sde/README.md`
