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

- [x] Repo + config (`config/default.yaml`, `config/paper_best.yaml`)
- [x] FRED data download + preprocess (start `2001-07-01`, drop incomplete rows before imputation)
- [x] PyTorch datasets/dataloaders (pointwise + latent windows)
- [x] Models: VAE, CVAE, Student-t CVAE, Neural SDE
- [x] Constraints: `bond_math`, `no_arbitrage_pde`, `jacobian_projection`
- [x] `training/train_stage_a.py` — full training loop
- [x] `training/train_stage_b.py` — full training + ablations
- [x] `notebooks/colab_train.ipynb` — Colab pipeline (data → Stage A → all ablations)
- [x] Evaluation scorecard, geometry metrics, baselines, frozen `reports/best_model/`

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

# Stage A / B (paper-primary config: config/paper_best.yaml; same hyperparams as default.yaml)
python training/train_stage_a.py
python training/train_stage_b.py --ablation sde_jacobian

# Evaluate (test split, multi-horizon + baselines)
python evaluation/evaluate_run.py --split test --output-dir reports/comparison

# Regenerate best-model figures from scorecard (optional; artifacts already in repo)
python evaluation/plot_from_scorecard.py

# Full comparison (eval + plots; add --train-stage-a/b to retrain)
python experiments/run_full_comparison.py
```

**Colab:** open `notebooks/colab_train.ipynb`, set runtime to **GPU** (not TPU for Stage B constraints). Run through cell 9 for evaluation scorecard.

## Training outputs

- Stage A: `reports/checkpoints/stage_a/`, `reports/latents/stage_a/`
- Stage B: `reports/checkpoints/stage_b/`, `reports/forecasts/stage_b/`
- Evaluation: `reports/comparison/scorecard.csv`, `summary.json`, `figures/`

## Code style preferences

- Readable, notebook-like Python
- Prefer simple functions over heavy abstraction
- Minimize scope — focused diffs only
- Comments only for non-obvious *why*; do not narrate obvious code or leave scaffold/TODO leftovers
- Do not commit unless user asks
- `__pycache__`, `data/raw/`, `data/processed/` are gitignored
- **Do not commit paper drafts:** `*.tex`, `Yield_Curve_Dynamics_Pred_Paper.pdf`, or unrelated notebooks (`GeoGuesser_Primary_Model.ipynb`) — these are gitignored

## Design notes for next agent

- **Stage A default:** `student_t_cvae` with `use_levelscript: true`
- **Neural SDE uses Tanh** (stability); VAE/CVAE use ReLU — not empirically ablated yet
- **Stage B fit loss:** `latent_fit_weight` * latent MSE + `curve_loss_weight` * curve MSE + optional constraints
- **Stage B training horizons:** random sample from `train_horizons` (default `[1,5,21]`); val uses `val_horizon` (default `1`)
- **Stage B checkpoint:** saved on `checkpoint_metric` (default `curve_rmse`)
- **SDE input:** last `latent_history_steps` latent vectors (default 10), not only z_t
- **LevelScript decode at forecast:** uses last known level from `y_hist`, not future `y_fut` (no lookahead)
- **Persistence-residual forecast (LOCKED):** `y_pred = y_last + decode(z_pred) - decode(z_last)` (matches persistence at zero drift). With `use_persistence_residual: true` (paper config), **constraint penalties also use this residual curve**, not linearized `D(z_t)+J δ`. Linearized `tilde y` is only used if persistence residual is off.
- **Constraint horizon gating:** PDE residual **and** diagnostic penalties (discount monotonicity + forward smoothness) apply on `pde_train_horizons` / `diag_train_horizons` (default `[1]`); Jacobian on `jacobian_train_horizons` (default `[1,5,21]`). Eval still reports all horizons. `sde_pde` / `sde_both` are this bundle, not a standalone PDE residual.
- **Constraint weights (tuned):** `pde_penalty_weight=0.001`, `jacobian_projection_weight=0.3`
- **Projection method:** `tangent` (direct decoder-tangent projection) — cleaner geometry gradient than `reencode`
- **Jacobian warmup:** `jacobian_warmup_epochs=40` to separate from `sde_only` after fit stabilizes
- **Why Jacobian can/can't beat `sde_only` on RMSE:** projecting a forecast onto the manifold moves it *toward* `D(E(y))`; if Stage A recon is poor, real (off-manifold) curves are far from the manifold and projection *hurts* RMSE. Constraints only help accuracy once Stage A recon is low — hence the Stage A capacity bump below.
- **Eval geometry tables:** `reports/comparison/constraint_ablation_h1.csv`, `h5.csv`, `h21.csv` + arb/manifold bar plots
- **Geometry metrics (non-redundant):**
  - `manifold_off_manifold_rmse` = `||y_pred - D(E(y_pred))||` (absolute; dominated by persistence anchor offset, looks flat across ablations)
  - `manifold_correction_gain` = `off_manifold(y_pred) - off_manifold(y_persist)` (isolates dynamics; **negative = forecast pulled closer to manifold than persistence**; Jacobian should be most negative)
  - `tangent_move_residual_rmse` = off-tangent component of decoded latent move at `z_last` (lower = move stays in decoder tangent space)
  - Superseded/removed: `manifold_delta_off_manifold_rmse` was algebraically identical to `manifold_off_manifold_rmse` (the `y_prev` term cancels) — do not reintroduce
- **Paper config file:** `config/paper_best.yaml` — frozen snapshot + reference soft scorecard (Jacobian beats `sde_only` at h=1/5/21). `default.yaml` training hyperparameters match.
- **Current best model (paper primary):** `reports/best_model/` — frozen scorecard, constraint tables, and `figures/*.png` (Jacobian ahead at h=1/5/21; Stage A recon 0.659). Config snapshot: `reports/best_model/config.yaml`. Retrain with `config/paper_best.yaml`; only replace this folder if a new run clearly wins. Same-config retrains often flip jac vs `sde_only` at the 4th–5th decimal — do not treat a later flip as a new winner.
- **Reference soft RMSE (best run, test split):** Stage A recon 0.659; h=1 jac 0.02917 vs only 0.02918; h=5 jac 0.06294 vs only 0.06301; h=21 jac 0.12212 vs only 0.12217 (both beat persistence 0.12375). h=63 extrapolation: only slightly ahead of jac.
- **Stage A capacity (LOCKED):** `latent_dim=5`, `hidden_dim=256`, `epochs_stage_a=200` — the run where `sde_jacobian` edged `sde_only` at trained horizons. Do not bump `latent_dim`; that dilutes the tangent constraint. The later 384/0.01/300 recon push did not widen RMSE separation and slightly hurt short-horizon RMSE — leave it archived, not default.
- **Keep `latent_dim` low (5), grow `hidden_dim`/epochs instead:** raising `latent_dim` enlarges the decoder tangent space the Jacobian projects onto, making the constraint *less* restrictive and diluting its effect.
- **Stage B convergence:** `epochs_stage_b=400`, `lr_schedule_stage_b=cosine` — constrained models need more/steadier training to pay off
- **Hard projection at eval (optional, non-primary):** `evaluation.compare_hard_project=true` emits extra `stage_b_*_hard` scorecard rows that project the forecast onto the manifold after soft decoding (`hard_project_method`, default `tangent`). Soft rows remain the paper primary. Hard projection needs **no retrain** — just re-run `evaluate_run.py`. Expect stronger geometry metrics; `curve_rmse` may worsen while Stage A recon is still ~0.66 off-manifold.
- **IMPORTANT:** changing `latent_dim`/`hidden_dim` invalidates ALL prior Stage A + Stage B checkpoints and latents — must retrain the full pipeline from Stage A
- **Constraints are soft penalties** during training; hard projection is eval-only and diagnostics are used at eval
- **AlphaVantage not recommended** — less tenor coverage than FRED, API limits
- **Local training not required** — Colab GPU preferred, especially for PDE/Jacobian ablations

## Suggested next tasks (priority)

1. Save Colab checkpoints from the best soft run to Drive; use `config/paper_best.yaml` to reproduce
2. Paper tables: soft primary results from `reports/best_model/`; hard projection supplementary only. Keep `.tex` drafts local (gitignored).

## References

- README: `yield-curve-geometric-sde/README.md`
- Frozen paper tables/figures: `yield-curve-geometric-sde/reports/best_model/`
