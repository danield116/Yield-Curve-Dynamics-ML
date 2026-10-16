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
| `sde_only` | no constraints (drift only; \(\sigma\) is not in the loss) |
| `sde_pde` | curve-shape bundle: simplified PDE residual + discount monotonicity + forward smoothness |
| `sde_jacobian` | ridge decoder-tangent penalty (drift only) |
| `sde_both` | bundle + Jacobian |

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
| Simplified PDE residual + mono/smooth diagnostics | `constraints/no_arbitrage_pde.py` |
| Jacobian projection | `constraints/jacobian_projection.py` |

Training scripts **call** these modules; they do not define the equations.

## Data conventions

- **Source:** FRED constant-maturity UST yields (11 tenors). These are par yields, not zeros.
- **Download start:** `2001-07-01`. After dropping incomplete rows the panel begins **2001-07-31** (first complete 1M day). End date is whatever FRED returns that day; the frozen scorecard was committed 2026-07-09, so the train cut is early January 2019.
- **Preprocess:** drop rows with any missing tenor, then business-day reindex + time interpolation (`limit_direction="both"`) and ffill/bfill. That fills holidays in **all** splits and can leak across a split boundary.
- **2002–2006 30Y:** Treasury discontinued DGS30, but current FRED still publishes estimated values. The pipeline does **not** invent a four-year linear hole. Those years sit in the training split as source estimates.
- **Split:** chronological 70/15/15, train-only robust scaling (median/IQR). LevelScript is applied **after** scaling, so subtracting scaled 1Y is not a raw parallel shift.
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
- **Stage B fit loss:** `latent_fit_weight` * latent MSE + `curve_loss_weight` * curve MSE + optional constraints. The latent term is the **same sampled horizon** as the curve term (h-step drift rollout vs \(z_{t+h}\)), not a one-step latent loss.
- **Stage B training horizons:** each batch draws uniformly from `train_horizons` (default `[1,5,21]` ≈ 1 day / 1 week / 1 month). Val and checkpointing use `val_horizon` (default `1`). Eval adds `h=63` (~1 quarter) as extrapolation.
- **Stage B checkpoint:** saved on `checkpoint_metric` (default `curve_rmse`)
- **SDE input:** last `latent_history_steps` latent vectors (default 10), not only z_t
- **LevelScript decode at forecast:** uses last known level from `y_hist`, not future `y_fut` (no lookahead)
- **Persistence-residual forecast (LOCKED):** `y_pred = y_last + decode(z_pred) - decode(z_last)` (matches persistence at zero drift).
- **What each penalty sees:** Jacobian **and** discount/forward diagnostics use that residual curve. The PDE residual does **not**: it is computed from `decoder(z_t)` with \(\mu_Q,\sigma\) from the current history. Linearized `D(z_t)+J δ` is only the constraint curve when persistence residual is off.
- **`sde_only` / `sde_jacobian` are drift networks:** no loss term depends on \(\sigma\). Diffusion is trained only when the bundle uses \(\mu_Q\).
- **Constraint horizon gating:** the bundle (PDE residual + discount monotonicity + forward smoothness) applies on `pde_train_horizons` / `diag_train_horizons` (default `[1]`); Jacobian on `jacobian_train_horizons` (default `[1,5,21]`). Eval still reports all horizons. Do not call `sde_pde` a no-arbitrage penalty: yields are robust-scaled par CMT, \(P=e^{-y\tau}\) is not bootstrapped, \(\partial_\tau P\) is implemented as \(-yP\), Hessian is off, and \(\Delta t=1\) day is mixed with annual \(\tau\).
- **Constraint weights (tuned):** `pde_penalty_weight=0.001`, `jacobian_projection_weight=0.3`
- **Projection method:** `tangent` (direct decoder-tangent projection) — cleaner geometry gradient than `reencode`
- **Jacobian warmup:** `jacobian_warmup_epochs=40` to separate from `sde_only` after fit stabilizes
- **Jacobian leading-order degeneracy (LOCKED interpretation):** the penalty is the off-tangent part of \(r_t + m_\psi\), with \(r_t=y_t-D(z_t)\) fixed (Stage A frozen) and \(m_\psi=D(\hat z)-D(z_t)\). To first order \(m_\psi=J\delta\) already lies in \(\mathrm{span}(J)\), so an exact projector gives Stage B no gradient. Ridge \(\varepsilon=10^{-5}\) leaves a weak cross-term and shrinkage; the decoder is ReLU so a smooth Hessian is not the main leftover. Do not describe the paper-primary penalty as re-encoding \(D(E(\hat y))\). High Stage A recon (~0.66) explains flat off-manifold RMSE; it does not by itself cause the degeneracy. Hard projection at eval is the separate case that can move forecasts toward the manifold and hurt RMSE.
- **Eval geometry tables:** `reports/comparison/constraint_ablation_h1.csv`, `h5.csv`, `h21.csv` + arb/manifold bar plots
- **Geometry metrics (non-redundant):**
  - `manifold_off_manifold_rmse` = `||y_pred - D(E(y_pred))||` (absolute; dominated by persistence anchor offset, looks flat across ablations)
  - `manifold_correction_gain` = `off_manifold(y_pred) - off_manifold(y_persist)` (isolates dynamics; negative = closer to the manifold than persistence). On the frozen run this does **not** order jac vs only the same way at every horizon.
  - `tangent_move_residual_rmse` = off-tangent component of decoded latent move at `z_last` (lower = move stays in decoder tangent space)
  - Superseded/removed: `manifold_delta_off_manifold_rmse` was algebraically identical to `manifold_off_manifold_rmse` (the `y_prev` term cancels) — do not reintroduce
- **Paper config file:** `config/paper_best.yaml` — frozen snapshot + reference soft scorecard (Jacobian beats `sde_only` at h=1/5/21). `default.yaml` training hyperparameters match.
- **Current best model (paper primary):** `reports/best_model/` — frozen scorecard, constraint tables, and `figures/*.png` (Jacobian ahead at h=1/5/21; Stage A recon 0.659). Config snapshot: `reports/best_model/config.yaml`. Retrain with `config/paper_best.yaml`; only replace this folder if a new run clearly wins. Same-config retrains often flip jac vs `sde_only` at the 4th–5th decimal — do not treat a later flip as a new winner.
- **Reference soft RMSE (best run, test split):** Stage A recon 0.659. Persistence is best at h=1 (0.02912) and h=5 (0.06233). Jac vs only: h=1 0.029167 vs 0.029175; h=5 0.062944 vs 0.063010; h=21 0.122123 vs 0.122173 (both beat persistence 0.12375). h=63: persistence 0.20239, only 0.23654, jac 0.24637. `sde_pde` / `sde_both` match (0.041 → 0.197 → 1.76 → 7.68); one arbitrage diagnostic differs by \(10^{-6}\) at h=21. Do not interpret `sde_both` as an independent result. Gaps of order \(10^{-5}\) are one seed, not a significance claim.
- **Stage A capacity (LOCKED):** `latent_dim=5`, `hidden_dim=256`, `epochs_stage_a=200` — the run where `sde_jacobian` edged `sde_only` at trained horizons. Do not bump `latent_dim`; that dilutes the tangent constraint. The later 384/0.01/300 recon push did not widen RMSE separation and slightly hurt short-horizon RMSE — leave it archived, not default.
- **Keep `latent_dim` low (5), grow `hidden_dim`/epochs instead:** raising `latent_dim` enlarges the decoder tangent space the Jacobian projects onto, making the constraint *less* restrictive and diluting its effect.
- **Stage B convergence:** `epochs_stage_b=400`, `lr_schedule_stage_b=cosine` — constrained models need more/steadier training to pay off
- **Hard projection at eval (optional, non-primary):** `evaluation.compare_hard_project=true` emits extra `stage_b_*_hard` scorecard rows that project the forecast onto the manifold after soft decoding (`hard_project_method`, default `tangent`). Soft rows remain the paper primary. Hard projection needs **no retrain** — just re-run `evaluate_run.py`. Expect stronger geometry metrics; `curve_rmse` may worsen while Stage A recon is still ~0.66 off-manifold.
- **IMPORTANT:** changing `latent_dim`/`hidden_dim` invalidates ALL prior Stage A + Stage B checkpoints and latents — must retrain the full pipeline from Stage A
- **Constraints are soft penalties** during training (they change the loss, not the delivered forecast). Hard projection is eval-only and was not the paper primary.
- **Baselines:** persistence is the valid external null. `pca_var` (rank 3) and NSS (4-factor Svensson; Diebold–Li-style VAR, not 3-factor NS) are implemented and appear on the scorecard, but the rolling refit `vstack`s each full lookback window onto history, so overlapping days are duplicated. Do not cite those RMSE rows as clean PCA–VAR or NSS performance. NSS aliases named `diebold_li` still call the NSS functions.
- **AlphaVantage not recommended** — less tenor coverage than FRED, API limits
- **Local training not required** — Colab GPU preferred, especially for PDE/Jacobian ablations

## Suggested next tasks (priority)

1. A non-degenerate geometric penalty (re-encoding, or a tangent check on a curve move that is not \(D(\hat z)-D(z_t)\)).
2. Residual-only ablation in correct units and matched horizons; do not treat the current bundle as a test of Luo & Geman's full PDE.
3. Multi-seed reruns and a paired test (Diebold–Mariano with HAC) before claiming RMSE gaps.
4. Optional data rebuild: start in 2006 or drop 30Y; impute after the split so holidays do not cross boundaries.

## References

- README: `yield-curve-geometric-sde/README.md`
- Frozen paper tables/figures: `yield-curve-geometric-sde/reports/best_model/`
