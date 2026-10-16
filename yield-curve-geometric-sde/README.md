# Geometric No-Arbitrage Yield Curve Dynamics

Two-stage yield-curve forecasting:

1. **Stage A** — Student-t CVAE manifold (LevelScript: level = 1Y, shape = curve − level)
2. **Stage B** — Neural SDE latent dynamics with optional constraint penalties

Jacobian projection is a geometric constraint on the decoder manifold, not a competing dynamics model.

## Ablations

| Ablation | Constraints |
|----------|-------------|
| `sde_only` | none (drift network; diffusion is not in the loss) |
| `sde_pde` | curve-shape bundle: simplified PDE residual + discount monotonicity + forward smoothness, trained at h=1 |
| `sde_jacobian` | ridge decoder-tangent penalty at h=1, 5, 21 |
| `sde_both` | bundle + Jacobian (frozen scorecard matches `sde_pde`; do not treat as a separate result) |

Paper-primary frozen results (soft forecasts) are in `reports/best_model/`. Retrain with `config/paper_best.yaml`.

## Data

FRED constant-maturity UST yields: `1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y`.  
Download starts `2001-07-01`; the cleaned panel begins `2001-07-31`. Chronological 70/15/15 split, train-only robust scaling, then LevelScript (1Y level). Business-day reindex fills holidays in every split. 2002–2006 30Y values in FRED are source estimates, not a hole filled by this repo.

Raw/processed data are gitignored.

## Quick start

```bash
cd yield-curve-geometric-sde
pip install -r requirements.txt

python data/download_fred_yields.py
python data/preprocess_curves.py --levelscript

python training/train_stage_a.py --config config/paper_best.yaml
python training/train_stage_b.py --config config/paper_best.yaml --ablation sde_jacobian

python evaluation/evaluate_run.py --split test --output-dir reports/comparison
```

**Colab:** open `notebooks/colab_train.ipynb` and set the runtime to **GPU**.

## Paper-primary config

Locked in `config/paper_best.yaml` (same training hyperparameters as `config/default.yaml`):

- `latent_dim=5`, `hidden_dim=256`, `epochs_stage_a=200`, `epochs_stage_b=400`
- Persistence-residual forecast: `y_pred = y_last + decode(z_pred) - decode(z_last)`
- Jacobian: `projection_method=tangent`, weight `0.3`, warmup 40 epochs
- Soft penalties during training; hard projection is eval-only and non-primary

Do not bump `latent_dim`: a larger latent space enlarges the decoder tangent plane and weakens the Jacobian constraint.

## Repo notes

Paper drafts (`.tex`, local PDFs) and unrelated notebooks stay out of git. Agent handoff lives in the repo-root `AGENTS.md`.
