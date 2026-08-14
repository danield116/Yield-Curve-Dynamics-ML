# Geometric No-Arbitrage Yield Curve Dynamics

Two-stage yield-curve forecasting:

1. **Stage A** — Student-t CVAE manifold (LevelScript: level = 1Y, shape = curve − level)
2. **Stage B** — Neural SDE latent dynamics with optional constraint penalties

Jacobian projection is a geometric constraint on the decoder manifold, not a competing dynamics model.

## Ablations

| Ablation | Constraints |
|----------|-------------|
| `sde_only` | none |
| `sde_pde` | no-arbitrage PDE residual + discount/forward diagnostics |
| `sde_jacobian` | decoder-Jacobian manifold projection |
| `sde_both` | PDE + Jacobian |

Paper-primary frozen results (soft forecasts) are in `reports/best_model/`. Retrain with `config/paper_best.yaml`.

## Data

FRED constant-maturity UST yields: `1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y`.  
Default start: `2001-07-01`. Chronological 70/15/15 split, train-only robust scaling.

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
