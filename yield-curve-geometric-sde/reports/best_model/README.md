# Current Best Model (paper primary)

Frozen **soft-forecast** evaluation from the locked config run where
`sde_jacobian` edged `sde_only` at trained horizons **h=1, 5, 21**.

Use this folder directly for the paper if later retrains do not beat it.
Checkpoints are **not** in git (Colab-only); tables and plots here are the
canonical reference.

## Config

See `config.yaml` (same as `config/paper_best.yaml`):
`latent_dim=5`, `hidden_dim=256`, `kl_weight=0.03`, `epochs_stage_a=200`,
`epochs_stage_b=400`, `jacobian_projection_weight=0.3`, `projection_method=tangent`.

## Key soft test RMSE

| Horizon | sde_jacobian | sde_only | persistence |
|--------|-------------|----------|-------------|
| h=1 | **0.029167** | 0.029175 | 0.029119 |
| h=5 | **0.062944** | 0.063010 | 0.062330 |
| h=21 | **0.122123** | 0.122173 | 0.123750 |
| h=63 | 0.246372 | **0.236540** | 0.202391 |

Stage A reconstruction RMSE: **0.659**

## Contents

| Path | Description |
|------|-------------|
| `scorecard.csv` / `scorecard.json` | Full test scorecard (soft only) |
| `constraint_ablation_h{1,5,21}.csv` | Stage B geometry / arbitrage tables |
| `figures/` | Bar charts + RMSE-vs-horizon PNGs |

## Updating this folder

Only replace after a retrain that clearly beats these numbers on the same
metrics. Regenerate from a new scorecard CSV:

```bash
python evaluation/plot_from_scorecard.py \
  --scorecard path/to/new_scorecard.csv \
  --output-dir reports/best_model
```
