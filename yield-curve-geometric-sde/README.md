# Geometric No-Arbitrage Yield Curve Dynamics

Backbone repository for a quantitative finance / machine learning project:

**Geometric No-Arbitrage Yield Curve Dynamics with Student-t CVAEs, Neural SDEs, and Jacobian Projection**

## Scope

This scaffold is intentionally lightweight and pseudocode-oriented.  
It contains:
- module layout and research pipeline structure,
- function/class skeletons,
- docstrings, TODO blocks, and tensor shape comments.

It does **not** contain full production implementations yet.

## Research Objective

Model and forecast yield-curve dynamics with:
1. Stage A: latent manifold learning with VAE/CVAE variants (including Student-t + LevelScript),
2. Stage B: latent continuous-time dynamics with Neural SDE,
3. constraint ablations:
   - unconstrained SDE,
   - SDE + no-arbitrage PDE penalty,
   - SDE + decoder-Jacobian projection,
   - SDE + PDE + Jacobian.

## Data (MVP)

FRED constant-maturity treasury yields:
`1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y`

## Milestones

1. Load/simulate yields, train basic VAE, evaluate reconstruction.
2. Add PCA + VAR baseline.
3. Add Student-t VAE and LevelScript.
4. Add latent Neural SDE dynamics.
5. Add no-arbitrage diagnostics/PDE penalty.
6. Add Jacobian projection and run full ablation.

## Quick Start

```bash
pip install -r requirements.txt
python experiments/run_full_comparison.py --config config/default.yaml
```

## Notes

- The initial files are pseudocode-heavy for planning and rapid iteration.
- Replace TODO blocks with concrete implementations incrementally.
