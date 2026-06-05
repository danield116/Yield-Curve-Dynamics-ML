"""Constraint modules: no-arbitrage and geometric projection."""

from constraints.bond_math import (
    bond_price_from_decoder,
    default_tau_grid,
    discount_to_instant_forward,
    short_rate_from_curve,
    yield_to_discount,
)
from constraints.jacobian_projection import (
    decoder_jacobian,
    manifold_projection_loss,
    project_curve_to_manifold,
    project_delta_to_tangent,
    reencode_projection,
)
from constraints.no_arbitrage_pde import (
    arbitrage_diagnostic_loss,
    discount_monotonicity_penalty,
    forward_smoothness_penalty,
    pde_penalty_loss,
    pde_residual,
    total_constraint_loss,
)

__all__ = [
    "yield_to_discount",
    "discount_to_instant_forward",
    "short_rate_from_curve",
    "bond_price_from_decoder",
    "default_tau_grid",
    "discount_monotonicity_penalty",
    "forward_smoothness_penalty",
    "arbitrage_diagnostic_loss",
    "pde_residual",
    "pde_penalty_loss",
    "total_constraint_loss",
    "decoder_jacobian",
    "project_delta_to_tangent",
    "reencode_projection",
    "project_curve_to_manifold",
    "manifold_projection_loss",
]
