"""Decoder-Jacobian projection scaffold for geometric constraints."""

import torch


def project_delta_to_tangent(jacobian: torch.Tensor, delta_y: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Project curve change into decoder tangent space.

    Formula:
    delta_y_proj = J (J^T J + eps I)^(-1) J^T delta_y

    Shapes:
    - jacobian: [B, N_tenors, latent_dim]
    - delta_y:  [B, N_tenors]
    - output:   [B, N_tenors]
    """
    jt = jacobian.transpose(-2, -1)
    gram = jt @ jacobian  # [B, latent_dim, latent_dim]
    i = torch.eye(gram.shape[-1], device=gram.device).unsqueeze(0)
    inv = torch.linalg.inv(gram + eps * i)
    proj_matrix = jacobian @ inv @ jt  # [B, N, N]
    return (proj_matrix @ delta_y.unsqueeze(-1)).squeeze(-1)


def reencode_projection(y_tilde: torch.Tensor, encoder, decoder) -> torch.Tensor:
    """Alternative manifold projection: y -> E(y) -> D(E(y))."""
    # TODO: decide detach policy and train/eval usage.
    z = encoder(y_tilde)
    if isinstance(z, tuple):
        z = z[0]
    return decoder(z)
