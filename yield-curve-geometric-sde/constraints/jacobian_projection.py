"""Decoder-Jacobian projection for geometric manifold constraints.

The decoder D(z) defines a learned yield-curve manifold:
    M = { D(z) : z in R^d }

An off-manifold curve y_tilde is any curve that is not exactly D(z) for some z.

Two usage modes:

1) Penalty mode (training, soft constraint)
   - Compute projected curve y_proj = Project(y_tilde)
   - Add loss: ||y_tilde - y_proj||^2 via manifold_projection_loss(...)
   - The optimizer learns to stay near the manifold instead of forcibly replacing outputs.

2) Explicit projection mode (inference / post-step correction)
   - Replace off-manifold output directly:
       y_tilde <- Project(y_tilde)
   - Use project_curve_to_manifold(...) after SDE decode steps when reporting paths.

Projection options:

- method="reencode" (nonlinear):
      y_proj = D(E(y_tilde))
  Uses the same Stage A encoder/decoder (not a new VAE). Encoder mean mu is used.

- method="tangent" (local linear):
      y_proj = D(z) + J(z) (J^T J + eps I)^(-1) J^T (y_tilde - D(z))
  Requires current latent z and decoder Jacobian J(z)=dD/dz.
"""

import torch


def _encode_to_latent(y, encoder):
    """Map curve batch to latent mean (deterministic projection path)."""
    if hasattr(encoder, "encode"):
        z = encoder.encode(y)
    else:
        z = encoder(y)

    if isinstance(z, tuple):
        # VAE-style encoder returns (mu, logvar); use mu for projection.
        z = z[0]
    return z


def decoder_jacobian(z, decoder):
    """Compute decoder Jacobian dD(z)/dz.

    Shapes:
    - z:        [B, latent_dim]
    - jacobian: [B, N_tenors, latent_dim]
    """
    z = z.detach().requires_grad_(True)
    y = decoder(z)
    batch_size, n_tenors = y.shape
    latent_dim = z.shape[1]
    jac = torch.zeros(batch_size, n_tenors, latent_dim, device=z.device, dtype=z.dtype)

    for tenor_idx in range(n_tenors):
        grad_z = torch.autograd.grad(
            y[:, tenor_idx].sum(),
            z,
            create_graph=True,
            retain_graph=True,
        )[0]
        jac[:, tenor_idx, :] = grad_z

    return jac


def project_delta_to_tangent(jacobian, delta_y, eps=1e-5):
    """Project curve change into decoder tangent space.

    delta_y_proj = J (J^T J + eps I)^(-1) J^T delta_y

    Shapes:
    - jacobian: [B, N_tenors, latent_dim]
    - delta_y:  [B, N_tenors]
    - output:   [B, N_tenors]
    """
    jt = jacobian.transpose(-2, -1)
    gram = jt @ jacobian
    identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).unsqueeze(0)
    inv = torch.linalg.inv(gram + eps * identity)
    proj_matrix = jacobian @ inv @ jt
    return (proj_matrix @ delta_y.unsqueeze(-1)).squeeze(-1)


def reencode_projection(y_tilde, encoder, decoder):
    """Nonlinear manifold projection: y_tilde -> E(y_tilde) -> D(E(y_tilde))."""
    z = _encode_to_latent(y_tilde, encoder)
    return decoder(z)


def project_curve_to_manifold(y_tilde, z, encoder, decoder, method="reencode", eps=1e-5):
    """Project an off-manifold curve back onto decoder manifold."""
    if method == "reencode":
        return reencode_projection(y_tilde, encoder, decoder)

    if method == "tangent":
        y_on_manifold = decoder(z)
        delta_y = y_tilde - y_on_manifold
        jac = decoder_jacobian(z, decoder)
        delta_proj = project_delta_to_tangent(jac, delta_y, eps=eps)
        return y_on_manifold + delta_proj

    raise ValueError(f"Unknown projection method: {method}")


def manifold_projection_loss(y_pred, z, encoder, decoder, method="reencode", eps=1e-5):
    """Penalty-mode Jacobian constraint for Stage B training.

    This does not overwrite y_pred; it only penalizes off-manifold distance.
    """
    y_proj = project_curve_to_manifold(
        y_pred,
        z,
        encoder,
        decoder,
        method=method,
        eps=eps,
    )
    return torch.mean((y_pred - y_proj) ** 2)
