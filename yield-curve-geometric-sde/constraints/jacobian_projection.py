"""Decoder-Jacobian projection for geometric manifold constraints."""

import torch


def _encode_to_latent(y, encoder=None, encode_fn=None):
    """Map curve batch to latent mean (deterministic projection path)."""
    if encode_fn is not None:
        return encode_fn(y)

    if encoder is None:
        raise ValueError("Provide encode_fn or encoder for manifold projection.")

    if hasattr(encoder, "encode"):
        z = encoder.encode(y)
    else:
        z = encoder(y)

    if isinstance(z, tuple):
        z = z[0]
    return z


def decoder_jacobian(z, decoder):
    """Compute decoder Jacobian dD(z)/dz."""
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
    """Project curve change into decoder tangent space."""
    jt = jacobian.transpose(-2, -1)
    gram = jt @ jacobian
    identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).unsqueeze(0)
    inv = torch.linalg.inv(gram + eps * identity)
    proj_matrix = jacobian @ inv @ jt
    return (proj_matrix @ delta_y.unsqueeze(-1)).squeeze(-1)


def reencode_projection(y_tilde, encoder=None, decoder=None, encode_fn=None, decode_fn=None):
    """Nonlinear manifold projection: y_tilde -> E(y_tilde) -> D(E(y_tilde))."""
    if encode_fn is not None and decode_fn is not None:
        return decode_fn(encode_fn(y_tilde))

    z = _encode_to_latent(y_tilde, encoder=encoder)
    return decoder(z)


def project_curve_to_manifold(
    y_tilde,
    z,
    encoder=None,
    decoder=None,
    encode_fn=None,
    decode_fn=None,
    method="reencode",
    eps=1e-5,
):
    """Project an off-manifold curve back onto decoder manifold."""
    if method == "reencode":
        return reencode_projection(
            y_tilde,
            encoder=encoder,
            decoder=decoder,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
        )

    if method == "tangent":
        decode = decode_fn if decode_fn is not None else decoder
        y_on_manifold = decode(z)
        delta_y = y_tilde - y_on_manifold
        jac = decoder_jacobian(z, decode)
        delta_proj = project_delta_to_tangent(jac, delta_y, eps=eps)
        return y_on_manifold + delta_proj

    raise ValueError(f"Unknown projection method: {method}")


def manifold_projection_loss(
    y_pred,
    z,
    encoder=None,
    decoder=None,
    encode_fn=None,
    decode_fn=None,
    method="reencode",
    eps=1e-5,
):
    """Penalty-mode Jacobian constraint for Stage B training."""
    y_proj = project_curve_to_manifold(
        y_pred,
        z,
        encoder=encoder,
        decoder=decoder,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        method=method,
        eps=eps,
    )
    return torch.mean((y_pred - y_proj) ** 2)
