"""
Selective scan — the core of Mamba's S6 layer.

Continuous-time SSM (per channel d):
    h'(t) = A h(t) + B x(t)
    y(t)  = C h(t) + D x(t)

Zero-order hold discretization with step size Δ_t (input-dependent in Mamba):
    Ā_t = exp(Δ_t * A)
    B̄_t = Δ_t * B_t          (simplified; matches the reference Mamba implementation)
    h_t = Ā_t h_{t-1} + B̄_t x_t
    y_t = C_t^T h_t + D x_t

When z is provided (Mamba block), output is gated: y_t * silu(z_t).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    *,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
    return_last_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Reference selective scan (sequential loop, easy to read).

    Shapes:
        u:     (B, D, L)  — input x after conv
        delta: (B, D, L)  — discretization step (before softplus + bias)
        A:     (D, N)     — state matrix (negative, diagonal in Mamba init)
        B:     (B, N, L)  — input matrix (input-dependent / selective)
        C:     (B, N, L)  — readout matrix (input-dependent)
        D:     (D,)       — skip connection
        z:     (B, D, L)  — gate branch (optional)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()

    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, seqlen = u.shape
    dstate = A.shape[1]

    # h: (B, D, N)
    h = torch.zeros(batch, dim, dstate, device=u.device, dtype=u.dtype)
    ys: list[torch.Tensor] = []

    # Precompute discretized dynamics for all timesteps
    # deltaA[b,d,l,n] = exp(delta[b,d,l] * A[d,n])
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    # deltaB_u[b,d,l,n] = delta[b,d,l] * B[b,n,l] * u[b,d,l]
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)

    for i in range(seqlen):
        h = deltaA[:, :, i] * h + deltaB_u[:, :, i]
        # y[b,d] = sum_n h[b,d,n] * C[b,n,i]
        y = torch.einsum("bdn,bn->bd", h, C[:, :, i])
        ys.append(y)

    y = torch.stack(ys, dim=2)  # (B, D, L)

    if D is not None:
        y = y + u * D.view(1, -1, 1)

    if z is not None:
        y = y * F.silu(z.float())

    y = y.to(dtype=dtype_in)

    if return_last_state:
        return y, h.to(dtype=dtype_in)
    return y
