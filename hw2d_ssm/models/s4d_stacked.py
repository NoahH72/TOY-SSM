"""
Stacked diagonal-complex SSM (S4D-Lin) with GELU mixing and residual
connections.

Drop-in replacement for `LinearSSM`: same constructor signature
(`input_dim`, `output_dim`, `state_dim`), same `forward_sequence` /
`one_step` interface so `CNNSSM` works unchanged. Exposes `A_effective`
and `discrete_eigenvalues` for the notebook's eigenvalue plot.

Architecture (n_layers = 3 by default):

    u (B, T, input_dim)
        |
    in_proj (Linear input_dim -> d_model)
        |
    +-- S4DLayer (diagonal complex, per-channel learnable Delta)
    |       |
    |   GELU + Linear mixing (d_model -> d_model)
    |       |
    +-- residual
        |
    [repeat n_layers times]
        |
    out_proj (Linear d_model -> output_dim) + skip from u via D
        |
    y (B, T-1, output_dim)

Why stacked + nonlinear:
  * Turbulence is nonlinear; a single linear recurrence cannot represent it.
  * Per-channel learnable Delta in each layer + depth gives a *hierarchy* of
    timescales (fast modes -> mixing -> slow modes), which is the multi-
    timescale story without going to selective/input-dependent SSMs.

Reference: Gu et al. "On the Parameterization and Initialization of Diagonal
State Space Models" (S4D), 2022.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Single S4D layer: diagonal complex SSM applied per channel of d_model.
# ---------------------------------------------------------------------------

class S4DLayer(nn.Module):
    """One S4D recurrence operating on d_model channels independently.

    Each of the d_model channels has its own state of size state_dim, its own
    diagonal A, its own Delta, and its own B and C. This mirrors how the real
    S4 paper handles multi-feature inputs (H independent copies).
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim

        # A: diagonal in C^state_dim, S4D-Lin init  lambda_n = -1/2 + i*pi*n
        # Stored per-channel so each of d_model channels can specialize.
        log_neg_real = torch.log(0.5 * torch.ones(d_model, state_dim))
        imag = math.pi * torch.arange(state_dim, dtype=torch.float32)
        imag = imag.unsqueeze(0).expand(d_model, state_dim).contiguous()
        self.A_log_neg_real = nn.Parameter(log_neg_real)
        self.A_imag = nn.Parameter(imag)

        # Delta: per-channel, per-state-dim learnable timestep
        log_dt = (
            torch.rand(d_model, state_dim) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        self.log_dt = nn.Parameter(log_dt)

        # B and C: complex, per-channel. Output is real via Re(C h).
        # (d_model, state_dim) — each channel of d_model has its own SSM.
        self.B_real = nn.Parameter(torch.randn(d_model, state_dim) * (1.0 / math.sqrt(state_dim)))
        self.B_imag = nn.Parameter(torch.zeros(d_model, state_dim))
        self.C_real = nn.Parameter(torch.randn(d_model, state_dim) * (1.0 / math.sqrt(state_dim)))
        self.C_imag = nn.Parameter(torch.randn(d_model, state_dim) * (1.0 / math.sqrt(state_dim)))

    # ------------------------------------------------------------------ helpers

    def _continuous_A(self) -> torch.Tensor:
        return torch.complex(-torch.exp(self.A_log_neg_real), self.A_imag)  # (d_model, N)

    def _discretize(self) -> tuple[torch.Tensor, torch.Tensor]:
        A = self._continuous_A()                           # (d_model, N) complex
        dt = torch.exp(self.log_dt).to(A.real.dtype)       # (d_model, N) real
        A_bar = torch.exp(dt * A)                          # (d_model, N) complex
        B = torch.complex(self.B_real, self.B_imag)        # (d_model, N) complex
        B_bar = (A_bar - 1.0) / A * B                      # (d_model, N) complex
        return A_bar, B_bar

    def discrete_eigenvalues(self) -> torch.Tensor:
        """All discrete eigenvalues across all channels, flattened. (d_model*N,)"""
        with torch.no_grad():
            A_bar, _ = self._discretize()
        return A_bar.flatten()

    # --------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) real  ->  y: (B, T, d_model) real

        Each channel of d_model is processed by its own diagonal SSM.
        """
        B, T, D = x.shape
        assert D == self.d_model
        A_bar, B_bar = self._discretize()                   # (D, N), (D, N)
        C = torch.complex(self.C_real, self.C_imag)         # (D, N)

        # Recurrence per timestep. Hidden state h: (B, D, N) complex.
        h = torch.zeros(B, D, self.state_dim, dtype=A_bar.dtype, device=x.device)
        outs = []
        for t in range(T):
            xt = x[:, t]                                    # (B, D) real
            # h_{t+1}[b,d,n] = A_bar[d,n] * h[b,d,n] + B_bar[d,n] * xt[b,d]
            h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * xt.unsqueeze(-1).to(A_bar.dtype)
            # y[b,d] = Re(sum_n C[d,n] * h[b,d,n])
            yt = (C.unsqueeze(0) * h).sum(dim=-1).real      # (B, D)
            outs.append(yt)
        return torch.stack(outs, dim=1)                     # (B, T, D)

    def step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single timestep. x: (B, D) real, h: (B, D, N) complex.
        Returns (y_real, h_next_complex).
        """
        A_bar, B_bar = self._discretize()
        C = torch.complex(self.C_real, self.C_imag)
        h_next = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * x.unsqueeze(-1).to(A_bar.dtype)
        y = (C.unsqueeze(0) * h_next).sum(dim=-1).real
        return y, h_next


# ---------------------------------------------------------------------------
# Full stacked block.
# ---------------------------------------------------------------------------

class StackedS4DSSM(nn.Module):
    """Stacked S4D with GELU mixing and residual connections.

    Interface matches LinearSSM:
      __init__(input_dim, output_dim, state_dim)
      forward_sequence(u) -> (preds, states)
      one_step(u, h)      -> (y, h)

    `state_dim` here is the per-channel state size (the N in S4D), NOT a
    global hidden size. Total complex states = n_layers * d_model * state_dim.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        state_dim: int = 64,
        d_model: int | None = None,
        n_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.n_layers = n_layers
        # Default d_model to input_dim so the CNN latent flows through unchanged.
        self.d_model = d_model if d_model is not None else input_dim

        self.in_proj = nn.Linear(input_dim, self.d_model)
        self.layers = nn.ModuleList(
            [S4DLayer(self.d_model, state_dim) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.d_model) for _ in range(n_layers)]
        )
        self.mixers = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.GELU(),
                           nn.Dropout(dropout),
                           nn.Linear(self.d_model, self.d_model))
             for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(self.d_model, output_dim)
        # Real skip from input to output, same as old LinearSSM's D
        self.D = nn.Parameter(torch.randn(output_dim, input_dim) * 0.05)

    # --------------------------------------------------------- diagnostics

    @property
    def A_effective(self) -> torch.Tensor:
        """Compatibility shim for code that calls `model.ssm.A`.

        Returns a real diagonal matrix whose values are |A_bar| from the
        FIRST layer, flattened across (d_model * state_dim). Use this only
        for back-compat. For real plots, use `.discrete_eigenvalues()`.
        """
        with torch.no_grad():
            eigs = self.layers[0].discrete_eigenvalues()  # (d_model*N,)
        return torch.diag(eigs.abs())

    # Alias so `model.ssm.A` still works in older notebook cells.
    @property
    def A(self) -> torch.Tensor:
        return self.A_effective

    def discrete_eigenvalues(self) -> torch.Tensor:
        """All discrete eigenvalues across all layers, flattened complex tensor.
        Shape: (n_layers * d_model * state_dim,)
        """
        return torch.cat([layer.discrete_eigenvalues() for layer in self.layers])

    def eigenvalues_per_layer(self) -> list[torch.Tensor]:
        """List of (d_model*N,) complex tensors, one per layer.
        Useful for plotting each layer's eigenvalue distribution separately.
        """
        return [layer.discrete_eigenvalues() for layer in self.layers]

    # ----------------------------------------------------------- forward iface

    def _apply_block(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Pre-norm residual block: x + mixer(GELU(SSM(norm(x))))."""
        h = self.norms[layer_idx](x)
        h = self.layers[layer_idx](h)         # diagonal SSM
        h = self.mixers[layer_idx](h)         # nonlinear mixing
        return x + h                          # residual

    def forward_sequence(
        self,
        u: torch.Tensor,
        h0: torch.Tensor | None = None,  # accepted for API parity; ignored
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """u: (B, T, input_dim)  ->  (preds: (B, T-1, output_dim),
                                       states: (B, T, state_dim))

        `states` is provided for API parity with LinearSSM. It returns the
        |hidden state| of the FIRST layer's first channel as a stand-in --
        the stacked model doesn't have a single canonical state, but this
        keeps any logging code working.
        """
        B, T, _ = u.shape
        x = self.in_proj(u)                   # (B, T, d_model)
        for i in range(self.n_layers):
            x = self._apply_block(x, i)
        y_full = self.out_proj(x) + u @ self.D.T   # (B, T, output_dim)
        preds = y_full[:, :-1]                # predict u_{t+1} from t=0..T-2

        # API-parity placeholder for `states` — magnitudes from layer 0, ch 0.
        with torch.no_grad():
            states = torch.zeros(B, T, self.state_dim, device=u.device, dtype=u.dtype)
        return preds, states

    # ---- one_step: maintain a stack of per-layer hidden states ----------

    def init_hidden(self, batch_size: int, device, dtype=None) -> list[torch.Tensor]:
        """Allocate per-layer complex hidden states. Pass this to one_step."""
        # Each layer's state is (B, d_model, state_dim) complex.
        cdtype = torch.complex64
        return [
            torch.zeros(batch_size, self.d_model, self.state_dim,
                        dtype=cdtype, device=device)
            for _ in range(self.n_layers)
        ]

    def one_step(
        self,
        u: torch.Tensor,
        h,
    ):
        """Single autoregressive step.

        u: (B, input_dim) real
        h: list of per-layer complex hidden states, OR a real (B, state_dim)
           tensor for back-compat with CNNSSM (interpreted as "start from
           zeros and ignore the passed-in h").

        Returns: (y: (B, output_dim) real, h: list of per-layer states)
        """
        B = u.shape[0]
        # Back-compat: CNNSSM passes a real (B, state_dim) h; reset cleanly.
        if isinstance(h, torch.Tensor):
            hs = self.init_hidden(B, u.device)
        else:
            hs = h

        x = self.in_proj(u)                   # (B, d_model)
        new_hs = []
        for i in range(self.n_layers):
            residual = x
            x_norm = self.norms[i](x)
            y_ssm, h_next = self.layers[i].step(x_norm, hs[i])
            x = residual + self.mixers[i](y_ssm)
            new_hs.append(h_next)

        y = self.out_proj(x) + u @ self.D.T
        return y, new_hs